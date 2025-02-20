import asyncio
from fastapi import FastAPI, UploadFile, File, HTTPException
from azure.storage.blob import BlobServiceClient, BlobSasPermissions, generate_blob_sas
from datetime import datetime, timedelta, timezone
from fastapi.responses import StreamingResponse
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv
import os
import io
import logging
import csv

app = FastAPI()
load_dotenv()

AZURE_CONNECTION_STRING = os.getenv("AZURE_CONNECTION_STRING")
blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
CONTAINER_NAME = "datasets"
container_client = blob_service_client.get_container_client(CONTAINER_NAME)

# Utility function to handle versioning
def get_versioned_filename(filename: str, version: str):
    name, ext = filename.rsplit(".", 1)  # Split at the last dot to separate name and extension
    return f"{name}_v{version}.{ext}"

@app.get("/list/")
async def list_datasets():
    """List all datasets stored in the container"""
    try:
        blobs = container_client.list_blobs()
        return {"datasets": [blob.name for blob in blobs]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get-all/{filename}")
async def stream_dataset(filename: str, version: str = "1"):
    """Stream dataset content from Azure Blob Storage (CSV, JSONL, or JSON)"""
    try:
        versioned_filename = get_versioned_filename(filename, version)
        blob_client = container_client.get_blob_client(versioned_filename)
        
        # Stream the dataset
        stream = blob_client.download_blob()
        content = stream.readall().decode("utf-8")
        ext = filename.split(".")[-1].lower()
        if ext == "csv":
            csv_reader = csv.DictReader(io.StringIO(content))
            columns = csv_reader.fieldnames
            rows = [row for row in csv_reader]
            return {"filename": versioned_filename, "type": "csv", "columns": columns, "rows": rows}
        
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{filename}")
async def generate_download_link(filename: str, version: str = "1"):
    """Generate a SAS link to download the dataset"""
    try:
        versioned_filename = get_versioned_filename(filename, version)
        blob_client = container_client.get_blob_client(versioned_filename)
        
        # Generate a SAS token with read permissions for 1 hour
        sas_token = generate_blob_sas(
            account_name=blob_service_client.account_name,
            container_name=CONTAINER_NAME,
            blob_name=versioned_filename,
            account_key=blob_service_client.credential.account_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.now(timezone.utc) + timedelta(hours=1)
        )
        
        return {"download_url": f"{blob_client.url}?{sas_token}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/delete/{filename}")
async def delete_dataset(filename: str, version: str = "1"):
    """Delete a specific version of a dataset"""
    try:
        versioned_filename = get_versioned_filename(filename, version)
        blob_client = container_client.get_blob_client(versioned_filename)
        blob_client.delete_blob()
        return {"message": "Dataset deleted successfully", "filename": versioned_filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload/")
async def upload_dataset(file: UploadFile = File(...), version: str = "1"):
    """Upload dataset to Azure Blob Storage with versioning"""
    try:
        filename = get_versioned_filename(file.filename, version)
        blob_client = container_client.get_blob_client(filename)
        
        ext = file.filename.split(".")[-1].lower()
        if ext != "csv":  # Add "json" to supported formats
            raise HTTPException(status_code=400, detail="Unsupported file format. Only CSV is allowed.")

        # Read file content
        content = await file.read()

        # Upload file to Azure Blob Storage
        blob_client.upload_blob(content, overwrite=True)
        
        return {"message": "Upload successful", "filename": filename, "version": version}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def generate_stream(stream):
    buffer = io.BytesIO()
    for chunk in stream.chunks():
        buffer.write(chunk)
        buffer.seek(0)
        while True:
            line = buffer.readline()
            if not line:
                break
            yield line
            await asyncio.sleep(1)

        
@app.get("/stream/{filename}")
async def stream_dataset(filename: str, version: str = "1"):
    """Stream dataset content from Azure Blob Storage (CSV, JSONL, or JSON)"""
    try:
        versioned_filename = get_versioned_filename(filename, version)
        blob_client = container_client.get_blob_client(versioned_filename)
        
        stream = blob_client.download_blob()
        ext = filename.split(".")[-1].lower()

        if ext == "csv":
            return StreamingResponse(content=generate_stream(stream))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format.")

    except Exception as e: 
        logging.exception("ERR")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/fetch")
async def fetch_and_store_dataset(repo_id: str, filename: str, version: str = "1"):
    """Fetch a dataset from Hugging Face and upload it to Azure Blob Storage"""
    try:
        # Download the dataset from Hugging Face
        file_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")

        ext = filename.split(".")[-1].lower()
        versioned_filename = get_versioned_filename(filename, version)
        blob_client = container_client.get_blob_client(versioned_filename)

        if ext == "csv":
            with open(file_path, "r", encoding="utf-8") as f:
                data = f.read()
            blob_client.upload_blob(data, overwrite=True)   
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format.")

        return {"message": f"Dataset {filename} fetched and uploaded as {versioned_filename}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
