from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
# Import your logic from the services folder
from services.document_extraction import extract_document_from_file, get_document, get_document_details, update_document_details

app = FastAPI()

# Your Vercel & Local URLs
origins = [
    "https://ltb-ocr.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def health_check():
    return {"status": "online", "message": "LTB OCR Backend is running"}

@app.post("/process-document")
async def extract_document_info(file: UploadFile = File(...)):
    # 1. Create a temporary path
    # temp_path = f"tmp_{file.filename}"
    filename = os.path.basename(file.filename)
    temp_path = os.path.join("/tmp", f"tmp_{filename}")
    
    try:
        # 2. Save file temporarily to disk (better for big files than keeping in RAM)
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 3. Call your processing script
        result = extract_document_from_file(temp_path, filename)
        
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # 4. Clean up the temp file so your Cloud Run doesn't run out of space
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
@app.get("/history-document")
async def list_document():
    return get_document()

@app.get("/history-document/{folder_id}")
async def list_document_details(folder_id: str):
    return get_document_details(folder_id)

@app.put("/edit-document/{folder_id}")
async def edit_document_details(folder_id: str, updated_data: dict = Body(...)):
    return update_document_details(folder_id, updated_data)
