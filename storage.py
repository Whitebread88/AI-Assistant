from google.cloud import storage
import os

BUCKET_NAME = os.environ.get("KNOWLEDGE_BUCKET")

storage_client = storage.Client()

def load_category_file(category: str) -> str:
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(f"knowledge/{category}.txt")
    try:
        return blob.download_as_text()
    except Exception:
        print(f"Warning: {category}.txt not found in GCS")
        return ""  # return empty string if file missing
