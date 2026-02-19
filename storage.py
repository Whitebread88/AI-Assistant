from google.cloud import storage
import os

BUCKET_NAME = os.environ.get("KNOWLEDGE_BUCKET")

storage_client = storage.Client()

def load_category_file(category: str) -> str:
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(f"aaron-knowledge/{category}.txt")
    return blob.download_as_text()
