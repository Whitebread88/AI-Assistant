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
        
def load_relevant_knowledge(question):
    combined_context = ""

    for category in CATEGORIES:
        file_text = load_category_file(category)
        if any(word.lower() in file_text.lower() for word in question.split()):
            combined_context += f"\n--- {category} ---\n{file_text}\n"

    return combined_context
