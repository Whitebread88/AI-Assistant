import os
from google.cloud import storage

from constants import CATEGORIES

BUCKET_NAME = os.environ.get("KNOWLEDGE_BUCKET")

storage_client = storage.Client()


def load_category_file(category: str) -> str:
    if not BUCKET_NAME:
        return ""

    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(f"knowledge/{category}.txt")
    try:
        return blob.download_as_text()
    except Exception:
        print(f"Warning: {category}.txt not found in GCS")
        return ""


def load_relevant_knowledge(question: str, category: str | None = None) -> str:
    """Loads relevant knowledge.

    If category is provided, only that category is loaded.
    Otherwise, performs a lightweight keyword match across categories.
    """
    categories = [category] if category in CATEGORIES else CATEGORIES

    combined_context = ""
    for current_category in categories:
        file_text = load_category_file(current_category)
        if not file_text:
            continue

        if category or any(word.lower() in file_text.lower() for word in question.split()):
            combined_context += f"\n--- {current_category} ---\n{file_text}\n"

    return combined_context
