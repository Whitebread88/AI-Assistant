import os
from google.cloud import storage

BUCKET_NAME = os.environ.get("KNOWLEDGE_BUCKET")
storage_client = storage.Client()


def _load_category_knowledge(category: str) -> str:
    if not BUCKET_NAME or not category:
        return ""

    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(f"knowledge/{category}.txt")

    try:
        return blob.download_as_text()
    except Exception:
        print(f"Warning: knowledge/{category}.txt not found in GCS")
        return ""


def load_relevant_knowledge(categories: list[str]) -> str:
    """Load and merge knowledge for all relevant categories."""
    sections: list[str] = []
    for category in categories:
        text = _load_category_knowledge(category)
        if text:
            sections.append(f"--- {category} ---\n{text}")
    return "\n\n".join(sections)
