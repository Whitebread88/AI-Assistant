import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

from google.cloud import storage

BUCKET_NAME = os.environ.get("KNOWLEDGE_BUCKET")
_storage_client: storage.Client | None = None


def _get_storage_client() -> storage.Client:
    global _storage_client
    if _storage_client is None:
        _storage_client = storage.Client()
    return _storage_client


@lru_cache(maxsize=64)
def _load_category_knowledge(category: str) -> str:
    if not BUCKET_NAME or not category:
        return ""

    bucket = _get_storage_client().bucket(BUCKET_NAME)
    blob = bucket.blob(f"knowledge/{category}.txt")

    try:
        return blob.download_as_text()
    except Exception:
        print(f"Warning: knowledge/{category}.txt not found in GCS")
        return ""


def load_relevant_knowledge(categories: list[str]) -> str:
    """Load and merge knowledge for all relevant categories."""
    unique_categories = list(dict.fromkeys(category for category in categories if category))
    sections: list[str] = []

    if len(unique_categories) <= 1:
        for category in unique_categories:
            text = _load_category_knowledge(category)
            if text:
                sections.append(f"--- {category} ---\n{text}")
        return "\n\n".join(sections)

    with ThreadPoolExecutor(max_workers=min(4, len(unique_categories))) as pool:
        texts = list(pool.map(_load_category_knowledge, unique_categories))

    for category, text in zip(unique_categories, texts):
        if text:
            sections.append(f"--- {category} ---\n{text}")
    return "\n\n".join(sections)
