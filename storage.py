import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Any

from google.cloud import storage

BUCKET_NAME = os.environ.get("KNOWLEDGE_BUCKET")

ARTICLE_PREFIX = "knowledge/articles/"
ARTICLE_INDEX_BLOB = f"{ARTICLE_PREFIX}index.json"
# Articles get edited and the index grows, so cache with a TTL instead of the
# process-lifetime lru_cache used for the (static) category files.
ARTICLE_CACHE_TTL_SECONDS = float(os.getenv("ARTICLE_CACHE_TTL_SECONDS", "600"))
ARTICLE_MAX_CHARS = int(os.getenv("ARTICLE_MAX_CHARS", "30000"))

_storage_client: storage.Client | None = None
_article_cache: dict[str, tuple[float, str]] = {}
_index_cache: tuple[float, list[dict[str, Any]]] | None = None


def _get_storage_client() -> storage.Client:
    global _storage_client
    if _storage_client is None:
        _storage_client = storage.Client()
    return _storage_client


def _download_text(blob_path: str) -> str:
    if not BUCKET_NAME or not blob_path:
        return ""

    bucket = _get_storage_client().bucket(BUCKET_NAME)
    try:
        return bucket.blob(blob_path).download_as_text()
    except Exception:
        print(f"Warning: {blob_path} not found in GCS")
        return ""


@lru_cache(maxsize=64)
def _load_category_knowledge(category: str) -> str:
    if not category:
        return ""
    return _download_text(f"knowledge/{category}.txt")


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


def load_article_text(slug: str) -> str:
    if not slug:
        return ""

    now = time.monotonic()
    cached = _article_cache.get(slug)
    if cached and now - cached[0] < ARTICLE_CACHE_TTL_SECONDS:
        return cached[1]

    text = _download_text(f"{ARTICLE_PREFIX}{slug}.txt")[:ARTICLE_MAX_CHARS]
    _article_cache[slug] = (now, text)
    return text


def load_article_index() -> list[dict[str, Any]]:
    """Load the article catalog (slug/title/summary entries) used for selection."""
    global _index_cache

    now = time.monotonic()
    if _index_cache and now - _index_cache[0] < ARTICLE_CACHE_TTL_SECONDS:
        return _index_cache[1]

    entries: list[dict[str, Any]] = []
    raw = _download_text(ARTICLE_INDEX_BLOB)
    if raw:
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                entries = [
                    entry for entry in data
                    if isinstance(entry, dict) and entry.get("slug")
                ]
        except json.JSONDecodeError:
            print(f"Warning: {ARTICLE_INDEX_BLOB} is not valid JSON")

    _index_cache = (now, entries)
    return entries


def build_article_context(current_slug: str | None, related_slugs: list[str]) -> str:
    """Load and label article texts, current article first."""
    slugs: list[str] = []
    for slug in [current_slug, *related_slugs]:
        if slug and slug not in slugs:
            slugs.append(slug)
    if not slugs:
        return ""

    entries = {entry["slug"]: entry for entry in load_article_index()}

    sections: list[str] = []
    for slug in slugs:
        text = load_article_text(slug)
        if not text:
            continue
        entry = entries.get(slug, {})
        label = f"Article: {entry.get('title') or slug}"
        if slug == current_slug:
            label += " (the reader is currently viewing this article)"
        url = entry.get("url", "")
        link_line = f"Link: {url}\n" if url else ""
        sections.append(f"--- {label} ---\n{link_line}{text}")
    return "\n\n".join(sections)
