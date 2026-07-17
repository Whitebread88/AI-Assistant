from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from google.cloud import firestore

MAX_TURNS = 4
_db_client: firestore.Client | None = None

@dataclass
class SessionState:
    history: list[dict[str, str]]
    summary: str
    active_categories: list[str]
    current_article: str | None = None
    related_articles: list[str] = field(default_factory=list)


def _db() -> firestore.Client:
    global _db_client
    if _db_client is None:
        _db_client = firestore.Client(database="conversation-metadata")
    return _db_client


def _normalize_categories(data: dict[str, Any]) -> list[str]:
    categories = data.get("active_categories") or []
    if categories:
        return [str(category) for category in categories]

    legacy_last_category = data.get("last_category", "")
    return [legacy_last_category] if legacy_last_category else []


def get_session(session_id: str) -> SessionState:
    doc = _db().collection("chats").document(session_id).get()
    if not doc.exists:
        return SessionState(history=[], summary="", active_categories=[])

    data: dict[str, Any] = doc.to_dict() or {}
    history = data.get("history") or data.get("messages") or []

    return SessionState(
        history=history,
        summary=data.get("summary", ""),
        active_categories=_normalize_categories(data),
        current_article=data.get("current_article") or None,
        related_articles=[str(slug) for slug in data.get("related_articles") or []],
    )


def save_session(session_id: str, session: SessionState) -> None:
    _db().collection("chats").document(session_id).set(
        {
            "history": session.history,
            "summary": session.summary,
            "active_categories": session.active_categories,
            "current_article": session.current_article,
            "related_articles": session.related_articles,
            "updated_at": datetime.utcnow(),
        }
    )


def trim_history(history: list[dict[str, str]]) -> list[dict[str, str]]:
    return history[-MAX_TURNS:]
