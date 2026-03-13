from dataclasses import dataclass
from datetime import datetime
from typing import Any

from google.cloud import firestore

MAX_TURNS = 4


@dataclass
class SessionState:
    history: list[dict[str, str]]
    summary: str
    active_categories: list[str]


def _db() -> firestore.Client:
    return firestore.Client(database="conversation-metadata")


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
    )


def save_session(session_id: str, session: SessionState) -> None:
    _db().collection("chats").document(session_id).set(
        {
            "history": session.history,
            "summary": session.summary,
            "active_categories": session.active_categories,
            "updated_at": datetime.utcnow(),
        }
    )


def trim_history(history: list[dict[str, str]]) -> list[dict[str, str]]:
    return history[-MAX_TURNS:]
