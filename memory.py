from datetime import datetime

from google.cloud import firestore

MAX_TURNS = 8


def _db():
    return firestore.Client(database="conversation-metadata")


def get_session(session_id: str):
    doc = _db().collection("chats").document(session_id).get()
    if doc.exists:
        data = doc.to_dict()
        history = data.get("history") or data.get("messages") or []
        return {
            "history": history,
            "summary": data.get("summary", ""),
            "last_category": data.get("last_category", ""),
        }
    return {"history": [], "summary": "", "last_category": ""}


def save_session(session_id: str, history, summary, last_category):
    _db().collection("chats").document(session_id).set({
        "history": history,
        "summary": summary,
        "last_category": last_category,
        "updated_at": datetime.utcnow()
    })


def trim_history(history):
    return history[-MAX_TURNS:]
