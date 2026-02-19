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
        return {"history": history, "summary": data.get("summary", "")}
    return {"history": [], "summary": ""}


def save_session(session_id: str, history, summary):
    _db().collection("chats").document(session_id).set({
        "history": history,
        "summary": summary,
        "updated_at": datetime.utcnow()
    })


def trim_history(history):
    return history[-MAX_TURNS:]
