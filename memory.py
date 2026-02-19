from google.cloud import firestore
from datetime import datetime

db = firestore.Client()

MAX_TURNS = 8

def get_session(session_id: str):
    doc = db.collection("chats").document(session_id).get()
    if doc.exists:
        return doc.to_dict()
    return {"messages": [], "summary": ""}

def save_session(session_id: str, messages, summary):
    db.collection("chats").document(session_id).set({
        "messages": messages,
        "summary": summary,
        "updated_at": datetime.utcnow()
    })

def trim_history(messages):
    return messages[-MAX_TURNS:]
