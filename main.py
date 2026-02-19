from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from llm import classify_question, generate_answer, summarize_conversation
from memory import get_session, save_session, trim_history
from storage import load_relevant_knowledge

app = FastAPI()


class ChatRequest(BaseModel):
    session_id: str
    message: str


def resolve_category(message: str, last_category: str) -> str:
    """Resolve category with follow-up continuity.

    If current message is irrelevant but we already have an active session category,
    continue using the previous category so short follow-ups still work.
    """
    category = classify_question(message)
    if category == "irrelevant" and last_category:
        return last_category
    return category


@app.get("/")
def root():
    return {"status": "running", "message": "Chatbot API is live"}


@app.post("/chat")
async def chat(req: ChatRequest):
    if not req.session_id.strip():
        raise HTTPException(status_code=400, detail="Missing session_id")
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Missing message")

    session = get_session(req.session_id)
    history = session.get("history", [])
    summary = session.get("summary", "")
    last_category = session.get("last_category", "")

    history.append({"role": "user", "content": req.message})

    category = resolve_category(req.message, last_category)
    context = ""
    if category != "irrelevant":
        context = load_relevant_knowledge(req.message, category=category)

    answer = generate_answer(
        question=req.message,
        context=context,
        summary=summary,
        history=history,
    )

    history.append({"role": "assistant", "content": answer})
    history = trim_history(history)

    summary = summarize_conversation(history)
    persisted_category = category if category != "irrelevant" else last_category
    save_session(req.session_id, history, summary, persisted_category)

    return {"answer": answer, "category": category}
