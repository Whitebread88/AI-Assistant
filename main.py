from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from llm import classify_question, generate_answer, summarize_conversation
from memory import get_session, save_session, trim_history
from storage import load_relevant_knowledge

app = FastAPI()


class ChatRequest(BaseModel):
    session_id: str
    message: str


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

    history.append({"role": "user", "content": req.message})

    category = classify_question(req.message)
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
    save_session(req.session_id, history, summary)

    return {"answer": answer, "category": category}
