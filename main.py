from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from storage import load_category_file
from memory import get_session, save_session, trim_history
from llm import classify_question, generate_answer, summarize_conversation

app = FastAPI()

class ChatRequest(BaseModel):
    session_id: str
    message: str

@app.post("/chat")
async def chat(req: ChatRequest):
    if not req.session_id:
        raise HTTPException(status_code=400, detail="Missing session_id")

    # Load session
    session = get_session(req.session_id)
    history = session["messages"]
    summary = session["summary"]

    # Step 1: Classify
    category = classify_question(req.message)

    # Step 2: Load relevant knowledge
    context = load_category_file(category)

    # Step 3: Generate answer
    trimmed_history = trim_history(history)
    answer = generate_answer(req.message, context, summary, trimmed_history)

    # Step 4: Update history
    history.append({"role": "user", "content": req.message})
    history.append({"role": "assistant", "content": answer})

    # Optional summarization trigger
    if len(history) > 12:
        summary = summarize_conversation(history)
        history = trim_history(history)

    save_session(req.session_id, history, summary)

    return {
        "reply": answer,
        "category_used": category
    }
