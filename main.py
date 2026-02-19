from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from storage import load_category_file
from memory import get_session, save_session, trim_history
from llm import classify_question, generate_answer, summarize_conversation

app = FastAPI()

class ChatRequest(BaseModel):
    session_id: str
    message: str


@app.get("/")
def root():
    return {"status": "running", "message": "Chatbot API is live"}


@app.post("/chat")
async def chat(req: ChatRequest):
    if not req.session_id:
        raise HTTPException(status_code=400, detail="Missing session_id")

    # 1️⃣ Load session
    session = get_session(req.session_id)
    history = session.get("history", [])
    summary = session.get("summary", "")
    
    # 2️⃣ Append user message first
    history.append({"role": "user", "content": req.message})
    
    # 3️⃣ Only classify if needed
    use_knowledge = True
    
    # If this is not the first message, allow conversational mode
    if len(history) > 1:
        use_knowledge = False
    
    if use_knowledge:
        context = load_relevant_knowledge(req.message)
    else:
        context = ""  # no strict knowledge constraint
    
    # 4️⃣ Generate answer
    answer = generate_answer(
        question=req.message,
        context=context,
        summary=summary,
        history=history
    )
    
    # 5️⃣ Append assistant reply
    history.append({"role": "assistant", "content": answer})
    
    # 6️⃣ Summarize and save
    summary = summarize_conversation(history)
    save_session(req.session_id, history, summary)
    
    return {"answer": answer}

