from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from llm import classify_categories, generate_answer, summarize_conversation
from memory import get_session, save_session, trim_history
from storage import load_relevant_knowledge

app = FastAPI()

# Your Vercel & Local URLs
origins = [
    "https://www.aaronfoong.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    session_id: str
    message: str


def _resolve_active_categories(message: str, previous: list[str]) -> list[str]:
    """Detect categories for this turn and preserve multi-turn context.

    - If current turn has categories, merge them with previously active categories.
    - If current turn has no category, keep previously active categories.
    """
    current = classify_categories(message)
    if not current:
        return previous

    merged = list(previous)
    for category in current:
        if category not in merged:
            merged.append(category)
    return merged


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "running", "message": "Chatbot API is live"}


@app.post("/chat")
async def chat(req: ChatRequest) -> dict:
    session_id = req.session_id.strip()
    message = req.message.strip()

    if not session_id:
        raise HTTPException(status_code=400, detail="Missing session_id")
    if not message:
        raise HTTPException(status_code=400, detail="Missing message")

    session = get_session(session_id)
    session.history.append({"role": "user", "content": message})

    session.active_categories = _resolve_active_categories(
        message,
        session.active_categories,
    )
    context = load_relevant_knowledge(session.active_categories)

    answer = generate_answer(
        question=message,
        context=context,
        summary=session.summary,
        history=session.history,
    )

    session.history.append({"role": "assistant", "content": answer})
    session.history = trim_history(session.history)
    session.summary = summarize_conversation(session.history)

    save_session(session_id, session)
    return {
        "answer": answer,
        "categories": session.active_categories,
    }
