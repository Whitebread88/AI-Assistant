from fastapi import Header, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import os
from fastapi.middleware.cors import CORSMiddleware
from llm import classify_categories, generate_answer, summarize_conversation
from memory import get_session, save_session, trim_history
from storage import load_relevant_knowledge
from slowapi import Limiter
from slowapi.middleware import SlowAPIMiddleware
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from pydantic import BaseModel, constr

app = FastAPI()


def get_real_ip(request: Request):
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0]
    return request.client.host
limiter = Limiter(key_func=get_real_ip)
app.state.limiter = limiter

app.add_middleware(SlowAPIMiddleware)

EXPECTED_SECRET = os.getenv("INTERNAL_CHAT_SECRET")

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
    message: constr(min_length=1, max_length=2000)

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

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request, exc):
    return JSONResponse(status_code=429, content={"detail": "Too many requests"})

@app.post("/chat")
@limiter.limit("7/minute")
async def chat(request: Request, data: ChatRequest, x_internal_secret: str = Header(None)):
    if x_internal_secret != EXPECTED_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden")
    session_id = data.session_id.strip()
    message = data.message.strip()

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

    # Reduce end-to-end latency by refreshing summary every other assistant turn.
    assistant_turns = sum(1 for item in session.history if item.get("role") == "assistant")
    if assistant_turns % 2 == 0 or not session.summary:
        session.summary = summarize_conversation(session.history)

    save_session(session_id, session)
    return {
        "answer": answer,
        "categories": session.active_categories,
    }
