import json
import os

from fastapi import Header, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, constr
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from llm import classify_categories, generate_answer, stream_answer_tokens, summarize_conversation
from memory import get_session, save_session, trim_history
from storage import load_relevant_knowledge

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

origins = ["https://www.aaronfoong.com"]

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


def _format_sse(event: str, data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def _resolve_active_categories(message: str, previous: list[str]) -> list[str]:
    current = classify_categories(message)
    if not current:
        return previous

    merged = list(previous)
    for category in current:
        if category not in merged:
            merged.append(category)
    return merged


def _update_summary(session_history: list[dict[str, str]], previous_summary: str) -> str:
    assistant_turns = sum(1 for item in session_history if item.get("role") == "assistant")
    if assistant_turns % 2 == 0 or not previous_summary:
        return summarize_conversation(session_history)
    return previous_summary


def _validate_chat_request(data: ChatRequest, x_internal_secret: str) -> tuple[str, str]:
    if x_internal_secret != EXPECTED_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden")

    session_id = data.session_id.strip()
    message = data.message.strip()

    if not session_id:
        raise HTTPException(status_code=400, detail="Missing session_id")
    if not message:
        raise HTTPException(status_code=400, detail="Missing message")

    return session_id, message


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "running", "message": "Chatbot API is live"}


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request, exc):
    return JSONResponse(status_code=429, content={"detail": "Too many requests"})


@app.post("/chat")
@limiter.limit("7/minute")
async def chat(request: Request, data: ChatRequest, x_internal_secret: str = Header(None)):
    session_id, message = _validate_chat_request(data, x_internal_secret)

    session = get_session(session_id)
    session.history.append({"role": "user", "content": message})

    session.active_categories = _resolve_active_categories(message, session.active_categories)
    context = load_relevant_knowledge(session.active_categories)

    answer = generate_answer(
        question=message,
        context=context,
        summary=session.summary,
        history=session.history,
    )

    session.history.append({"role": "assistant", "content": answer})
    session.history = trim_history(session.history)
    session.summary = _update_summary(session.history, session.summary)

    save_session(session_id, session)
    return {
        "answer": answer,
        "categories": session.active_categories,
    }


@app.post("/chat/stream")
@limiter.limit("7/minute")
async def chat_stream(request: Request, data: ChatRequest, x_internal_secret: str = Header(None)):
    session_id, message = _validate_chat_request(data, x_internal_secret)

    session = get_session(session_id)
    session.history.append({"role": "user", "content": message})

    session.active_categories = _resolve_active_categories(message, session.active_categories)
    context = load_relevant_knowledge(session.active_categories)

    def event_generator():
        full_answer_parts: list[str] = []
        try:
            yield _format_sse("metadata", {"categories": session.active_categories})
            for token in stream_answer_tokens(
                question=message,
                context=context,
                summary=session.summary,
                history=session.history,
            ):
                full_answer_parts.append(token)
                yield _format_sse("token", {"text": token})

            answer = "".join(full_answer_parts).strip()
            session.history.append({"role": "assistant", "content": answer})
            session.history = trim_history(session.history)
            session.summary = _update_summary(session.history, session.summary)
            save_session(session_id, session)

            yield _format_sse("done", {"answer": answer, "categories": session.active_categories})
        except Exception as exc:
            yield _format_sse("error", {"detail": str(exc)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
