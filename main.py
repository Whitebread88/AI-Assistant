import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

from fastapi import Header, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, constr
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from llm import (
    classify_categories,
    finalize_answer_markdown,
    generate_answer,
    select_relevant_articles,
    stream_answer_tokens,
    summarize_conversation,
)
from constants import ARTICLES_CATEGORY
from conversation_logger import build_conversation_event, conversation_logger
from memory import SessionState, get_session, save_session, trim_history
from storage import build_article_context, load_article_index, load_relevant_knowledge

app = FastAPI()

SUMMARY_REFRESH_TURNS = 4
SMALL_TALK_PATTERN = re.compile(
    r"^(hi|hello|hey|yo|sup|good\s+(morning|afternoon|evening))[!.?\s]*$",
    re.IGNORECASE,
)
ARTICLE_SLUG_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,100}$")
MAX_RELATED_ARTICLES = 2

def get_real_ip(request: Request):
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0]
    return request.client.host


limiter = Limiter(key_func=get_real_ip)
app.state.limiter = limiter

app.add_middleware(SlowAPIMiddleware)

EXPECTED_SECRET = os.getenv("INTERNAL_CHAT_SECRET")

origins = ["https://www.aaronfoong.com",
          "https://aaronfoong.com"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    session_id: str
    message: constr(min_length=1, max_length=4000)
    article_slug: str | None = None


def _format_sse(event: str, data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def _sanitize_article_slug(slug: str | None) -> str | None:
    if not slug:
        return None
    slug = slug.strip()
    return slug if ARTICLE_SLUG_PATTERN.match(slug) else None


def _active_articles(session: SessionState) -> list[str]:
    articles: list[str] = []
    for slug in [session.current_article, *session.related_articles]:
        if slug and slug not in articles:
            articles.append(slug)
    return articles


def _prepare_turn_context(
    session: SessionState, message: str, article_slug: str | None
) -> str:
    """Update routing state (categories + articles) and assemble the prompt context."""
    if article_slug:
        session.current_article = article_slug

    exclude_slugs = _active_articles(session)

    def _select_articles() -> list[str]:
        return select_relevant_articles(
            question=message,
            index_entries=load_article_index(),
            exclude_slugs=exclude_slugs,
        )

    # Category classification and article selection are independent LLM calls;
    # run them concurrently so selecting on every message adds no latency.
    with ThreadPoolExecutor(max_workers=2) as pool:
        categories_future = pool.submit(classify_categories, message)
        selection_future = pool.submit(_select_articles)
        current_categories = categories_future.result()
        selected = selection_future.result()

    merged = list(session.active_categories)
    for category in current_categories:
        if category not in merged:
            merged.append(category)
    session.active_categories = merged

    if selected:
        remaining = [s for s in session.related_articles if s not in selected]
        session.related_articles = (selected + remaining)[:MAX_RELATED_ARTICLES]

    sections: list[str] = []
    knowledge_categories = [c for c in merged if c != ARTICLES_CATEGORY]
    category_context = load_relevant_knowledge(knowledge_categories)
    if category_context:
        sections.append(category_context)

    article_context = build_article_context(
        session.current_article, session.related_articles
    )
    if article_context:
        sections.append(article_context)

    return "\n\n".join(sections)


def _update_summary(session_history: list[dict[str, str]], previous_summary: str) -> str:
    assistant_turns = sum(1 for item in session_history if item.get("role") == "assistant")
    if assistant_turns % SUMMARY_REFRESH_TURNS == 0 or not previous_summary:
        return summarize_conversation(session_history)
    return previous_summary

def _is_small_talk(message: str) -> bool:
    normalized = message.strip().lower()
    if not normalized:
        return False
    if SMALL_TALK_PATTERN.match(normalized):
        return True
    return len(normalized.split()) <= 3 and normalized in {"hi", "hello", "hey", "yo"}


def _small_talk_response() -> str:
    return (
        "Hey there 👋 I'm Ava. I can walk you through Aaron's professional life - "
        "what are you most curious about?"
    )
    
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
def root(x_internal_secret: str = Header(None)) -> dict[str, str]:
    # Ensure only your Vercel Proxy can wake up the container
    if EXPECTED_SECRET and x_internal_secret != EXPECTED_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden")
        
    return {"status": "running", "message": "Chatbot API is live"}

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request, exc):
    return JSONResponse(status_code=429, content={"detail": "Too many requests"})


@app.post("/chat")
@limiter.limit("7/minute")
async def chat(request: Request, data: ChatRequest, x_internal_secret: str = Header(None)):
    session_id, message = _validate_chat_request(data, x_internal_secret)
    article_slug = _sanitize_article_slug(data.article_slug)

    session = get_session(session_id)
    session.history.append({"role": "user", "content": message})

    context = _prepare_turn_context(session, message, article_slug)

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
    conversation_logger.log(
        build_conversation_event(
            session_id=session_id,
            user_message=message,
            assistant_answer=answer,
            categories=session.active_categories,
            endpoint="/chat",
        )
    )
    return {
        "answer": answer,
        "categories": session.active_categories,
        "articles": _active_articles(session),
    }


@app.post("/chat/stream")
@limiter.limit("10/minute")
async def chat_stream(request: Request, data: ChatRequest, x_internal_secret: str = Header(None)):
    session_id, message = _validate_chat_request(data, x_internal_secret)
    article_slug = _sanitize_article_slug(data.article_slug)

    session = get_session(session_id)
    session.history.append({"role": "user", "content": message})

    if article_slug:
        session.current_article = article_slug

    if _is_small_talk(message):
        answer = _small_talk_response()
        session.history.append({"role": "assistant", "content": answer})
        session.history = trim_history(session.history)
        session.summary = _update_summary(session.history, session.summary)
        save_session(session_id, session)
        conversation_logger.log(
            build_conversation_event(
                session_id=session_id,
                user_message=message,
                assistant_answer=answer,
                categories=session.active_categories,
                endpoint="/chat/stream",
            )
        )

        def small_talk_event_generator():
            yield _format_sse("metadata", {"categories": session.active_categories})
            yield _format_sse("token", {"text": answer})
            yield _format_sse("done", {"answer": answer, "categories": session.active_categories})

        return StreamingResponse(
            small_talk_event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
        
    context = _prepare_turn_context(session, message, article_slug)

    def event_generator():
        full_answer_parts: list[str] = []
        try:
            yield _format_sse(
                "metadata",
                {
                    "categories": session.active_categories,
                    "articles": _active_articles(session),
                },
            )
            for token in stream_answer_tokens(
                question=message,
                context=context,
                summary=session.summary,
                history=session.history,
            ):
                full_answer_parts.append(token)
                yield _format_sse("token", {"text": token})

            answer = finalize_answer_markdown("".join(full_answer_parts))
            session.history.append({"role": "assistant", "content": answer})
            session.history = trim_history(session.history)
            session.summary = _update_summary(session.history, session.summary)
            save_session(session_id, session)
            conversation_logger.log(
                build_conversation_event(
                    session_id=session_id,
                    user_message=message,
                    assistant_answer=answer,
                    categories=session.active_categories,
                    endpoint="/chat/stream",
                )
            )

            yield _format_sse(
                "done",
                {
                    "answer": answer,
                    "categories": session.active_categories,
                    "articles": _active_articles(session),
                },
            )
        except Exception as exc:
            yield _format_sse("error", {"detail": str(exc)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache, no-transform", # Added no-transform here too
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
