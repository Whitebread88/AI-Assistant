# What
Production-ready AI persona assistant designed to act as an interactive, context-aware layer over a professional portfolio.

Unlike a standard "GPT-wrapper," this service demonstrates a sophisticated Retrieval-Augmented Generation (RAG) architecture. It allows users to query professional background, technical expertise, and project history through a natural conversation, while maintaining strict control over data accuracy and operational costs.

## Key Value Propositions:
- **Contextual Intelligence:** Uses a custom classification layer to dynamically fetch relevant knowledge snippets from Google Cloud Storage, ensuring responses are grounded in my actual work history.

- **Production-Grade Architecture:** Features a serverless, streaming-first design on Google Cloud Run, incorporating stateful conversation memory (Firestore), rate limiting, and asynchronous telemetry (BigQuery).

- **Optimized UX:** Provides a "live" feel via Server-Sent Events (SSE), allowing for real-time token streaming and responsive UI rendering.

## Cloud Run

A FastAPI backend for an AI persona assistant (“Ava”) that answers questions about Aaron using category-specific knowledge files, Gemini models, and persistent conversation state in Firestore.

## What this service does

This API powers a portfolio-focused chatbot with two response modes:

- **`POST /chat`**: standard JSON response.
- **`POST /chat/stream`**: Server-Sent Events (SSE) token streaming for live UI rendering.

For each user message, the service:

1. Validates the request and internal shared secret.
2. Loads session state from Firestore.
3. Classifies the message into one or more portfolio categories.
4. Loads matching knowledge snippets from Google Cloud Storage.
5. Generates an answer with Gemini using:
   - category context,
   - rolling conversation summary,
   - recent chat history.
6. Stores updated session history + summary back to Firestore.
7. Asynchronously logs conversation events to BigQuery.

## Repository structure

- `main.py` — FastAPI app, middleware, auth check, `/chat` and `/chat/stream` endpoints, SSE formatting, session update flow.
- `llm.py` — Gemini model setup, category classifier, prompt builder, streaming token generation, markdown post-processing, conversation summarization.
- `memory.py` — Firestore-backed session storage and history trimming logic.
- `storage.py` — GCS knowledge loader by category with in-process caching and concurrent fetch for multi-category requests; article + article-index loaders with TTL caching.
- `conversation_logger.py` — non-blocking BigQuery logging queue + background flusher.
- `constants.py` — category taxonomy used by the classifier and knowledge lookup.
- `requirements.txt` — Python dependencies.
- `Dockerfile` — container build and runtime command.

## Runtime architecture and workflow

### 1) Request ingress and guardrails

- CORS is restricted to `https://www.aaronfoong.com`.
- SlowAPI rate limits are applied per IP:
  - `/chat`: `7/minute`
  - `/chat/stream`: `10/minute`
- Every chat request must include `x-internal-secret` matching `INTERNAL_CHAT_SECRET`.

### 2) Session lifecycle

Session data is keyed by `session_id` and stored in Firestore (`chats` collection in database `conversation-metadata`):

- `history`: list of `{role, content}` entries.
- `summary`: compressed conversation state for long context continuity.
- `active_categories`: cumulative category set inferred across the conversation.
- `current_article`: slug of the article page the reader is on (replaced on navigation).
- `related_articles`: up to 2 article slugs pulled in by the index selector.
- `updated_at`: UTC timestamp.

History is trimmed to the last **8 turns** (`MAX_TURNS`) to control prompt size.

### 3) Category routing and knowledge retrieval

The classifier model maps each question to a subset of fixed categories in `constants.py`.

The resulting categories are merged into the session’s existing `active_categories` (no duplicates), then used to load text files from GCS:

- Bucket: `KNOWLEDGE_BUCKET`
- Object path pattern: `knowledge/<Category>.txt`

Loaded texts are concatenated into labeled sections (`--- Category ---`) and supplied to the answer prompt.

### 3b) Article grounding

Readers can chat about website articles. Two mechanisms feed article content into the prompt — neither requires the user to pick anything:

1. **Current article (frontend-passed).** The chat widget sends an optional `article_slug` with each request — the slug of the article page the reader is on. The backend loads `knowledge/articles/<slug>.txt` and injects it as a labeled context section marked as "currently viewing". A new slug replaces the previous one, so navigating between articles never accumulates stale content.
2. **Invisible index selection.** On every non-small-talk message, a flash-lite call reads the article catalog `knowledge/articles/index.json` (slug, title, summary, url per entry) and picks up to 2 articles relevant to the question, excluding those already in context (the selector answers "none" when nothing fits). It runs concurrently with the category classifier, so it adds no serial latency. Selected slugs persist in the session (`related_articles`, capped at 2, newest first) for follow-up questions.

Each article's context section includes its `Link:` URL (from the index), and the answer prompt instructs the model to cite articles it draws on as markdown links and to recommend related articles from context.

`Articles` remains a routing-only category in the classifier taxonomy — it has no `knowledge/Articles.txt` file and is excluded from category knowledge loading.

GCS layout:

```
knowledge/<Category>.txt          # fixed persona categories (cached for process lifetime)
knowledge/articles/<slug>.txt     # one plain-text/markdown file per article (TTL cache)
knowledge/articles/index.json     # article catalog used by the selector (TTL cache)
```

`index.json` format:

```json
[
  {
    "slug": "why-i-built-ava",
    "title": "Why I Built Ava",
    "summary": "A dense description of what the article covers; the selector judges relevance from this text.",
    "url": "https://www.aaronfoong.com/archive/why-i-built-ava"
  }
]
```

`url` is optional; when present it is surfaced to the model as the article's `Link:` line so answers can cite the article.

Publishing a new article requires only uploading its `.txt` file and re-uploading `index.json` — no redeploy. Article slugs must match `^[A-Za-z0-9_-]{1,100}$`; invalid slugs in requests are ignored. Article text is truncated to `ARTICLE_MAX_CHARS` and both articles and the index are cached in-process for `ARTICLE_CACHE_TTL_SECONDS`.

### 4) LLM answer generation

The answer prompt in `llm.py` includes:

- persona instructions (assistant identity, tone, constraints),
- category knowledge context,
- conversation summary,
- recent turn history,
- current user query.

Responses are generated by Gemini with streaming token support. A final markdown cleanup step closes unbalanced backticks/fences to avoid frontend markdown rendering issues.

### 5) Summarization cadence

The conversation summary is refreshed:

- on first summary creation, and
- every second assistant turn thereafter.

This balances summary freshness and cost.

### 6) Analytics logging

Conversation events are enqueued and batched to BigQuery asynchronously so response latency is not blocked by logging I/O.

If project env vars are missing, BigQuery logging auto-disables safely.

## API contract

## `GET /`

Health check:

```json
{
  "status": "running",
  "message": "Chatbot API is live"
}
```

## `POST /chat`

Headers:

- `x-internal-secret: <INTERNAL_CHAT_SECRET>`

Body (`article_slug` is optional — the slug of the article page the reader is chatting from):

```json
{
  "session_id": "abc123",
  "message": "Tell me about Aaron's projects",
  "article_slug": "why-i-built-ava"
}
```

Response:

```json
{
  "answer": "...",
  "categories": ["Project", "Skills"],
  "articles": ["why-i-built-ava"]
}
```

## `POST /chat/stream`

Headers and body are the same as `/chat`.

SSE events emitted in sequence:

- `metadata` → current categories + active article slugs
- `token` → incremental answer chunks
- `done` → final answer + categories + active article slugs
- `error` → error payload if generation fails

## Environment variables

### Core

- `INTERNAL_CHAT_SECRET` — required shared secret for chat endpoints.
- `GEMINI_API_KEY` — required for Gemini model access.

### Model selection (optional)

- `ANSWER_MODEL` (default: `gemini-3.1-flash-lite`)
- `CLASSIFIER_MODEL` (default: `gemini-3.1-flash-lite`)
- `SUMMARY_MODEL` (default: `gemini-3.1-flash-lite`)
- `GEMINI_THINKING_LEVEL` (default: `minimal`) — Gemini 3 reasoning level (`minimal`, `low`, `medium`, `high`); `minimal` keeps latency lowest.

### Storage and persistence

- `KNOWLEDGE_BUCKET` — GCS bucket containing `knowledge/*.txt` files and `knowledge/articles/`.
- `ARTICLE_CACHE_TTL_SECONDS` (default: `600`) — in-process cache lifetime for article files and the article index.
- `ARTICLE_MAX_CHARS` (default: `30000`) — per-article truncation limit for prompt context.
- Google Cloud credentials/env needed for Firestore and GCS clients.
