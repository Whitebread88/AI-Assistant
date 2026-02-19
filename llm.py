import os

import google.generativeai as genai

from constants import CATEGORIES

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")


def classify_categories(question: str) -> list[str]:
    prompt = f"""
You are a classifier.
Select all relevant categories from this list:
{", ".join(CATEGORIES)}

Return ONLY a comma-separated list of category names.
If none are relevant, return "none".

Question: {question}
"""

    response = model.generate_content(prompt, generation_config={"temperature": 0.0})
    raw = (response.text or "").strip()
    if not raw or raw.lower() == "none":
        return []

    selected: list[str] = []
    for part in raw.split(","):
        candidate = part.strip()
        if candidate in CATEGORIES and candidate not in selected:
            selected.append(candidate)
    return selected


def generate_answer(
    question: str,
    context: str,
    summary: str,
    history: list[dict[str, str]],
) -> str:
    history_text = "\n".join(
        f"{message['role'].upper()}: {message['content']}" for message in history
    )

    prompt = f"""
You are a helpful portfolio AI assistant on behalf of Aaron. Aaron is your creator.

Behavior Rules:
- Maintain natural conversation.
- Use conversation history and summary for continuity.
- If relevant Knowledge Context is provided, use it for factual accuracy.
- If knowledge is missing for a factual question, communicate that you don't have the information currently and encourage to reach out to Aaron.
- Do not fabricate business facts.
- Be concise but natural.

Conversation Summary:
{summary}

Recent Conversation:
{history_text}

Knowledge Context:
{context}

User Question:
{question}
"""

    response = model.generate_content(
        prompt,
        generation_config={"temperature": 0.7, "max_output_tokens": 1000},
    )
    return (response.text or "").strip()


def summarize_conversation(messages: list[dict[str, str]]) -> str:
    conversation_text = "\n".join(
        f"{message['role'].upper()}: {message['content']}" for message in messages
    )

    prompt = f"""
Summarize the following conversation briefly.
Focus only on important facts and user intent.

Conversation:
{conversation_text}
"""

    response = model.generate_content(
        prompt,
        generation_config={"temperature": 0.0, "max_output_tokens": 200},
    )
    return (response.text or "").strip()
