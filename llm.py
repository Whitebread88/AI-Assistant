import os
import google.generativeai as genai

# Configure API key
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

MODEL_NAME = "gemini-2.5-flash"

model = genai.GenerativeModel(MODEL_NAME)

CATEGORIES = [
    "Drinks",
    "Food",
    "Desserts"
]


# -----------------------------------
# 1️⃣ CLASSIFIER
# -----------------------------------
def classify_question(question: str) -> str:
    prompt = f"""
You are a classifier.

Select the single most relevant category from:

{", ".join(CATEGORIES)}

Return ONLY the category name.
IF none of the category is relevant, return "irrelevant".

Question: {question}
"""

    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": 0.0
        }
    )
    category = response.text.strip()
    # Fallback if unexpected category
    if category not in CATEGORIES:
        category = "technical_support"

    return category

# -----------------------------------
# 2️⃣ GENERATE ANSWER
# -----------------------------------
def generate_answer(question, context, summary, history):
    history_text = ""
    for msg in history:
        history_text += f"{msg['role'].upper()}: {msg['content']}\n"

    prompt = f"""
You are a helpful assistant.

Rules:
- Answer ONLY using the provided knowledge context.
- If the answer is not found, say:
  "I do not have that information."
- Do not fabricate.
- Be concise.

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
        generation_config={
            "temperature": 0.3,
            "max_output_tokens": 512
        }
    )

    return response.text.strip()


# -----------------------------------
# 3️⃣ SUMMARIZE CONVERSATION
# -----------------------------------
def summarize_conversation(messages):
    conversation_text = ""
    for msg in messages:
        conversation_text += f"{msg['role'].upper()}: {msg['content']}\n"

    prompt = f"""
Summarize the following conversation briefly.
Focus only on important facts and user intent.

Conversation:
{conversation_text}
"""

    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": 0.0,
            "max_output_tokens": 200
        }
    )

    return response.text.strip()
