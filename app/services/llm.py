# """
# Step 5 of the pipeline: send the question + retrieved chunks to an LLM
# and get back a grounded answer.

# Uses Groq's free-tier API by default (OpenAI-compatible format) - swap
# the URL/model in .env to use OpenAI, Together, or any compatible provider.
# """
# from typing import List
# import requests
# from app.config import settings

# GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


# def generate_answer(question: str, context_chunks: List[str]) -> str:
#     context = "\n\n".join(context_chunks)

#     prompt = (
#     "Answer the question using only the context below, in plain "
#     "conversational sentences - like you're explaining it out loud to a "
#     "student. Do not use any Markdown formatting: no **bold**, no bullet "
#     "points, no numbered lists, no headings, no tables. Just write it as "
#     "normal flowing paragraphs. If the answer isn't in the context, say "
#     "you don't have enough information.\n\n"
#     f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
#     )

#     response = requests.post(
#         GROQ_URL,
#         headers={"Authorization": f"Bearer {settings.LLM_API_KEY}"},
#         json={
#             "model": settings.LLM_MODEL,
#             "messages": [{"role": "user", "content": prompt}],
#             "temperature": 0.2,
#         },
#         timeout=30,
#     )
#     response.raise_for_status()
#     data = response.json()
#     return data["choices"][0]["message"]["content"]

"""
Step 5 of the pipeline: send the question + retrieved chunks to an LLM
and get back a grounded answer.

Uses Groq's free-tier API by default (OpenAI-compatible format) - swap
the URL/model in .env to use OpenAI, Together, or any compatible provider.
"""
from typing import List, Optional
import requests
from app.config import settings

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


def generate_answer(
    question: str,
    context_chunks: List[str],
    history: Optional[List[dict]] = None,
) -> str:
    context = "\n\n".join(context_chunks)

    history_text = ""
    if history:
        recent = history[-5:]  # cap context size regardless of how much is sent in
        lines = [f"Q: {t['question']}\nA: {t['answer']}" for t in recent]
        history_text = "Previous conversation in this session:\n" + "\n\n".join(lines) + "\n\n"

    prompt = (
        "Answer the question using only the context below, and taking the "
        "previous conversation into account if it's relevant (e.g. if the "
        "question refers to 'it' or 'that', check what it likely means from "
        "the conversation above). Write in clear, well-organized Markdown: "
        "use a numbered or bulleted list when the question asks for "
        "multiple items, steps, or topics; use short paragraphs otherwise. "
        "For bold text, always use matching double asterisks like **this** "
        "- never a single asterisk, and never mismatched. If the answer "
        "isn't in the context, say you don't have enough information.\n\n"
        f"{history_text}"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )

    response = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {settings.LLM_API_KEY}"},
        json={
            "model": settings.LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        },
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]