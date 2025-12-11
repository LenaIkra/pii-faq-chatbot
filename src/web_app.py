from __future__ import annotations

from typing import Optional, Dict, Any

from fastapi import FastAPI
from pydantic import BaseModel

from src.faq_bot import FAQBot
from src.logger import log_interaction

app = FastAPI(title="PII FAQ Chatbot API")

# Инициализируем два бота: для студентов и абитуриентов
bots = {
    "students": FAQBot(segment="students", sim_threshold=0.5, top_k=3),
    "applicants": FAQBot(segment="applicants", sim_threshold=0.5, top_k=3),
}


class ChatRequest(BaseModel):
    segment: str = "students"  # "students" или "applicants"
    question: str


class ChatResponse(BaseModel):
    segment: str
    question: str
    answer: str
    matched_question: Optional[str] = None
    similarity: Optional[float] = None


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    segment = req.segment.lower()
    if segment not in ("students", "applicants"):
        segment = "students"

    bot = bots[segment]
    answer, match = bot.get_answer(req.question, debug=False)

    matched_q = None
    similarity = None
    if match is not None:
        matched_q = getattr(match, "question", None)
        similarity = getattr(match, "similarity", None)

    meta: Dict[str, Any] = {
        "matched_question": matched_q,
        "similarity": similarity,
    }
    log_interaction("web", segment, req.question, answer, meta)

    return ChatResponse(
        segment=segment,
        question=req.question,
        answer=answer,
        matched_question=matched_q,
        similarity=similarity,
    )