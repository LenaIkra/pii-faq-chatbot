from __future__ import annotations
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "interactions.csv"


def log_interaction(
    source: str,
    segment: str,
    question: str,
    answer: str,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Пишет строку в logs/interactions.csv
    source  — где спросили: "cli", "web", "telegram"
    segment — "students" или "applicants"
    meta    — например: similarity, matched_question и т.п.
    """
    meta = meta or {}
    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "source": source,
        "segment": segment,
        "question": question,
        "answer": answer,
        "meta": json.dumps(meta, ensure_ascii=False),
    }

    file_exists = LOG_FILE.exists()
    with LOG_FILE.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["timestamp", "source", "segment", "question", "answer", "meta"],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)