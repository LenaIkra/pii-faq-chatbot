from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Match:
    index: int
    question: str
    answer: str
    similarity: float  # итоговый гибридный скор


class FAQBot:
    """
    FAQ-бот для магистратуры ПИИ.

    Режимы:
      - students    — действующие студенты
      - applicants  — абитуриенты

    Гибридный поиск:
      • эмбеддинги (sentence-transformers)
      • TF-IDF (1–2-граммы по вопросам)
      • пересечение токенов запроса и вопроса
    """

    def __init__(
        self,
        segment: str = "students",
        sim_threshold: float = 0.35,  # чуть смягчили порог
        top_k: int = 5,
    ) -> None:
        self.segment = segment if segment in ("students", "applicants") else "students"
        self.sim_threshold = sim_threshold
        self.top_k = top_k

        self.no_answer_text = (
            "Я пока не нашёл точного ответа в базе. "
            "Этот вопрос можно передать координатору программы."
        )

        base_dir = Path(__file__).resolve().parent.parent / "data"

        if self.segment == "students":
            kb_csv = base_dir / "faq_kb_students.csv"
            emb_npy = base_dir / "faq_embeddings_students.npy"
        else:
            kb_csv = base_dir / "faq_kb_applicants.csv"
            emb_npy = base_dir / "faq_embeddings_applicants.npy"

        print(f"[FAQBot] Режим: {self.segment}")
        print(f"[FAQBot] Загружаю базу: {kb_csv}")
        self.df = pd.read_csv(kb_csv)

        print(f"[FAQBot] Вопросов в базе: {len(self.df)}")

        print(f"[FAQBot] Загружаю эмбеддинги: {emb_npy}")
        self.embeddings = np.load(emb_npy)

        if self.embeddings.ndim != 2:
            raise ValueError("Эмбеддинги должны быть 2D-матрицей.")
        if len(self.df) != self.embeddings.shape[0]:
            raise ValueError(
                f"Несовпадение размеров: в CSV {len(self.df)} строк, "
                f"в эмбеддингах {self.embeddings.shape[0]} векторов."
            )

        # модель эмбеддингов
        print("[FAQBot] Загружаю модель эмбеддингов…")
        self.model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

        # список вопросов
        self.questions: List[str] = self.df["question"].astype(str).tolist()

        # словарь предметной области
        print("[FAQBot] Строю словарь домена…")
        self.domain_vocab = self._build_domain_vocab()

        # TF-IDF по вопросам
        print("[FAQBot] Строю TF-IDF по вопросам…")
        self.tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.9,
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.questions)

    # ---------------- UTILS ---------------- #

    def _normalize(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^а-яa-z0-9ё\s]", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _tokenize(self, text: str) -> List[str]:
        return self._normalize(text).split()

    def _build_domain_vocab(self) -> set:
        vocab = set()
        for _, row in self.df.iterrows():
            for tok in self._tokenize(str(row["question"])):
                vocab.add(tok)
            for tok in self._tokenize(str(row["answer"])):
                vocab.add(tok)
        return vocab

    def _in_domain(self, query: str) -> bool:
        tokens = self._tokenize(query)
        return any(tok in self.domain_vocab for tok in tokens)

    # ---------------- RANKER ---------------- #

    def _rank(self, query: str, debug: bool = False) -> List[Match]:
        """
        Гибридный ранкер:
          • эмбеддинги (dense)
          • TF-IDF (lexical)
          • пересечение токенов (overlap)
        """
        if not self.questions:
            return []

        # ---- эмбеддинги ----
        q_emb = self.model.encode([query], normalize_embeddings=True)[0]
        sim_dense = self.embeddings @ q_emb  # (N,)

        # ---- TF-IDF ----
        q_tfidf = self.tfidf_vectorizer.transform([query])
        sim_lex = cosine_similarity(q_tfidf, self.tfidf_matrix)[0]  # (N,)

        # ---- нормализация [0..1] ----
        def _norm(x: np.ndarray) -> np.ndarray:
            x_min = float(x.min())
            x_max = float(x.max())
            if x_max - x_min < 1e-6:
                return np.zeros_like(x)
            return (x - x_min) / (x_max - x_min)

        sim_dense_n = _norm(sim_dense)
        sim_lex_n = _norm(sim_lex)

        # ---- пересечение токенов ----
        q_tokens = self._tokenize(query)
        num_tokens = len(q_tokens)
        overlap_scores = np.zeros(len(self.questions), dtype=float)

        if q_tokens:
            q_token_set = set(q_tokens)
            for i, q_text in enumerate(self.questions):
                q2_tokens = set(self._tokenize(q_text))
                common = len(q_token_set & q2_tokens)
                if common > 0:
                    overlap_scores[i] = common / len(q_token_set)

        overlap_n = overlap_scores  # уже в [0..1]

        # ---- смешиваем сигналы ----
        if num_tokens <= 3:
            alpha_dense_vs_tfidf = 0.2  # короткие вопросы → TF-IDF важнее
        elif num_tokens <= 6:
            alpha_dense_vs_tfidf = 0.4
        else:
            alpha_dense_vs_tfidf = 0.6  # длинные → эмбеддинги важнее

        mixed = (
            alpha_dense_vs_tfidf * sim_dense_n
            + (1.0 - alpha_dense_vs_tfidf) * sim_lex_n
        )

        gamma = 0.3  # вклад overlap
        final = (1.0 - gamma) * mixed + gamma * overlap_n

        idx_sorted = np.argsort(-final)

        matches: List[Match] = []
        for idx in idx_sorted[: self.top_k]:
            matches.append(
                Match(
                    index=int(idx),
                    question=self.questions[idx],
                    answer=str(self.df.iloc[idx]["answer"]),
                    similarity=float(final[idx]),
                )
            )

        if debug:
            print("--- HYBRID RANK + OVERLAP ---")
            print("Query:", query)
            print("Tokens:", q_tokens, f"(n={num_tokens})")
            print("Top candidates:")
            for m in matches:
                print(f"{m.similarity:.3f} → {m.question[:80]}")
            print("-----------------------------")

        return matches

    # ---------------- PUBLIC API ---------------- #

    def get_answer(self, query: str, debug: bool = False) -> tuple[str, Optional[Match]]:
        """
        Основной метод: (ответ, match).

        Логика:
          1) Проверяем, что вопрос в домене (есть слова из словаря базы).
          2) Берём top_k кандидатов из _rank (ембеддинги + TF-IDF + overlap).
          3) Если лучший similarity ≥ порога → берём его.
          4) Иначе:
               – среди всех top_k ищем того, у кого максимальное пересечение токенов;
               – если overlap ≥ 0.5 → берём этого кандидата;
          5) Иначе — честно говорим, что точного ответа нет.
        """
        query = (query or "").strip()
        if not query:
            return self.no_answer_text, None

        # 1) грубый фильтр по домену
        if not self._in_domain(query):
            if debug:
                print("[FAQBot] Вопрос вне домена.")
            return self.no_answer_text, None

        # 2) ранжирование
        matches = self._rank(query, debug=debug)
        if not matches:
            return self.no_answer_text, None

        best = matches[0]

        # 3) базовая проверка по порогу similarity
        if best.similarity >= self.sim_threshold:
            if debug:
                print(
                    f"[FAQBot] Берём лучший ответ по similarity: "
                    f"{best.similarity:.3f} ≥ {self.sim_threshold}"
                )
            return best.answer, best

        # 4) fallback: ищем ЛУЧШИЙ по overlap среди top_k
        q_tokens = set(self._tokenize(query))
        best_overlap_match: Optional[Match] = None
        best_overlap_ratio = 0.0

        for m in matches:
            m_tokens = set(self._tokenize(m.question))
            overlap = len(q_tokens & m_tokens)
            ratio = overlap / len(q_tokens) if q_tokens else 0.0

            if ratio > best_overlap_ratio:
                best_overlap_ratio = ratio
                best_overlap_match = m

        if debug:
            print(
                f"[FAQBot] Лучший по similarity ниже порога "
                f"(sim={best.similarity:.3f} < {self.sim_threshold})."
            )
            if best_overlap_match is not None:
                print(
                    f"[FAQBot] Лучший по overlap: "
                    f"overlap_ratio={best_overlap_ratio:.2f} → "
                    f"{best_overlap_match.question[:80]}"
                )

        # если хотя бы половина токенов совпала — считаем, что это наш вопрос
        if best_overlap_match is not None and best_overlap_ratio >= 0.5:
            if debug:
                print("[FAQBot] Берём ответ по высокому совпадению токенов.")
            return best_overlap_match.answer, best_overlap_match

        # 5) совсем не уверены — отвечаем нейтрально
        return self.no_answer_text, best
