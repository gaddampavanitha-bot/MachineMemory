"""
machine_memory.py — a small, dependency-free memory system for an AI agent/chatbot.

Design
------
- Short-term memory: a rolling buffer of the most recent conversation turns.
- Long-term memory: persistent SQLite store of "memories" (facts, events, preferences).
- Retrieval: each memory is scored by  relevance * w_rel + recency * w_rec + importance * w_imp.
  Relevance here is TF-IDF-style keyword overlap (stdlib only). To upgrade, swap
  `_relevance` for cosine similarity over embeddings from any embedding model.
- Consolidation: old, low-value memories are pruned; duplicates are merged.

Usage
-----
    mem = MachineMemory("memory.db")
    mem.add_turn("user", "I'm allergic to peanuts and I live in Hyderabad.")
    mem.remember("User is allergic to peanuts", kind="fact", importance=0.9)
    context = mem.build_context("What snacks should I suggest?")
    # -> put `context` into your LLM's system prompt
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "is", "are", "was", "were", "be", "to", "of",
    "in", "on", "at", "for", "with", "i", "you", "it", "this", "that", "my", "me", "we",
    "do", "does", "did", "have", "has", "had", "as", "by", "from", "so", "if", "not", "user",
}


def tokenize(text: str) -> List[str]:
    words = re.findall(r"[a-z0-9']+", text.lower())
    words = [w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w for w in words]  # crude plural stemming
    return [w for w in words if w not in STOPWORDS and len(w) > 1]


@dataclass
class Memory:
    id: int
    text: str
    kind: str
    importance: float
    created_at: float
    last_accessed: float
    access_count: int
    meta: Dict = field(default_factory=dict)


class MachineMemory:
    def __init__(
        self,
        db_path: str = "machine_memory.db",
        short_term_size: int = 10,
        w_relevance: float = 0.6,
        w_recency: float = 0.25,
        w_importance: float = 0.15,
        recency_half_life_days: float = 14.0,
    ):
        self.short_term: Deque[Tuple[str, str]] = deque(maxlen=short_term_size)
        self.w_rel, self.w_rec, self.w_imp = w_relevance, w_recency, w_importance
        self.half_life = recency_half_life_days * 86400
        self.db = sqlite3.connect(db_path)
        self.db.row_factory = sqlite3.Row
        self._init_db()

    # ------------------------------------------------------------------ storage
    def _init_db(self) -> None:
        self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                text          TEXT NOT NULL,
                kind          TEXT NOT NULL DEFAULT 'fact',
                importance    REAL NOT NULL DEFAULT 0.5,
                created_at    REAL NOT NULL,
                last_accessed REAL NOT NULL,
                access_count  INTEGER NOT NULL DEFAULT 0,
                meta          TEXT NOT NULL DEFAULT '{}'
            )
            """
        )
        self.db.commit()

    @staticmethod
    def _row_to_memory(row: sqlite3.Row) -> Memory:
        return Memory(
            id=row["id"], text=row["text"], kind=row["kind"], importance=row["importance"],
            created_at=row["created_at"], last_accessed=row["last_accessed"],
            access_count=row["access_count"], meta=json.loads(row["meta"]),
        )

    def _all(self) -> List[Memory]:
        return [self._row_to_memory(r) for r in self.db.execute("SELECT * FROM memories")]

    # ------------------------------------------------------------ short-term API
    def add_turn(self, role: str, content: str) -> None:
        """Record a conversation turn in the rolling short-term buffer."""
        self.short_term.append((role, content))

    # ------------------------------------------------------------- long-term API
    def remember(
        self, text: str, kind: str = "fact", importance: float = 0.5, meta: Optional[dict] = None
    ) -> int:
        """Store a memory. If a near-duplicate exists, reinforce it instead of adding."""
        importance = min(max(importance, 0.0), 1.0)
        dup = self._find_duplicate(text)
        now = time.time()
        if dup:
            self.db.execute(
                "UPDATE memories SET importance = MAX(importance, ?), last_accessed = ?, "
                "access_count = access_count + 1 WHERE id = ?",
                (importance, now, dup.id),
            )
            self.db.commit()
            return dup.id
        cur = self.db.execute(
            "INSERT INTO memories (text, kind, importance, created_at, last_accessed, meta) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (text.strip(), kind, importance, now, now, json.dumps(meta or {})),
        )
        self.db.commit()
        return cur.lastrowid

    def forget(self, memory_id: int) -> None:
        self.db.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self.db.commit()

    def _find_duplicate(self, text: str, threshold: float = 0.85) -> Optional[Memory]:
        new = set(tokenize(text))
        if not new:
            return None
        for m in self._all():
            old = set(tokenize(m.text))
            if old and len(new & old) / len(new | old) >= threshold:
                return m
        return None

    # ---------------------------------------------------------------- retrieval
    def _relevance(self, query_tokens: List[str], mem: Memory, idf: Dict[str, float]) -> float:
        if not query_tokens:
            return 0.0
        tf = Counter(tokenize(mem.text))
        if not tf:
            return 0.0
        score = sum(idf.get(t, 0.0) * (1 + math.log(tf[t])) for t in set(query_tokens) if t in tf)
        max_score = sum(idf.get(t, 0.0) for t in set(query_tokens)) or 1.0
        return min(score / max_score, 1.0)

    def _recency(self, mem: Memory, now: float) -> float:
        age = max(now - mem.last_accessed, 0.0)
        return 0.5 ** (age / self.half_life)  # exponential decay

    def recall(self, query: str, k: int = 5, min_score: float = 0.15) -> List[Tuple[Memory, float]]:
        """Return the top-k memories for `query`, best first, as (memory, score) pairs."""
        memories = self._all()
        if not memories:
            return []
        q_tokens = tokenize(query)
        n = len(memories)
        df = Counter(t for m in memories for t in set(tokenize(m.text)))
        idf = {t: math.log((n + 1) / (c + 0.5)) + 1 for t, c in df.items()}

        now = time.time()
        scored = []
        for m in memories:
            rel = self._relevance(q_tokens, m, idf)
            score = self.w_rel * rel + self.w_rec * self._recency(m, now) + self.w_imp * m.importance
            # Require some topical relevance so important-but-unrelated memories don't flood results.
            if rel > 0 and score >= min_score:
                scored.append((m, score))
        scored.sort(key=lambda p: p[1], reverse=True)
        top = scored[:k]

        for m, _ in top:  # reinforce retrieved memories
            self.db.execute(
                "UPDATE memories SET last_accessed = ?, access_count = access_count + 1 WHERE id = ?",
                (now, m.id),
            )
        self.db.commit()
        return top

    # ------------------------------------------------------------ prompt building
    def build_context(self, query: str, k: int = 5) -> str:
        """Build a text block to inject into your LLM prompt."""
        parts = []
        recalled = self.recall(query, k=k)
        if recalled:
            parts.append("Relevant long-term memories:")
            parts += [f"- [{m.kind}] {m.text}" for m, _ in recalled]
        if self.short_term:
            parts.append("\nRecent conversation:")
            parts += [f"{role}: {content}" for role, content in self.short_term]
        return "\n".join(parts)

    # ------------------------------------------------------------- maintenance
    def consolidate(self, max_age_days: float = 90.0, min_value: float = 0.2) -> int:
        """Prune stale, low-importance, rarely-used memories. Returns number removed."""
        cutoff = time.time() - max_age_days * 86400
        cur = self.db.execute(
            "DELETE FROM memories WHERE last_accessed < ? AND importance < ? AND access_count < 3",
            (cutoff, min_value),
        )
        self.db.commit()
        return cur.rowcount

    def close(self) -> None:
        self.db.close()


# ------------------------------------------------------------------------- demo
if __name__ == "__main__":
    mem = MachineMemory(":memory:")

    mem.remember("User is allergic to peanuts", kind="fact", importance=0.95)
    mem.remember("User prefers concise answers with code examples", kind="preference", importance=0.7)
    mem.remember("User is building a project called machine memory", kind="fact", importance=0.8)
    mem.remember("User asked about the weather last Tuesday", kind="event", importance=0.1)

    mem.add_turn("user", "Suggest a snack for my road trip.")

    print(mem.build_context("can I suggest peanut snacks?"))
    print("\n--- second query ---")
    print(mem.build_context("how should I format code answers?"))
