"""
environments/trace_env/agents/memory.py

MemoryAgent — episodic + semantic memory store for the Trace agent.

Stores findings across steps without centralizing raw data.
Provides a compressed summary for the observation prompt.
"""

from __future__ import annotations
from typing import Any
from collections import deque


class MemoryAgent:
    MAX_ENTRIES = 20  # hard cap to prevent memory stuffing

    def __init__(self, config: dict):
        self.config = config
        self._episodic: deque = deque(maxlen=self.MAX_ENTRIES)
        self._semantic: dict[str, Any] = {}

    def reset(self):
        self._episodic.clear()
        self._semantic.clear()

    def store(self, content: str, metadata: dict = None):
        """Store a finding into episodic memory."""
        entry = {"content": content, "metadata": metadata or {}}
        self._episodic.append(entry)

        # Build semantic index (simple keyword extraction)
        for word in content.lower().split():
            if len(word) > 4:
                self._semantic.setdefault(word, []).append(len(self._episodic) - 1)

    def recall(self, query: str, top_k: int = 3) -> list[str]:
        """Retrieve relevant memories by keyword match."""
        scores = {}
        for word in query.lower().split():
            if word in self._semantic:
                for idx in self._semantic[word]:
                    scores[idx] = scores.get(idx, 0) + 1

        top = sorted(scores, key=scores.get, reverse=True)[:top_k]
        return [self._episodic[i]["content"] for i in top if i < len(self._episodic)]

    def summarize(self) -> str:
        """Return a compressed summary for the observation prompt."""
        if not self._episodic:
            return "(empty memory)"
        entries = list(self._episodic)[-5:]  # last 5 entries
        lines = [f"- {e['content'][:80]}" for e in entries]
        return "\n".join(lines)

    def size(self) -> int:
        return len(self._episodic)
