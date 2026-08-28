"""Intelligent document chunking with metadata propagation."""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from src.ingestion.loader import Document

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A single chunk derived from a document."""

    content: str
    metadata: dict[str, str | int | float] = field(default_factory=dict)
    chunk_id: str = ""
    doc_id: str = ""


def _compute_chunk_id(content: str, doc_id: str, index: int) -> str:
    """Deterministic chunk ID for idempotent upserts."""
    raw = f"{doc_id}::chunk-{index}::{content[:50]}"
    return hashlib.md5(raw.encode()).hexdigest()


class Chunker:
    """Splits documents into chunks with configurable overlap.

    Supports multiple splitting strategies:
      - "recursive": split on paragraph/sentence boundaries (default, recommended)
      - "fixed": fixed-size character chunks
      - "sentence": sentence-boundary aware chunks
    """

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        strategy: str = "recursive",
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError(f"chunk_overlap ({chunk_overlap}) must be < chunk_size ({chunk_size})")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy

        self._strategies: dict[str, Callable[[str], list[str]]] = {
            "recursive": self._recursive_split,
            "fixed": self._fixed_size_split,
            "sentence": self._sentence_split,
        }

        if strategy not in self._strategies:
            raise ValueError(
                f"Unknown chunking strategy '{strategy}'. "
                f"Choose from: {list(self._strategies.keys())}"
            )

    def chunk(self, document: Document) -> list[Chunk]:
        """Split a single document into chunks."""
        raw_chunks = self._strategies[self.strategy](document.content)
        chunks: list[Chunk] = []

        for i, text in enumerate(raw_chunks):
            text = text.strip()
            if not text:
                continue

            chunk = Chunk(
                content=text,
                metadata={**document.metadata, "chunk_index": i, "total_chunks": len(raw_chunks)},
                chunk_id=_compute_chunk_id(text, document.doc_id, i),
                doc_id=document.doc_id,
            )
            chunks.append(chunk)

        logger.debug("Split doc %s into %d chunks", document.doc_id, len(chunks))
        return chunks

    def chunk_many(self, documents: list[Document]) -> list[Chunk]:
        """Split multiple documents into chunks."""
        chunks: list[Chunk] = []
        for doc in documents:
            chunks.extend(self.chunk(doc))
        logger.info("Created %d chunks from %d documents", len(chunks), len(documents))
        return chunks

    @staticmethod
    def _split_keep_separator(text: str, pattern: str) -> list[str]:
        """Split on ``pattern``, keeping each delimiter attached to the segment
        before it, so that ``"".join(segments) == text``.

        A plain ``re.split`` discards the whitespace it matches on. Because
        ``_merge_chunks`` re-joins segments with no separator, that lost the
        boundary entirely and welded neighbours together — a heading and its
        body became "Enhanced RetrievalHybrid search combines...". That hurts
        embedding quality, fuses two words into one unmatchable BM25 token,
        and renders citation snippets to users with words run together.
        """
        parts = re.split(f"({pattern})", text)
        segments: list[str] = []
        for i in range(0, len(parts), 2):
            body = parts[i]
            separator = parts[i + 1] if i + 1 < len(parts) else ""
            if body or separator:
                segments.append(body + separator)
        return segments

    def _recursive_split(self, text: str) -> list[str]:
        """Recursively split on paragraph then sentence boundaries."""
        if len(text) <= self.chunk_size:
            return [text]

        # Try paragraph split first, then break down any paragraph that is
        # itself larger than chunk_size — otherwise a single long paragraph
        # becomes one oversized chunk that overruns the context budget.
        segments: list[str] = []
        for paragraph in self._split_keep_separator(text, r"\n\s*\n"):
            if len(paragraph) <= self.chunk_size:
                segments.append(paragraph)
            else:
                segments.extend(self._split_oversized(paragraph))
        return self._merge_chunks(segments)

    def _split_oversized(self, text: str) -> list[str]:
        """Break an over-long segment down to sentences, then to hard character
        windows for run-on text with no sentence punctuation."""
        segments: list[str] = []
        for sentence in self._split_keep_separator(text, r"(?<=[.!?])\s+"):
            if len(sentence) <= self.chunk_size:
                segments.append(sentence)
            else:
                # Non-overlapping windows: overlap is applied later by
                # _merge_chunks, so adding it here would double-count it.
                segments.extend(
                    sentence[i : i + self.chunk_size]
                    for i in range(0, len(sentence), self.chunk_size)
                )
        return segments

    def _fixed_size_split(self, text: str) -> list[str]:
        """Fixed-size character-level splitting."""
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])
            start += self.chunk_size - self.chunk_overlap
        return chunks

    def _sentence_split(self, text: str) -> list[str]:
        """Split on sentence boundaries."""
        sentences = self._split_keep_separator(text, r"(?<=[.!?])\s+")
        return self._merge_chunks(sentences)

    def _merge_chunks(self, segments: list[str]) -> list[str]:
        """Merge segments into chunks respecting chunk_size and overlap."""
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for seg in segments:
            seg_len = len(seg)
            if current_len + seg_len > self.chunk_size and current:
                chunks.append("".join(current))
                # Keep overlap trailing content
                overlap_text = ""
                overlap_chars = 0
                for s in reversed(current):
                    if overlap_chars + len(s) > self.chunk_overlap:
                        needed = self.chunk_overlap - overlap_chars
                        if needed > 0:
                            overlap_text = s[-needed:] + overlap_text
                        break
                    overlap_text = s + overlap_text
                    overlap_chars += len(s)
                current = [overlap_text] if overlap_text else []
                current_len = len(overlap_text)

            current.append(seg)
            current_len += seg_len

        if current:
            chunks.append("".join(current))
        return chunks
