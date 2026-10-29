"""Text preprocessing and chunking module for semantic search."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

import pandas as pd


def clean_text(text: str) -> str:
    """Normalize raw text by collapsing whitespace and stripping control characters."""
    if not isinstance(text, str):
        return ""
    cleaned = re.sub(r"\s+", " ", text)
    return cleaned.strip()


def chunk_text(text: str, chunk_size: int = 180, chunk_overlap: int = 35) -> List[str]:
    """Split text into overlapping character chunks using a sliding window.

    Args:
        text: Source text to split.
        chunk_size: Maximum character length for each segment.
        chunk_overlap: Overlapping character count between adjacent chunks.

    Returns:
        List of cleaned chunk strings.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []

    if len(cleaned) <= chunk_size:
        return [cleaned]

    chunks: List[str] = []
    start = 0
    text_length = len(cleaned)
    step = max(1, chunk_size - chunk_overlap)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_length:
            break
        start += step

    return chunks


def prepare_faq_documents(
    df: pd.DataFrame,
    chunk_size: int = 200,
    chunk_overlap: int = 40,
) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
    """Transform FAQ DataFrame rows into chunked documents with rich metadata and IDs.

    Args:
        df: Pandas DataFrame containing 'id', 'question', 'answer', and 'category'.
        chunk_size: Target chunk size in characters.
        chunk_overlap: Number of characters to overlap between adjacent chunks.

    Returns:
        Tuple containing (documents, metadatas, ids).
    """
    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    ids: List[str] = []

    for _, row in df.iterrows():
        faq_id = int(row["id"])
        question = clean_text(str(row["question"]))
        answer = clean_text(str(row["answer"]))
        category = clean_text(str(row.get("category", "General")))

        # Contextual representation merging intent + detail
        combined_text = f"Question: {question} Answer: {answer}"
        chunks = chunk_text(combined_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        for chunk_idx, chunk in enumerate(chunks):
            doc_id = f"faq_{faq_id}_chunk_{chunk_idx}"
            documents.append(chunk)
            metadatas.append({
                "faq_id": faq_id,
                "question": question,
                "category": category,
                "chunk_index": chunk_idx,
                "total_chunks": len(chunks),
            })
            ids.append(doc_id)

    return documents, metadatas, ids
