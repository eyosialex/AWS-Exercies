"""Mini Semantic Search Engine - AWS Week 2 Lab

This module provides reusable, modular functions to:
1. Load FAQ dataset from CSV or JSON using Pandas.
2. Clean and chunk textual content with configurable overlap.
3. Generate vector embeddings and store them in ChromaDB.
4. Perform semantic search queries and return top matches with similarity scores.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# ============================================================================
# 1. DATA LOADING FUNCTIONS
# ============================================================================

def load_faqs(filepath: str) -> pd.DataFrame:
    """Load FAQ dataset from either a CSV or JSON file into a Pandas DataFrame.

    Args:
        filepath: Absolute or relative path to the CSV/JSON file.

    Returns:
        pd.DataFrame containing the FAQ records.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(filepath)
    elif ext == ".json":
        df = pd.read_json(filepath)
    else:
        raise ValueError(f"Unsupported file format '{ext}'. Expected .csv or .json")

    # Validate essential columns
    required_cols = {"question", "answer"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns: {required_cols}. Found: {list(df.columns)}")

    # Ensure id column exists
    if "id" not in df.columns:
        df["id"] = range(1, len(df) + 1)

    # Ensure category column exists
    if "category" not in df.columns:
        df["category"] = "General"

    return df


# ============================================================================
# 2. TEXT CLEANING & CHUNKING FUNCTIONS
# ============================================================================

def clean_text(text: str) -> str:
    """Clean and normalize raw text.

    Removes excessive whitespace, line breaks, and stray non-standard characters.
    """
    if not isinstance(text, str):
        return ""
    # Normalize multiple whitespace and newlines to a single space
    cleaned = re.sub(r"\s+", " ", text)
    # Strip leading and trailing whitespace
    return cleaned.strip()


def chunk_text(text: str, chunk_size: int = 150, chunk_overlap: int = 30) -> List[str]:
    """Split a string into overlapping character/word chunks.

    Args:
        text: The source text to chunk.
        chunk_size: Maximum character length for each chunk.
        chunk_overlap: Overlap length in characters between adjacent chunks.

    Returns:
        List of text chunks.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []

    # If the text is shorter than or equal to chunk_size, return it as single chunk
    if len(cleaned) <= chunk_size:
        return [cleaned]

    chunks = []
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
    """Clean, chunk, and package FAQ entries into documents, metadata, and unique IDs.

    Args:
        df: Pandas DataFrame containing 'id', 'question', 'answer', and 'category'.
        chunk_size: Target chunk size in characters.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        Tuple of (documents, metadatas, ids) ready for ChromaDB.
    """
    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    ids: List[str] = []

    for _, row in df.iterrows():
        faq_id = int(row["id"])
        question = clean_text(str(row["question"]))
        answer = clean_text(str(row["answer"]))
        category = clean_text(str(row.get("category", "General")))

        # Combined text captures both the user's intent and the detailed explanation
        full_content = f"Question: {question} Answer: {answer}"
        chunks = chunk_text(full_content, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

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


# ============================================================================
# 3. CHROMADB VECTOR STORE FUNCTIONS
# ============================================================================

def get_embedding_function(model_name: str = "all-MiniLM-L6-v2"):
    """Instantiate a sentence-transformers embedding function compatible with ChromaDB.

    Falls back to Chroma's default embedding function if sentence_transformers
    is not installed.
    """
    try:
        from chromadb.utils import embedding_functions
        return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
    except Exception:
        try:
            import chromadb.utils.embedding_functions as ef
            return ef.DefaultEmbeddingFunction()
        except Exception:
            return None


def initialize_chroma_collection(
    collection_name: str = "aws_faq_search",
    persist_directory: Optional[str] = None,
    embedding_function=None,
):
    """Initialize a ChromaDB client and collection.

    Args:
        collection_name: Name of the collection.
        persist_directory: Optional disk directory to persist the vector database.
        embedding_function: Embedding function to vectorize texts.

    Returns:
        ChromaDB Collection instance.
    """
    import chromadb

    if persist_directory:
        os.makedirs(persist_directory, exist_ok=True)
        client = chromadb.PersistentClient(path=persist_directory)
    else:
        client = chromadb.Client()

    if embedding_function is None:
        embedding_function = get_embedding_function()

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def index_faqs_to_chroma(
    collection,
    documents: List[str],
    metadatas: List[Dict[str, Any]],
    ids: List[str],
    batch_size: int = 64,
) -> int:
    """Index prepared documents and their metadata into ChromaDB in batches.

    Returns:
        Total number of documents indexed.
    """
    total = len(documents)
    for i in range(0, total, batch_size):
        end = min(i + batch_size, total)
        collection.upsert(
            documents=documents[i:end],
            metadatas=metadatas[i:end],
            ids=ids[i:end],
        )
    return total


# ============================================================================
# 4. QUERYING & RESULT DISPLAY FUNCTIONS
# ============================================================================

def query_faqs(
    collection,
    query_text: str,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """Query ChromaDB collection and return top K matching FAQ chunks with similarity score.

    Args:
        collection: The ChromaDB collection to query.
        query_text: Natural language user question or phrase.
        top_k: Number of matches to retrieve (default is 3 as required by lab).

    Returns:
        List of result dictionaries sorted by relevance.
    """
    cleaned_query = clean_text(query_text)
    results = collection.query(
        query_texts=[cleaned_query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    formatted_results: List[Dict[str, Any]] = []
    if not results or not results["documents"] or not results["documents"][0]:
        return formatted_results

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    for rank, (doc, meta, dist) in enumerate(zip(docs, metas, distances), start=1):
        # For cosine distance: distance is in [0, 2], similarity = 1 - distance
        similarity = max(0.0, 1.0 - dist) if dist is not None else 0.0
        formatted_results.append({
            "rank": rank,
            "score": round(similarity, 4),
            "distance": round(dist, 4) if dist is not None else None,
            "question": meta.get("question", "N/A"),
            "category": meta.get("category", "General"),
            "faq_id": meta.get("faq_id"),
            "matched_chunk": doc,
        })

    return formatted_results


def print_search_results(query_text: str, results: List[Dict[str, Any]]) -> None:
    """Print formatted search results to the console."""
    print("=" * 80)
    print(f"QUERY: '{query_text}'")
    print("=" * 80)
    if not results:
        print("No matches found.")
        return

    for res in results:
        print(f"\n[Match #{res['rank']}] | Similarity Score: {res['score']:.4f} (Distance: {res['distance']})")
        print(f" Category : {res['category']}")
        print(f" Question : {res['question']}")
        print(f" Matched  : \"{res['matched_chunk']}\"")
    print("-" * 80)


# ============================================================================
# 5. MAIN DEMO EXECUTION
# ============================================================================

def main():
    """Run an end-to-end demonstration of the Mini Semantic Search pipeline."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data", "faqs.csv")
    chroma_dir = os.path.join(base_dir, "chroma_db")

    print("[Step 1] Loading FAQs dataset with Pandas...")
    df = load_faqs(data_path)
    print(f"  -> Successfully loaded {len(df)} FAQ entries from '{data_path}'.")

    print("\n[Step 2] Cleaning and chunking text...")
    documents, metadatas, ids = prepare_faq_documents(df, chunk_size=220, chunk_overlap=40)
    print(f"  -> Generated {len(documents)} text chunks with overlap from {len(df)} FAQs.")

    print("\n[Step 3] Initializing ChromaDB vector store and embedding function...")
    embedding_func = get_embedding_function("all-MiniLM-L6-v2")
    collection = initialize_chroma_collection(
        collection_name="aws_faq_search",
        persist_directory=chroma_dir,
        embedding_function=embedding_func,
    )
    indexed_count = index_faqs_to_chroma(collection, documents, metadatas, ids)
    print(f"  -> Successfully indexed {indexed_count} chunks into Chroma collection '{collection.name}'.")

    print("\n[Step 4] Querying Mini Semantic Search (Top 3 Matches)...")
    sample_queries = [
        "How can I deploy and train machine learning models on AWS?",
        "What is the timeout limit for serverless functions?",
        "How does vector similarity search work compared to keyword matching?",
    ]

    for q in sample_queries:
        top_matches = query_faqs(collection, query_text=q, top_k=3)
        print_search_results(q, top_matches)


if __name__ == "__main__":
    main()
