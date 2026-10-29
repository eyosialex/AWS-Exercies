"""Mini Semantic Search Engine - AWS Week 2 Lab

This module provides reusable, modular functions to:
1. Load FAQ dataset from CSV or JSON using Pandas.
2. Clean and chunk textual content with configurable overlap.
3. Generate vector embeddings and store them in ChromaDB (with robust offline fallback).
4. Perform semantic search queries and return top 3 matches with similarity scores.
"""

from __future__ import annotations

import os
import re
import ssl
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Workaround for Windows SSL certificate verification when reaching HuggingFace / PyPI
try:
    ssl._create_default_https_context = ssl._create_unverified_context
    os.environ["CURL_CA_BUNDLE"] = ""
except Exception:
    pass

# Check for native ChromaDB
try:
    import chromadb
    from chromadb.utils import embedding_functions
    HAS_CHROMADB = True
except ModuleNotFoundError:
    HAS_CHROMADB = False


# ============================================================================
# 1. DATA LOADING FUNCTIONS
# ============================================================================

def load_faqs(filepath: str) -> pd.DataFrame:
    """Load FAQ dataset from CSV or JSON into a clean Pandas DataFrame.

    Args:
        filepath: Absolute or relative path to the CSV/JSON file.

    Returns:
        pd.DataFrame containing the FAQ records.
    """
    if not os.path.exists(filepath):
        # Look in sibling data folder or current directory
        candidate = os.path.join(os.path.dirname(__file__), "data", os.path.basename(filepath))
        if os.path.exists(candidate):
            filepath = candidate
        else:
            raise FileNotFoundError(f"File not found: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(filepath, quotechar='"', skipinitialspace=True)
    elif ext == ".json":
        df = pd.read_json(filepath)
    else:
        raise ValueError(f"Unsupported file format '{ext}'. Expected .csv or .json")

    required_cols = {"question", "answer"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns: {required_cols}. Found: {list(df.columns)}")

    if "id" not in df.columns:
        df["id"] = range(1, len(df) + 1)

    if "category" not in df.columns:
        df["category"] = "General"

    return df


# ============================================================================
# 2. TEXT CLEANING & CHUNKING FUNCTIONS
# ============================================================================

def clean_text(text: str) -> str:
    """Clean and normalize raw text, stripping extra whitespace and control chars."""
    if not isinstance(text, str):
        return ""
    cleaned = re.sub(r"\s+", " ", text)
    return cleaned.strip()


def chunk_text(text: str, chunk_size: int = 180, chunk_overlap: int = 35) -> List[str]:
    """Split text into overlapping character chunks using a sliding window algorithm."""
    cleaned = clean_text(text)
    if not cleaned:
        return []

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
    """Clean, chunk, and package FAQ entries into documents, metadata, and unique IDs."""
    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    ids: List[str] = []

    for _, row in df.iterrows():
        faq_id = int(row["id"])
        question = clean_text(str(row["question"]))
        answer = clean_text(str(row["answer"]))
        category = clean_text(str(row.get("category", "General")))

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
# 3. VECTOR STORE & EMBEDDING ENGINES
# ============================================================================

class ChromaCompatCollection:
    """ChromaDB-compatible collection using local vector embeddings (SentenceTransformers or TF-IDF)."""

    def __init__(self, name: str = "aws_faq_search"):
        self.name = name
        self.documents: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        self.ids: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self.encoder_type = "sentence-transformers"

        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            # Fully offline vector embedder via Scikit-Learn
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.model = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
            self.encoder_type = "tfidf"

    def upsert(self, documents: List[str], metadatas: List[Dict[str, Any]], ids: List[str]):
        self.documents = list(documents)
        self.metadatas = list(metadatas)
        self.ids = list(ids)

        if self.encoder_type == "sentence-transformers":
            embs = self.model.encode(self.documents, convert_to_numpy=True, normalize_embeddings=True)
            self.embeddings = embs
        else:
            embs = self.model.fit_transform(self.documents).toarray()
            # Normalize for cosine similarity
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self.embeddings = embs / norms

    def query(self, query_texts: List[str], n_results: int = 3, include=None) -> Dict[str, Any]:
        if self.encoder_type == "sentence-transformers":
            q_embs = self.model.encode(query_texts, convert_to_numpy=True, normalize_embeddings=True)
        else:
            q_embs = self.model.transform(query_texts).toarray()
            norms = np.linalg.norm(q_embs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            q_embs = q_embs / norms

        sims = np.dot(q_embs, self.embeddings.T)[0]
        top_k_indices = np.argsort(sims)[::-1][:n_results]

        res_docs = [self.documents[i] for i in top_k_indices]
        res_metas = [self.metadatas[i] for i in top_k_indices]
        res_dists = [float(max(0.0, 1.0 - sims[i])) for i in top_k_indices]

        return {
            "documents": [res_docs],
            "metadatas": [res_metas],
            "distances": [res_dists],
        }

    def count(self) -> int:
        return len(self.documents)


def initialize_chroma_collection(
    collection_name: str = "aws_faq_search",
    persist_directory: Optional[str] = None,
):
    """Initialize ChromaDB collection or compatible vector store."""
    if HAS_CHROMADB:
        if persist_directory:
            os.makedirs(persist_directory, exist_ok=True)
            client = chromadb.PersistentClient(path=persist_directory)
        else:
            client = chromadb.Client()

        try:
            embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            return client.get_or_create_collection(
                name=collection_name,
                embedding_function=embedding_fn,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception:
            return client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
    else:
        print("[Notice] Using Chroma-compatible vector store engine.")
        return ChromaCompatCollection(name=collection_name)


def index_faqs_to_chroma(
    collection,
    documents: List[str],
    metadatas: List[Dict[str, Any]],
    ids: List[str],
) -> int:
    """Index documents and metadata into the collection."""
    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )
    return len(documents)


# ============================================================================
# 4. QUERYING & RESULT DISPLAY FUNCTIONS
# ============================================================================

def query_faqs(
    collection,
    query_text: str,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """Query the vector collection and return top K matching FAQ chunks with similarity scores."""
    cleaned_query = clean_text(query_text)
    results = collection.query(
        query_texts=[cleaned_query],
        n_results=top_k,
    )

    formatted_results: List[Dict[str, Any]] = []
    if not results or not results["documents"] or not results["documents"][0]:
        return formatted_results

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    for rank, (doc, meta, dist) in enumerate(zip(docs, metas, distances), start=1):
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
    documents, metadatas, ids = prepare_faq_documents(df, chunk_size=200, chunk_overlap=40)
    print(f"  -> Generated {len(documents)} text chunks with overlap from {len(df)} FAQs.")

    print("\n[Step 3] Initializing Vector Store...")
    collection = initialize_chroma_collection(
        collection_name="aws_faq_search",
        persist_directory=chroma_dir,
    )
    indexed_count = index_faqs_to_chroma(collection, documents, metadatas, ids)
    print(f"  -> Successfully indexed {indexed_count} chunks into collection '{collection.name}'.")

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
