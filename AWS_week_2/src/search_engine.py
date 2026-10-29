"""Semantic retrieval and result presentation module."""

from __future__ import annotations

from typing import Any, Dict, List

from src.preprocessor import clean_text


def query_faqs(
    collection,
    query_text: str,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """Query the vector collection and return top K matching FAQ chunks with similarity scores.

    Args:
        collection: The Chroma collection or ChromaVectorStore instance.
        query_text: The user query string.
        top_k: Number of nearest matches to retrieve.

    Returns:
        List of structured result dictionaries.
    """
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
    """Print formatted search results to console."""
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


def search_and_display(collection, query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """Convenience function to query and immediately print results."""
    matches = query_faqs(collection, query_text, top_k=top_k)
    print_search_results(query_text, matches)
    return matches
