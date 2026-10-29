"""Main application entry point for AWS Week 2 Mini Semantic Search.

Demonstrates the full end-to-end semantic search pipeline using the modular
components in the `src` package.
"""

from __future__ import annotations

import os
import sys

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import (
    load_faqs,
    prepare_faq_documents,
    initialize_chroma_collection,
    index_faqs_to_chroma,
    search_and_display,
)


def run_pipeline():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_csv = os.path.join(base_dir, "data", "faqs.csv")
    chroma_dir = os.path.join(base_dir, "chroma_db")

    print("\n" + "=" * 60)
    print(" [AWS Week 2] Mini Semantic Search Engine")
    print("=" * 60 + "\n")

    # Step 1: Ingestion
    print("[Step 1] Loading FAQ dataset using Pandas...")
    df = load_faqs(data_csv)
    print(f"  -> Successfully loaded {len(df)} FAQ entries from '{os.path.basename(data_csv)}'.")

    # Step 2: Preprocessing & Chunking
    print("\n[Step 2] Cleaning text and creating overlapping chunks...")
    documents, metadatas, ids = prepare_faq_documents(df, chunk_size=200, chunk_overlap=40)
    print(f"  -> Generated {len(documents)} text chunks from {len(df)} FAQs.")

    # Step 3: Vector Database Initialization & Indexing
    print("\n[Step 3] Initializing Vector Store & indexing embeddings...")
    collection = initialize_chroma_collection(
        collection_name="aws_faq_search",
        persist_directory=chroma_dir,
    )
    indexed_count = index_faqs_to_chroma(collection, documents, metadatas, ids)
    print(f"  -> Successfully indexed {indexed_count} chunks into collection '{collection.name}'.")

    # Step 4: Semantic Retrieval
    print("\n[Step 4] Running Semantic Search Queries (Top 3 Matches)...")
    sample_queries = [
        "How can I deploy and train machine learning models on AWS?",
        "What is the timeout limit for serverless functions?",
        "How does vector similarity search work compared to keyword matching?",
        "Which GPU instance family is optimized for deep learning training?",
    ]

    for query in sample_queries:
        search_and_display(collection, query, top_k=3)

    print("\n[Done] Mini Semantic Search pipeline completed successfully!\n")


if __name__ == "__main__":
    run_pipeline()
