"""Mini Semantic Search Package - AWS Week 2

Provides modular components for:
- Data loading and schema validation (data_loader)
- Text cleaning and sliding-window chunking (preprocessor)
- Vector store management and embedding indexing (vector_store)
- Semantic query execution and formatted presentation (search_engine)
"""

from src.data_loader import load_faqs
from src.preprocessor import clean_text, chunk_text, prepare_faq_documents
from src.vector_store import (
    initialize_chroma_collection,
    index_faqs_to_chroma,
    ChromaVectorStore,
)
from src.search_engine import query_faqs, print_search_results, search_and_display

__all__ = [
    "load_faqs",
    "clean_text",
    "chunk_text",
    "prepare_faq_documents",
    "initialize_chroma_collection",
    "index_faqs_to_chroma",
    "ChromaVectorStore",
    "query_faqs",
    "print_search_results",
    "search_and_display",
]
