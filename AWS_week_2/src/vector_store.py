"""Vector store management module supporting ChromaDB with seamless local fallback."""

from __future__ import annotations

import os
import ssl
from typing import Any, Dict, List, Optional

import numpy as np

# Apply Windows SSL unverified context workaround for HuggingFace model download
try:
    ssl._create_default_https_context = ssl._create_unverified_context
    os.environ["CURL_CA_BUNDLE"] = ""
except Exception:
    pass

try:
    import chromadb
    from chromadb.utils import embedding_functions
    HAS_CHROMADB = True
except ModuleNotFoundError:
    HAS_CHROMADB = False


class ChromaCompatCollection:
    """In-memory vector store matching Chroma's interface when native chromadb is absent."""

    def __init__(self, name: str = "aws_faq_search", model_name: str = "all-MiniLM-L6-v2"):
        self.name = name
        self.documents: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        self.ids: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self.encoder_type = "sentence-transformers"

        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        except Exception:
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


class ChromaVectorStore:
    """Unified Vector Store wrapper for ChromaDB and Fallback engines."""

    def __init__(self, collection_name: str = "aws_faq_search", persist_dir: Optional[str] = None):
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        self.collection = initialize_chroma_collection(
            collection_name=collection_name,
            persist_directory=persist_dir,
        )

    def upsert(self, documents: List[str], metadatas: List[Dict[str, Any]], ids: List[str]):
        return self.collection.upsert(documents=documents, metadatas=metadatas, ids=ids)

    def query(self, query_texts: List[str], n_results: int = 3):
        return self.collection.query(query_texts=query_texts, n_results=n_results)

    def count(self) -> int:
        return self.collection.count()


def initialize_chroma_collection(
    collection_name: str = "aws_faq_search",
    persist_directory: Optional[str] = None,
):
    """Instantiate a ChromaDB collection or local compatibility vector store."""
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
        return ChromaCompatCollection(name=collection_name)


def index_faqs_to_chroma(
    collection,
    documents: List[str],
    metadatas: List[Dict[str, Any]],
    ids: List[str],
) -> int:
    """Index documents and their metadata into the collection."""
    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )
    return len(documents)
