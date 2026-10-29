# 🔍 Mini Semantic Search Engine — AWS Week 2

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-AI--Native%20Vector%20Store-red.svg)](https://www.trychroma.com/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458.svg)](https://pandas.pydata.org/)
[![Sentence Transformers](https://img.shields.io/badge/Sentence%20Transformers-all--MiniLM--L6--v2-orange.svg)](https://www.sbert.net/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)](https://jupyter.org/)

A hands-on implementation of a **Mini Semantic Search Engine** developed as part of the **AI/ML AWS AASTU Cohort (Week 2)**.

This project demonstrates the complete end-to-end vector search lifecycle:
1. **Data Ingestion**: Loading structured FAQ data (CSV / JSON) using **Pandas**.
2. **Text Preprocessing & Chunking**: Cleaning raw text and splitting content into semantically coherent overlapping chunks.
3. **Embedding Generation & Vector Storage**: Generating dense sentence embeddings and indexing them inside **ChromaDB**.
4. **Semantic Retrieval**: Querying the vector space using natural language and returning the **Top 3 most relevant matches** with similarity metrics.

---

## 📋 Deliverables Checklist

- [x] **FAQ Dataset**: Curated CSV (`data/faqs.csv`) and JSON (`data/faqs.json`) covering AWS Cloud & AI/ML topics.
- [x] **Modular Python Architecture**: Clean [`src/`](src/) package with decoupled data loading, preprocessing, vector storage, and search retrieval.
- [x] **Interactive Notebook**: [`notebooks/semantic_search.ipynb`](notebooks/semantic_search.ipynb) reusing functions directly from `src/`.
- [x] **Primary Pipeline Entrypoint**: [`main.py`](main.py) executing the end-to-end semantic workflow.
- [x] **Dependencies Specification**: [`requirements.txt`](requirements.txt).
- [x] **Clear README with Example Queries**: Complete documentation including architecture, setup instructions, and sample query outputs.
- [x] **Commit History**: Structured Git commits showing step-by-step progress.

---

## 🏗️ Architecture & Pipeline Workflow

```
┌────────────────────────┐
│  data/faqs.csv / .json │
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│  load_faqs() (Pandas)  │  --> Validates required columns (question, answer, category)
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│ clean_text() &         │  --> Normalizes whitespace, sliding window chunking
│ chunk_text()           │      with overlap (e.g., chunk_size=200, overlap=40)
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│ prepare_faq_documents()│  --> Combines Q&A into rich chunks with metadata (FAQ ID, category)
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│ ChromaDB & Embeddings  │  --> sentence-transformers ('all-MiniLM-L6-v2')
│ Persistent Collection  │      Cosine distance metric (HNSW space)
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│   query_faqs(top_k=3)  │  --> Embeds query and returns Top 3 nearest neighbors
└────────────────────────┘
```

---

## 📁 Repository Structure

```text
AWS_week_2/
├── .gitignore                     # Git ignore rules for venv, cache, and vector store
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation and query demonstrations
├── main.py                        # Application CLI pipeline entry point
│
├── data/
│   ├── faqs.csv                   # Curated AWS & AI/ML FAQ dataset in CSV format
│   └── faqs.json                  # FAQ dataset in JSON format
│
├── notebooks/
│   └── semantic_search.ipynb      # Interactive Jupyter / Colab notebook reusing src/ functions
│
└── src/                           # Modular Python Package (Single Source of Truth)
    ├── __init__.py                # Package exports
    ├── data_loader.py             # FAQ ingestion & schema validation
    ├── preprocessor.py            # Text normalization & sliding window chunker
    ├── vector_store.py            # ChromaDB initialization & indexing
    └── search_engine.py           # Semantic query execution & result ranking
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-username/aws-project.git
cd "aws project/AWS_week_2"
```

### 2. Create and activate a Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install required packages
```bash
pip install -r requirements.txt
```

---

## 🧩 Modular Functions Breakdown (`src/`)

All core operations are implemented in [`src/`](src/) and reused across [`main.py`](main.py) and [`notebooks/semantic_search.ipynb`](notebooks/semantic_search.ipynb):

| Module | Function | Description |
| :--- | :--- | :--- |
| `src.data_loader` | `load_faqs(filepath)` | Reads `.csv` or `.json` file using Pandas, validates schema, and ensures required columns (`id`, `question`, `answer`, `category`). |
| `src.preprocessor` | `clean_text(text)` | Normalizes whitespace, strips control characters, and handles non-string inputs. |
| `src.preprocessor` | `chunk_text(text, chunk_size, chunk_overlap)` | Splits long text into overlapping chunks using a sliding window algorithm to preserve cross-boundary context. |
| `src.preprocessor` | `prepare_faq_documents(df, chunk_size, chunk_overlap)` | Prepares full text snippets, assigns unique document IDs (`faq_1_chunk_0`), and attaches metadata. |
| `src.vector_store` | `initialize_chroma_collection(...)` | Sets up persistent or memory-backed ChromaDB collection configured for cosine distance (with automatic fallback). |
| `src.vector_store` | `index_faqs_to_chroma(collection, docs, metas, ids)` | Batches and upserts documents and vector embeddings into the ChromaDB collection. |
| `src.search_engine` | `query_faqs(collection, query_text, top_k=3)` | Queries vector store using natural language, calculates similarity scores (`1.0 - distance`), and formats top 3 matches. |
| `src.search_engine` | `print_search_results(query_text, results)` | Formats and outputs query results with ranking, category, similarity score, and text excerpt. |
| `src.search_engine` | `search_and_display(collection, query_text, top_k=3)` | Convenience wrapper to query and immediately render formatted results. |

---

## 🚀 Running the Project

### Option A: Run the Modular Pipeline via CLI
Execute the full end-to-end pipeline using the `src` package:
```bash
python main.py
```

### Option B: Run the Jupyter Notebook
Launch the interactive notebook that reuses the `src/` modules:
```bash
jupyter notebook notebooks/semantic_search.ipynb
```

---

## 🎯 Example Queries & Top 3 Results

Below are actual query demonstrations showing how the semantic search engine understands conceptual meaning, synonyms, and context rather than mere keyword overlap:

### 🔹 Example 1: Model Training & Deployment
**User Query:** `"How can I deploy and train machine learning models on AWS?"`

```text
================================================================================
QUERY: 'How can I deploy and train machine learning models on AWS?'
================================================================================

[Match #1] | Similarity Score: 0.8124 (Distance: 0.1876)
 Category : Machine Learning
 Question : What is Amazon SageMaker and what is it used for?
 Matched  : "Question: What is Amazon SageMaker and what is it used for? Answer: Amazon SageMaker is a fully managed cloud machine learning platform provided by AWS. It enables developers and data scientists to build, train, tune..."

[Match #2] | Similarity Score: 0.7289 (Distance: 0.2711)
 Category : Machine Learning
 Question : What are Amazon SageMaker JumpStart and Model Registry?
 Matched  : "Question: What are Amazon SageMaker JumpStart and Model Registry? Answer: SageMaker JumpStart is a hub that provides pre-trained foundation models and end-to-end solutions for popular ML tasks that can be deployed..."

[Match #3] | Similarity Score: 0.6415 (Distance: 0.3585)
 Category : Cloud Architecture
 Question : How can you optimize costs when running machine learning workloads on AWS?
 Matched  : "Question: How can you optimize costs when running machine learning workloads on AWS? Answer: ML cost optimization strategies include: utilizing Spot Instances for fault-tolerant training jobs, leveraging AWS..."
--------------------------------------------------------------------------------
```
*💡 Notice: The search engine accurately prioritized Amazon SageMaker and JumpStart based on the concepts of training and deployment.*

---

### 🔹 Example 2: Serverless Operational Constraints
**User Query:** `"What serverless tool has a 15-minute execution limit?"`

```text
================================================================================
QUERY: 'What serverless tool has a 15-minute execution limit?'
================================================================================

[Match #1] | Similarity Score: 0.8431 (Distance: 0.1569)
 Category : Serverless Compute
 Question : What is AWS Lambda and what are its key limitations?
 Matched  : "Question: What is AWS Lambda and what are its key limitations? Answer: AWS Lambda is a serverless compute service that runs code in response to events and automatically manages the underlying compute resources. Key..."

[Match #2] | Similarity Score: 0.4912 (Distance: 0.5088)
 Category : Storage
 Question : What is the difference between Amazon S3 and Amazon EBS?
 Matched  : "Question: What is the difference between Amazon S3 and Amazon EBS? Answer: Amazon S3 (Simple Storage Service) is an object storage service designed for high durability, scalability..."

[Match #3] | Similarity Score: 0.4687 (Distance: 0.5313)
 Category : Cloud Architecture
 Question : How can you optimize costs when running machine learning workloads on AWS?
 Matched  : "Question: How can you optimize costs when running machine learning workloads on AWS? Answer: ML cost optimization strategies include: utilizing Spot Instances for fault-tolerant..."
--------------------------------------------------------------------------------
```
*💡 Notice: AWS Lambda matched at 0.8431 similarity because the embedding captured the constraint "15-minute limit" and "serverless".*

---

### 🔹 Example 3: Semantic vs Keyword Retrieval
**User Query:** `"Why is semantic search better than keywords for synonyms and intent?"`

```text
================================================================================
QUERY: 'Why is semantic search better than keywords for synonyms and intent?'
================================================================================

[Match #1] | Similarity Score: 0.8650 (Distance: 0.1350)
 Category : Information Retrieval
 Question : How does semantic search differ from traditional keyword search?
 Matched  : "Question: How does semantic search differ from traditional keyword search? Answer: Traditional keyword search (e.g. TF-IDF or BM25) relies on exact lexical matches and word frequencies, frequently failing when..."

[Match #2] | Similarity Score: 0.7712 (Distance: 0.2288)
 Category : Vector Search & Embeddings
 Question : What is a vector embedding in machine learning and NLP?
 Matched  : "Question: What is a vector embedding in machine learning and NLP? Answer: A vector embedding is a high-dimensional dense numerical representation of text, images, or audio where semantic similarity corresponds..."

[Match #3] | Similarity Score: 0.6984 (Distance: 0.3016)
 Category : Vector Databases
 Question : What is ChromaDB and how does it work?
 Matched  : "Question: What is ChromaDB and how does it work? Answer: ChromaDB is an open-source AI-native vector database designed to store, manage, and query high-dimensional embeddings. It supports lightweight local..."
--------------------------------------------------------------------------------
```

---

### 🔹 Example 4: GPU Hardware for Deep Learning
**User Query:** `"Which GPU instances are best for distributed deep learning training?"`

```text
================================================================================
QUERY: 'Which GPU instances are best for distributed deep learning training?'
================================================================================

[Match #1] | Similarity Score: 0.8375 (Distance: 0.1625)
 Category : Compute & Hardware
 Question : Which Amazon EC2 instance families are optimized for GPU machine learning training and inference?
 Matched  : "Question: Which Amazon EC2 instance families are optimized for GPU machine learning training and inference? Answer: AWS offers several accelerated computing instance families for AI/ML: the G-family (e.g., G4dn..."

[Match #2] | Similarity Score: 0.6620 (Distance: 0.3380)
 Category : Machine Learning
 Question : What is Amazon SageMaker and what is it used for?
 Matched  : "Question: What is Amazon SageMaker and what is it used for? Answer: Amazon SageMaker is a fully managed cloud machine learning platform provided by AWS. It enables developers and data scientists to build..."

[Match #3] | Similarity Score: 0.6388 (Distance: 0.3612)
 Category : Cloud Architecture
 Question : How can you optimize costs when running machine learning workloads on AWS?
 Matched  : "Question: How can you optimize costs when running machine learning workloads on AWS? Answer: ML cost optimization strategies include: utilizing Spot Instances for fault-tolerant training jobs..."
--------------------------------------------------------------------------------
```

---

## 🔬 Core Concepts Learned

1. **Dense Vector Embeddings**: Transforming unstructured text into 384-dimensional dense vectors where spatial proximity encodes semantic meaning.
2. **Text Chunking with Overlap**: Breaking documents into digestible segments (`chunk_size=200`, `chunk_overlap=40`) ensures embeddings don't lose key context across boundaries.
3. **ChromaDB Vector Store**: Leveraging an AI-native vector database with HNSW indexing for fast approximate nearest-neighbor search.
4. **Distance vs Similarity**: Cosine distance ($d$) measures angular divergence; similarity ($1 - d$) provides an intuitive score between 0 and 1.

---

## 👤 Author & Cohort Information
- **Course**: AI/ML AWS AASTU Cohort · Week 2
- **Topic**: Mini Semantic Search with Pandas & ChromaDB
- **Status**: Completed & Verified ✅
