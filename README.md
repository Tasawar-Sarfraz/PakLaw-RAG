# 🏛️ Government Multi-Department RAG

An enterprise-oriented **Retrieval-Augmented Generation (RAG)** application designed to search and answer questions from government documents across multiple departments.

The system processes government PDFs, preserves document structure including tables, creates sequential chunks, generates embeddings, stores them in a FAISS vector index, and uses **Groq GPT-OSS 120B** to generate grounded answers through a Streamlit interface.

---

## 🚀 Project Overview

Government information is often distributed across multiple PDF documents and departments. Finding relevant information manually can be time-consuming.

This project provides a centralized AI-powered knowledge assistant that allows users to ask questions in natural language and retrieve relevant information from government documents.

### Core Pipeline

```text
Government PDFs
      ↓
PDF Text & Table Extraction
      ↓
Structured Document Content
      ↓
Sequential Word-Safe Chunking
      ↓
BGE-M3 Embeddings
      ↓
FAISS Vector Database
      ↓
Semantic Retrieval
      ↓
Relevant Context
      ↓
Groq GPT-OSS 120B
      ↓
Grounded Answer
      ↓
Streamlit UI
```

---

## ✨ Key Features

### 📚 Multi-Department Knowledge Base

The system supports documents from multiple government departments.

Each chunk contains metadata such as:

* Department
* PDF source
* Page number
* Chunk ID
* Chunk order
* File path
* Start/end position

---

### 📄 PDF Document Processing

The ingestion pipeline extracts content directly from PDF documents using **PyMuPDF**.

The system processes documents page-by-page to maintain document order.

---

### 📊 Table Preservation

Detected PDF tables are converted into Markdown table format.

Example:

```text
[TABLE]

| Department | Budget | Year |
| --- | --- | --- |
| Finance | 100M | 2026 |
| Health | 80M | 2026 |

[/TABLE]
```

This allows the LLM to understand tabular information more effectively than treating table content as unstructured text.

---

### 🔗 Sequential Chunking

The chunking system processes content sequentially.

It avoids arbitrary word cuts and maintains:

* Document order
* Page order
* Chunk order
* Word boundaries
* Chunk overlap

For example:

```text
Page 1
 ├── Chunk 1
 ├── Chunk 2
 └── Chunk 3

Page 2
 ├── Chunk 1
 ├── Chunk 2
 └── Chunk 3
```

The system completes the chunks of one page before moving to the next page.

---

### 🧠 Semantic Embeddings

The project uses:

```text
BAAI/bge-m3
```

to convert document chunks into numerical vector representations.

Embeddings are generated **once during the ingestion/indexing process**.

The Streamlit application does not regenerate document embeddings.

Only the user's query is embedded during search.

---

### ⚡ FAISS Vector Search

The generated embeddings are stored using:

```text
FAISS IndexFlatIP
```

Because the embeddings are normalized, Inner Product is used as cosine similarity.

This allows the system to retrieve semantically relevant document chunks for user questions.

---

### 🤖 Grounded LLM Responses

The application uses:

```text
openai/gpt-oss-120b
```

through Groq.

The LLM receives the retrieved government document context and is instructed to answer using the provided documents rather than inventing government policies or information.

---

### 💬 Streamlit Interface

The frontend provides:

* Natural-language question input
* Department filtering
* Configurable retrieval count
* AI-generated answers
* Source documents
* Page numbers
* Similarity scores
* Retrieved document previews

---

## 🏗️ Project Structure

```text
paklaw-rag/
│
├── app.py
│
├── requirements.txt
│
├── README.md
│
├── rag/
│   ├── __init__.py
│   ├── config.py
│   ├── retriever.py
│   └── llm.py
│
├── vectorstore/
│   ├── faiss.index
│   ├── metadata.pkl
│   └── config.json
│
└── .streamlit/
    └── secrets.toml
```

---

## 🛠️ Technology Stack

| Technology            | Purpose                             |
| --------------------- | ----------------------------------- |
| Python                | Application development             |
| Streamlit             | Interactive web interface           |
| PyMuPDF               | PDF processing                      |
| Sentence Transformers | Embeddings                          |
| BGE-M3                | Embedding model                     |
| FAISS                 | Vector similarity search            |
| Groq                  | LLM inference                       |
| GPT-OSS 120B          | Generative AI                       |
| Google Colab          | One-time embedding/index generation |
| GitHub                | Source code management              |
| Streamlit Cloud       | Application deployment              |

---

## 🔄 Embedding & Retrieval Architecture

### One-Time Indexing

The Colab pipeline performs:

```text
PDF
 ↓
Text/Table Extraction
 ↓
Chunking
 ↓
BGE-M3
 ↓
Embeddings
 ↓
FAISS
 ↓
vectorstore/
```

The generated vectorstore contains:

```text
faiss.index
metadata.pkl
config.json
```

These files are then included with the application.

---

### Runtime Retrieval

When a user asks:

```text
What is the government policy regarding X?
```

the application performs:

```text
User Question
      ↓
Query Embedding
      ↓
FAISS Search
      ↓
Relevant Chunks
      ↓
Department/Page/Source Metadata
      ↓
LLM Context
      ↓
Grounded Answer
```

Document embeddings are **not regenerated at application startup**.

---

## ⚙️ Configuration

`rag/config.py` contains the main configuration:

```python
EMBEDDING_MODEL = "BAAI/bge-m3"

GROQ_MODEL = "openai/gpt-oss-120b"

TOP_K = 6

VECTORSTORE_PATH = "vectorstore"
```

---

## 🔐 Environment Variables / Secrets

The Groq API key should not be hardcoded.

For Streamlit deployment, add the secret:

```toml
GROQ_API_KEY = "your_groq_api_key"
```

Do not commit:

```text
.streamlit/secrets.toml
```

to GitHub.

Add it to `.gitignore`.

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/Tasawar-Sarfraz/paklaw-rag.git
```

Move into the project:

```bash
cd paklaw-rag
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run Locally

Start Streamlit:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## ☁️ Streamlit Deployment

The application can be deployed using Streamlit Cloud.

Deployment requirements:

1. Push the project to GitHub.
2. Select the repository.
3. Select `app.py` as the main file.
4. Configure `GROQ_API_KEY` in Streamlit Secrets.
5. Deploy the application.

The `vectorstore` directory must contain:

```text
faiss.index
metadata.pkl
config.json
```

---

## 📈 Retrieval Metadata

Every retrieved chunk contains information such as:

```text
Department
Source
Page
Chunk ID
Chunk Order
Similarity Score
```

This provides traceability between an AI response and the source government document.

---

## 🧪 Example Query

A user can ask:

```text
What are the eligibility requirements mentioned in the document?
```

The system:

1. Converts the question into an embedding.
2. Searches the FAISS index.
3. Retrieves relevant chunks.
4. Builds the document context.
5. Sends the context to GPT-OSS 120B.
6. Generates a grounded answer.
7. Displays the relevant sources.

---

## 🎯 Design Goals

The project focuses on:

* Grounded AI responses
* Semantic document retrieval
* Multi-department knowledge access
* Source traceability
* Structured PDF processing
* Table preservation
* Sequential chunking
* Reusable vector indexes
* Efficient runtime retrieval

---

## 🔮 Future Improvements

Potential improvements include:

* Advanced two-column PDF layout detection
* Hybrid keyword + semantic search
* Cross-encoder reranking
* Query rewriting
* Conversation memory
* Citation-level answer grounding
* Document versioning
* Access control by department
* OCR support for scanned PDFs
* Advanced table understanding
* Evaluation datasets and retrieval metrics
* Monitoring and observability
* Production-grade vector database integration

---

## 👨‍💻 Author

**Tasawar Sarfraz**

Data Science & AI Developer

Focused on building practical solutions using:

* Artificial Intelligence
* Generative AI
* RAG
* Machine Learning
* Data Science
* LLM Applications

---

## 📄 License

This project is intended for educational and development purposes.
