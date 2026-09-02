# LangChainDS

A learning project exploring Retrieval-Augmented Generation (RAG) with [LangChain](https://python.langchain.com/), built entirely with local, API-key-free models (HuggingFace embeddings + Ollama for generation).

The work lives in [notebook/document.ipynb](notebook/document.ipynb) and walks through a full RAG pipeline step by step, from loading raw documents to answering questions grounded in retrieved context.

## What's been built

**Document loading**
- Plain text via `TextLoader`.
- PDFs via `PyPDFLoader` / `PyMuPDFLoader`, including bulk loading of a whole directory with `DirectoryLoader` and a custom `process_all_pdfs()` helper that tags each document with `source_file` and `file_type` metadata.

**Chunking** — two strategies, compared side by side:
- Fixed-size splitting with `RecursiveCharacterTextSplitter` (1000 chars, 200 overlap).
- Semantic chunking with `SemanticChunker`, which embeds sentences and splits where topic similarity actually drops, rather than at a fixed character count. Uses a local `sentence-transformers` model (`all-MiniLM-L6-v2`), so no API key is required.

**Vector storage & retrieval**
- Semantic chunks are embedded and persisted to a local [Chroma](https://www.trychroma.com/) store (`data/vector_store/semantic_chunks`).
- Retrieval uses MMR (Maximal Marginal Relevance) search to balance relevance against diversity, so an answer spread across multiple slides doesn't get buried under near-duplicate chunks.

**RAG chain**
- Retriever wired to a locally running `llama3.2` model via `ChatOllama`.
- A prompt template that instructs the model to answer only from retrieved context, with deduplication of repeated chunks and normalization of question casing (e.g. "Intelligent Agent" vs "intelligent agent" are treated as the same concept).
- Verified end-to-end on sample questions (e.g. "What is an intelligent agent?", "What is the 8-queens problem?") against a small set of AI-course lecture PDFs.

## Stack

Python 3.12+, managed with [uv](https://docs.astral.sh/uv/) (`pyproject.toml` / `uv.lock`). Key libraries: `langchain`, `langchain-community`, `langchain-experimental`, `langchain-chroma`, `langchain-huggingface`, `langchain-ollama`, `sentence-transformers`, `pypdf` / `pymupdf`.

## Constraints & things learned along the way

- **Everything runs locally, on purpose.** Embeddings use a local `sentence-transformers` model and generation uses a local Ollama model, so the whole pipeline works offline with no API keys or paid usage — at the cost of noticeably slower inference than a hosted API model.
- **`langchain-community` and `langchain-experimental` are both in sunset/deprecation status** upstream (used here for `TextLoader`/`DirectoryLoader`/`PyPDFLoader` and `SemanticChunker` respectively). They still work but emit deprecation warnings; a future iteration should migrate to their standalone replacement packages as those stabilize.
- **Chroma's `from_documents()` appends rather than replaces** an existing persisted collection. Re-running the indexing cell without clearing `persist_directory` first silently duplicates every chunk on each run, which degrades retrieval quality — the notebook now clears the directory before each re-index.

## Project structure

```
data/
  pdf_files/        # source PDFs (local only, not committed)
  text_files/        # sample text documents
  vector_store/       # persisted Chroma index (local only, not committed)
notebook/
  document.ipynb      # the RAG pipeline, step by step
main.py              # placeholder entry point
```
