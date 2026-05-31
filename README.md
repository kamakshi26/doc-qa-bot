# 📄 Document Q&A Bot

A RAG-powered web app that answers questions from any PDF with cited sources. Upload a document, ask questions in natural language, and get answers grounded exclusively in your document — no hallucinations.

## Live Demo
[Add your Streamlit URL here]

## What it does
- Upload any text-based PDF
- Ask questions in natural language
- Get answers grounded in the document with exact source citations
- Blocks hallucinations — won't answer questions outside the document
- Maintains conversation history within a session

## Tech stack
- **Frontend** — Streamlit
- **RAG pipeline** — LangChain + Qdrant (in-memory vector store)
- **Embeddings** — sentence-transformers/all-MiniLM-L6-v2
- **LLM** — GPT-4o via GitHub Models API
- **PDF parsing** — PyPDF

## Architecture
