# 📄 Document Q&A Bot

A RAG-powered web app that answers questions from any PDF with cited sources. Upload a document, ask questions in natural language, and get answers grounded exclusively in your document — no hallucinations.

## Live Demo
[(https://doc-app-bot-ymcjaelzxkg7wbzw6q8ktz.streamlit.app/)]

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
PDF upload → text extraction (PyPDF) → chunking (RecursiveCharacterTextSplitter, 500 chars, 50 overlap)
→ embedding (MiniLM-L6-v2, 384 dimensions) → storage (Qdrant in-memory)
→ semantic search (cosine similarity, top-3 chunks) → prompt injection → GPT-4o → cited answer

## How RAG works here
1. The uploaded PDF is split into 500-character overlapping chunks
2. Each chunk is converted into a 384-dimension vector using sentence-transformers
3. Vectors are stored in a Qdrant in-memory vector database
4. When a question is asked, it is also converted to a vector
5. The top 3 most semantically similar chunks are retrieved
6. Those chunks are injected into the LLM prompt as context
7. The LLM answers using only that context and cites the exact source sentence

## What I learned building this
- Chunking strategy directly affects retrieval quality — tested fixed, paragraph, page, and recursive strategies on a real 59-page IRS document
- PDF text extraction produces noise (page headers, dot leaders, hyphenated line breaks) that must be cleaned before embedding or retrieval quality degrades
- Query specificity matters more than query length — "what is the penalty for filing taxes late" retrieves better than "what happens if I miss the deadline"
- Scanned PDFs extract 0 text — OCR support is a planned enhancement
- LLM confidence doesn't equal retrieval accuracy — a confident wrong answer usually means the wrong chunk was retrieved, not that the LLM hallucinated

## Sample questions to try
Upload any IRS publication and ask:
- "What is the standard deduction for a single filer?"
- "What is the penalty for filing taxes late?"
- "What is the FUTA tax rate?"
- "Who is the CEO of Apple?" — tests hallucination guard

## Limitations
- Text-based PDFs only — scanned or image-based PDFs are not supported
- Single PDF per session — uploading a new PDF clears the previous conversation
- Retrieval struggles with tabular data such as tax bracket tables — rows get separated across chunks
- In-memory Qdrant means vectors are not persisted between sessions — each upload re-indexes

## Run locally
1. Clone the repo
```bash
   git clone https://github.com/yourusername/doc-qa-bot.git
   cd doc-qa-bot
```
2. Install dependencies
```bash
   pip install -r requirements.txt
```
3. Add your API key — create `.streamlit/secrets.toml`
```toml
   GITHUB_TOKEN = "your_github_personal_access_token"
```
4. Run the app
```bash
   streamlit run app.py
```

## Requirements
streamlit
langchain
langchain-community
langchain-huggingface
langchain-openai
pypdf
qdrant-client
sentence-transformers
