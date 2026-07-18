# 📚 Multi-PDF Chatbot

A RAG (Retrieval-Augmented Generation) chatbot that answers questions from any uploaded PDF(s) — with source citations for every answer.

🔗 Live app: https://nanda-pdf-chatbot.streamlit.app
📂 Repo: https://github.com/27122003-nanda/pdf-multi-chatbot

## What it does

Upload one or more PDFs, then ask questions in plain English to get accurate answers grounded only in the uploaded documents, with the exact source chunks shown for transparency. Supports multiple documents at once, with answers indicating which file they came from.

## Architecture

PDF Upload (one or more files) leads to Text Extraction using pdfplumber, then Chunking using LangChain's RecursiveCharacterTextSplitter (500 characters, 50 overlap), then Embeddings using sentence-transformers (all-MiniLM-L6-v2), then a per-document FAISS index for balanced retrieval across all uploaded files. When a user asks a question, it is embedded and the top matching chunks are retrieved per document, then the context and question are sent to the Groq LLM (Llama 3.3 70B), which returns an answer along with the source chunks used, displayed in the chat UI.

## Tech stack

- PDF parsing: pdfplumber, for reliable text extraction
- Chunking: LangChain text splitters, to keep chunks semantically coherent
- Embeddings: sentence-transformers (MiniLM), free and lightweight, runs on CPU
- Vector search: FAISS, fast with no external database needed
- LLM: Groq API (Llama 3.3 70B), free tier with very low latency
- UI and hosting: Streamlit and Streamlit Community Cloud, free and simple to deploy from GitHub

## Key features

- Multi-PDF support: upload several documents; retrieval pulls from each one individually so small files aren't drowned out by larger ones
- Source transparency: every answer shows which document and chunk it came from
- Graceful fallback: if the answer isn't in the document(s), the bot says so instead of guessing
- Session-based: no data is stored beyond your browser session; nothing persists after you close the tab

## Possible extensions

- OCR support for scanned or image-only PDFs
- Persistent chat history across sessions
- Support for other file types such as docx and txt

Built as part of an internship project exploring applied RAG systems.
