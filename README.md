# AI Document Intelligence Platform

A Retrieval-Augmented Generation (RAG) application that allows users to upload multiple PDFs, query them with grounded, citation-backed answers, generate structured summaries, and analyze resumes — built on Google's Gemini models and deployed on Streamlit Cloud.

---

## Overview

Most "chat with your PDF" demos break down the moment you give them more than one document — they either lose track of which file an answer came from, or silently ignore half the uploaded content. This project was built to solve that specific problem: a RAG pipeline that genuinely reasons across multiple documents, attributes every answer to its exact source and page, and degrades gracefully under real-world constraints like API rate limits and model deprecations.

## Features

| Feature | Description |
|---|---|
| **Multi-PDF Knowledge Base** | Upload and query multiple PDFs simultaneously. Uses MMR (Maximal Marginal Relevance) retrieval instead of plain similarity search, specifically to avoid one document dominating the retrieved context when documents are semantically similar. |
| **Source Citations** | Every answer is traceable to the exact filename and page number it was derived from. No citation is shown if the answer doesn't draw from the documents — the model is instructed to say "not found" rather than hallucinate a source. |
| **PDF Summarization** | Per-document structured summaries (Overview, Key Points, Conclusion), retrieved with metadata filtering so each file is summarized from its own content rather than a generic top-k match. |
| **Resume Analyzer** | Structured candidate evaluation: skills, education, experience, strengths, gaps, and an ATS-friendliness rating — built as a distinct prompt strategy, not a relabeled summarizer. |
| **Cloud Deployment** | Live on Streamlit Cloud with secrets-based API key management (no credentials in source control). |

## Architecture

PDF Upload → PyMuPDF text extraction (per page, with metadata) → RecursiveCharacterTextSplitter (chunking) → Gemini Embeddings → FAISS vector store → MMR Retriever (k=10, fetch_k=20) → Gemini 2.5 Flash (context-grounded generation) → Answer + Source citations

## Engineering Decisions

**Why MMR over plain similarity search?**
Default top-k similarity search returns the n closest chunks overall — which, across multiple documents, can mean all retrieved chunks come from a single file if that file happens to embed closer to the query. MMR retrieves a larger candidate pool and selects for diversity, which was necessary to make true cross-document comparison questions ("what's similar between these two PDFs?") actually work.

**Why batch the embedding calls?**
Google's free-tier embedding API enforces a strict per-minute request quota. Sending all chunks in a single call reliably triggered 429 RESOURCE_EXHAUSTED errors on multi-PDF uploads. The pipeline now batches chunks, paces requests under the quota window, and retries with backoff on transient rate-limit failures — rather than just catching the error and giving up.

**Why per-file metadata filtering for summarization/resume analysis?**
A single retrieval query against a multi-document vector store doesn't guarantee even coverage. Filtering by source metadata before retrieval guarantees every uploaded file is represented, instead of relying on embedding similarity to "happen" to surface it.

## Challenges & Fixes

- **Model deprecation mid-development** — gemini-1.5-flash, then gemini-2.0-flash were both retired by Google during the build. Solved by isolating the model name behind a single config point (qa_engine.py) and structuring the embedding/generation logic so a model swap requires a one-line change.
- **Silent multi-file ingestion failures** — Streamlit's file-pointer state across reruns occasionally caused a PDF's bytes to read as empty. Added a seek(0) guard and per-file page-count logging so any silent extraction failure is immediately visible in logs instead of failing invisibly.
- **Rate-limit cascading failures** — Initial batching logic didn't account for actual API request-per-batch-element behavior, causing the first "fix" to still exceed quota. Corrected by measuring real request volume against the documented quota window rather than guessing a delay value.

## Tech Stack

- Frontend: Streamlit
- PDF Parsing: PyMuPDF (fitz)
- Orchestration: LangChain
- Embeddings: Google Generative AI (gemini-embedding-001)
- Vector Store: FAISS
- LLM: Google Gemini (gemini-2.5-flash)
- Deployment: Streamlit Community Cloud

## Setup

Clone the repo, then run these commands in your terminal:

git clone https://github.com/sarani-nowbottu/ai-pdf-chat-app.git
cd ai-pdf-chat-app
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

Create a .env file in the project root with this line:

GOOGLE_API_KEY=your_api_key_here

Run locally:

streamlit run app.py

## Project Structure

ai-pdf-chat-app/
- app.py — Streamlit UI, session state, retrieval orchestration
- pdf_processor.py — PDF parsing, chunking, batched embedding pipeline
- qa_engine.py — Gemini LLM configuration
- requirements.txt
- README.md

## Limitations & Future Work

- Currently relies on Gemini's free tier, which imposes daily and per-minute quotas — a production version would need a paid tier or a queuing system for high-volume use.
- No persistent storage between sessions; the vector store is rebuilt on every upload rather than cached.
- Resume Analyzer currently handles one resume at a time — batch comparison across multiple candidates is a natural next feature.
- Scanned/image-based PDFs aren't supported yet; adding OCR (e.g. via Tesseract) would extend coverage to non-text PDFs.

## License

MIT
