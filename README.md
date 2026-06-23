# AI PDF Chat with Gemini

A Retrieval-Augmented Generation (RAG) application that allows users to upload PDFs and ask questions about their content.

## Features

- Upload PDF documents
- Extract text using PyMuPDF
- Split text into chunks
- Generate embeddings using Gemini
- Store embeddings in FAISS
- Ask questions about uploaded PDFs
- Receive context-aware answers

## Tech Stack

- Python
- Streamlit
- LangChain
- Google Gemini API
- FAISS
- PyMuPDF

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Project Flow

PDF
↓
Extract Text
↓
Create Embeddings
↓
Store in FAISS
↓
Ask Questions
↓
Gemini Answers
