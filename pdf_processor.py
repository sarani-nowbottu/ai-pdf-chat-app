import fitz
import time
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()


def process_pdfs(uploaded_files):
    docs = []
    pages_per_file = {}

    for pdf in uploaded_files:
        filename = pdf.name

        pdf.seek(0)
        pdf_bytes = pdf.read()

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        page_count_for_this_file = 0

        for page_num, page in enumerate(doc):
            text = page.get_text("text").strip()

            if text:
                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": filename,
                            "page": page_num + 1,
                            "total_pages": doc.page_count
                        }
                    )
                )
                page_count_for_this_file += 1

        pages_per_file[filename] = page_count_for_this_file

    print("Pages extracted per file:")
    for fname, count in pages_per_file.items():
        print(f"  {fname}: {count} pages")

    print(f"Documents extracted: {len(docs)}")

    if len(docs) == 0:
        raise ValueError(
            "No text could be extracted from the uploaded PDF. "
            "The PDF may be scanned or image-based."
        )

    # FIX: bigger chunk_size means FEWER total chunks,
    # which means fewer embedding requests overall.
    # This reduces how often we even hit the rate limit.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    print(f"Chunks created: {len(chunks)}")

    if len(chunks) == 0:
        raise ValueError(
            "Text extraction succeeded but no chunks were created."
        )

    sources_in_chunks = set(chunk.metadata["source"] for chunk in chunks)
    print(f"Sources present in final chunks: {sources_in_chunks}")

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )

    # FIX: the free tier allows 100 embedding requests PER MINUTE.
    # batch_size=80 stays safely under that limit per batch.
    # time.sleep(65) waits longer than a full minute between batches,
    # so the rate limit window completely resets before the next batch.
    # This is slower, but it actually respects the quota instead of
    # just hoping a short delay is enough.
    batch_size = 80
    wait_between_batches = 65  # seconds

    vector_store = None
    total_batches = (len(chunks) // batch_size) + 1

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_number = i // batch_size + 1

        print(f"Embedding batch {batch_number}/{total_batches} ({len(batch)} chunks)...")

        retries = 3
        for attempt in range(retries):
            try:
                if vector_store is None:
                    vector_store = FAISS.from_documents(batch, embeddings)
                else:
                    vector_store.add_documents(batch)
                break
            except Exception as e:
                if "RESOURCE_EXHAUSTED" in str(e) and attempt < retries - 1:
                    print("Hit rate limit, waiting 65 seconds before retry...")
                    time.sleep(65)
                else:
                    raise

        # only wait if there's another batch coming
        if i + batch_size < len(chunks):
            print(f"Waiting {wait_between_batches}s before next batch (rate limit cooldown)...")
            time.sleep(wait_between_batches)

    return vector_store