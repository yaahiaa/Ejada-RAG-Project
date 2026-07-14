import os
import json
from pathlib import Path
import pymupdf
import chromadb
import hashlib
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv


embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(path="chroma_db")

collection = chroma_client.get_or_create_collection(name="books",metadata={"hnsw:space": "cosine"})

def extract_text(file_path: str) -> list[dict]:
    pages = []

    with pymupdf.open(file_path) as doc:
        for page_number, page in enumerate(doc, start=1):
            text = page.get_text().strip()

            if text:
                pages.append({
                    "page": page_number,
                    "content": text
                })

    return pages

def chunk_text(pages: list[dict],chunk_size: int = 800,overlap: int = 100) -> list[dict]:
   
    chunks = []
    chunk_id = 0

    for page in pages:
        page_number = page["page"]
        text = page["content"].strip()

        if not text:
            continue

        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end]

            # Try to finish the chunk at a natural boundary.
            if end < len(text):
                possible_breaks = [
                    chunk.rfind("\n\n"),
                    chunk.rfind(". "),
                    chunk.rfind("? "),
                    chunk.rfind("! "),
                    chunk.rfind("\n"),
                ]

                last_break = max(possible_breaks)

                # Avoid producing an excessively short chunk.
                if last_break >= chunk_size // 2:
                    end = start + last_break + 1
                    chunk = text[start:end]

            chunk = chunk.strip()

            if chunk:
                chunks.append({
                    "chunk_id": chunk_id,
                    "page": page_number,
                    "content": chunk
                })
                chunk_id += 1

            if end >= len(text):
                break

            start = end - overlap

    return chunks

def index_pdf(file_path: str) -> int:

    pdf_path = Path(file_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")

    pages = extract_text(str(pdf_path))
    chunks = chunk_text(pages)

    if not chunks:
        print(f"No readable text found in: {pdf_path.name}")
        return 0

    texts = [chunk["content"] for chunk in chunks]

    embeddings = embedding_model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    ).tolist()

    file_hash = hashlib.sha256(
        pdf_path.read_bytes()
    ).hexdigest()[:16]

    ids = [
        f"{file_hash}_chunk_{chunk['chunk_id']}"
        for chunk in chunks
    ]

    metadatas = [
        {
            "source": pdf_path.name,
            "page": chunk["page"],
            "chunk_id": chunk["chunk_id"],
            "file_hash": file_hash
        }
        for chunk in chunks
    ]

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas
    )

    print(f"Indexed {pdf_path.name}: {len(chunks)} chunks")

    return len(chunks)


def main():
    pdf_path = "books/Harry_Potter_and_the_Sorcerer's_Stone.pdf"

    try:
        indexed_chunks = index_pdf(pdf_path)
        print(f"\nCollection contains {collection.count()} chunks.")

    except Exception as error:
        print(f"Error: {error}")


if __name__ == "__main__":
    main()