import os

import chromadb
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer


load_dotenv()

embedding_model = SentenceTransformer(
    os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
)

chroma_client = chromadb.PersistentClient(
    path=os.getenv("CHROMA_DB_PATH", "chroma_db")
)

collection = chroma_client.get_or_create_collection(
    name="books",
    metadata={"hnsw:space": "cosine"}
)

groq_client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))
TOP_K = int(os.getenv("TOP_K", 4))


def retrieve_chunks(question: str, top_k: int = TOP_K) -> list[dict]:
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    collection_size = collection.count()

    if collection_size == 0:
        raise ValueError("No PDF has been indexed yet.")

    top_k = min(top_k, collection_size)

    question_embedding = embedding_model.encode(
        question,
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    retrieved_chunks = []

    for document, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        retrieved_chunks.append({
            "content": document,
            "page": metadata["page"],
            "source": metadata["source"],
            "chunk_id": metadata["chunk_id"],
            "distance": distance
        })

    return retrieved_chunks


def generate_answer(question: str,retrieved_chunks: list[dict]) -> str:
    
    if not retrieved_chunks:
        return "I could not find relevant information in the book."

    context_parts = []

    for index, chunk in enumerate(retrieved_chunks, start=1):
        context_parts.append(
            f"[Source {index} | Page {chunk['page']}]\n"
            f"{chunk['content']}"
        )

    context = "\n\n".join(context_parts)

    system_prompt = """
You are a question-answering assistant for an uploaded book.

Answer using only the provided book context.

Rules:
- Do not use outside knowledge.
- Do not invent information.
- If the answer is not supported by the context, say:
  "I could not find this information in the book."
- Cite supporting pages using the format (Page 12).
- Keep the answer clear and concise.
"""

    user_prompt = f"""
BOOK CONTEXT:

{context}

QUESTION:

{question}
"""

    response = groq_client.chat.completions.create(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    )

    return response.choices[0].message.content


def ask_book(question: str,top_k: int = TOP_K) -> dict:

    retrieved_chunks = retrieve_chunks(
        question=question,
        top_k=top_k
    )

    answer = generate_answer(
        question=question,
        retrieved_chunks=retrieved_chunks
    )

    return {
        "answer": answer,
        "sources": retrieved_chunks
    }