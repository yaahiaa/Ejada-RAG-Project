import os

import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from llm_service import LLMService

load_dotenv()

class Rag:
    def __init__(self):
        self.embedding_model = SentenceTransformer(os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
        self.chroma_client = chromadb.PersistentClient(path=os.getenv("CHROMA_DB_PATH", "chroma_db"))
        collection_name = os.getenv("COLLECTION_NAME")
        self.collection = self.chroma_client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
        self.top_k = int(os.getenv("TOP_K", 4))

        self.llm_service = LLMService()


    def retrieve_chunks(self, question: str, book_name: str) -> list[dict]:
        if not question.strip():
            raise ValueError("Question cannot be empty.")
        
        if not book_name.strip():
            raise ValueError("Book name cannot be empty.")

        book_records = self.collection.get(
            where={"source": book_name},
            include=[]
        )

        book_chunk_count = len(book_records["ids"])

        if book_chunk_count == 0:
            raise ValueError(
                f"No indexed chunks were found for '{book_name}'."
            )

        top_k = min(self.top_k, book_chunk_count)

        question_embedding = self.embedding_model.encode(
            question,
            normalize_embeddings=True
        ).tolist()

        results = self.collection.query(
            query_embeddings=[question_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
            where={"source": book_name}
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

    def ask_book(self, question: str, book_name:str) -> dict:

        retrieved_chunks = self.retrieve_chunks(question=question, book_name=book_name)

        answer = self.llm_service.generate_answer(
            question=question,
            retrieved_chunks=retrieved_chunks
        )

        return {
            "answer": answer,
            "sources": retrieved_chunks
        }