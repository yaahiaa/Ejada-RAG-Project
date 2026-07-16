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
        self.collection = self.chroma_client.get_or_create_collection(name="books",metadata={"hnsw:space": "cosine"})
        self.chunk_size = int(os.getenv("CHUNK_SIZE", 800))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 100))
        self.top_k = int(os.getenv("TOP_K", 4))

        self.llm_service = LLMService()


    def retrieve_chunks(self, question: str) -> list[dict]:
        if not question.strip():
            raise ValueError("Question cannot be empty.")

        collection_size = self.collection.count()

        if collection_size == 0:
            raise ValueError("No PDF has been indexed yet.")

        top_k = min(self.top_k, collection_size)

        question_embedding = self.embedding_model.encode(
            question,
            normalize_embeddings=True
        ).tolist()

        results = self.collection.query(
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

    def ask_book(self, question: str) -> dict:

        retrieved_chunks = self.retrieve_chunks(question=question)

        answer = self.llm_service.generate_answer(
            question=question,
            retrieved_chunks=retrieved_chunks
        )

        return {
            "answer": answer,
            "sources": retrieved_chunks
        }