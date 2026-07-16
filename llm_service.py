import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

class LLMService:
    def __init__(self):
        self.model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError("GROQ_API_KEY is missing from the .env file.")
        
        self.client = Groq(api_key=api_key)

    def generate_answer(self, question: str,retrieved_chunks: list[dict]) -> str:
        
        if not question.strip():
            raise ValueError("Question cannot be empty.")

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

        response = self.client.chat.completions.create(
            model=self.model,
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

        answer = response.choices[0].message.content

        if not answer:
            raise RuntimeError("Groq returned an empty response.")

        return answer

    