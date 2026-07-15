from ingest import index_pdf
from rag import ask_book
import os


def main():
    BOOKS_DIR = "books"

    pdf_files = [
        os.path.join(BOOKS_DIR, file)
        for file in os.listdir(BOOKS_DIR)
        if file.lower().endswith(".pdf")
    ]

    index_pdf(pdf_files[0])  

    while True:
        question = input(
            "\nAsk a question about the book "
            "(or type 'exit' to quit): "
        ).strip()

        if question.lower() in {"exit", "quit"}:
            break

        result = ask_book(question)

        print("\nAnswer:\n")
        print(result["answer"])

        print("\nSources:\n")

        for source in result["sources"]:
            print(
                f"- Page {source['page']} "
                f"| Distance: {source['distance']:.4f}"
            )


if __name__ == "__main__":
    main()