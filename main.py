from ingest import Ingestion
from rag import Rag
import os


def find_pdf() -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    books_dir = os.path.join(base_dir, "books")

    if not os.path.exists(books_dir):
        raise FileNotFoundError("The books directory does not exist.")

    pdf_files = [
        os.path.join(books_dir, file_name)
        for file_name in os.listdir(books_dir)
        if file_name.lower().endswith(".pdf")
    ]

    if not pdf_files:
        raise FileNotFoundError(
            "No PDF file was found in the books directory."
        )
    return pdf_files[0]

def run_chat(rag: Rag) -> None:
    while True:
        question = input(
            "\nAsk a question about the book "
            "(or type 'exit' to quit): "
        ).strip()

        if question.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        if not question:
            print("Please enter a question.")
            continue

        try:
            result = rag.ask_book(question)

            print("\nAnswer:\n")
            print(result["answer"])

            print("\nSources:")

            for source in result["sources"]:
                print(
                    f"- Page {source['page']} "
                    f"| Distance: {source['distance']:.4f}"
                )

        except Exception as error:
            print(f"Error while answering: {error}")

def run_application() -> None:
    pdf_path = find_pdf()

    ingest = Ingestion()
    rag = Rag()

    chunk_count = ingest.index_pdf(pdf_path)
    run_chat(rag)

def main():
    try:
        run_application()
    except Exception as error:
        print(f"Error: {error}")

if __name__ == "__main__":
    main()