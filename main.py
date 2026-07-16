from dotenv import load_dotenv

from ingest import Ingestion
from rag import Rag
import os

load_dotenv()

def find_pdfs() -> list[str]:

    BOOK_DIR = os.getenv("BOOK_DIR")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    books_dir = os.path.join(base_dir, BOOK_DIR)

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
    return pdf_files

def select_book(pdf_files: list[str]) -> str:
    print("\nAvailable books:")

    for index, pdf_path in enumerate(pdf_files, start=1):
        print(f"{index}. {os.path.basename(pdf_path)}")

    while True:
        choice = input("\nSelect a book number: ").strip()

        if not choice.isdigit():
            print("Please enter a valid number.")
            continue

        selected_index = int(choice) - 1

        if 0 <= selected_index < len(pdf_files):
            return os.path.basename(pdf_files[selected_index])

        print("Selection is out of range.")

def run_chat(rag: Rag, book_name: str) -> None:
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
            result = rag.ask_book(question=question, book_name=book_name)

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
    pdf_paths = find_pdfs()

    ingest = Ingestion()
    rag = Rag()

    chunk_count = ingest.index_pdfs(pdf_paths)

    selected_book = select_book(pdf_paths)
    run_chat(rag, book_name=selected_book)

def main():
    try:
        run_application()
    except Exception as error:
        print(f"Error: {error}")

if __name__ == "__main__":
    main()