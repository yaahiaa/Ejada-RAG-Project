from ingest import index_pdf
from rag import ask_book


def main():
    pdf_path = "books/Harry_Potter_and_the_Sorcerer's_Stone.pdf"

    index_pdf(pdf_path)

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