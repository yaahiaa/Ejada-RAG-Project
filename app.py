import os

import streamlit as st
from dotenv import load_dotenv

from ingest import Ingestion
from rag import Rag


load_dotenv()


# ---------------------------------------------------------
# Streamlit page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Chat with Your Books",
    page_icon="📚",
    layout="centered"
)


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

BOOKS_DIR_NAME = os.getenv("BOOKS_DIR", "books")

BOOKS_DIR = (
    BOOKS_DIR_NAME
    if os.path.isabs(BOOKS_DIR_NAME)
    else os.path.join(BASE_DIR, BOOKS_DIR_NAME)
)

os.makedirs(BOOKS_DIR, exist_ok=True)


# ---------------------------------------------------------
# Load backend services
# ---------------------------------------------------------

@st.cache_resource
def load_services() -> tuple[Ingestion, Rag]:
    """
    Create and cache the ingestion and RAG services.

    Streamlit reruns the script after every interaction.
    Caching prevents the embedding model and database clients
    from being reloaded on every rerun.
    """

    ingestion = Ingestion()
    rag = Rag()

    return ingestion, rag


ingestion, rag = load_services()


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def find_books() -> list[str]:
    """
    Return the names of all PDF files in the books directory.
    """

    books = [
        file_name
        for file_name in os.listdir(BOOKS_DIR)
        if file_name.lower().endswith(".pdf")
        and os.path.isfile(os.path.join(BOOKS_DIR, file_name))
    ]

    return sorted(books)


def save_uploaded_books(uploaded_files) -> list[str]:
    """
    Save uploaded PDF files into the local books directory.

    Returns the full paths of the saved files.
    """

    saved_paths = []

    for uploaded_file in uploaded_files:
        safe_file_name = os.path.basename(uploaded_file.name)

        file_path = os.path.join(
            BOOKS_DIR,
            safe_file_name
        )

        with open(file_path, "wb") as pdf_file:
            pdf_file.write(uploaded_file.getbuffer())

        saved_paths.append(file_path)

    return saved_paths


def index_uploaded_books(file_paths: list[str]) -> list[dict]:
    """
    Index all provided PDF files and return the indexing results.
    """

    if not file_paths:
        return []

    return ingestion.index_pdfs(file_paths)


def clear_chat() -> None:
    """
    Remove all messages from the current conversation.
    """

    st.session_state.messages = []


def display_sources(sources: list[dict]) -> None:
    """
    Display retrieved book passages below an assistant answer.
    """

    if not sources:
        return

    with st.expander("View retrieved sources"):
        for index, source in enumerate(sources, start=1):
            st.markdown(
                f"**Source {index}: "
                f"{source['source']} — Page {source['page']}**"
            )

            st.caption(
                f"Similarity distance: "
                f"{source['distance']:.4f}"
            )

            st.write(source["content"])

            if index < len(sources):
                st.divider()


def display_chat_history() -> None:
    """
    Display all stored user and assistant messages.
    """

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant":
                display_sources(
                    message.get("sources", [])
                )


# ---------------------------------------------------------
# Session state
# ---------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_book" not in st.session_state:
    st.session_state.selected_book = None

if "index_results" not in st.session_state:
    st.session_state.index_results = []


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:
    st.header("📚 Book Library")

    uploaded_files = st.file_uploader(
        "Upload PDF books",
        type=["pdf"],
        accept_multiple_files=True
    )

    index_button = st.button(
        "Save and Index Books",
        use_container_width=True,
        disabled=not uploaded_files
    )

    if index_button:
        try:
            saved_paths = save_uploaded_books(uploaded_files)

            with st.spinner("Indexing books..."):
                st.session_state.index_results = (
                    index_uploaded_books(saved_paths)
                )

            st.success("Indexing completed.")

        except Exception as error:
            st.error(
                f"An error occurred while indexing: {error}"
            )

    if st.session_state.index_results:
        for result in st.session_state.index_results:
            source = result.get("source", "Unknown file")
            status = result.get("status", "Unknown status")
            chunk_count = result.get("chunks_indexed", 0)

            if chunk_count > 0:
                st.success(
                    f"{source}: {chunk_count} chunks indexed."
                )
            else:
                st.warning(
                    f"{source}: {status}"
                )

    st.divider()

    available_books = find_books()

    if available_books:
        if (
            st.session_state.selected_book
            not in available_books
        ):
            st.session_state.selected_book = available_books[0]

        selected_index = available_books.index(
            st.session_state.selected_book
        )

        selected_book = st.selectbox(
            "Select a book",
            options=available_books,
            index=selected_index
        )

        if selected_book != st.session_state.selected_book:
            st.session_state.selected_book = selected_book
            clear_chat()
            st.rerun()

        st.caption(
            f"Selected book: "
            f"**{st.session_state.selected_book}**"
        )

    else:
        st.session_state.selected_book = None

        st.info(
            "Upload and index at least one PDF book."
        )

    st.divider()

    if st.button(
        "Clear Chat",
        use_container_width=True,
        disabled=not st.session_state.messages
    ):
        clear_chat()
        st.rerun()


# ---------------------------------------------------------
# Main chatbot interface
# ---------------------------------------------------------

st.title("📚 Chat with Your Books")

if st.session_state.selected_book:
    st.caption(
        f"Currently chatting with "
        f"**{st.session_state.selected_book}**"
    )
else:
    st.info(
        "Upload, index, and select a book from the sidebar "
        "to begin asking questions."
    )


display_chat_history()


question = st.chat_input(
    "Ask a question about the selected book...",
    disabled=st.session_state.selected_book is None
)


if question:
    user_message = {
        "role": "user",
        "content": question
    }

    st.session_state.messages.append(user_message)

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the book..."):
                result = rag.ask_book(
                    question=question,
                    book_name=st.session_state.selected_book
                )

            answer = result["answer"]
            sources = result["sources"]

            st.markdown(answer)
            display_sources(sources)

        except Exception as error:
            answer = (
                "An error occurred while answering the question: "
                f"{error}"
            )
            sources = []

            st.error(answer)

    assistant_message = {
        "role": "assistant",
        "content": answer,
        "sources": sources
    }

    st.session_state.messages.append(
        assistant_message
    )