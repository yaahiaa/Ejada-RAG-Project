import tempfile
from pathlib import Path

import streamlit as st

from ingest import index_pdf
from rag import ask_book


st.set_page_config(
    page_title="Chat with a Book",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Chat with a Book")

st.write(
    "Upload a PDF, let it be indexed, then ask questions about its content."
)

# -------------------------
# Upload PDF
# -------------------------

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)

if uploaded_file is not None:

    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)

    pdf_path = temp_dir / uploaded_file.name

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.button("Index Book"):

        with st.spinner("Indexing book..."):

            chunk_count = index_pdf(str(pdf_path))

        st.success(f"Indexed {chunk_count} chunks.")

# -------------------------
# Ask Question
# -------------------------

question = st.text_input("Ask a question about the book")

if st.button("Ask"):

    if not question.strip():
        st.warning("Please enter a question.")

    else:

        with st.spinner("Thinking..."):

            result = ask_book(question)

        st.subheader("Answer")

        st.write(result["answer"])

        st.subheader("Sources")

        for source in result["sources"]:

            with st.expander(
                f"Page {source['page']} | Distance {source['distance']:.4f}"
            ):
                st.write(source["content"])