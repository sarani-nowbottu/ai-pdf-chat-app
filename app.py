import streamlit as st
import tempfile
import os

from pdf_processor import extract_text_from_pdf
from qa_engine import get_qa_chain

st.title("AI PDF Chat with Gemini")

uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf"
)

if uploaded_file:

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp:

        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    text = extract_text_from_pdf(pdf_path)

    os.unlink(pdf_path)

    st.success("PDF processed successfully!")

    question = st.text_input(
        "Ask a question about the PDF"
    )

    if question:

        llm = get_qa_chain()

        prompt = f"""
        Answer only from this document.

        DOCUMENT:
        {text[:15000]}

        QUESTION:
        {question}
        """

        response = llm.invoke(prompt)

        st.subheader("Answer")
        st.write(response.content)