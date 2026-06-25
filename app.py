import streamlit as st
from pdf_processor import process_pdfs
from qa_engine import get_qa_chain

st.set_page_config(page_title="AI PDF Chat")
st.title("AI PDF Chat")

# ==========================
# SESSION STATE
# ==========================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

if "last_answer" not in st.session_state:
    st.session_state.last_answer = None

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

# ==========================
# SIDEBAR
# ==========================
mode = st.sidebar.radio(
    "Mode",
    ["Chat PDF", "Summarize PDF", "Resume Analyzer"]
)

# ==========================
# FILE UPLOAD
# ==========================
uploaded_files = st.file_uploader(
    "Upload PDF files",
    type="pdf",
    accept_multiple_files=True
)

# ==========================
# PROCESS PDFS
# ==========================
if uploaded_files:
    file_names = [f.name for f in uploaded_files]
    if st.session_state.processed_files != file_names:
        with st.spinner("Processing PDFs..."):
            st.session_state.vector_store = process_pdfs(uploaded_files)
            st.session_state.processed_files = file_names
            st.session_state.last_answer = None
            st.session_state.last_sources = []
            st.success("PDFs processed successfully!")

# ==========================
# RETRIEVER
# ==========================
if st.session_state.vector_store is not None:
    retriever = st.session_state.vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 10, "fetch_k": 20}
    )
else:
    retriever = None

# ==========================
# LLM
# ==========================
llm = get_qa_chain()

# ==========================
# CHAT MODE
# ==========================
if mode == "Chat PDF":

    question = st.text_input("Ask a question", key="question_input")
    ask_button = st.button("Ask")

    if ask_button and question.strip():
        if not retriever:
            st.error("Please upload PDF files first.")
        else:
            with st.spinner("Thinking..."):
                try:
                    docs = retriever.invoke(question)
                    context = "\n".join([doc.page_content for doc in docs])

                    prompt = f"""Answer using only this context.

Context:
{context}

Question:
{question}
"""
                    response = llm.invoke(prompt)

                    st.session_state.last_answer = response.content
                    st.session_state.chat_history.append({
                        "question": question,
                        "answer": response.content
                    })

                    sources = []
                    shown = set()
                    for doc in docs:
                        source = doc.metadata.get("source", "Unknown")
                        page = doc.metadata.get("page", "?")
                        src = (source, page)
                        if src not in shown:
                            shown.add(src)
                            sources.append(f"{source} (Page {page})")
                    st.session_state.last_sources = sources

                except Exception as e:
                    st.error(f"Error: {e}")

    if st.session_state.last_answer:
        st.subheader("Answer")
        st.write(st.session_state.last_answer)

        if st.session_state.last_sources:
            with st.expander("📄 Sources"):
                for src in st.session_state.last_sources:
                    st.write(src)

    if st.session_state.chat_history:
        st.subheader("Chat History")
        for chat in reversed(st.session_state.chat_history):
            st.markdown(f"**You:** {chat['question']}")
            st.markdown(f"**AI:** {chat['answer']}")
            st.divider()

        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.session_state.last_answer = None
            st.session_state.last_sources = []
            st.rerun()

# ==========================
# SUMMARIZE MODE
# ==========================
elif mode == "Summarize PDF":

    generate_button = st.button("Generate Summary")

    if generate_button:
        if not retriever:
            st.error("Please upload PDF files first.")
        else:
            with st.spinner("Generating summary..."):
                try:
                    all_docs = []
                    for filename in st.session_state.processed_files:
                        file_docs = st.session_state.vector_store.similarity_search(
                            "Summarize this document",
                            k=5,
                            filter={"source": filename}
                        )
                        all_docs.extend(file_docs)

                    context = "\n".join([doc.page_content for doc in all_docs])

                    prompt = f"""Summarize this document.

Give:
1. Overview
2. Key Points
3. Conclusion

Context:
{context}
"""
                    summary = llm.invoke(prompt)

                    st.subheader("Summary")
                    st.write(summary.content)

                    with st.expander("📄 Sources"):
                        shown = set()
                        for doc in all_docs:
                            source = doc.metadata.get("source", "Unknown")
                            page = doc.metadata.get("page", "?")
                            src = (source, page)
                            if src not in shown:
                                shown.add(src)
                                st.write(f"{source} (Page {page})")

                except Exception as e:
                    st.error(f"Error while generating summary: {e}")

# ==========================
# RESUME ANALYZER MODE
# ==========================
elif mode == "Resume Analyzer":

    if not st.session_state.processed_files:
        st.error("Please upload a resume PDF first.")
    else:
        # if more than one PDF is uploaded, let the user pick which
        # one is the resume to analyze
        selected_resume = st.selectbox(
            "Select the resume to analyze",
            st.session_state.processed_files
        )

        analyze_button = st.button("Analyze Resume")

        if analyze_button:
            with st.spinner("Analyzing resume..."):
                try:
                    # pull ALL chunks belonging to this one file
                    # k=50 is high on purpose, resumes are short so
                    # this basically grabs the whole document
                    resume_docs = st.session_state.vector_store.similarity_search(
                        "resume skills experience education",
                        k=50,
                        filter={"source": selected_resume}
                    )

                    context = "\n".join([doc.page_content for doc in resume_docs])

                    prompt = f"""You are a resume reviewer. Read the resume below
and give a structured analysis.

Give your answer in this format:

1. Candidate Summary (2-3 lines)
2. Key Skills (as a list)
3. Education
4. Work Experience / Projects
5. Strengths
6. Areas to Improve
7. ATS Friendliness (rate as Good / Average / Poor, with a short reason)

Resume:
{context}
"""
                    analysis = llm.invoke(prompt)

                    st.subheader("Resume Analysis")
                    st.write(analysis.content)

                    with st.expander("📄 Source"):
                        st.write(selected_resume)

                except Exception as e:
                    st.error(f"Error while analyzing resume: {e}")