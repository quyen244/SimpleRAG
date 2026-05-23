import tempfile
from pathlib import Path

import streamlit as st

from config import EMBEDDING_MODEL
from src.chain.rag_chain import build_chain
from src.ingestion.chunker import chunk_documents
from src.ingestion.loader import load_documents
from src.retrieval.embeddings import MultilingualE5Embeddings
from src.retrieval.vectorstore import get_retriever, get_vectorstore

st.set_page_config(page_title="RAG Chatbot", page_icon="📚", layout="wide")
st.title("📚 RAG Chatbot")


@st.cache_resource
def _init_store():
    embeddings = MultilingualE5Embeddings(model_name=EMBEDDING_MODEL)
    vectorstore = get_vectorstore(embeddings)
    return vectorstore


vectorstore = _init_store()

if "chain" not in st.session_state:
    retriever = get_retriever(vectorstore)
    st.session_state.chain = build_chain(retriever)

if "messages" not in st.session_state:
    st.session_state.messages = []


# ── Sidebar: file uploader ────────────────────────────────────────────────────
with st.sidebar:
    st.header("Upload Documents")
    uploaded = st.file_uploader(
        "PDF, Markdown, or TXT",
        type=["pdf", "md", "txt"],
        accept_multiple_files=True,
    )
    if uploaded and st.button("Ingest", type="primary"):
        with st.spinner("Ingesting documents..."):
            with tempfile.TemporaryDirectory() as tmpdir:
                for f in uploaded:
                    (Path(tmpdir) / f.name).write_bytes(f.read())
                docs = load_documents(Path(tmpdir))
                chunks = chunk_documents(docs)
            vectorstore.add_documents(chunks)
        st.success(f"Added {len(chunks)} chunks to ChromaDB.")

    st.divider()
    if st.button("Clear conversation"):
        st.session_state.messages = []
        retriever = get_retriever(vectorstore)
        st.session_state.chain = build_chain(retriever)
        st.rerun()


# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for src in msg["sources"]:
                    label = Path(src["file"]).name
                    section = src.get("section", "")
                    st.markdown(f"- **{label}**{f' — {section}' if section else ''}")


# ── Input ─────────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Hỏi gì đó..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Đang suy nghĩ..."):
            result = st.session_state.chain.invoke({"question": prompt})

        answer = result["answer"]
        source_docs = result.get("source_documents", [])

        sources = []
        seen = set()
        for doc in source_docs:
            meta = doc.metadata
            file = meta.get("source", "Unknown")
            section = meta.get("Header 1", meta.get("Header 2", str(meta.get("page", ""))))
            key = f"{file}|{section}"
            if key not in seen:
                seen.add(key)
                sources.append({"file": file, "section": section})

        st.markdown(answer)
        if sources:
            with st.expander("Sources"):
                for src in sources:
                    label = Path(src["file"]).name
                    section = src.get("section", "")
                    st.markdown(f"- **{label}**{f' — {section}' if section else ''}")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
