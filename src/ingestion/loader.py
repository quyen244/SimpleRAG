from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader


def load_documents(data_dir: Path) -> List[Document]:
    docs = []
    for path in data_dir.rglob("*.pdf"):
        loader = PyPDFLoader(str(path))
        docs.extend(loader.load())
    for ext in ("*.md", "*.txt"):
        for path in data_dir.rglob(ext):
            loader = TextLoader(str(path), encoding="utf-8")
            docs.extend(loader.load())
    return docs
