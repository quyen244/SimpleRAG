from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP

_recursive = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

_md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ],
)


def _chunk_markdown(doc: Document) -> List[Document]:
    source = doc.metadata.get("source", "")
    md_chunks = _md_splitter.split_text(doc.page_content)
    result = []
    for chunk in md_chunks:
        chunk.metadata["source"] = source
        if len(chunk.page_content) > CHUNK_SIZE:
            result.extend(_recursive.split_documents([chunk]))
        else:
            result.append(chunk)
    return result


def chunk_documents(docs: List[Document]) -> List[Document]:
    chunks = []
    for doc in docs:
        source = doc.metadata.get("source", "")
        if source.endswith(".md"):
            chunks.extend(_chunk_markdown(doc))
        else:
            chunks.extend(_recursive.split_documents([doc]))
    return chunks
