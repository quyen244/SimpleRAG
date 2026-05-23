"""
Run this script to bulk-ingest documents into ChromaDB.
Usage: python ingest.py [--data-dir path/to/docs]
"""
import argparse
from pathlib import Path
from config import DATA_DIR, EMBEDDING_MODEL
from src.ingestion.loader import load_documents
from src.ingestion.chunker import chunk_documents
from src.retrieval.embeddings import MultilingualE5Embeddings
from src.retrieval.vectorstore import get_vectorstore


def ingest(data_dir: Path) -> None:
    print(f"Loading documents from: {data_dir}")
    docs = load_documents(data_dir)
    if not docs:
        print("No documents found. Drop PDFs or Markdown files into the data/ folder.")
        return
    print(f"Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks")

    print("Embedding and storing in ChromaDB...")
    embeddings = MultilingualE5Embeddings(model_name=EMBEDDING_MODEL)
    vectorstore = get_vectorstore(embeddings)
    vectorstore.add_documents(chunks)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    args = parser.parse_args()
    ingest(Path(args.data_dir))
