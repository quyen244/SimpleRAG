from langchain_chroma import Chroma
from config import CHROMA_DIR, EMBEDDING_MODEL, RETRIEVER_K, RETRIEVER_FETCH_K
from src.retrieval.embeddings import MultilingualE5Embeddings


def get_vectorstore(embeddings=None) -> Chroma:
    if embeddings is None:
        embeddings = MultilingualE5Embeddings(model_name=EMBEDDING_MODEL)
    return Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )


def get_retriever(vectorstore: Chroma):
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": RETRIEVER_K, "fetch_k": RETRIEVER_FETCH_K},
    )
