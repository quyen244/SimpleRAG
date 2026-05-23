from typing import List
from langchain_huggingface import HuggingFaceEmbeddings


class MultilingualE5Embeddings(HuggingFaceEmbeddings):
    """Wraps multilingual-e5 models with required query/passage prefixes."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        prefixed = [f"passage: {t}" for t in texts]
        return super().embed_documents(prefixed)

    def embed_query(self, text: str) -> List[float]:
        return super().embed_query(f"query: {text}")
