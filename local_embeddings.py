from chromadb.api.types import DefaultEmbeddingFunction
from langchain_core.embeddings import Embeddings


class ChromaDefaultEmbeddings(Embeddings):
    def __init__(self):
        self.embedding_function = DefaultEmbeddingFunction()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        embeddings = self.embedding_function(texts)
        return [embedding.tolist() for embedding in embeddings]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
