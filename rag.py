import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langsmith import traceable

from local_embeddings import ChromaDefaultEmbeddings


load_dotenv()

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME = "naive_rag"
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


@traceable
def query_rag(question: str) -> str:
    embeddings = ChromaDefaultEmbeddings()
    vector_store = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )

    docs = vector_store.similarity_search(question, k=5)
    context = "\n\n".join(doc.page_content for doc in docs)

    llm = ChatGroq(model=GROQ_MODEL)
    response = llm.invoke(
        [
            (
                "system",
                "You are a helpful assistant. Answer using only the context provided.",
            ),
            (
                "user",
                f"Context:\n{context}\n\nQuestion: {question}",
            ),
        ]
    )

    return response.content
