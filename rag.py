import os
import shutil
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langsmith import traceable

from local_embeddings import ChromaDefaultEmbeddings


load_dotenv()

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME = "naive_rag"
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


def get_chroma_persist_dir() -> str:
    source_dir = Path(CHROMA_PERSIST_DIR)

    if not os.getenv("VERCEL"):
        return str(source_dir)

    target_dir = Path(tempfile.gettempdir()) / "chroma_db"

    if not target_dir.exists() and source_dir.exists():
        shutil.copytree(source_dir, target_dir)

    return str(target_dir)


@traceable
def query_rag(question: str) -> str:
    embeddings = ChromaDefaultEmbeddings()
    vector_store = Chroma(
        persist_directory=get_chroma_persist_dir(),
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
