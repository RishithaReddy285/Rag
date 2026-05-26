import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from local_embeddings import ChromaDefaultEmbeddings


load_dotenv()

DOCS_DIR = Path(os.getenv("DOCS_DIR", "./docs"))
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME = "naive_rag"


def load_documents():
    documents = []

    for file_path in DOCS_DIR.rglob("*"):
        if not file_path.is_file():
            continue

        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            loader = PyPDFLoader(str(file_path))
        elif suffix == ".txt":
            loader = TextLoader(str(file_path), encoding="utf-8")
        else:
            continue

        documents.extend(loader.load())

    return documents


def main():
    documents = load_documents()

    if not documents:
        print(f"No PDF or TXT files found in {DOCS_DIR.resolve()}")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    chunks = [chunk for chunk in chunks if chunk.page_content.strip()]

    if not chunks:
        print(f"No non-empty text found in PDF or TXT files under {DOCS_DIR.resolve()}")
        return

    embeddings = ChromaDefaultEmbeddings()
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name=COLLECTION_NAME,
    )

    print(f"Stored {len(chunks)} chunks in ChromaDB collection '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    main()
