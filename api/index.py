import os
import re
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from langsmith import traceable
from pydantic import BaseModel


try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None


load_dotenv()

DOCS_DIR = Path(os.getenv("DOCS_DIR", "./docs"))
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

app = FastAPI(title="Naive RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str


def split_text(text: str) -> list[str]:
    clean_text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    start = 0

    while start < len(clean_text):
        end = start + CHUNK_SIZE
        chunk = clean_text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(clean_text):
            break

        start = max(0, end - CHUNK_OVERLAP)

    return chunks


def load_text_from_file(file_path: Path) -> str:
    if file_path.suffix.lower() == ".txt":
        return file_path.read_text(encoding="utf-8", errors="ignore")

    if file_path.suffix.lower() == ".pdf" and PdfReader is not None:
        reader = PdfReader(str(file_path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    return ""


def load_chunks() -> list[str]:
    chunks = []

    if not DOCS_DIR.exists():
        return chunks

    for file_path in DOCS_DIR.rglob("*"):
        if file_path.suffix.lower() not in {".pdf", ".txt"}:
            continue

        chunks.extend(split_text(load_text_from_file(file_path)))

    return chunks


def tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z0-9]+", text.lower())
        if len(token) > 2
    }


def retrieve_context(question: str, k: int = 5) -> str:
    question_tokens = tokenize(question)
    scored_chunks = []

    for chunk in load_chunks():
        chunk_tokens = tokenize(chunk)
        score = len(question_tokens & chunk_tokens)
        scored_chunks.append((score, chunk))

    scored_chunks.sort(key=lambda item: item[0], reverse=True)
    top_chunks = [chunk for score, chunk in scored_chunks[:k] if score > 0]

    if not top_chunks:
        top_chunks = [chunk for _, chunk in scored_chunks[:k]]

    return "\n\n".join(top_chunks)


@traceable
def query_rag(question: str) -> str:
    context = retrieve_context(question, k=5)
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer using only the context provided.",
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ],
    )

    return response.choices[0].message.content or ""


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/query", response_model=QueryResponse)
def query(request: QueryRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question is required.")

    return QueryResponse(answer=query_rag(question))
