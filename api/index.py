import os
import re
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from langsmith import traceable
from pydantic import BaseModel


try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = Path(os.getenv("DOCS_DIR", str(BASE_DIR / "docs")))
FRONTEND_DIST = BASE_DIR / "frontend" / "dist"
FRONTEND_INDEX = FRONTEND_DIST / "index.html"
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

if (FRONTEND_DIST / "assets").exists():
    app.mount(
        "/assets",
        StaticFiles(directory=str(FRONTEND_DIST / "assets")),
        name="assets",
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


def get_groq_api_key() -> str:
    api_key = os.getenv("GROQ_API_KEY", "").strip()

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is not configured in this Vercel project.",
        )

    return api_key


@traceable
def query_rag(question: str) -> str:
    context = retrieve_context(question, k=5)
    client = Groq(api_key=get_groq_api_key())

    try:
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
    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail=f"Groq request failed. Check GROQ_API_KEY and GROQ_MODEL. {error}",
        ) from error

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


def fallback_frontend() -> HTMLResponse:
    return HTMLResponse(
        """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Naive RAG</title>
    <style>
      body { margin: 0; font-family: Arial, sans-serif; background: #f2f5f1; color: #1d2430; }
      main { min-height: 100vh; display: grid; place-items: center; padding: 24px; }
      section { width: min(760px, 100%); background: white; border: 1px solid #dce3da; border-radius: 8px; padding: 20px; }
      h1 { margin: 0 0 6px; font-size: 24px; }
      p { line-height: 1.5; }
      form { display: grid; grid-template-columns: 1fr 44px; gap: 10px; margin-top: 18px; }
      input { height: 44px; border: 1px solid #bdcabc; border-radius: 8px; padding: 0 12px; }
      button { height: 44px; border: 0; border-radius: 8px; color: white; background: #315f46; cursor: pointer; }
      #answer { margin-top: 18px; white-space: pre-wrap; }
    </style>
  </head>
  <body>
    <main>
      <section>
        <h1>Naive RAG</h1>
        <p>Ask a question about the documents deployed with this project.</p>
        <form id="form">
          <input id="question" placeholder="Ask about your docs" />
          <button>Send</button>
        </form>
        <p id="answer"></p>
      </section>
    </main>
    <script>
      document.getElementById("form").addEventListener("submit", async (event) => {
        event.preventDefault();
        const answer = document.getElementById("answer");
        const question = document.getElementById("question").value.trim();
        if (!question) return;
        answer.textContent = "Searching...";
        const response = await fetch("/api/query", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question })
        });
        const data = await response.json();
        answer.textContent = response.ok ? data.answer : (data.detail || "Request failed");
      });
    </script>
  </body>
</html>
        """.strip()
    )


@app.get("/")
def serve_frontend_root():
    if FRONTEND_INDEX.exists():
        return FileResponse(FRONTEND_INDEX)

    return fallback_frontend()


@app.get("/{full_path:path}")
def serve_frontend(full_path: str):
    if full_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="Not Found")

    file_path = FRONTEND_DIST / full_path
    if file_path.is_file():
        return FileResponse(file_path)

    if FRONTEND_INDEX.exists():
        return FileResponse(FRONTEND_INDEX)

    return fallback_frontend()
