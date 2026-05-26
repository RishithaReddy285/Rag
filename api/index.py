import os
import re
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pydantic import BaseModel


try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = Path(os.getenv("DOCS_DIR", str(BASE_DIR / "docs")))
PUBLIC_DIR = BASE_DIR / "public"
FRONTEND_DIST = PUBLIC_DIR if PUBLIC_DIR.exists() else BASE_DIR / "frontend" / "dist"
FRONTEND_INDEX = FRONTEND_DIST / "index.html"
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
CHUNK_SIZE = 900
CHUNK_OVERLAP = 180
TOP_K = 5
RERANK_POOL_SIZE = 18
MMR_LAMBDA = 0.72

app = FastAPI(title="Advanced RAG API")

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


@dataclass(frozen=True)
class DocumentChunk:
    source: str
    index: int
    text: str
    tokens: tuple[str, ...]
    token_set: frozenset[str]


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str


def split_text(text: str) -> list[str]:
    clean_text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    start = 0

    while start < len(clean_text):
        end = min(len(clean_text), start + CHUNK_SIZE)
        if end < len(clean_text):
            sentence_boundary = max(
                clean_text.rfind(". ", start, end),
                clean_text.rfind("? ", start, end),
                clean_text.rfind("! ", start, end),
            )
            if sentence_boundary > start + int(CHUNK_SIZE * 0.55):
                end = sentence_boundary + 1

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


def normalize_token(token: str) -> str:
    token = token.lower()

    for suffix in ("ing", "ed", "es", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[: -len(suffix)]

    return token


def tokenize_list(text: str) -> list[str]:
    return [
        normalize_token(token)
        for token in re.findall(r"[a-zA-Z0-9]+", text.lower())
        if len(token) > 2
    ]


def load_chunks() -> list[DocumentChunk]:
    chunks = []

    if not DOCS_DIR.exists():
        return chunks

    for file_path in sorted(DOCS_DIR.rglob("*")):
        if file_path.suffix.lower() not in {".pdf", ".txt"}:
            continue

        source = str(file_path.relative_to(DOCS_DIR)).replace("\\", "/")
        for index, chunk in enumerate(split_text(load_text_from_file(file_path))):
            tokens = tuple(tokenize_list(chunk))
            chunks.append(
                DocumentChunk(
                    source=source,
                    index=index,
                    text=chunk,
                    tokens=tokens,
                    token_set=frozenset(tokens),
                )
            )

    return chunks


@lru_cache(maxsize=1)
def load_index() -> tuple[list[DocumentChunk], dict[str, int], float]:
    chunks = load_chunks()
    document_frequency: dict[str, int] = {}

    for chunk in chunks:
        for token in chunk.token_set:
            document_frequency[token] = document_frequency.get(token, 0) + 1

    average_length = (
        sum(len(chunk.tokens) for chunk in chunks) / len(chunks) if chunks else 0.0
    )

    return chunks, document_frequency, average_length


def expand_query(question: str) -> list[str]:
    tokens = tokenize_list(question)
    expanded = list(tokens)

    synonyms = {
        "doc": ("document", "file", "context"),
        "docs": ("document", "file", "context"),
        "information": ("info", "detail", "summary"),
        "rag": ("retrieval", "generation", "context"),
        "project": ("system", "application", "app"),
    }

    for token in tokens:
        expanded.extend(synonyms.get(token, ()))

    return expanded


def bm25_score(
    query_tokens: list[str],
    chunk: DocumentChunk,
    document_frequency: dict[str, int],
    average_length: float,
    total_chunks: int,
) -> float:
    if not query_tokens or not chunk.tokens:
        return 0.0

    token_counts: dict[str, int] = {}
    for token in chunk.tokens:
        token_counts[token] = token_counts.get(token, 0) + 1

    score = 0.0
    k1 = 1.4
    b = 0.72
    chunk_length = len(chunk.tokens)

    for token in query_tokens:
        frequency = token_counts.get(token, 0)
        if not frequency:
            continue

        df = document_frequency.get(token, 0)
        idf = math.log(1 + (total_chunks - df + 0.5) / (df + 0.5))
        denominator = frequency + k1 * (1 - b + b * chunk_length / average_length)
        score += idf * ((frequency * (k1 + 1)) / denominator)

    return score


def phrase_boost(question: str, chunk: DocumentChunk) -> float:
    clean_question = re.sub(r"\s+", " ", question.lower()).strip()
    clean_chunk = chunk.text.lower()
    boost = 0.0

    if clean_question and clean_question in clean_chunk:
        boost += 3.0

    query_terms = tokenize_list(question)
    for size in (4, 3, 2):
        for index in range(0, max(0, len(query_terms) - size + 1)):
            phrase = " ".join(query_terms[index : index + size])
            if phrase in clean_chunk:
                boost += 0.35 * size

    return boost


def similarity(left: DocumentChunk, right: DocumentChunk) -> float:
    union = left.token_set | right.token_set
    if not union:
        return 0.0

    return len(left.token_set & right.token_set) / len(union)


def select_diverse_chunks(
    scored_chunks: list[tuple[float, DocumentChunk]],
    k: int,
) -> list[DocumentChunk]:
    selected: list[DocumentChunk] = []
    remaining = scored_chunks[:]

    while remaining and len(selected) < k:
        best_position = 0
        best_score = float("-inf")

        for position, (relevance, chunk) in enumerate(remaining):
            diversity_penalty = max(
                (similarity(chunk, chosen) for chosen in selected),
                default=0.0,
            )
            mmr_score = MMR_LAMBDA * relevance - (1 - MMR_LAMBDA) * diversity_penalty

            if mmr_score > best_score:
                best_position = position
                best_score = mmr_score

        _, best_chunk = remaining.pop(best_position)
        selected.append(best_chunk)

    return selected


def retrieve_context(question: str, k: int = 5) -> str:
    chunks, document_frequency, average_length = load_index()
    if not chunks:
        return "No documents were found in the configured docs directory."

    query_tokens = expand_query(question)
    scored_chunks = []

    for chunk in chunks:
        lexical_score = bm25_score(
            query_tokens,
            chunk,
            document_frequency,
            average_length,
            len(chunks),
        )
        overlap_score = len(set(query_tokens) & chunk.token_set) / max(
            len(set(query_tokens)),
            1,
        )
        score = lexical_score + overlap_score + phrase_boost(question, chunk)
        scored_chunks.append((score, chunk))

    scored_chunks.sort(key=lambda item: item[0], reverse=True)
    candidates = [(score, chunk) for score, chunk in scored_chunks[:RERANK_POOL_SIZE] if score > 0]

    if not candidates:
        candidates = scored_chunks[:RERANK_POOL_SIZE]

    selected_chunks = select_diverse_chunks(candidates, k)

    return "\n\n".join(
        f"Source: {chunk.source} | Chunk: {chunk.index + 1}\n{chunk.text}"
        for chunk in selected_chunks
    )


def get_groq_api_key() -> str:
    api_key = os.getenv("GROQ_API_KEY", "").strip()

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is not configured in this Vercel project.",
        )

    return api_key


def query_rag(question: str) -> str:
    context = retrieve_context(question, k=TOP_K)
    client = Groq(api_key=get_groq_api_key())

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an advanced RAG assistant. Answer only from the "
                        "provided context. If the context does not contain the "
                        "answer, say that the documents do not provide enough "
                        "information. Cite source names when useful."
                    ),
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
    return {"status": "ok", "rag": "advanced", "version": "2026-05-26.2"}


@app.get("/api/debug")
def debug():
    chunks, _, _ = load_index()

    return {
        "status": "ok",
        "rag": "advanced",
        "chunks": len(chunks),
        "docs_dir": str(DOCS_DIR),
        "groq_api_key_configured": bool(os.getenv("GROQ_API_KEY", "").strip()),
        "groq_model": GROQ_MODEL,
    }


@app.get("/api/check-groq")
def check_groq():
    try:
        get_groq_api_key()
    except HTTPException as error:
        return JSONResponse(
            status_code=error.status_code,
            content={"detail": error.detail},
        )

    return {"status": "ok", "groq_api_key_configured": True}


def answer_question(question: str):
    question = question.strip()

    if not question:
        return JSONResponse(
            status_code=400,
            content={"detail": "Question is required."},
        )

    try:
        get_groq_api_key()
        return QueryResponse(answer=query_rag(question))
    except HTTPException as error:
        return JSONResponse(
            status_code=error.status_code,
            content={"detail": error.detail},
        )
    except Exception as error:
        return JSONResponse(
            status_code=500,
            content={"detail": f"Advanced RAG query failed. {error}"},
        )


@app.get("/api/query")
def query_get(question: str = ""):
    return answer_question(question)


@app.get("/api/ask")
def ask_get(question: str = ""):
    return answer_question(question)


@app.post("/api/query")
async def query(request: Request):
    try:
        payload = await request.json()
        parsed_request = QueryRequest(**payload)
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"detail": "Request body must include a JSON question field."},
        )

    return answer_question(parsed_request.question)


def fallback_frontend() -> HTMLResponse:
    return HTMLResponse(
        """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Advanced RAG</title>
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
        <h1>Advanced RAG</h1>
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
        const contentType = response.headers.get("content-type") || "";
        const data = contentType.includes("application/json")
          ? await response.json()
          : { detail: await response.text() };
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
