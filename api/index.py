import os
import sys
from pathlib import Path

# Add root folder to path to allow importing rag.py on Vercel/local
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import RAG features from modular rag module
from rag import (
    RetrievalOptions,
    query_rag,
    clear_index_cache,
    normalize_options,
)

load_dotenv()

PUBLIC_DIR = BASE_DIR / "public"
FRONTEND_DIST = PUBLIC_DIR if PUBLIC_DIR.exists() else BASE_DIR / "frontend" / "dist"
FRONTEND_INDEX = FRONTEND_DIST / "index.html"
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

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


class QueryRequest(BaseModel):
    question: str
    optimization_profile: str | None = None
    top_k: int | None = None
    rerank_pool_size: int | None = None
    mmr_lambda: float | None = None
    lexical_weight: float | None = None
    semantic_weight: float | None = None
    min_score: float | None = None
    max_context_chars: int | None = None
    adaptive_weights: bool | None = None
    reranking_strategy: str | None = None


class QueryResponse(BaseModel):
    answer: str
    retrieval: list[dict[str, object]] = []
    summary: dict[str, object] = {}


def build_options(
    optimization_profile: str | None = None,
    top_k: int | None = None,
    rerank_pool_size: int | None = None,
    mmr_lambda: float | None = None,
    lexical_weight: float | None = None,
    semantic_weight: float | None = None,
    min_score: float | None = None,
    max_context_chars: int | None = None,
    adaptive_weights: bool | None = None,
    reranking_strategy: str | None = None,
) -> RetrievalOptions:
    return normalize_options(
        RetrievalOptions(),
        optimization_profile=optimization_profile,
        top_k=top_k,
        rerank_pool_size=rerank_pool_size,
        mmr_lambda=mmr_lambda,
        lexical_weight=lexical_weight,
        semantic_weight=semantic_weight,
        min_score=min_score,
        max_context_chars=max_context_chars,
        adaptive_weights=adaptive_weights,
        reranking_strategy=reranking_strategy,
    )


@app.get("/api/health")
def health():
    return {"status": "ok", "rag": "advanced", "version": "2026-05-27.1"}


@app.get("/api/debug")
def debug():
    from rag import load_index, DOCS_DIR
    try:
        chunks, _, _ = load_index()
        num_chunks = len(chunks)
    except Exception as e:
        num_chunks = 0
        print(f"Error loading index: {e}")

    options = normalize_options()

    return {
        "status": "ok",
        "rag": "advanced",
        "chunks": num_chunks,
        "docs_dir": str(DOCS_DIR),
        "groq_api_key_configured": bool(os.getenv("GROQ_API_KEY", "").strip()),
        "groq_model": GROQ_MODEL,
        "retrieval_options": options.__dict__,
    }


@app.get("/api/check-groq")
def check_groq():
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        return {"status": "missing", "detail": "GROQ_API_KEY is not configured."}
    return {"status": "ok", "groq_api_key_configured": True}


@app.post("/api/reload")
def reload_documents():
    """Wipes the cache, triggering re-indexing of documents on next search."""
    clear_index_cache()
    return {"status": "ok", "detail": "Document index cache cleared successfully."}


def answer_question(question: str, options: RetrievalOptions):
    question = question.strip()

    if not question:
        return JSONResponse(
            status_code=400,
            content={"detail": "Question is required."},
        )

    try:
        # Check Groq key before running LLM
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            raise HTTPException(
                status_code=400,
                detail="GROQ_API_KEY is missing or unconfigured.",
            )

        result = query_rag(question, options, True)
        return QueryResponse(
            answer=result["answer"],
            retrieval=result["retrieval"],
            summary=result["summary"],
        )
    except HTTPException as error:
        return QueryResponse(
            answer=(
                "Advanced RAG retrieval is working, but generation cannot run yet. "
                f"{error.detail}"
            )
        )
    except Exception as error:
        return QueryResponse(
            answer=f"Advanced RAG query failed before generation completed. {error}"
        )


@app.get("/api/query")
def query_get(
    question: str = "",
    optimization_profile: str | None = None,
    top_k: int | None = None,
    rerank_pool_size: int | None = None,
    mmr_lambda: float | None = None,
    lexical_weight: float | None = None,
    semantic_weight: float | None = None,
    min_score: float | None = None,
    max_context_chars: int | None = None,
    adaptive_weights: bool | None = None,
    reranking_strategy: str | None = None,
):
    options = build_options(
        top_k=top_k,
        optimization_profile=optimization_profile,
        rerank_pool_size=rerank_pool_size,
        mmr_lambda=mmr_lambda,
        lexical_weight=lexical_weight,
        semantic_weight=semantic_weight,
        min_score=min_score,
        max_context_chars=max_context_chars,
        adaptive_weights=adaptive_weights,
        reranking_strategy=reranking_strategy,
    )
    return answer_question(question, options)


@app.get("/api/ask")
def ask_get(
    question: str = "",
    optimization_profile: str | None = None,
    top_k: int | None = None,
    rerank_pool_size: int | None = None,
    mmr_lambda: float | None = None,
    lexical_weight: float | None = None,
    semantic_weight: float | None = None,
    min_score: float | None = None,
    max_context_chars: int | None = None,
    adaptive_weights: bool | None = None,
    reranking_strategy: str | None = None,
):
    options = build_options(
        top_k=top_k,
        optimization_profile=optimization_profile,
        rerank_pool_size=rerank_pool_size,
        mmr_lambda=mmr_lambda,
        lexical_weight=lexical_weight,
        semantic_weight=semantic_weight,
        min_score=min_score,
        max_context_chars=max_context_chars,
        adaptive_weights=adaptive_weights,
        reranking_strategy=reranking_strategy,
    )
    return answer_question(question, options)


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

    options = build_options(
        top_k=parsed_request.top_k,
        optimization_profile=parsed_request.optimization_profile,
        rerank_pool_size=parsed_request.rerank_pool_size,
        mmr_lambda=parsed_request.mmr_lambda,
        lexical_weight=parsed_request.lexical_weight,
        semantic_weight=parsed_request.semantic_weight,
        min_score=parsed_request.min_score,
        max_context_chars=parsed_request.max_context_chars,
        adaptive_weights=parsed_request.adaptive_weights,
        reranking_strategy=parsed_request.reranking_strategy,
    )
    return answer_question(parsed_request.question, options)


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
