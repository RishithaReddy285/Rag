from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from rag import RetrievalOptions, query_rag, normalize_options, clear_index_cache

load_dotenv()

app = FastAPI(title="Advanced RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5175",
        "http://localhost:5176",
        "http://127.0.0.1:5176",
    ],
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    return {"status": "ok", "rag": "advanced"}


@app.post("/api/reload")
def reload_documents():
    """Wipes the cache, triggering re-indexing of documents on next search."""
    clear_index_cache()
    return {"status": "ok", "detail": "Document index cache cleared successfully."}


async def answer_question(question: str, options: RetrievalOptions):
    if not question:
        raise HTTPException(status_code=400, detail="Question is required.")

    result = await run_in_threadpool(query_rag, question, options, True)
    return QueryResponse(
        answer=result["answer"],
        retrieval=result["retrieval"],
        summary=result["summary"],
    )


@app.get("/api/ask", response_model=QueryResponse)
async def ask(
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
    return await answer_question(question.strip(), options)


@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    options = build_options(
        top_k=request.top_k,
        optimization_profile=request.optimization_profile,
        rerank_pool_size=request.rerank_pool_size,
        mmr_lambda=request.mmr_lambda,
        lexical_weight=request.lexical_weight,
        semantic_weight=request.semantic_weight,
        min_score=request.min_score,
        max_context_chars=request.max_context_chars,
        adaptive_weights=request.adaptive_weights,
        reranking_strategy=request.reranking_strategy,
    )
    return await answer_question(request.question.strip(), options)
