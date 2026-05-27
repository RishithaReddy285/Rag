import math
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langsmith import traceable

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

# Optional ChromaDB imports to allow serverless fallback on Vercel
try:
    import chromadb
    from langchain_chroma import Chroma
    from local_embeddings import ChromaDefaultEmbeddings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

docs_env = os.getenv("DOCS_DIR", "./docs")
if not Path(docs_env).is_absolute():
    DOCS_DIR = (BASE_DIR / docs_env).resolve()
else:
    DOCS_DIR = Path(docs_env).resolve()

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "180"))
TOP_K = int(os.getenv("RAG_TOP_K", "5"))
RERANK_POOL_SIZE = int(os.getenv("RAG_RERANK_POOL_SIZE", "18"))
MMR_LAMBDA = float(os.getenv("RAG_MMR_LAMBDA", "0.72"))
RRF_K = int(os.getenv("RAG_RRF_K", "60"))
LEXICAL_WEIGHT = float(os.getenv("RAG_LEXICAL_WEIGHT", "0.58"))
SEMANTIC_WEIGHT = float(os.getenv("RAG_SEMANTIC_WEIGHT", "0.42"))

persist_env = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
if not Path(persist_env).is_absolute():
    CHROMA_PERSIST_DIR = str((BASE_DIR / persist_env).resolve())
else:
    CHROMA_PERSIST_DIR = str(Path(persist_env).resolve())

COLLECTION_NAME = "naive_rag"

OPTIMIZATION_PROFILES = {
    "balanced": {},
    "precision": {
        "top_k": 4,
        "rerank_pool_size": 16,
        "mmr_lambda": 0.82,
        "lexical_weight": 0.66,
        "semantic_weight": 0.34,
        "min_score": 0.08,
    },
    "recall": {
        "top_k": 7,
        "rerank_pool_size": 36,
        "mmr_lambda": 0.62,
        "lexical_weight": 0.48,
        "semantic_weight": 0.52,
        "min_score": 0.0,
    },
    "semantic": {
        "top_k": 5,
        "rerank_pool_size": 28,
        "mmr_lambda": 0.7,
        "lexical_weight": 0.32,
        "semantic_weight": 0.68,
    },
    "keyword": {
        "top_k": 5,
        "rerank_pool_size": 22,
        "mmr_lambda": 0.78,
        "lexical_weight": 0.76,
        "semantic_weight": 0.24,
    },
}


@dataclass(frozen=True)
class DocumentChunk:
    source: str
    index: int
    text: str
    tokens: tuple[str, ...]
    token_set: frozenset[str]
    token_counts: dict[str, int]
    tfidf_vector: dict[str, float]
    tfidf_norm: float


@dataclass(frozen=True)
class RetrievalOptions:
    top_k: int = TOP_K
    rerank_pool_size: int = RERANK_POOL_SIZE
    mmr_lambda: float = MMR_LAMBDA
    lexical_weight: float = LEXICAL_WEIGHT
    semantic_weight: float = SEMANTIC_WEIGHT
    min_score: float = 0.0
    max_context_chars: int = 6000
    adaptive_weights: bool = True
    optimization_profile: str = "balanced"
    reranking_strategy: str = "rrf"  # "rrf", "proximity", "llm"


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: DocumentChunk
    score: float
    lexical_score: float
    semantic_score: float
    rank: int


@lru_cache(maxsize=1)
def get_chroma_db():
    """Lazily load and cache the Chroma vector store if available."""
    if not CHROMA_AVAILABLE:
        return None
    try:
        embeddings = ChromaDefaultEmbeddings()
        persist_path = Path(CHROMA_PERSIST_DIR).resolve()
        if persist_path.exists():
            return Chroma(
                persist_directory=str(persist_path),
                embedding_function=embeddings,
                collection_name=COLLECTION_NAME,
            )
    except Exception as e:
        print(f"ChromaDB initialization failed: {e}")
    return None


def clamp_number(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def normalize_options(options: RetrievalOptions | None = None, **overrides) -> RetrievalOptions:
    base = options or RetrievalOptions()
    profile_name = str(
        overrides.pop("optimization_profile", None)
        or base.optimization_profile
        or "balanced"
    ).lower()
    profile = OPTIMIZATION_PROFILES.get(profile_name, OPTIMIZATION_PROFILES["balanced"])
    values = {
        "top_k": base.top_k,
        "rerank_pool_size": base.rerank_pool_size,
        "mmr_lambda": base.mmr_lambda,
        "lexical_weight": base.lexical_weight,
        "semantic_weight": base.semantic_weight,
        "min_score": base.min_score,
        "max_context_chars": base.max_context_chars,
        "adaptive_weights": base.adaptive_weights,
        "optimization_profile": profile_name,
        "reranking_strategy": base.reranking_strategy,
    }
    values.update(profile)
    values.update({key: value for key, value in overrides.items() if value is not None})

    top_k = int(clamp_number(float(values["top_k"]), 1, 12))
    rerank_pool_size = int(clamp_number(float(values["rerank_pool_size"]), top_k, 80))
    lexical_weight = clamp_number(float(values["lexical_weight"]), 0.0, 1.0)
    semantic_weight = clamp_number(float(values["semantic_weight"]), 0.0, 1.0)
    total_weight = lexical_weight + semantic_weight
    if total_weight <= 0:
        lexical_weight, semantic_weight = LEXICAL_WEIGHT, SEMANTIC_WEIGHT
        total_weight = lexical_weight + semantic_weight

    reranking_strategy = str(values.get("reranking_strategy", "rrf")).lower()
    if reranking_strategy not in {"rrf", "proximity", "llm"}:
        reranking_strategy = "rrf"

    return RetrievalOptions(
        top_k=top_k,
        rerank_pool_size=rerank_pool_size,
        mmr_lambda=clamp_number(float(values["mmr_lambda"]), 0.05, 0.95),
        lexical_weight=lexical_weight / total_weight,
        semantic_weight=semantic_weight / total_weight,
        min_score=clamp_number(float(values["min_score"]), 0.0, 1.0),
        max_context_chars=int(clamp_number(float(values["max_context_chars"]), 1200, 16000)),
        adaptive_weights=bool(values["adaptive_weights"]),
        optimization_profile=(
            profile_name if profile_name in OPTIMIZATION_PROFILES else "balanced"
        ),
        reranking_strategy=reranking_strategy,
    )


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

        next_start = max(0, end - CHUNK_OVERLAP)
        if next_start > 0:
            word_boundary = clean_text.find(
                " ",
                next_start,
                min(len(clean_text), next_start + 80),
            )
            if word_boundary != -1:
                next_start = word_boundary + 1

        start = next_start

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
            token_counts = count_tokens(tokens)
            chunks.append(
                DocumentChunk(
                    source=source,
                    index=index,
                    text=chunk,
                    tokens=tokens,
                    token_set=frozenset(tokens),
                    token_counts=token_counts,
                    tfidf_vector={},
                    tfidf_norm=0.0,
                )
            )

    return chunks


@lru_cache(maxsize=1)
def load_index() -> tuple[list[DocumentChunk], dict[str, int], float]:
    raw_chunks = load_chunks()
    document_frequency: dict[str, int] = {}

    for chunk in raw_chunks:
        for token in chunk.token_set:
            document_frequency[token] = document_frequency.get(token, 0) + 1

    average_length = (
        sum(len(chunk.tokens) for chunk in raw_chunks) / len(raw_chunks)
        if raw_chunks
        else 0.0
    )
    chunks = [
        DocumentChunk(
            source=chunk.source,
            index=chunk.index,
            text=chunk.text,
            tokens=chunk.tokens,
            token_set=chunk.token_set,
            token_counts=chunk.token_counts,
            tfidf_vector=build_tfidf_vector(
                chunk.token_counts,
                document_frequency,
                len(raw_chunks),
            ),
            tfidf_norm=0.0,
        )
        for chunk in raw_chunks
    ]
    chunks = [
        DocumentChunk(
            source=chunk.source,
            index=chunk.index,
            text=chunk.text,
            tokens=chunk.tokens,
            token_set=chunk.token_set,
            token_counts=chunk.token_counts,
            tfidf_vector=chunk.tfidf_vector,
            tfidf_norm=vector_norm(chunk.tfidf_vector),
        )
        for chunk in chunks
    ]

    return chunks, document_frequency, average_length


def clear_index_cache():
    """Wipes the global cache so documents can be re-indexed if modified."""
    load_index.cache_clear()
    get_chroma_db.cache_clear()


def count_tokens(tokens: tuple[str, ...] | list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1

    return counts


def inverse_document_frequency(
    token: str,
    document_frequency: dict[str, int],
    total_chunks: int,
) -> float:
    df = document_frequency.get(token, 0)
    return math.log(1 + (total_chunks - df + 0.5) / (df + 0.5))


def build_tfidf_vector(
    token_counts: dict[str, int],
    document_frequency: dict[str, int],
    total_chunks: int,
) -> dict[str, float]:
    if not token_counts or total_chunks <= 0:
        return {}

    return {
        token: (1 + math.log(frequency))
        * inverse_document_frequency(token, document_frequency, total_chunks)
        for token, frequency in token_counts.items()
    }


def vector_norm(vector: dict[str, float]) -> float:
    return math.sqrt(sum(weight * weight for weight in vector.values()))


def cosine_similarity(
    left_vector: dict[str, float],
    left_norm: float,
    right_vector: dict[str, float],
    right_norm: float,
) -> float:
    if not left_vector or not right_vector or left_norm <= 0 or right_norm <= 0:
        return 0.0

    if len(left_vector) > len(right_vector):
        left_vector, right_vector = right_vector, left_vector

    dot_product = sum(
        weight * right_vector.get(token, 0.0)
        for token, weight in left_vector.items()
    )
    return dot_product / (left_norm * right_norm)


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
    if not query_tokens or not chunk.tokens or average_length <= 0:
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

        idf = inverse_document_frequency(token, document_frequency, total_chunks)
        denominator = frequency + k1 * (1 - b + b * chunk_length / average_length)
        score += idf * ((frequency * (k1 + 1)) / denominator)

    return score


def semantic_score(
    query_tokens: list[str],
    chunk: DocumentChunk,
    document_frequency: dict[str, int],
    total_chunks: int,
) -> float:
    query_counts = count_tokens(query_tokens)
    query_vector = build_tfidf_vector(query_counts, document_frequency, total_chunks)
    return cosine_similarity(
        query_vector,
        vector_norm(query_vector),
        chunk.tfidf_vector,
        chunk.tfidf_norm,
    )


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


def normalized_scores(scored_chunks: list[tuple[float, DocumentChunk]]) -> dict[tuple[str, int], float]:
    if not scored_chunks:
        return {}

    values = [score for score, _ in scored_chunks]
    low = min(values)
    high = max(values)
    span = high - low

    if span <= 0:
        return {
            (chunk.source, chunk.index): 1.0 if score > 0 else 0.0
            for score, chunk in scored_chunks
        }

    return {
        (chunk.source, chunk.index): (score - low) / span
        for score, chunk in scored_chunks
    }


def adaptive_hybrid_weights(
    question: str,
    query_tokens: list[str],
    options: RetrievalOptions,
) -> tuple[float, float]:
    if not options.adaptive_weights:
        return options.lexical_weight, options.semantic_weight

    lexical_weight = options.lexical_weight
    semantic_weight = options.semantic_weight
    has_exact_phrase = '"' in question or len(query_tokens) <= 4
    has_long_query = len(query_tokens) >= 9

    if has_exact_phrase:
        lexical_weight += 0.12
        semantic_weight -= 0.12
    if has_long_query:
        lexical_weight -= 0.1
        semantic_weight += 0.1

    lexical_weight = clamp_number(lexical_weight, 0.2, 0.8)
    semantic_weight = clamp_number(semantic_weight, 0.2, 0.8)
    total = lexical_weight + semantic_weight
    return lexical_weight / total, semantic_weight / total


def select_diverse_chunks(
    scored_chunks: list[tuple[float, DocumentChunk, float, float]],
    options: RetrievalOptions,
) -> list[RetrievedChunk]:
    selected: list[RetrievedChunk] = []
    remaining = scored_chunks[:]
    used_chars = 0

    while remaining and len(selected) < options.top_k:
        best_position = 0
        best_score = float("-inf")

        for position, (relevance, chunk, lexical_score, semantic_score) in enumerate(remaining):
            diversity_penalty = max(
                (similarity(chunk, chosen.chunk) for chosen in selected),
                default=0.0,
            )
            mmr_score = (
                options.mmr_lambda * relevance
                - (1 - options.mmr_lambda) * diversity_penalty
            )

            if mmr_score > best_score:
                best_position = position
                best_score = mmr_score

        relevance, best_chunk, lexical_score, semantic_score = remaining.pop(best_position)
        projected_chars = used_chars + len(best_chunk.text)
        if selected and projected_chars > options.max_context_chars:
            continue

        used_chars = projected_chars
        selected.append(
            RetrievedChunk(
                chunk=best_chunk,
                score=relevance,
                lexical_score=lexical_score,
                semantic_score=semantic_score,
                rank=len(selected) + 1,
            )
        )

    return selected


def reciprocal_rank_fusion(
    lexical_scores: list[tuple[float, DocumentChunk]],
    semantic_scores: list[tuple[float, DocumentChunk]],
    options: RetrievalOptions,
    query_tokens: list[str],
    question: str,
) -> list[tuple[float, DocumentChunk, float, float]]:
    fused: dict[tuple[str, int], tuple[float, DocumentChunk, float, float]] = {}
    lexical_normalized = normalized_scores(lexical_scores)
    semantic_normalized = normalized_scores(semantic_scores)
    lexical_weight, semantic_weight = adaptive_hybrid_weights(
        question,
        query_tokens,
        options,
    )

    for weight, ranked_scores in (
        (lexical_weight, lexical_scores),
        (semantic_weight, semantic_scores),
    ):
        for rank, (_, chunk) in enumerate(ranked_scores, start=1):
            key = (chunk.source, chunk.index)
            previous_score, _, lexical_score, semantic_score = fused.get(
                key,
                (
                    0.0,
                    chunk,
                    lexical_normalized.get(key, 0.0),
                    semantic_normalized.get(key, 0.0),
                ),
            )
            normalized_component = (
                lexical_weight * lexical_normalized.get(key, 0.0)
                + semantic_weight * semantic_normalized.get(key, 0.0)
            )
            rank_component = weight / (RRF_K + rank)
            fused[key] = (
                previous_score + rank_component + normalized_component,
                chunk,
                lexical_score,
                semantic_score,
            )

    return sorted(fused.values(), key=lambda item: item[0], reverse=True)


def find_matching_chunk(chroma_text: str, chroma_source: str, cached_chunks: list[DocumentChunk]) -> DocumentChunk | None:
    """Matches a retrieved document from ChromaDB back to the in-memory DocumentChunk cache."""
    normalized_chroma_source = chroma_source.replace("\\", "/").lower()
    
    # Extract file name (e.g. sample.txt) from chroma_source
    file_name = Path(normalized_chroma_source).name.lower()
    if not file_name:
        return None

    # Filter cached chunks by the same file name
    candidates = [
        c for c in cached_chunks 
        if c.source.lower() == file_name or c.source.lower().endswith(file_name) or file_name.endswith(c.source.lower())
    ]

    if not candidates:
        return None

    # Find the candidate chunk that has the highest keyword overlap or Jaccard similarity
    chroma_tokens = set(tokenize_list(chroma_text))
    if not chroma_tokens:
        return candidates[0]

    best_chunk = None
    best_overlap = -1

    for chunk in candidates:
        overlap = len(chunk.token_set & chroma_tokens)
        if overlap > best_overlap:
            best_overlap = overlap
            best_chunk = chunk

    return best_chunk


def compute_proximity_score(question: str, text: str) -> float:
    """Computes a proximity relevance boost based on how close query keywords appear in the chunk text."""
    query_words = [w.lower() for w in re.findall(r"[a-zA-Z0-9]+", question) if len(w) > 2]
    if not query_words:
        return 0.0

    text_words = [w.lower() for w in re.findall(r"[a-zA-Z0-9]+", text)]
    if not text_words:
        return 0.0

    # Find the positions of query terms in the text
    word_indices = {}
    for i, w in enumerate(text_words):
        for qw in query_words:
            if w == qw or w.startswith(qw) or qw.startswith(w):
                if qw not in word_indices:
                    word_indices[qw] = []
                word_indices[qw].append(i)

    # Need at least two query terms to evaluate spacing
    if len(word_indices) < 2:
        exact_count = sum(1 for qw in query_words if qw in text.lower())
        return exact_count * 0.1

    all_hits = []
    for qw, idxs in word_indices.items():
        for idx in idxs:
            all_hits.append((idx, qw))
    all_hits.sort()

    min_window = float("inf")
    for i in range(len(all_hits)):
        seen = set()
        seen.add(all_hits[i][1])
        for j in range(i + 1, len(all_hits)):
            seen.add(all_hits[j][1])
            if len(seen) == len(word_indices):
                window = all_hits[j][0] - all_hits[i][0]
                if window < min_window:
                    min_window = window
                break

    if min_window == float("inf"):
        return 0.0

    # Exponential decay based on distance: smaller spans indicate higher structural alignment
    ratio = len(word_indices) / len(query_words)
    proximity = ratio * math.exp(-min_window / 25.0)
    return proximity


def groq_llm_rerank(question: str, candidates: list[tuple[float, DocumentChunk, float, float]]) -> list[tuple[float, DocumentChunk, float, float]]:
    """Uses Groq to score and rerank the top candidate chunks for maximum precision."""
    if not candidates:
        return candidates

    rerank_limit = min(len(candidates), 10)
    candidates_to_rerank = candidates[:rerank_limit]
    remaining_candidates = candidates[rerank_limit:]

    # Prepare document content with clear block identifiers
    context_str = ""
    for idx, (_, chunk, _, _) in enumerate(candidates_to_rerank):
        context_str += f"[ID: {idx}] Document: {chunk.source} (Index: {chunk.index})\nText: {chunk.text}\n---\n"

    prompt = f"""You are an advanced search relevance grader. You must rank the following document chunks based on their direct relevance to answering the User Question.

User Question: {question}

Document Chunks to rank:
{context_str}
For each chunk ID, output a numeric relevance score from 0.0 (completely irrelevant) to 10.0 (perfectly relevant and directly answers the question).
Respond ONLY with a valid JSON object mapping ID strings to float scores, exactly like this:
{{
  "0": 8.5,
  "1": 4.0
}}
Do not write any introductory or concluding text, only the raw JSON code block."""

    try:
        llm = ChatGroq(model=GROQ_MODEL, temperature=0.0)
        response = llm.invoke(prompt)
        content = response.content.strip()

        # Extract the JSON block
        json_match = re.search(r"\{.*?\}", content, re.DOTALL)
        if json_match:
            # Safely parse JSON structure
            import json
            scores_dict = json.loads(json_match.group(0))
            
            scored_candidates = []
            for idx, (score, chunk, l_score, s_score) in enumerate(candidates_to_rerank):
                str_idx = str(idx)
                llm_score_raw = scores_dict.get(str_idx, scores_dict.get(idx, 5.0))
                llm_score = float(llm_score_raw)
                
                # Blend the retrieval score and LLM relevance score
                normalized_llm = llm_score / 10.0
                blended_score = 0.7 * normalized_llm + 0.3 * score
                scored_candidates.append((blended_score, chunk, l_score, s_score))

            scored_candidates.sort(key=lambda x: x[0], reverse=True)
            return scored_candidates + remaining_candidates
    except Exception as e:
        print(f"Error in Groq LLM reranking: {e}. Falling back to default scoring.")

    return candidates


def retrieve_chunks(
    question: str,
    options: RetrievalOptions | None = None,
) -> list[RetrievedChunk]:
    options = normalize_options(options)
    chunks, document_frequency, average_length = load_index()
    if not chunks:
        return []

    query_tokens = expand_query(question)
    lexical_scores = []
    semantic_scores = []

    # 1. Dense Semantic Retrieval via ChromaDB (if available and running locally)
    use_chroma = False
    chroma_results = []
    if CHROMA_AVAILABLE:
        db = get_chroma_db()
        if db is not None:
            try:
                # Retrieve slightly more than pool size to allow robust rank-merging
                search_k = min(len(chunks), max(options.rerank_pool_size * 2, 40))
                chroma_results = db.similarity_search_with_score(question, k=search_k)
                if chroma_results:
                    use_chroma = True
            except Exception as e:
                print(f"ChromaDB dense query failed: {e}. Falling back to in-memory TF-IDF.")

    # 2. Populating Semantic Scores
    if use_chroma:
        matched_chunks_set = set()
        for doc, distance in chroma_results:
            source = doc.metadata.get("source", "")
            chunk_obj = find_matching_chunk(doc.page_content, source, chunks)
            if chunk_obj is not None:
                key = (chunk_obj.source, chunk_obj.index)
                if key not in matched_chunks_set:
                    matched_chunks_set.add(key)
                    # Convert distance securely: 1.0 / (1.0 + distance)
                    score_val = 1.0 / (1.0 + float(distance))
                    semantic_scores.append((score_val, chunk_obj))

        # Fill in unmatched chunks with default 0.0 semantic relevance
        for chunk in chunks:
            key = (chunk.source, chunk.index)
            if key not in matched_chunks_set:
                semantic_scores.append((0.0, chunk))
    else:
        # High-performance in-memory TF-IDF semantic fallback
        for chunk in chunks:
            semantic_scores.append(
                (
                    semantic_score(
                        query_tokens,
                        chunk,
                        document_frequency,
                        len(chunks),
                    ),
                    chunk,
                )
            )

    # 3. Sparse Lexical Scoring (BM25 + Phrase Boost)
    for chunk in chunks:
        bm25 = bm25_score(
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
        lexical_score = bm25 + overlap_score + phrase_boost(question, chunk)
        lexical_scores.append((lexical_score, chunk))

    lexical_scores.sort(key=lambda item: item[0], reverse=True)
    semantic_scores.sort(key=lambda item: item[0], reverse=True)

    # 4. Rank Merging (Reciprocal Rank Fusion)
    scored_chunks = reciprocal_rank_fusion(
        lexical_scores[: options.rerank_pool_size],
        semantic_scores[: options.rerank_pool_size],
        options,
        query_tokens,
        question,
    )

    candidates = [
        (score, chunk, lexical_score, semantic_score)
        for score, chunk, lexical_score, semantic_score in scored_chunks[
            : options.rerank_pool_size
        ]
        if score >= options.min_score
    ]

    if not candidates:
        lexical_normalized = normalized_scores(lexical_scores)
        candidates = [
            (
                lexical_normalized.get((chunk.source, chunk.index), 0.0),
                chunk,
                lexical_normalized.get((chunk.source, chunk.index), 0.0),
                0.0,
            )
            for _, chunk in lexical_scores[: options.rerank_pool_size]
        ]

    # 5. Advanced Hybrid Reranking Optimization
    if options.reranking_strategy == "proximity":
        # Multi-term structural word distance reranking
        proximity_scored = []
        for score, chunk, l_score, s_score in candidates:
            prox_boost = compute_proximity_score(question, chunk.text)
            blended = 0.65 * score + 0.35 * prox_boost
            proximity_scored.append((blended, chunk, l_score, s_score))
        proximity_scored.sort(key=lambda item: item[0], reverse=True)
        candidates = proximity_scored

    elif options.reranking_strategy == "llm":
        # Deep semantic Groq LLM scoring and sorting
        candidates = groq_llm_rerank(question, candidates)

    return select_diverse_chunks(candidates, options)


def retrieval_metadata(retrieved_chunks: list[RetrievedChunk]) -> list[dict[str, object]]:
    return [
        {
            "rank": item.rank,
            "source": item.chunk.source,
            "chunk": item.chunk.index + 1,
            "score": round(item.score, 4),
            "lexical_score": round(item.lexical_score, 4),
            "semantic_score": round(item.semantic_score, 4),
            "preview": item.chunk.text[:220],
        }
        for item in retrieved_chunks
    ]


def retrieval_summary(
    retrieved_chunks: list[RetrievedChunk],
    options: RetrievalOptions,
) -> dict[str, object]:
    if not retrieved_chunks:
        return {
            "profile": options.optimization_profile,
            "reranking_strategy": options.reranking_strategy,
            "selected_chunks": 0,
            "avg_score": 0.0,
            "max_score": 0.0,
            "lexical_share": 0.0,
            "semantic_share": 0.0,
        }

    lexical_total = sum(item.lexical_score for item in retrieved_chunks)
    semantic_total = sum(item.semantic_score for item in retrieved_chunks)
    signal_total = lexical_total + semantic_total

    return {
        "profile": options.optimization_profile,
        "reranking_strategy": options.reranking_strategy,
        "selected_chunks": len(retrieved_chunks),
        "avg_score": round(
            sum(item.score for item in retrieved_chunks) / len(retrieved_chunks),
            4,
        ),
        "max_score": round(max(item.score for item in retrieved_chunks), 4),
        "lexical_share": round(lexical_total / signal_total, 4)
        if signal_total
        else 0.0,
        "semantic_share": round(semantic_total / signal_total, 4)
        if signal_total
        else 0.0,
    }


def retrieve_context(
    question: str,
    k: int = TOP_K,
    options: RetrievalOptions | None = None,
) -> str:
    options = normalize_options(options, top_k=k)
    selected_chunks = retrieve_chunks(question, options)
    if not selected_chunks:
        return "No documents were found in the configured docs directory."

    return "\n\n".join(
        f"Source: {item.chunk.source} | Chunk: {item.chunk.index + 1}\n{item.chunk.text}"
        for item in selected_chunks
    )


@traceable
def query_rag(
    question: str,
    options: RetrievalOptions | None = None,
    include_metadata: bool = False,
) -> str | dict[str, object]:
    options = normalize_options(options)
    selected_chunks = retrieve_chunks(question, options)
    context = (
        "\n\n".join(
            f"Source: {item.chunk.source} | Chunk: {item.chunk.index + 1}\n{item.chunk.text}"
            for item in selected_chunks
        )
        if selected_chunks
        else "No documents were found in the configured docs directory."
    )
    llm = ChatGroq(model=GROQ_MODEL)
    response = llm.invoke(
        [
            (
                "system",
                (
                    "You are an advanced RAG assistant. Answer only from the "
                    "provided context. If the context does not contain the "
                    "answer, say that the documents do not provide enough "
                    "information. Cite source names when useful."
                ),
            ),
            (
                "user",
                f"Context:\n{context}\n\nQuestion: {question}",
            ),
        ]
    )

    if include_metadata:
        return {
            "answer": response.content,
            "retrieval": retrieval_metadata(selected_chunks),
            "summary": retrieval_summary(selected_chunks, options),
            "options": normalize_options(options).__dict__,
        }

    return response.content

