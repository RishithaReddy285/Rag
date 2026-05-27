import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Gauge,
  Loader2,
  Send,
  Server,
  SlidersHorizontal,
  Sparkles,
} from "lucide-react";
import "./styles.css";

const API_URL =
  import.meta.env.VITE_API_URL ||
  (import.meta.env.DEV ? "http://localhost:8099" : "");

const optimizerProfiles = {
  balanced: {
    topK: 5,
    rerankPoolSize: 24,
    lexicalWeight: 0.58,
    semanticWeight: 0.42,
    mmrLambda: 0.72,
    minScore: 0,
  },
  precision: {
    topK: 4,
    rerankPoolSize: 16,
    lexicalWeight: 0.66,
    semanticWeight: 0.34,
    mmrLambda: 0.82,
    minScore: 0.08,
  },
  recall: {
    topK: 7,
    rerankPoolSize: 36,
    lexicalWeight: 0.48,
    semanticWeight: 0.52,
    mmrLambda: 0.62,
    minScore: 0,
  },
  semantic: {
    topK: 5,
    rerankPoolSize: 28,
    lexicalWeight: 0.32,
    semanticWeight: 0.68,
    mmrLambda: 0.7,
    minScore: 0,
  },
  keyword: {
    topK: 5,
    rerankPoolSize: 22,
    lexicalWeight: 0.76,
    semanticWeight: 0.24,
    mmrLambda: 0.78,
    minScore: 0,
  },
};

function App() {
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content:
        "Ask a question about the documents available to this RAG system.",
    },
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const [status, setStatus] = useState("checking");
  const [retrieval, setRetrieval] = useState([]);
  const [summary, setSummary] = useState(null);
  const [options, setOptions] = useState({
    optimizationProfile: "balanced",
    topK: 5,
    rerankPoolSize: 24,
    lexicalWeight: 0.58,
    semanticWeight: 0.42,
    mmrLambda: 0.72,
    minScore: 0,
    adaptiveWeights: true,
    rerankingStrategy: "rrf",
  });
  const inputRef = useRef(null);

  useEffect(() => {
    fetch(`${API_URL}/api/health`)
      .then((response) => {
        if (!response.ok) {
          throw new Error("Backend health check failed");
        }
        setStatus("online");
      })
      .catch(() => setStatus("offline"));
  }, []);

  async function askQuestion(event) {
    event.preventDefault();

    const trimmed = question.trim();
    if (!trimmed || isLoading) {
      return;
    }

    setQuestion("");
    setIsLoading(true);
    setMessages((current) => [
      ...current,
      { role: "user", content: trimmed },
    ]);

    try {
      const response = await fetch(`${API_URL}/api/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: trimmed,
          optimization_profile: options.optimizationProfile,
          top_k: options.topK,
          rerank_pool_size: options.rerankPoolSize,
          lexical_weight: options.lexicalWeight,
          semantic_weight: options.semanticWeight,
          mmr_lambda: options.mmrLambda,
          min_score: options.minScore,
          adaptive_weights: options.adaptiveWeights,
          reranking_strategy: options.rerankingStrategy,
        }),
      });

      const contentType = response.headers.get("content-type") || "";
      const data = contentType.includes("application/json")
        ? await response.json()
        : { detail: await response.text() };

      if (!response.ok) {
        const requestError = new Error(data.detail || "Request failed");
        requestError.backendReachable = true;
        throw requestError;
      }

      setMessages((current) => [
        ...current,
        { role: "assistant", content: data.answer },
      ]);
      setRetrieval(data.retrieval || []);
      setSummary(data.summary || null);
      setStatus("online");
    } catch (error) {
      setStatus(error.backendReachable ? "online" : "offline");
      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          content:
            error.message ||
            "The backend did not return an answer. Check that the API server is running.",
        },
      ]);
    } finally {
      setIsLoading(false);
      inputRef.current?.focus();
    }
  }

  function updateOption(key, value) {
    setOptions((current) => ({ ...current, [key]: value }));
  }

  function applyProfile(profileName) {
    setOptions((current) => ({
      ...current,
      ...optimizerProfiles[profileName],
      optimizationProfile: profileName,
    }));
  }

  return (
    <main className="app-shell">
      <section className="workspace" aria-label="Advanced RAG chat">
        <header className="topbar">
          <div>
            <h1>Advanced RAG</h1>
            <p>Hybrid retrieval, adaptive fusion, MMR reranking</p>
          </div>
          <div className={`status status-${status}`}>
            <Server size={16} aria-hidden="true" />
            <span>{status}</span>
          </div>
        </header>

        <div className="work-area">
          <div className="messages" aria-live="polite">
            {messages.map((message, index) => (
              <article
                className={`message message-${message.role}`}
                key={`${message.role}-${index}`}
              >
                <div className="message-icon">
                  {message.role === "assistant" ? (
                    <Sparkles size={16} aria-hidden="true" />
                  ) : (
                    <span aria-hidden="true">Q</span>
                  )}
                </div>
                <p>{message.content}</p>
              </article>
            ))}
            {isLoading ? (
              <article className="message message-assistant">
                <div className="message-icon">
                  <Loader2 className="spin" size={16} aria-hidden="true" />
                </div>
                <p>Searching the document context...</p>
              </article>
            ) : null}
          </div>

          <aside className="optimizer" aria-label="Hybrid reranking optimizer">
            <div className="optimizer-title" style={{ display: "flex", justifyContent: "space-between", alignItems: "center", width: "100%" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                <SlidersHorizontal size={17} aria-hidden="true" />
                <h2>Optimizer</h2>
              </div>
              <button
                onClick={async () => {
                  try {
                    const res = await fetch(`${API_URL}/api/reload`, { method: "POST" });
                    if (res.ok) alert("Cache cleared! Documents will be re-indexed on next query.");
                  } catch (err) {
                    console.error("Failed to reload documents:", err);
                  }
                }}
                title="Reload documents from disk and clear caches"
                style={{
                  border: "1px solid #cdd8d5",
                  borderRadius: "6px",
                  padding: "2px 8px",
                  background: "#ffffff",
                  fontSize: "0.72rem",
                  fontWeight: "700",
                  color: "#53665a",
                  cursor: "pointer",
                }}
                type="button"
              >
                Reload Docs
              </button>
            </div>

            <div className="profile-group" aria-label="Optimization profile">
              {Object.keys(optimizerProfiles).map((profileName) => (
                <button
                  className={
                    options.optimizationProfile === profileName
                      ? "profile-active"
                      : ""
                  }
                  key={profileName}
                  onClick={() => applyProfile(profileName)}
                  type="button"
                >
                  {profileName}
                </button>
              ))}
            </div>

            <label className="toggle-row">
              <span>Reranker Mode</span>
              <select
                value={options.rerankingStrategy}
                onChange={(event) =>
                  updateOption("rerankingStrategy", event.target.value)
                }
                style={{
                  border: "1px solid #cdd8d5",
                  borderRadius: "6px",
                  padding: "4px 8px",
                  background: "#ffffff",
                  fontSize: "0.82rem",
                  fontWeight: "600",
                  color: "#31414b",
                  outline: "none",
                  cursor: "pointer",
                }}
              >
                <option value="rrf">RRF Only</option>
                <option value="proximity">Proximity Boosted</option>
                <option value="llm">Groq LLM Rerank</option>
              </select>
            </label>

            <label className="toggle-row">
              <span>Adaptive weights</span>
              <input
                checked={options.adaptiveWeights}
                onChange={(event) =>
                  updateOption("adaptiveWeights", event.target.checked)
                }
                type="checkbox"
              />
            </label>

            <label className="control-row">
              <span>Top K</span>
              <output>{options.topK}</output>
              <input
                max="10"
                min="1"
                onChange={(event) =>
                  updateOption("topK", Number(event.target.value))
                }
                type="range"
                value={options.topK}
              />
            </label>

            <label className="control-row">
              <span>Rerank pool</span>
              <output>{options.rerankPoolSize}</output>
              <input
                max="60"
                min="5"
                onChange={(event) =>
                  updateOption("rerankPoolSize", Number(event.target.value))
                }
                type="range"
                value={options.rerankPoolSize}
              />
            </label>

            <label className="control-row">
              <span>Lexical</span>
              <output>{options.lexicalWeight.toFixed(2)}</output>
              <input
                max="1"
                min="0"
                onChange={(event) => {
                  const lexicalWeight = Number(event.target.value);
                  updateOption("lexicalWeight", lexicalWeight);
                  updateOption("semanticWeight", 1 - lexicalWeight);
                }}
                step="0.01"
                type="range"
                value={options.lexicalWeight}
              />
            </label>

            <label className="control-row">
              <span>Semantic</span>
              <output>{options.semanticWeight.toFixed(2)}</output>
              <input
                max="1"
                min="0"
                onChange={(event) => {
                  const semanticWeight = Number(event.target.value);
                  updateOption("semanticWeight", semanticWeight);
                  updateOption("lexicalWeight", 1 - semanticWeight);
                }}
                step="0.01"
                type="range"
                value={options.semanticWeight}
              />
            </label>

            <label className="control-row">
              <span>MMR</span>
              <output>{options.mmrLambda.toFixed(2)}</output>
              <input
                max="0.95"
                min="0.05"
                onChange={(event) =>
                  updateOption("mmrLambda", Number(event.target.value))
                }
                step="0.01"
                type="range"
                value={options.mmrLambda}
              />
            </label>

            <label className="control-row">
              <span>Min score</span>
              <output>{options.minScore.toFixed(2)}</output>
              <input
                max="1"
                min="0"
                onChange={(event) =>
                  updateOption("minScore", Number(event.target.value))
                }
                step="0.01"
                type="range"
                value={options.minScore}
              />
            </label>

            <div className="source-panel">
              <div className="source-title">
                <Gauge size={16} aria-hidden="true" />
                <h2>Sources</h2>
              </div>
              {summary ? (
                <div className="summary-grid">
                  <span>Avg {Number(summary.avg_score || 0).toFixed(2)}</span>
                  <span>Lex {Number(summary.lexical_share || 0).toFixed(2)}</span>
                  <span>Sem {Number(summary.semantic_share || 0).toFixed(2)}</span>
                </div>
              ) : null}
              {retrieval.length ? (
                retrieval.map((item) => (
                  <article className="source-item" key={`${item.source}-${item.chunk}`}>
                    <div>
                      <strong>
                        {item.rank}. {item.source}
                      </strong>
                      <span>Chunk {item.chunk}</span>
                    </div>
                    <meter max="1.1" min="0" value={item.score} />
                    <p>{item.preview}</p>
                  </article>
                ))
              ) : (
                <p className="empty-sources">No ranked sources yet.</p>
              )}
            </div>
          </aside>
        </div>

        <form className="composer" onSubmit={askQuestion}>
          <input
            aria-label="Question"
            disabled={isLoading}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="Ask about your indexed docs"
            ref={inputRef}
            value={question}
          />
          <button
            aria-label="Send question"
            disabled={isLoading || !question.trim()}
            title="Send question"
            type="submit"
          >
            {isLoading ? (
              <Loader2 className="spin" size={18} aria-hidden="true" />
            ) : (
              <Send size={18} aria-hidden="true" />
            )}
          </button>
        </form>
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
