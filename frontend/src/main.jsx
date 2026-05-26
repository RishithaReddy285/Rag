import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { Loader2, Send, Server, Sparkles } from "lucide-react";
import "./styles.css";

const API_URL =
  import.meta.env.VITE_API_URL ||
  (import.meta.env.DEV ? "http://localhost:8010" : "");

function App() {
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content:
        "Ask a question about the documents indexed in your local Chroma database.",
    },
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const [status, setStatus] = useState("checking");
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
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ question: trimmed }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Request failed");
      }

      setMessages((current) => [
        ...current,
        { role: "assistant", content: data.answer },
      ]);
      setStatus("online");
    } catch (error) {
      setStatus("offline");
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

  return (
    <main className="app-shell">
      <section className="workspace" aria-label="Naive RAG chat">
        <header className="topbar">
          <div>
            <h1>Naive RAG</h1>
            <p>Groq generation with local Chroma retrieval</p>
          </div>
          <div className={`status status-${status}`}>
            <Server size={16} aria-hidden="true" />
            <span>{status}</span>
          </div>
        </header>

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
              <p>Searching the vector database...</p>
            </article>
          ) : null}
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
