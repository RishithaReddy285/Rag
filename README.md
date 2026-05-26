# Advanced RAG

Step 1: `pip install -r requirements.txt`

Step 2: Copy `.env.example` to `.env` and fill in your Groq and LangSmith keys

Step 3: Add PDF or TXT files to the `docs/` folder

Step 4: Run `python ingest.py` to build the vector database

Step 5: Run `python main.py` to start asking questions

The local CLI workflow uses Chroma similarity search. The Vercel API uses an advanced lightweight RAG retriever with sentence-aware chunking, query expansion, BM25-style scoring, phrase boosts, source labels, and diversity reranking before calling Groq.

To use the React frontend:

Step 1: Run `uvicorn local_server:app --reload --host 0.0.0.0 --port 8010`

Step 2: In another terminal, run `cd frontend && npm install && npm run dev`

Step 3: Open http://localhost:5173

To deploy on Vercel:

Step 1: Import the `naive-rag` folder as the Vercel project root. Do not set the root directory to `api/` or `frontend/`.

Step 2: Set the production environment variables in Vercel from `.env.example`. Leave `VITE_API_URL` empty in production so the frontend calls the same Vercel domain.

Step 3: Use these Vercel build settings:

- Framework Preset: `FastAPI`
- Install Command: `pip install -r requirements-vercel.txt && npm install`
- Build Command: `npm run build`
- Output Directory: `frontend/dist`

Step 4: Vercel builds the React app from `frontend/`, serves the frontend at `/`, and serves the advanced RAG backend through `/api`.

LangSmith traces will appear at https://smith.langchain.com
