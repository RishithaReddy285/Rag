# Naive RAG

Step 1: `pip install -r requirements.txt`

Step 2: Copy `.env.example` to `.env` and fill in your Groq and LangSmith keys

Step 3: Add PDF or TXT files to the `docs/` folder

Step 4: Run `python ingest.py` to build the vector database

Step 5: Run `python main.py` to start asking questions

To use the React frontend:

Step 1: Run `uvicorn server:app --reload --host 0.0.0.0 --port 8010`

Step 2: In another terminal, run `cd frontend && npm install && npm run dev`

Step 3: Open http://localhost:5173

To deploy on Vercel:

Step 1: Set the production environment variables in Vercel from `.env.example`

Step 2: Vercel runs `python ingest.py` during the build to create `chroma_db`

Step 3: Vercel builds the React app from `frontend/` and serves the FastAPI app through `/api`

LangSmith traces will appear at https://smith.langchain.com
