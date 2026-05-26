# Naive RAG

Step 1: `pip install -r requirements.txt`

Step 2: Copy `.env.example` to `.env` and fill in your Groq and LangSmith keys

Step 3: Add PDF or TXT files to the `docs/` folder

Step 4: Run `python ingest.py` to build the vector database

Step 5: Run `python main.py` to start asking questions

To use the React frontend:

Step 1: Run `uvicorn api:app --reload --host 0.0.0.0 --port 8010`

Step 2: In another terminal, run `cd frontend && npm install && npm run dev`

Step 3: Open http://localhost:5173

LangSmith traces will appear at https://smith.langchain.com
