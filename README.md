# LiveKit RAG Voice Assistant

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![LiveKit](https://img.shields.io/badge/LiveKit_Agents-1.5-0F172A?logo=livekit&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.116%2B-009688?logo=fastapi&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-16-000000?logo=nextdotjs&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green?logo=open-source-initiative&logoColor=white)

LiveKit RAG Voice Assistant is a decoupled realtime AI voice project built from:

- `apps/web`: a Next.js LiveKit frontend
- `apps/agent`: a Python LiveKit voice agent worker
- `services/rag-backend`: a FastAPI RAG backend with PostgreSQL and pgvector

The voice agent keeps normal conversation working, adds grounded document Q&A through the RAG backend, and includes a weather tool demo using Open-Meteo.

## Architecture

```text
apps/web (Next.js)
  -> requests LiveKit token
  -> joins LiveKit room
  -> streams microphone audio
  -> renders transcript, controls, and session status

apps/agent (Python LiveKit worker)
  -> handles VAD, STT, LLM, TTS, interruptions, and text fallback
  -> decides when to call tools
  -> ask_knowledge_base(question) -> services/rag-backend POST /retrieval/context
  -> synthesizes the final grounded answer in the voice agent LLM
  -> get_current_weather(city) -> Open-Meteo APIs

services/rag-backend (FastAPI)
  -> ingests text / PDF / DOCX
  -> stores embeddings in PostgreSQL + pgvector
  -> runs retrieval, reranking, and context preparation
  -> keeps POST /chat/ask for backend-only testing and compatibility
```

## Project Structure

```text
.
+-- apps
|   +-- agent
|   +-- web
+-- docs
|   +-- sample-faq
|       +-- company-faq.txt
+-- services
|   +-- rag-backend
+-- docker-compose.yml
+-- README.md
```

## Prerequisites

- Node.js 18+
- Python 3.11+
- Docker Desktop
- A LiveKit Cloud project
- LiveKit credentials for `apps/web` and `apps/agent`
- A valid chat provider key in `services/rag-backend/.env`

Important:

- `docker compose up --build` starts only `postgres` and `rag-backend`
- you still run `apps/agent` and `apps/web` manually
- RAG answers can feel slower than normal chat because they add an HTTP hop plus retrieval and grounded generation

## Environment Setup

### 1. Web app

```powershell
cd apps/web
Copy-Item .env.example .env.local
```

Set your LiveKit values in `apps/web/.env.local`.

### 2. Voice agent

```powershell
cd apps/agent
Copy-Item .env.example .env
```

Set your LiveKit values in `apps/agent/.env`.

The new agent config includes:

```env
LIVEKIT_AGENT_NAME=livekit-rag-voice-agent
RAG_BACKEND_URL=http://localhost:8000
RAG_CONTEXT_PATH=/retrieval/context
RAG_CHAT_PATH=/chat/ask
```

### 3. RAG backend

```powershell
cd services/rag-backend
Copy-Item .env.example .env
```

Then update `services/rag-backend/.env` with a working provider configuration.

The copied backend currently defaults to:

- local embeddings
- Groq chat provider
- PostgreSQL at `postgres:5432` when run through Docker Compose

If you keep the default chat provider, make sure `GROQ_API_KEY` is set.

If you want to run the copied backend unit tests locally outside Docker, also create a Python environment in `services/rag-backend`:

```powershell
cd services/rag-backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt -r requirements-test.txt
```

## Run The Backend Services

From the repository root:

```powershell
docker compose up --build
```

This starts:

- `postgres` on `localhost:5432`
- `rag-backend` on `localhost:8000`

Useful backend URLs:

- Health: [http://localhost:8000/health](http://localhost:8000/health)
- Readiness: [http://localhost:8000/ready](http://localhost:8000/ready)
- Docs: [http://localhost:8000/api/v1/docs](http://localhost:8000/api/v1/docs)

## Run The Voice Agent

```powershell
cd apps/agent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python agent.py download-files
python agent.py dev
```

## Run The Frontend

```powershell
cd apps/web
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Ingest The Sample FAQ

After the backend is running, ingest the sample FAQ from the repository root:

```powershell
Invoke-RestMethod `
  -Uri http://localhost:8000/documents/ingest/file `
  -Method Post `
  -Form @{ file = Get-Item "docs\\sample-faq\\company-faq.txt" }
```

You can also ingest raw text:

```powershell
Invoke-RestMethod `
  -Uri http://localhost:8000/documents/ingest/text `
  -Method Post `
  -ContentType "application/json" `
  -Body (@{ text = "We build LiveKit voice assistants with RAG."; filename = "quick-note.txt" } | ConvertTo-Json)
```

## Demo Questions

- `What services do you offer?`
- `What is the weather in Lahore?`
- `Explain what FastAPI is.`

Expected behavior:

- company or FAQ questions use the RAG backend
- weather questions use the weather tool
- general questions are answered normally without tools

## Useful Commands

From the repository root:

```powershell
npm run lint:web
npm run typecheck:web
npm run build:web
npm run test:agent
npm run test:rag-backend
```

## Notes

- The integrated RAG backend was extracted into `services/rag-backend`, while the original source copy remains in `services/RAG_Chatbot` as a reference.
- The voice worker uses LiveKit Inference for its normal conversational voice pipeline.
- The RAG backend now powers retrieval/context for the voice agent, while `/chat/ask` remains available for direct backend chat testing.
