# AirbnbPT Insight Agent

A fullstack Airbnb listing explorer for Portugal. This agentic workflow leverages vector similarity search (Qdrant), knowledge graph (Neo4j), and LLMs (Flan-T5 via Hugging Face). Built with FastAPI on the backend and React + Tailwind on the frontend

---

## Features
- Semantic search of Airbnb reviews using vector similarity
- Context reasoning with Neo4j for amenities, locations, and price levels
- Natural language query support via local Flan-T5 LLM
- FastAPI backend with REST endpoint for queries
- React frontend with TailwindCSS for styling

---

## Teach Stack
- **Backend**: Python, FastAPI, LangChain, Neo4j, Qdrant, HuggingFace Transformers
- **Frontend**: React, TailwindCSS, Vite
- **VectorDB**: Qdrant
- **GraphDB**: Neo4j
- **LLM**: Flan-T5 (via Hugging Face)

---

## Run with Docker Compose

### 1. Clone the repo
```bash
git clone https://github.com/albertolovell/airbnbpt-insight-agent.git
cd airbnbpt-insight-agent
```

### 2. Create a .env file at the root (example.env)
For Docker Compose service-to-service networking, use:
```
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=your_username
NEO4J_PASSWORD=your_secure_password
QDRANT_HOST=qdrant
QDRANT_PORT=6333
```

### 3. Run the full stack with Docker Compose
```bash
docker-compose up --build
```
This will:
- build and start the fastAPI backend
- start neo4j with persistent data
- start qdrant
- serve the react frontend

### 4. Open the UI (Docker)
Visit `http://localhost:5173` for the Airbnb Portugal clone with the chat window.
The frontend container proxies `/ask` to the backend service.

---

## Run locally (reproducible)
This runs Neo4j + Qdrant in Docker and the backend on your host.

### 1. Create a local .env
```bash
cp example.env .env
```
Set these for local host access:
```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=your_username
NEO4J_PASSWORD=your_secure_password
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

### 2. Start services (Neo4j + Qdrant)
```bash
docker-compose up -d neo4j qdrant
```

### 3. Install backend deps and run API
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn src.agent_api:app --host 0.0.0.0 --port 8000
```

### 4. Optional: run frontend locally
```bash
cd app
npm install
npm run dev
```

### 5. Open the UI
Visit `http://localhost:5173` for the Airbnb Portugal clone with the chat window.
The Vite dev server proxies `/ask` to `http://localhost:8000`, so the frontend can call the API without hardcoding the backend URL.
Use the `Dashboard` toggle in the header to open the analytics view with filters for neighborhood, room type, property type, superhost, price level, and minimum accommodates.
Use the `Update listings` button in the header to:
- fetch latest Portugal city datasets from Inside Airbnb
- ingest and rebuild processed files
- refresh Neo4j and Qdrant data

Update status modal states:
- `database update pending` while processing is running
- `already up to date` if no new files are available
- `database update completed` when refresh finishes

### Future Ideas
- Deeper feature engineering and extraction
- Add login and personalization
- Support multiple languages
- Store past queries for analytics
- Add map view with coordinates from Neo4J

**Contact me if you would like to implement a similar workflow on your dataset**
