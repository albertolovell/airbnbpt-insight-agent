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

### Clone the repo
```bash
git clone https://github.com/albertolovell/airbnbpt-insight-agent.git
cd airbnbpt-insight-agent
```

### Create a .env file at the root (example.env)

### Run the full stack with Docker Compose
```bash
docker-compose up --build
```
This will:
- build and start the fastAPI backend
- start neo4j with persistent data
- start qdrant
- serve the react frontend

### Future Ideas
- Add login and personalization
- Support multiple languages
- Store past queries for analytics
- Add map view with coordinates from Neo4J
