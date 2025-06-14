from fastapi import FastAPI
from pydantic import BaseModel
from src.agent_core import run_agent

app = FastAPI()

class QueryRequest(BaseModel):
  query: str

@app.post('/ask')
async def ask_agent(req: QueryRequest):
  result = run_agent(req.query)
  return {'answer': f"recieved: {req.query}"}