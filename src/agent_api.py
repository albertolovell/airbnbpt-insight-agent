from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class QueryRequest(BaseModel):
  query: str

_agent_runner = None

def get_agent_runner():
  global _agent_runner
  if _agent_runner is None:
    print('loading agent for the first request.....')
    from src.agent_core import run_agent
    _agent_runner = run_agent
  return _agent_runner

@app.post('/ask')
async def ask_agent(req: QueryRequest):
  agent_runner = get_agent_runner()
  result = agent_runner(req.query)
  return agent_runner(req.query)
  # return {'answer': f"recieved: {req.query}"}