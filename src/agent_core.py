from langchain_core.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from qdrant_client import QdrantClient
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
import torch

load_dotenv()
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USER = os.getenv('NEO4J_USER')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
qdrant = QdrantClient(host='localhost', port=6333)
neo4j = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

llama_model_name = 'meta-llama/Llama-2-7b-hf'
tokenizer = AutoTokenizer.from_pretrained(llama_model_name)
model = AutoModelForCausalLM.from_pretrained(
  llama_model_name,
  torch_dtype=torch.float16,
  device_map='auto'
)

llama_pipe = pipeline(
  'text-generation',
  model=model,
  tokenizer=tokenizer,
  max_length=512,
  temperature=0.0,
  do_sample=False
)

llm = HuggingFacePipeline(pipeline=llama_pipe)

def lookup_neo4j(listing_id: str) -> str:
  with neo4j.session() as session:
    res = session.run("""
      MATCH (l:Listing {id: $lid})
      OPTIONAL MATCH (l)-[:HAS_AMENITY]->(a:Amenity)
      OPTIONAL MATCH (l)-[:IN_NEIGHBORHOOD]->(n:Neighborhood)
      OPTIONAL MATCH (l)-[:PRICE_LEVEL]->(p:PriceLevel)
      RETURN collect(DISTINCT a.name) AS amenities,
        collect(DISTINCT n.name) AS neighborhoods,
        collect(DISTINCT p.level) AS price_levels
    """, lid=listing_id).single()
    return f"Amenities: {res['amenities']}, Neighborhoods: {res['neighborhoods']}, Price Levels: {res['price_levels']}"

neo4j_tool = Tool(
  name='ListingMetadataLookup',
  func=lookup_neo4j,
  description='get listing metadata given a listing_id'
)

def search_qdrant(query: str) -> str:
  vec = embed_model.encode([query])[0]
  hits = qdrant.search('airbnb_reviews', query_vector=vec, limit=5)
  return '\n'.join([f"Review: {h.payload['text']}" for h in hits])

qdrant_tool = Tool(
  name='ReviewSearch',
  func=search_qdrant,
  description='semantic search of airbnb reviews'
)

agent = initialize_agent(
  tools=[qdrant_tool, neo4j_tool],
  llm=llm,
  agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
  verbose=True
)

def run_agent(query: str):
  return agent.run(query)