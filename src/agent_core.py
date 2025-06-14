from langchain_community.llms import HuggingFacePipeline
from langchain.vectorstores import Qdrant as LCQdrant
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
import torch

load_dotenv()
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USER = os.getenv('NEO4J_USER')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

qdrant_client = QdrantClient(host='localhost', port=6333)
neo4j = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

vector_store = LCQdrant(
  client=qdrant_client,
  collection_name='airbnb_reviews',
  embedding_function=lambda texts: embed_model.encode(texts).tolist()
)

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

prompt = PromptTemplate.from_template("""
  You are an Airbnb assistant. Given the context, answer the user's query helpfully.

  User query: {query}

  Context:
  {context}

  Answer:
""")

chain = LLMChain(llm=llm, prompt=prompt)

def query_neo4j(listing_id: str):
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
    return res

def build_context(docs, metas):
  context = ''
  for doc, meta in zip(docs, metas):
    context += f"Review: {doc.page_content}\n"
    context += f"Amenities: {meta['amenities']}\n"
    context += f"Neighborhoods: {meta['neighborhoods']}\n"
    context += f"Price Level: {meta['price_levels']}\n\n"
  return context

def run_agent(query: str):
  docs = vector_store.similarity_search(query, k=3)
  metas = [query_neo4j(doc.metadata.get('listing_id')) for doc in docs]
  context = build_context(docs, metas)
  return chain.run(query=query, context=context)