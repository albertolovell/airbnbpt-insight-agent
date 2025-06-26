from langchain_community.llms import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_qdrant import Qdrant as LCQdrant
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
# from langchain_community.llms import huggingface_pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from qdrant_client import QdrantClient
# from qdrant_client.http.models import Distance, VectorParams
from neo4j import GraphDatabase
# from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
import torch
import time

load_dotenv()
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USER = os.getenv('NEO4J_USER')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

qdrant_client = QdrantClient(host='localhost', port=6333, timeout=60.0)
neo4j = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

vector_store = QdrantVectorStore(
  client=qdrant_client,
  collection_name='airbnb_reviews',
  embedding=embeddings,
  content_payload_key='text',
  metadata_payload_key=['listing_id']
)

# llama_model_name = 'meta-llama/Llama-2-7b-hf' #llama7b 1/2
model_name = 'distilgpt2'

tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(
#   llama_model_name,
#   torch_dtype=torch.float16,
#   device_map='auto'
# ) # llama7b 2/2
model = AutoModelForCausalLM.from_pretrained(model_name)

llm_pipe = pipeline(
  'text-generation',
  model=model,
  tokenizer=tokenizer,
  max_new_tokens=64,
  do_sample=False
)

llm = HuggingFacePipeline(pipeline=llm_pipe)

prompt = PromptTemplate.from_template("""
  You are an Airbnb assistant. Given the user's query and context provided, generate a helpful, concise answer. If context is missing, say: 'Sorry, no data found."

  User query:
  {query}

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
    if res:
      return {
        'amenities': res['amenities'],
        'neighborhoods': res['neighborhoods'],
        'price_levels': res['price_levels']
      }
    else:
      return {
      'amenities': [],
      'neighborhoods': [],
      'price_levels': []
    }

def build_context(docs, metas):
  context = ''
  for doc, meta in zip(docs, metas):
    context += f"Review: {doc.page_content}\n"
    context += f"Amenities: {meta['amenities']}\n"
    context += f"Neighborhoods: {meta['neighborhoods']}\n"
    context += f"Price Level: {meta['price_levels']}\n\n"
  return context.strip()

def run_agent(query: str):
  timings = {}

  # qdrant
  t0 = time.time()
  docs = vector_store.similarity_search(query, k=1)
  timings['qdrant_search'] = time.time() - t0
  if not docs:
    return{'answer': "Sorry, I couldn't find any relevant data for that query."}

  # neo4j
  t0 = time.time()
  metas = [query_neo4j(doc.metadata.get('listing_id')) for doc in docs]
  timings['neo4j_lookup'] = time.time() - t0

  # build context
  context = build_context(docs, metas)

  # llm
  t0 = time.time()
  result = chain.run(query=query, context=context)
  timings['llm_inference'] = time.time() - t0

  print(f"timings: {timings}")
  with open('data/timings.log', 'a') as f:
    f.write(f"{query}\n step_times: {timings}\n")

  return {'answer': result}