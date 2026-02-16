from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from qdrant_client import QdrantClient
from neo4j import GraphDatabase
# from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from datetime import datetime
import os, time

load_dotenv()
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USER = os.getenv('NEO4J_USER')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
QDRANT_HOST = os.getenv('QDRANT_HOST', 'localhost')
QDRANT_PORT = int(os.getenv('QDRANT_PORT') or '6333')

qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=60.0)
neo4j = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

vector_store = QdrantVectorStore(
  client=qdrant_client,
  collection_name='airbnb_reviews',
  embedding=embeddings,
  content_payload_key='text'
)

# llama_model_name = 'meta-llama/Llama-2-7b-hf' #llama7b 1/2
model_name = 'google/flan-t5-base'
# model_name = 'distilgpt2'

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
llm_pipe = pipeline(
  'text2text-generation',
  model=model,
  tokenizer=tokenizer,
  max_new_tokens=512
)
llm = HuggingFacePipeline(pipeline=llm_pipe)

prompt = PromptTemplate.from_template(
  "You are an assistant helping users query Airbnb data.\n"
  "Use the context to answer the user's question concisely and clearly.\n\n"
  "-If the user asks for a specific number of items (e.g., top 3 amenities), return **only that number**.\n"
  "-If the user requests a **listing ID**, include the corresponding ID(s) in your answer.\n"
  "-If the user asks for a specific location like Lisbon, prioritize matching that.\n"
  "-If no relevant data is found, reply: 'No relevant data found.'\n\n"
  "Context:\n{context}\n\n"
  "User Query:\n{query}\n\n"
  "Answer:"
)

chain = LLMChain(llm=llm, prompt=prompt)

def query_neo4j(listing_id: str):
  with neo4j.session() as session:
    res = session.run("""
      MATCH (l:Listing {id: $lid})
      OPTIONAL MATCH (l)-[:HAS_AMENITY]->(a:Amenity)
      OPTIONAL MATCH (l)-[:IN_NEIGHBORHOOD]->(n:Neighborhood)
      OPTIONAL MATCH (l)-[:PRICE_LEVEL]->(p:PriceLevel)
      RETURN l.id AS listing_id,
        collect(DISTINCT a.name) AS amenities,
        collect(DISTINCT n.name) AS neighborhoods,
        collect(DISTINCT p.level) AS price_levels
    """, lid=listing_id).single()
    if res:
      return {
        'listing_id': res.get('listing_id', listing_id),
        'amenities': res.get('amenities', []),
        'neighborhoods': res.get('neighborhoods', []),
        'price_levels': res.get('price_levels', [])
      }
    else:
      return {
        'listing_id': listing_id,
        'amenities': [],
        'neighborhoods': [],
        'price_levels': []
      }

def extract_listing_id(doc):
  if not doc.metadata:
    return None
  if 'listing_id' in doc.metadata:
    return doc.metadata.get('listing_id')
  nested = doc.metadata.get('metadata')
  if isinstance(nested, dict):
    return nested.get('listing_id')
  return None

def build_context(docs, metas, listing_ids):
  context = ''
  for doc, meta, lid in zip(docs, metas, listing_ids):
    context += f"Review: {doc.page_content}\n"
    context += f"Listing ID: {lid}\n"
    amenities = meta.get('amenities', [])
    neighborhoods = meta.get('neighborhoods', [])
    price_levels = meta.get('price_levels', [])
    context += f"Amenities: {', '.join(amenities[:10])}\n"
    context += f"Neighborhoods: {neighborhoods[0] if neighborhoods else 'N/A'}\n"
    context += f"Price Level: {price_levels}\n\n"
  return context.strip()

def run_agent(query: str):
  timings = {}

  # qdrant
  t0 = time.time()
  docs = vector_store.similarity_search(query, k=1)
  timings['qdrant_search'] = time.time() - t0
  if not docs:
    return{'answer': "Sorry, no relevant data found."}

  listing_ids = []
  for doc in docs:
    listing_id = extract_listing_id(doc)
    if listing_id is None:
      point_id = None
      if doc.metadata:
        point_id = doc.metadata.get('_id') or doc.metadata.get('id')
      if point_id is not None:
        payload = qdrant_client.retrieve(
          collection_name='airbnb_reviews',
          ids=[point_id],
          with_payload=True,
          with_vectors=False
        )[0].payload or {}
        listing_id = payload.get('listing_id') or (payload.get('metadata') or {}).get('listing_id')
    listing_ids.append(listing_id)

  # neo4j
  t0 = time.time()
  metas = [query_neo4j(str(lid)) if lid else {} for lid in listing_ids]
  timings['neo4j_lookup'] = time.time() - t0

  # build context
  context = build_context(docs, metas, listing_ids)

  # llm
  t0 = time.time()
  result = chain.invoke({'query': query, 'context': context})
  timings['llm_inference'] = time.time() - t0

  print(f"timings: {timings}")
  os.makedirs('data', exist_ok=True)
  with open('data/timings.log', 'a') as f:
    f.write(f"{query}\ndate: [{datetime.now().isoformat()}] \nstep_times: {timings}\n")

  return {'answer': result.get('text', '').strip()}
