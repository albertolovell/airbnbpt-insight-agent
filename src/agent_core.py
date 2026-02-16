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
import re

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
  do_sample=False,
  num_beams=4,
  max_new_tokens=120,
  repetition_penalty=1.1,
  no_repeat_ngram_size=3
)
llm = HuggingFacePipeline(pipeline=llm_pipe)

prompt = PromptTemplate.from_template(
  "You are an Airbnb Portugal data assistant.\n"
  "Answer using only the context below.\n"
  "Do not repeat or quote instructions.\n"
  "If context is insufficient, answer exactly: No relevant data found.\n"
  "Keep the answer concise and factual.\n\n"
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
  context_lines = []
  for doc, meta, lid in zip(docs, metas, listing_ids):
    snippet = (doc.page_content or '')[:320]
    context_lines.append(f"Review: {snippet}")
    context_lines.append(f"Listing ID: {lid}")
    amenities = meta.get('amenities', [])
    neighborhoods = meta.get('neighborhoods', [])
    price_levels = meta.get('price_levels', [])
    context_lines.append(f"Amenities: {', '.join(amenities[:8])}")
    context_lines.append(f"Neighborhoods: {neighborhoods[0] if neighborhoods else 'N/A'}")
    context_lines.append(f"Price Level: {price_levels}")
    context_lines.append('')
  return '\n'.join(context_lines).strip()

def extract_top_n(query: str):
  match = re.search(r'\btop\s+(\d+)\b', query.lower())
  if match:
    return int(match.group(1))
  return None

def query_structured_listings(query: str):
  q = query.lower()
  amenities = ['pool', 'wifi', 'kitchen', 'washer', 'balcony', 'parking', 'pet-friendly']
  amenity = next((a for a in amenities if a in q), None)
  if not amenity:
    return None

  neighborhood = None
  for loc in ['lisbon', 'porto', 'aveiro']:
    if loc in q:
      neighborhood = loc.capitalize()
      break

  top_n = extract_top_n(query) or 5
  with neo4j.session() as session:
    if neighborhood:
      rows = session.run(
        """
        MATCH (l:Listing)-[:HAS_AMENITY]->(a:Amenity)
        MATCH (l)-[:IN_NEIGHBORHOOD]->(n:Neighborhood {name: $neighborhood})
        WHERE toLower(a.name) CONTAINS $amenity
        RETURN l.id AS listing_id
        LIMIT $limit
        """,
        amenity=amenity,
        neighborhood=neighborhood,
        limit=top_n
      )
    else:
      rows = session.run(
        """
        MATCH (l:Listing)-[:HAS_AMENITY]->(a:Amenity)
        WHERE toLower(a.name) CONTAINS $amenity
        RETURN l.id AS listing_id
        LIMIT $limit
        """,
        amenity=amenity,
        limit=top_n
      )
    listing_ids = [r['listing_id'] for r in rows]

  if not listing_ids:
    return {'answer': 'No relevant data found.'}
  loc_part = f" in {neighborhood}" if neighborhood else ""
  return {'answer': f"Found {len(listing_ids)} listing(s) with {amenity}{loc_part}: {', '.join(str(x) for x in listing_ids)}"}

def is_low_signal_answer(answer: str):
  low = answer.lower().strip()
  bad_patterns = [
    'if the user asks',
    'if the user requests',
    'use the context',
    'answer:',
    'user query:'
  ]
  return any(p in low for p in bad_patterns)

def retrieve_docs(query: str, k: int = 8, min_relevance: float = 0.2):
  try:
    scored = vector_store.similarity_search_with_relevance_scores(query, k=k)
    filtered = [doc for doc, score in scored if score >= min_relevance]
    if filtered:
      return filtered
    return [doc for doc, _ in scored[:3]]
  except Exception:
    return vector_store.similarity_search(query, k=5)

def run_agent(query: str):
  timings = {}

  # structured fallback for amenity/location-type queries
  structured = query_structured_listings(query)
  if structured is not None:
    return structured

  # qdrant
  t0 = time.time()
  docs = retrieve_docs(query, k=8, min_relevance=0.2)
  timings['qdrant_search'] = time.time() - t0
  if not docs:
    return {'answer': "No relevant data found."}

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

  deduped = []
  seen = set()
  for doc, lid in zip(docs, listing_ids):
    key = str(lid) if lid else f"doc:{hash(doc.page_content)}"
    if key in seen:
      continue
    seen.add(key)
    deduped.append((doc, lid))
    if len(deduped) >= 5:
      break
  docs = [d for d, _ in deduped]
  listing_ids = [lid for _, lid in deduped]

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

  answer = result.get('text', '').strip()
  if not answer or is_low_signal_answer(answer):
    return {'answer': 'No relevant data found.'}
  return {'answer': answer}
