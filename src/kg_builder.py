import ast
import pandas as pd
from pathlib import Path
from typing import List, Tuple

from langchain_community.llms import HuggingFacePipeline
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'
KG_DIR = BASE_DIR / 'data' / 'kg'

LISTINGS_PATH = PROCESSED_DIR / 'listings.parquet'
REVIEW_CHUNKS_PATH = PROCESSED_DIR / 'review_chunks.parquet'

KG_DIR.mkdir(parents=True, exist_ok=True)

def parse_amenities(amenities_str: str) -> List[str]:
  try:
    cleaned = amenities_str.replace('u"', '"').replace('{', '[').replace('}', ']')
    return ast.literal_eval(cleaned)
  except Exception:
    return []

def extract_metadata_triples(listings_df: pd.DataFrame) -> List[Tuple[str, str, str]]:
  """
  For each listing output triples eg:
    (listing_id, 'has_amenity', amenity)
    (listing_id, 'in_neighborhood', neighborhood)
    (listing_id, 'price_level', 'low'/'medium'/'high')
  """
  triples = []
  for _, row in listings_df.iterrows():
    lid = str(row["id"])

    neighborhood = row.get("neighbourhood_cleansed") or row.get("neighbourhood")
    if isinstance(neighborhood, str) and neighborhood.strip():
      triples.append((lid, 'in_neighborhood', neighborhood.strip()))

    amenities_list = parse_amenities(row.get('amenities', '[]'))
    for am in amenities_list:
      triples.append((lid, 'has_amenity', am))
    try:
      price = float(row.get('price', 0).replace('$', '').replace(',', '')) if isinstance(row.get('price'), str) else float(row.get('price', 0))
      if price < 50:
        level = 'low'
      elif price < 150:
        level = 'medium'
      else:
        level = 'high'
      triples.append((lid, 'price_level', level))
    except Exception:
      pass
  return triples


class ReviewTriple(BaseModel):
  subject: str = Field(..., description='the subject entity')
  predicate: str = Field(..., description='the relationship')
  object: str = Field(..., description='the object entity or value')


PROMPT_TEMPLATE = PromptTemplate(
  template=(
    "You are given a chunk of an Airbnb review. Extract all subject-predicate-object triples. "
    "Format as a JSON array of objects [{{\"subject\":..., \"predicate\":..., \"object\":...}}, ...].\n\n"
    "Review Chunk:\n\"\"\"\n{chunk}\n\"\"\"\n\n"
    "Ensure each triple is clear and relevant to the review."
  ),
  input_variables=['chunk']
)

def load_llama_pipeline(model_name: str = 'meta-llama/Llama-2-7b-hf') -> HuggingFacePipeline:
  from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
  import torch

  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map='auto'
  )

  hf_pipe = pipeline(
    'text-generation',
    model=model,
    tokenizer=tokenizer,
    max_length=512,
    temperature=0.0,
    do_sample=False
  )
  return HuggingFacePipeline(pipeline=hf_pipe)


def extract_review_triples(
    review_chunks_df: pd.DataFrame,
    llama_pipe: HuggingFacePipeline,
    parser: PydanticOutputParser,
    prompt_template: PromptTemplate) -> List[Tuple[str, str, str]]:
  """
  Iterate over each chunk, run the LLaMA-based extraction chain, and collect triples
  """
  triples: List[Tuple[str, str, str]] = []

  for idx, row in review_chunks_df.iterrows():
    lid   = str(row['listing_id'])
    chunk = row['text']

    if not isinstance(chunk, str) or not chunk.strip():
      continue
    prompt = prompt_template.format(chunk=chunk)

    try:
      raw_out = llama_pipe(prompt)
      generated_text = raw_out[0]['generated_text'].strip()
      parsed = parser.parse(generated_text)
      parsed_list = parsed if isinstance(parsed, list) else [parsed]

      for triple_obj in parsed_list:
        subject = f"{lid}:{triple_obj.subject}"
        triples.append((subject, triple_obj.predicate, triple_obj.object))
    except Exception:
      continue
  return triples


if __name__ == "__main__":
  print('loading processed listings...')
  listings_df = pd.read_parquet(LISTINGS_PATH)
  print('loading processed review chunks...')
  review_chunks_df = pd.read_parquet(REVIEW_CHUNKS_PATH)

  print('extracting metadata triples (amenities, neighborhoods, price levels)...')
  meta_triples = extract_metadata_triples(listings_df)
  print(f"found {len(meta_triples)} metadata triples")

  meta_df = pd.DataFrame(meta_triples, columns=['subject', 'predicate', 'object'])
  meta_out = KG_DIR / 'metadata_triples.csv'
  meta_df.to_csv(meta_out, index=False)
  print(f"saved metadata triples to {meta_out}")

  print('loading llama-2-7b pipeline (may take a minute)...')
  llama_pipe = load_llama_pipeline()
  parser = PydanticOutputParser(pydantic_object=ReviewTriple)

  print('extracting review triples with llama (this will be slow)...')
  review_triples = extract_review_triples(
    review_chunks_df,
    llama_pipe=llama_pipe,
    parser=parser,
    prompt_template=PROMPT_TEMPLATE
  )
  print(f"found {len(review_triples)} review triples")

  review_df = pd.DataFrame(review_triples, columns=['subject', 'predicate', 'object'])
  review_out = KG_DIR / 'review_triples.csv'
  review_df.to_csv(review_out, index=False)
  print(f"saved review triples to {review_out}")

  print('knowledge graph construction complete')

# deprecated