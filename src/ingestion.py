import os
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter

RAW_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')

def load_csv(name: str) -> pd.DataFrame:
  path = os.path.join(RAW_DIR, name)
  return pd.read_csv(path)

def chunk_reviews(df: pd.DataFrame, chunk_size=1000, overlap=200):
  splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
  docs = []
  for _, row in df.iterrows():
    text = row['review_text']
    for i, chunk in enumerate(splitter.split_text(text)):
      docs.append({
        'listing_id': row['listing_id'],
        'chunk_id': f"{row['review_id']}_{i}",
        'text': chunk
      })
  return pd.DataFrame(docs)

if __name__=='__main__':
  os.makedirs(PROCESSED_DIR, exist_ok=True)
  reviews = load_csv('reviews.csv')
  review_chunks = chunk_reviews(reviews)
  review_chunks.to_parquet(os.path.join(PROCESSED_DIR, 'review_chunks.parquet'), index=False)
  print('Review chunks saved: ', review_chunks.shape)