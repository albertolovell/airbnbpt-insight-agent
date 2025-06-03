import os
import glob
import pandas as pd
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter

# create listings metadata csv for neo4j

BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / 'data' / 'raw'
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'
TEMP_DIR = BASE_DIR / 'data' / 'temp'

LISTING_PATTERNS = [
  'lisbon_listings_*.csv',
  'porto_listings_*.csv'
]
REVIEW_PATTERNS = [
  'lisbon_reviews_*.csv',
  'porto_reviews_*.csv'
]

CHUNK_SIZE = 1000
OVERLAP = 200

def find_all_csvs(patterns_list):
  files = []
  for pat in patterns_list:
    files.extend(glob.glob(str(RAW_DIR / pat)))
  return sorted(files)

def concat_csvs(file_list, usecols=None, dtype=None):
  dfs = []
  for fp in file_list:
    print(f"reading {os.path.basename(fp)}")
    df = pd.read_csv(fp, usecols=usecols, dtype=dtype)
    dfs.append(df)
  combined = pd.concat(dfs, ignore_index=True)
  return combined

def chunk_reviews(df_reviews: pd.DataFrame):
  splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=OVERLAP)
  rows = []
  for idx, row in df_reviews.iterrows():
    listing_id = row['listing_id']
    review_id = row['id']
    text = row['comments']

    for i, chunk in enumerate(splitter.split_text(text)):
      rows.append({
        'listing_id': listing_id,
        'chunk_id': f"{review_id}_{i}",
        'text': chunk
      })
  return pd.DataFrame(rows)

if __name__=='__main__':
  PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
  TEMP_DIR.mkdir(parents=True, exist_ok=True)

  # concat listing csvs
  listing_files = find_all_csvs(LISTING_PATTERNS)
  if not listing_files:
    raise FileNotFoundError('no listing csvs found in data/raw')
  print(f"found {len(listing_files)} listing files. concatenating...")

  listings_df = concat_csvs(listing_files)
  print(f"total listings rows: {listings_df.shape}")

  listings_parquet = PROCESSED_DIR / 'listings.parquet'
  listings_df.to_parquet(str(listings_parquet), index=False)
  print(f"saved all listings to {listings_parquet}")

  # concat review csvs
  review_files = find_all_csvs(REVIEW_PATTERNS)
  if not review_files:
    raise FileNotFoundError('no review csvs found in data/raw')
  print(f"found {len(review_files)} review files. concatenating...")

  reviews_df = concat_csvs(review_files, usecols=['listing_id', 'id', 'comments'])
  reviews_df = reviews_df.dropna(subset=['comments'])
  print(f"total review rows after dropna: {reviews_df.shape}")

  print('chunking reviews (this can take a minute) ...')
  review_chunks_df = chunk_reviews(reviews_df)
  print(f"total review chunks: {review_chunks_df.shape}")

  review_chunks_parquet = PROCESSED_DIR / 'review_chunks.parquet'
  review_chunks_df.to_parquet(str(review_chunks_parquet), index=False)
  print(f"saved review chunks to {review_chunks_parquet}")
  print('* ingestion and chunking complete *')