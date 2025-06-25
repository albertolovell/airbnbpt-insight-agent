from qdrant_client import QdrantClient
import pandas as pd
# from qdrant_client.http import models as qdrant_models
from tqdm import tqdm
from pathlib import Path

DATA_PATH = Path('data/processed/review_chunks.parquet')
STATE_PATH = Path('data/processed/qdrant_payload_checkpoint.txt')
COLLECTION_NAME = 'airbnb_reviews'
BATCH_SIZE = 256

client = QdrantClient(host='localhost', port=6333)
df = pd.read_parquet(DATA_PATH)
df = df[df['text'].notnull()].reset_index(drop=True)

start_idx = 0
if STATE_PATH.exists():
  with open(STATE_PATH, 'r') as f:
    start_idx = int(f.read().strip())

for i in tqdm(range(start_idx, len(df), BATCH_SIZE)):
  batch = df.iloc[i:i+BATCH_SIZE]
  texts = batch['text'].tolist()
  listing_ids = batch['listing_id'].tolist()
  point_ids = [int(i + j) for j in range(len(batch))]

  payloads = [
    {'text': texts[j], 'listing_id': listing_ids[j]}
    for j in range(len(batch))
  ]

  client.set_payload(
    collection_name=COLLECTION_NAME,
    payload=payloads,
    points=point_ids
  )

  with open(STATE_PATH, 'w') as f:
    f.write(str(i + BATCH_SIZE))

print('* payload-only upsert to qdrant complete *')
