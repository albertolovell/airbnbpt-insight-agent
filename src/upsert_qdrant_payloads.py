import os
from qdrant_client import QdrantClient
import pandas as pd
from qdrant_client.http import models as qdrant_models
from tqdm import tqdm
from pathlib import Path
# 39288 docs upserted (original run count)

DATA_PATH = Path('data/processed/review_chunks.parquet')
STATE_PATH = Path('data/processed/qdrant_payload_checkpoint.txt')
COLLECTION_NAME = 'airbnb_reviews'
BATCH_SIZE = 256
QDRANT_HOST = os.getenv('QDRANT_HOST', 'localhost')
QDRANT_PORT = int(os.getenv('QDRANT_PORT', '6333'))

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
df = pd.read_parquet(DATA_PATH)
df = df[df['text'].notnull()].reset_index(drop=True)

start_idx = 0
if STATE_PATH.exists():
  with open(STATE_PATH, 'r') as f:
    start_idx = int(f.read().strip())

for i in tqdm(range(start_idx, len(df), BATCH_SIZE)):
  batch = df.iloc[i:i+BATCH_SIZE]
  old_ids = [int(i + j) for j in range(len(batch))]
  listing_ids = batch['listing_id'].tolist()
  texts = batch['text'].tolist()
  id_to_payload = {
    int(i + j): {'listing_id': listing_ids[j], 'text': texts[j]}
    for j in range(len(batch))
  }

  retrieved = client.retrieve(
    collection_name=COLLECTION_NAME,
    ids = old_ids,
    with_vectors=True,
    with_payload=True
  )

  new_points = []
  for point in retrieved:
    if point.vector is None:
      continue
    merged_payload = dict(point.payload or {})
    merged_payload.update(id_to_payload.get(int(point.id), {}))
    new_points.append(qdrant_models.PointStruct(
      id=int(point.id),
      vector=point.vector,
      payload=merged_payload
    ))

  if new_points:
    client.upsert(
      collection_name=COLLECTION_NAME,
      points=new_points
    )

  with open(STATE_PATH, 'w') as f:
    f.write(str(i + BATCH_SIZE))

print('* qdrant payload backfill complete *')
