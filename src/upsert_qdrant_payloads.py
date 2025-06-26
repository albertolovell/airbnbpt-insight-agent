from qdrant_client import QdrantClient
import pandas as pd
from qdrant_client.http import models as qdrant_models
from tqdm import tqdm
from pathlib import Path
# 39288 docs upserted

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
  old_ids = [int(i + j) for j in range(len(batch))]

  retrieved = client.retrieve(
    collection_name=COLLECTION_NAME,
    ids = old_ids,
    with_vectors=True,
    with_payload=True
  )

  new_points = []
  for point in retrieved:
    listing_id = point.payload.get('listing_id')
    if listing_id and point.vector:
      new_points.append(qdrant_models.PointStruct(
        id=listing_id,
        vector=point.vector,
        payload=point.payload
      ))

  if new_points:
    client.upsert(
      collection_name=COLLECTION_NAME,
      points=new_points
    )

  # upsert text
  # texts = batch['text'].tolist()
  # listing_ids = batch['listing_id'].tolist()
  # point_ids = [int(i + j) for j in range(len(batch))]

  # for j, point_id in enumerate(point_ids):
  #   client.set_payload(
  #     collection_name=COLLECTION_NAME,
  #     payload={
  #       'text': texts[j], 'listing_id': listing_ids[j]
  #     },
  #     points=[point_id]
  #   )

  with open(STATE_PATH, 'w') as f:
    f.write(str(i + BATCH_SIZE))

print('* listing_id upsert to qdrant complete *')
