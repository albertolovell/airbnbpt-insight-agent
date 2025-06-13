import os
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from pathlib import Path

DATA_PATH = Path('data/processed/review_chunks.parquet')
STATE_PATH = Path('data/processed/qdrant_checkpoint.txt')
COLLECTION_NAME = 'airbnb_reviews'
BATCH_SIZE = 256

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
client = QdrantClient(host='localhost', port=6333)

if not client.collection_exists(collection_name=COLLECTION_NAME):
  client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=qdrant_models.VectorParams(
      size=model.get_sentence_embedding_dimension(),
      distance=qdrant_models.Distance.COSINE
    )
  )

df = pd.read_parquet(DATA_PATH)
df = df[df['text'].notnull()].reset_index(drop=True)

# resume logic
start_idx = 0
if STATE_PATH.exists():
  with open(STATE_PATH, 'r') as f:
    start_idx = int(f.read().strip())

# batch loop
for i in tqdm(range(start_idx, len(df), BATCH_SIZE)):
  batch = df.iloc[i:i+BATCH_SIZE]
  texts = batch['text'].tolist()
  listing_ids = batch['listing_id'].tolist()
  embeddings = model.encode(texts, show_progress_bar=False)

  client.upsert(
    collection_name=COLLECTION_NAME,
    points=[
      qdrant_models.PointStruct(
        id=int(i + j),
        vector=embeddings[j],
        payload={'listing_id': listing_ids[j]}
      )
      for j in range(len(texts))
    ]
  )

  with open(STATE_PATH, 'w') as f:
    f.write(str(i + BATCH_SIZE))

print('* embeddings loaded into Qdrant *')