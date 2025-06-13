import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance

df = pd.read_parquet('data/processed/review_chunks.parquet')
texts = df['text'].dropna().tolist()

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(texts, show_progress_bar=True)

client = QdrantClient(host='localhost', port=6333)

collection_name = 'airbnb_reviews'
client.recreate_collection(
  collection_name=collection_name,
  vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)

points = [PointStruct(id=i, vector=vec.tolist(), payload={'text': text})
          for i, (vec, text) in enumerate(zip(embeddings, texts))]

client.upsert(collection_name=collection_name, points=points)