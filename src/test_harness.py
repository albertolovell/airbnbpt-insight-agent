from agent_core import run_agent, vector_store, query_neo4j
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

def main():
  # one-time cleanup to remove old point ids during embedding
  # client = QdrantClient(host='localhost', port=6333)
  # old_ids = list(range(0, 39288))
  # client.delete(
  #   collection_name='airbnb_reviews',
  #   points_selector=qdrant_models.PointIdsList(points=old_ids)
  # )
  # print('* old point ids deleted *')

  test_queries = [
    # 'show me one airbnb listing',
    # 'which listings have wifi and are in Lisbon',
    # 'find a pet-friendly listing with self check-in',
    # 'which neighborhood has the highest prices'
    # 'list all price levels for listings with balconies'
    # 'is there a listing that mentions ants?'
    # 'what is the most luxurious listing in terms of amenities',
    # 'find a listing with a washer and kitchen'
    # 'find one listing that mentions noisy and give me the listing id'
    # 'what are the available price levels'
    # 'tell me top 3 amenities for listings where price level is high'
    'show me one listing in Aveiro'
  ]

# test qdrant output for listing_id
#   client = QdrantClient(host='localhost', port=6333)
#   points = client.retrieve(
#     collection_name='airbnb_reviews',
#     ids=[604554],
#     with_payload=True,
#     with_vectors=False)
#   for point in points:
#     print(point.payload)

#   points = client.scroll(
#     collection_name='airbnb_reviews',
#     limit=5,
#     with_payload=True
#     )
#   for pt in points[0]:
#     print(pt.id, pt.payload)

# # query neo4j
#   result = query_neo4j('604554')
#   print('neo4j result' , result)

# # test qdrant output
#   print('running test harness for Qdrant output...')
#   client = QdrantClient(host='localhost', port=6333)

#   for query in test_queries:
#     print('\n==============================')
#     print(f"Query: {query}")

#     docs = vector_store.similarity_search(query, k=1)
#     if not docs:
#       print('No matching docuents found in Qdrant')
#       continue

#     for doc in docs:
#       print(f"Matched Doc Content: {doc.page_content}")
#       print(f"Metadata: {doc.metadata}")

#       #test neo4j output
#       print('running test harness for neo4j output')

#       point_id = doc.metadata.get('_id')
#       if point_id:
#         full_payload = client.retrieve(
#           collection_name='airbnb_reviews',
#           ids=[point_id],
#           with_payload=True,
#           with_vectors=False
#         )[0].payload

#         listing_id = full_payload.get('listing_id')
#         print(f"listing_id field: {listing_id if listing_id else 'N/A'}")

#         if listing_id:
#           neo4j_meta = query_neo4j(str(listing_id))
#           print(f"neo4j data: {neo4j_meta}")
#         else:
#           print('no listing_id found in metadata')
#       else:
#         print('no _id in metadata')


# test llm output
  print('running test harness for llm output...')
  for query in test_queries:
    result = run_agent(query)
    print(f"\n==================================")
    print(f"Query: {query}")
    print(f"Result:\n{result['answer']}")

if __name__ == "__main__":
  main()
