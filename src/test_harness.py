from agent_core import run_agent, vector_store, query_neo4j
from qdrant_client import QdrantClient

def main():

  test_queries = [
    'show me one airbnb listing',
    'which listings have wifi and are in Lisbon',
    'what is the most recent listing',
    'find a quiet place with good reviews'
  ]

# test qdrant output for listing_id
  client = QdrantClient(host='localhost', port=6333)
  points = client.retrieve(
    collection_name='airbnb_reviews',
    ids=[5844519],
    with_payload=True)
  for point in points:
    print(point.payload)

# test qdrant output
  # print('running test harness for Qdrant + neo4j...')

  # for query in test_queries:
  #   print('\n==============================')
  #   print(f"Query: {query}")

  #   docs = vector_store.similarity_search(query, k=1)
  #   if not docs:
  #     print('No matching docuents found in Qdrant')
  #     continue
  #   else:
  #     for doc in docs:
  #       print(f"Matched Doc Content: {doc.page_content}")
  #       print(f"Metadata: {doc.metadata}")

  #       listing_id = doc.metadata.get('listing_id')
  #       if listing_id:
  #         neo4j_meta = query_neo4j(str(listing_id))
  #         print(f"neo4j data: {neo4j_meta}")
  #       else:
  #         print('no listing_id found in metadata')


# test llm output
  # print('running test harness for llm output...')
    # result = run_agent(query)
    # print(f"Result:\n{result['answer']}")

if __name__ == "__main__":
  main()
