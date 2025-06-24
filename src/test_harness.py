from agent_core import run_agent, vector_store

def main():
  print('running test harness for Qdrant similarity search...')

  test_queries = [
    'show me one airbnb listing',
    'which listings have wifi and are in Lisbon',
    'what is the most recent listing',
    'find a quiet place with good reviews'
  ]

  for query in test_queries:
    print('\n==============================')
    print(f"Query: {query}")

    docs = vector_store.similarity_search(query, k=1)
    if not docs:
      print('No matching docuents found in Qdrant')
    else:
      for doc in docs:
        print(f"Matched Doc Content: {doc.page_content}")
        print(f"Metadata: {doc.metadata}")

    # result = run_agent(query)
    # print(f"Result:\n{result['answer']}")

if __name__ == "__main__":
  main()
