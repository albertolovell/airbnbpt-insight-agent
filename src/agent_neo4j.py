import os
import re
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USER = os.getenv('NEO4J_USER')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

class KGAgentNeo4j:
  def __init__(self, uri, user, pwd):
    self.driver = GraphDatabase.driver(uri, auth=(user, pwd))

  def close(self):
    self.driver.close()

  def answer_with_kg(self, query: str) -> str:
    session = self.driver.session()

    if "pool" in query.lower():
      neighborhood = None
      match = re.search(r"in ([A-Za-z]+)", query, re.IGNORECASE)
      if match:
        neighborhood = match.group(1).capitalize()

      if neighborhood:
        cypher = """
        MATCH (l:Listing)-[:HAS_AMENITY]->(:Amenity {name: $amenity})
              -[:HAS_AMENITY]-()  // ensures the amenity relationship
        MATCH (l)-[:IN_NEIGHBORHOOD]->(n:Neighborhood {name: $neighborhood})
        RETURN l.id AS listing_id LIMIT 10
        """
        result = session.run(cypher, amenity='pool', neighborhood=neighborhood)
      else:
        cypher = """
        MATCH (l:Listing)-[:HAS_AMENITY]->(:Amenity {name: $amenity})
        RETURN l.id AS listing_id LIMIT 10
        """
        result = session.run(cypher, amenity='pool')

      listings = [record['listing_id'] for record in result]
      session.close()
      if listings:
        return (f"listings with pool{' in '+neighborhood if neighborhood else ''}: {listings}")
      else:
        return (f"no listings found with pool{' in '+neighborhood if neighborhood else ''}")
      # add other structured patters for price_level, wifi etc

    session.close()
    return "I'm not sure how to answer that with structured data"

  def is_structured_query(self, q: str) -> bool:
    keywords = ['amenity', 'pool', 'wifi', 'neighborhood', 'price', 'low', 'high']
    return any(k in q.lower() for k in keywords)

  def agent_answer(self, query: str, rag_func):
    if self.is_structured_query(query):
      return self.answer_with_kg(query)
    else:
      return rag_func(query)
