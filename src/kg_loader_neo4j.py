import os
import pandas as pd
from neo4j import GraphDatabase
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent
KG_PATH = BASE_DIR / 'data' / 'kg' / 'metadata_triples.csv'

NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USER = os.getenv('NEO4J_USER')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')


def create_neo4j_driver(uri: str, user: str, pwd: str):
  return GraphDatabase.driver(uri, auth=(user, pwd))


def clear_neo4j(driver):
  with driver.session() as session:
    session.run('MATCH (n) DETACH DELETE n')


def ingest_metadata_triples(drive, csv_path: Path):
  """
  Reads metadata_triples.csv (subject,predicate,object) and creates:
    (l:Listing {id: <subject>})
    (a:Amenity {name: <object>})  for predicate='has_amenity'
    (n:Neighborhood {name: <object>}) for predicate='in_neighborhood'
    (p:PriceLevel {level: <object>}) for predicate='price_level'
    Relationships:
        (l:Listing)-[:HAS_AMENITY]->(a:Amenity)
        (l:Listing)-[:IN_NEIGHBORHOOD]->(n:Neighborhood)
        (l:Listing)-[:PRICE_LEVEL]->(p:PriceLevel)
    """
  df = pd.read_csv(csv_path)
  with driver.session() as session:
    session.run('CREATE CONSTRAINT IF NOT EXISTS ON (l:Listing) ASSERT l.id IS UNIQUE')
    session.run('CREATE CONSTRAINT IF NOT EXISTS ON (a:Amenity) ASSERT a.name IS UNIQUE')
    session.run('CREATE CONSTRAINT IF NOT EXISTS ON (n:Neighborhood) ASSERT n.name IS UNIQUE')
    session.run('CREATE CONSTRAINT IF NOT EXISTS ON (p:PriceLevel) ASSERT p.level IS UNIQUE')

    tx = session.begin_transaction()
    for _, row in df.iterrows():
      subj = str(row['subject'])
      pred = row['predicate']
      obj = row['object']

      tx.run(
        'MERGE (l:Listing {id: $lid})',
        lid=subj
      )

      if pred == 'has_amenity':
        tx.run(lid=subj, obj=obj)
      elif pred == 'in_neighborhood':
        tx.run(lid=subj, obj=obj)
      elif pred == 'price_level':
        tx.run(lid=subj, obj=obj)
    tx.commit()
    print(f"ingested {len(df)} metadata triples into neo4j")

if __name__=='__main__':
  driver = create_neo4j_driver(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

  print('clearing any existing data in neo4j...')
  clear_neo4j(driver)

  print('ingesting metadata triples into neo4j...')
  ingest_metadata_triples(driver, KG_PATH)

  driver.close()
  print('* kg loader complete *')