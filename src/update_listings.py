import gzip
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / 'data' / 'raw'
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'
KG_DIR = BASE_DIR / 'data' / 'kg'
CHECKPOINT_FILES = [
  PROCESSED_DIR / 'qdrant_checkpoint.txt',
  PROCESSED_DIR / 'qdrant_payload_checkpoint.txt',
]
GET_DATA_URL = 'https://insideairbnb.com/get-the-data/'
SNAPSHOT_MANIFEST_PATH = PROCESSED_DIR / 'source_snapshot_manifest.json'


def _slugify_city(city_value: str) -> str:
  lowered = city_value.strip().lower()
  slug = re.sub(r'[^a-z0-9]+', '_', lowered).strip('_')
  return slug or 'unknown_city'


def _to_abs_data_url(href: str) -> str:
  if href.startswith('http://') or href.startswith('https://'):
    return href
  if href.startswith('/'):
    return f"https://data.insideairbnb.com{href}"
  return f"https://data.insideairbnb.com/{href}"


def discover_portugal_pairs():
  response = requests.get(GET_DATA_URL, timeout=60)
  response.raise_for_status()
  hrefs = re.findall(r'href="([^"]+)"', response.text)

  candidates = []
  for href in hrefs:
    normalized = _to_abs_data_url(href)
    lower_url = normalized.lower()
    if '/portugal/' not in lower_url:
      continue
    if not (lower_url.endswith('/listings.csv.gz') or lower_url.endswith('/reviews.csv.gz')):
      continue
    candidates.append(normalized)

  grouped = {}
  for url in candidates:
    key = url.rsplit('/', 1)[0]
    grouped.setdefault(key, {})
    if url.lower().endswith('/listings.csv.gz'):
      grouped[key]['listings'] = url
    elif url.lower().endswith('/reviews.csv.gz'):
      grouped[key]['reviews'] = url

  pairs = []
  for base_path, files in grouped.items():
    if 'listings' not in files or 'reviews' not in files:
      continue
    parts = [p for p in urlparse(base_path).path.split('/') if p]
    city = None
    date = None
    for idx, part in enumerate(parts):
      if re.fullmatch(r'\d{4}-\d{2}-\d{2}', part):
        date = part
        if idx > 0:
          city = parts[idx - 1]
        break
    if not city:
      continue
    pairs.append({
      'city_slug': _slugify_city(city.replace('-', ' ')),
      'date': date or 'unknown_date',
      'listings_url': files['listings'],
      'reviews_url': files['reviews'],
    })
  return pairs


def latest_snapshot_by_city():
  pairs = discover_portugal_pairs()
  latest = {}
  for entry in pairs:
    city = entry['city_slug']
    date_text = entry['date']
    try:
      parsed = datetime.strptime(date_text, '%Y-%m-%d')
    except ValueError:
      parsed = datetime.min
    existing = latest.get(city)
    if existing is None or parsed > existing['_parsed_date']:
      latest[city] = {
        **entry,
        '_parsed_date': parsed
      }
  for city in list(latest.keys()):
    latest[city].pop('_parsed_date', None)
  return latest


def local_latest_snapshot_by_city():
  latest = {}
  for listings_file in RAW_DIR.glob('*_listings_*.csv'):
    match = re.match(r'^(?P<city>.+)_listings_(?P<date>\d{4}-\d{2}-\d{2})\.csv$', listings_file.name)
    if not match:
      continue
    city_slug = match.group('city')
    date_text = match.group('date')
    reviews_file = RAW_DIR / f'{city_slug}_reviews_{date_text}.csv'
    if not reviews_file.exists():
      continue
    try:
      parsed = datetime.strptime(date_text, '%Y-%m-%d')
    except ValueError:
      parsed = datetime.min
    existing = latest.get(city_slug)
    if existing is None or parsed > existing['_parsed_date']:
      latest[city_slug] = {
        'city_slug': city_slug,
        'date': date_text,
        'listings_url': 'local',
        'reviews_url': 'local',
        '_parsed_date': parsed
      }
  for city in list(latest.keys()):
    latest[city].pop('_parsed_date', None)
  return latest


def _download_and_extract_gzip(url: str, destination_csv: Path):
  tmp_gz = destination_csv.with_suffix(destination_csv.suffix + '.gz')
  with requests.get(url, stream=True, timeout=120) as response:
    response.raise_for_status()
    with open(tmp_gz, 'wb') as out_f:
      for chunk in response.iter_content(chunk_size=1024 * 1024):
        if chunk:
          out_f.write(chunk)
  with gzip.open(tmp_gz, 'rb') as src_f, open(destination_csv, 'wb') as dst_f:
    shutil.copyfileobj(src_f, dst_f)
  tmp_gz.unlink(missing_ok=True)


def _expected_snapshot_manifest(snapshot_by_city):
  return {
    city: {
      'date': info['date'],
      'listings_url': info['listings_url'],
      'reviews_url': info['reviews_url']
    }
    for city, info in snapshot_by_city.items()
  }


def _load_snapshot_manifest():
  if not SNAPSHOT_MANIFEST_PATH.exists():
    return {}
  try:
    with open(SNAPSHOT_MANIFEST_PATH, 'r', encoding='utf-8') as f:
      return json.load(f)
  except Exception:
    return {}


def _save_snapshot_manifest(manifest):
  PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
  with open(SNAPSHOT_MANIFEST_PATH, 'w', encoding='utf-8') as f:
    json.dump(manifest, f, indent=2, sort_keys=True)


def sync_latest_portugal_files(snapshot_by_city):
  RAW_DIR.mkdir(parents=True, exist_ok=True)
  downloaded = []
  expected_files = set()

  for city_slug, entry in snapshot_by_city.items():
    listings_name = f"{entry['city_slug']}_listings_{entry['date']}.csv"
    reviews_name = f"{entry['city_slug']}_reviews_{entry['date']}.csv"
    listings_path = RAW_DIR / listings_name
    reviews_path = RAW_DIR / reviews_name
    expected_files.add(listings_name)
    expected_files.add(reviews_name)

    if not listings_path.exists():
      _download_and_extract_gzip(entry['listings_url'], listings_path)
      downloaded.append(listings_name)
    if not reviews_path.exists():
      _download_and_extract_gzip(entry['reviews_url'], reviews_path)
      downloaded.append(reviews_name)

  # Remove older redundant snapshots for cities in the latest set.
  for city_slug in snapshot_by_city.keys():
    for obsolete in RAW_DIR.glob(f'{city_slug}_listings_*.csv'):
      if obsolete.name not in expected_files:
        obsolete.unlink(missing_ok=True)
    for obsolete in RAW_DIR.glob(f'{city_slug}_reviews_*.csv'):
      if obsolete.name not in expected_files:
        obsolete.unlink(missing_ok=True)

  return downloaded


def build_metadata_triples():
  KG_DIR.mkdir(parents=True, exist_ok=True)
  listings_path = PROCESSED_DIR / 'listings.parquet'
  if not listings_path.exists():
    raise FileNotFoundError(f'Missing processed listings parquet: {listings_path}')

  df = pd.read_parquet(listings_path)
  triples = []
  for _, row in df.iterrows():
    listing_id = str(row.get('id'))
    if not listing_id or listing_id == 'nan':
      continue

    neighborhood = row.get('neighbourhood_cleansed') or row.get('neighbourhood')
    if isinstance(neighborhood, str) and neighborhood.strip():
      triples.append((listing_id, 'in_neighborhood', neighborhood.strip()))

    amenities_raw = row.get('amenities', '[]')
    amenities_text = str(amenities_raw)
    amenities = re.findall(r'"([^"]+)"', amenities_text)
    for amenity in amenities:
      if amenity.strip():
        triples.append((listing_id, 'has_amenity', amenity.strip()))

    price_val = row.get('price')
    if isinstance(price_val, str):
      cleaned = price_val.replace('$', '').replace(',', '').strip()
      try:
        price_num = float(cleaned)
      except ValueError:
        price_num = None
    else:
      price_num = float(price_val) if pd.notna(price_val) else None
    if price_num is not None:
      level = 'low' if price_num < 50 else ('medium' if price_num < 150 else 'high')
      triples.append((listing_id, 'price_level', level))

  out_path = KG_DIR / 'metadata_triples.csv'
  pd.DataFrame(triples, columns=['subject', 'predicate', 'object']).to_csv(out_path, index=False)
  return out_path


def _run_python_script(script_name: str):
  script_path = BASE_DIR / 'src' / script_name
  venv_python = BASE_DIR / 'venv' / 'bin' / 'python'
  python_bin = str(venv_python) if venv_python.exists() else sys.executable
  env = os.environ.copy()
  env.setdefault('NEO4J_URI', 'bolt://localhost:7687')
  env.setdefault('QDRANT_HOST', 'localhost')
  env.setdefault('QDRANT_PORT', '6333')
  cmd = [python_bin, str(script_path)]
  result = subprocess.run(cmd, cwd=BASE_DIR, capture_output=True, text=True, env=env)
  if result.returncode != 0:
    combined = f"{result.stdout}\n{result.stderr}".lower()
    if script_name == 'kg_loader_neo4j.py' and ('connection refused' in combined or 'serviceunavailable' in combined):
      neo4j_uri = env.get('NEO4J_URI', 'bolt://localhost:7687')
      raise RuntimeError(
        f"kg_loader_neo4j.py failed: Neo4j is not reachable at {neo4j_uri}. "
        "Start Neo4j and verify bolt port 7687."
      )
    raise RuntimeError(f"{script_name} failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")


def _reset_qdrant_checkpoints():
  PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
  for checkpoint in CHECKPOINT_FILES:
    checkpoint.unlink(missing_ok=True)


def run_full_update():
  RAW_DIR.mkdir(parents=True, exist_ok=True)
  latest_snapshot = local_latest_snapshot_by_city()
  downloaded = []

  # Only fetch from remote when local raw files are missing.
  if not latest_snapshot:
    remote_snapshot = latest_snapshot_by_city()
    downloaded = sync_latest_portugal_files(remote_snapshot)
    latest_snapshot = local_latest_snapshot_by_city()

  if not latest_snapshot:
    return {'status': 'error', 'message': 'No local raw files found and no remote files downloaded.'}

  expected_manifest = _expected_snapshot_manifest(latest_snapshot)
  prior_manifest = _load_snapshot_manifest()
  if expected_manifest == prior_manifest and not downloaded:
    status_message = 'already up to date'
  else:
    status_message = 'database update completed'

  _reset_qdrant_checkpoints()
  _run_python_script('ingestion.py')
  build_metadata_triples()
  _run_python_script('kg_loader_neo4j.py')
  _run_python_script('embed_qdrant.py')
  _save_snapshot_manifest(expected_manifest)

  return {
    'status': 'completed' if status_message != 'already up to date' else 'up_to_date',
    'message': status_message,
    'downloaded_pairs': len(downloaded) // 2 if downloaded else 0,
  }


if __name__ == '__main__':
  result = run_full_update()
  print(result)
