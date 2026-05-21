from fastapi import FastAPI
from fastapi import Query
from fastapi import BackgroundTasks
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import pandas as pd
import threading
from datetime import datetime, timezone
from src.update_listings import run_full_update

app = FastAPI()

app.add_middleware(
  CORSMiddleware,
  allow_origins=['http://localhost:5173', 'http://localhost:3000'],
  allow_credentials=True,
  allow_methods=['*'],
  allow_headers=['*']
)

class QueryRequest(BaseModel):
  query: str

_agent_runner = None
_dashboard_df = None
_dashboard_options = None
_update_lock = threading.Lock()
_update_state = {
  'status': 'idle',
  'message': 'idle',
  'last_success_at': None
}


def _to_float(value):
  if value is None:
    return None
  if isinstance(value, (int, float)):
    return float(value)
  if isinstance(value, str):
    cleaned = value.replace('$', '').replace(',', '').strip()
    if not cleaned:
      return None
    try:
      return float(cleaned)
    except ValueError:
      return None
  return None


def _price_bucket(price):
  if price is None:
    return None
  if price < 50:
    return 'low'
  if price < 150:
    return 'medium'
  return 'high'


def _round_or_none(value, digits=2):
  if value is None or pd.isna(value):
    return None
  return round(float(value), digits)

def _normalize_city(value):
  if value is None or pd.isna(value):
    return None
  city = str(value).strip()
  if not city:
    return None
  return city.title()


def get_dashboard_df():
  global _dashboard_df, _dashboard_options
  if _dashboard_df is not None:
    return _dashboard_df, _dashboard_options

  data_path = Path(__file__).resolve().parents[1] / 'data' / 'processed' / 'listings.parquet'
  if not data_path.exists():
    data_path = Path('data/processed/listings.parquet')
  if not data_path.exists():
    raise FileNotFoundError('listings.parquet not found in data/processed')

  cols = [
    'id', 'neighbourhood_group_cleansed', 'neighbourhood_cleansed', 'room_type', 'property_type', 'price',
    'availability_365', 'estimated_occupancy_l365d', 'estimated_revenue_l365d',
    'reviews_per_month', 'review_scores_rating', 'accommodates', 'host_is_superhost', 'bedrooms', 'beds',
    'first_review', 'last_review', 'number_of_reviews_l30d', 'number_of_reviews_ltm'
  ]
  df = pd.read_parquet(data_path, columns=cols)
  df['price_num'] = df['price'].apply(_to_float)
  df['occupancy_num'] = pd.to_numeric(df['estimated_occupancy_l365d'], errors='coerce')
  df['availability_num'] = pd.to_numeric(df['availability_365'], errors='coerce')
  df['revenue_num'] = pd.to_numeric(df['estimated_revenue_l365d'], errors='coerce')
  df['reviews_pm_num'] = pd.to_numeric(df['reviews_per_month'], errors='coerce')
  df['rating_num'] = pd.to_numeric(df['review_scores_rating'], errors='coerce')
  df['accommodates_num'] = pd.to_numeric(df['accommodates'], errors='coerce')
  df['bedrooms_num'] = pd.to_numeric(df['bedrooms'], errors='coerce')
  df['beds_num'] = pd.to_numeric(df['beds'], errors='coerce')
  df['reviews_l30d_num'] = pd.to_numeric(df['number_of_reviews_l30d'], errors='coerce')
  df['reviews_ltm_num'] = pd.to_numeric(df['number_of_reviews_ltm'], errors='coerce')
  df['superhost_bool'] = df['host_is_superhost'].astype(str).str.lower().isin(['t', 'true', '1', 'yes'])
  df['price_level'] = df['price_num'].apply(_price_bucket)
  df['city_name'] = df['neighbourhood_group_cleansed'].apply(_normalize_city)
  df['city_name'] = df['city_name'].fillna('Unknown')
  df['first_review_dt'] = pd.to_datetime(df['first_review'], errors='coerce')
  df['last_review_dt'] = pd.to_datetime(df['last_review'], errors='coerce')
  df['last_review_month'] = df['last_review_dt'].dt.to_period('M')

  cities = sorted([c for c in df['city_name'].dropna().unique().tolist() if str(c).strip()])
  room_types = sorted([r for r in df['room_type'].dropna().unique().tolist() if str(r).strip()])
  property_types = sorted([p for p in df['property_type'].dropna().unique().tolist() if str(p).strip()])
  _dashboard_options = {
    'cities': cities,
    'room_types': room_types,
    'property_types': property_types
  }
  _dashboard_df = df
  return _dashboard_df, _dashboard_options

def get_agent_runner():
  global _agent_runner
  if _agent_runner is None:
    print('loading agent for the first request.....')
    from src.agent_core import run_agent
    _agent_runner = run_agent
  return _agent_runner

@app.post('/ask')
async def ask_agent(req: QueryRequest):
  agent_runner = get_agent_runner()
  result = agent_runner(req.query)
  return result


def _run_update_job():
  global _update_state
  try:
    result = run_full_update()
    with _update_lock:
      last_success_at = _update_state.get('last_success_at')
      if result.get('status') == 'completed':
        last_success_at = datetime.now(timezone.utc).isoformat()
      _update_state = {
        'status': result.get('status', 'completed'),
        'message': result.get('message', 'database update completed'),
        'downloaded_pairs': result.get('downloaded_pairs', 0),
        'last_success_at': last_success_at
      }
  except Exception as exc:
    with _update_lock:
      _update_state = {
        'status': 'error',
        'message': f'update failed: {exc}',
        'last_success_at': _update_state.get('last_success_at')
      }


@app.post('/update-listings')
async def update_listings(background_tasks: BackgroundTasks):
  global _update_state
  with _update_lock:
    current_status = _update_state.get('status')
    if current_status == 'pending':
      return {'status': 'pending', 'message': 'database update pending'}
    _update_state = {'status': 'pending', 'message': 'database update pending'}
  background_tasks.add_task(_run_update_job)
  return {'status': 'pending', 'message': 'database update pending'}


@app.get('/update-listings/status')
async def update_listings_status():
  with _update_lock:
    return dict(_update_state)


@app.get('/dashboard')
async def dashboard_data(
  city: str = Query(default='all'),
  room_type: str = Query(default='all'),
  property_type: str = Query(default='all'),
  superhost: str = Query(default='all'),
  price_level: str = Query(default='all'),
  min_accommodates: int = Query(default=1, ge=1),
  min_bedrooms: int = Query(default=0, ge=0),
  min_beds: int = Query(default=0, ge=0)
):
  df, options = get_dashboard_df()
  filtered = df.copy()

  if city != 'all':
    filtered = filtered[filtered['city_name'] == city]
  if room_type != 'all':
    filtered = filtered[filtered['room_type'] == room_type]
  if property_type != 'all':
    filtered = filtered[filtered['property_type'] == property_type]
  if price_level != 'all':
    filtered = filtered[filtered['price_level'] == price_level]
  if superhost == 'yes':
    filtered = filtered[filtered['superhost_bool'] == True]
  elif superhost == 'no':
    filtered = filtered[filtered['superhost_bool'] == False]
  if min_accommodates > 1:
    filtered = filtered[filtered['accommodates_num'] >= min_accommodates]
  if min_bedrooms > 0:
    filtered = filtered[filtered['bedrooms_num'] >= min_bedrooms]
  if min_beds > 0:
    filtered = filtered[filtered['beds_num'] >= min_beds]

  metrics = {
    'listing_count': int(len(filtered)),
    'avg_price': _round_or_none(filtered['price_num'].mean()),
    'median_price': _round_or_none(filtered['price_num'].median()),
    'avg_occupancy_pct': _round_or_none(filtered['occupancy_num'].mean()),
    'avg_availability_days': _round_or_none(filtered['availability_num'].mean()),
    'avg_revenue_l365d': _round_or_none(filtered['revenue_num'].mean()),
    'avg_reviews_per_month': _round_or_none(filtered['reviews_pm_num'].mean()),
    'avg_rating': _round_or_none(filtered['rating_num'].mean()),
    'superhost_share_pct': _round_or_none(filtered['superhost_bool'].mean() * 100 if len(filtered) else None),
    'avg_bedrooms': _round_or_none(filtered['bedrooms_num'].mean()),
    'avg_beds': _round_or_none(filtered['beds_num'].mean())
  }

  if filtered.empty:
    city_breakdown = []
    room_type_breakdown = []
    time_series = []
    occupancy_band_chart = []
    room_type_metric_chart = []
  else:
    city_group = (
      filtered.groupby('city_name', dropna=True)
      .agg(
        listing_count=('id', 'count'),
        avg_price=('price_num', 'mean'),
        avg_occupancy_pct=('occupancy_num', 'mean'),
        avg_reviews_per_month=('reviews_pm_num', 'mean')
      )
      .reset_index()
      .sort_values('listing_count', ascending=False)
      .head(12)
    )
    city_breakdown = [
      {
        'city': row['city_name'],
        'listing_count': int(row['listing_count']),
        'avg_price': _round_or_none(row['avg_price']),
        'avg_occupancy_pct': _round_or_none(row['avg_occupancy_pct']),
        'avg_reviews_per_month': _round_or_none(row['avg_reviews_per_month'])
      }
      for _, row in city_group.iterrows()
    ]

    room_group = (
      filtered.groupby('room_type', dropna=True)
      .agg(
        listing_count=('id', 'count'),
        avg_price=('price_num', 'mean'),
        avg_occupancy_pct=('occupancy_num', 'mean')
      )
      .reset_index()
      .sort_values('listing_count', ascending=False)
    )
    room_type_breakdown = [
      {
        'room_type': row['room_type'],
        'listing_count': int(row['listing_count']),
        'avg_price': _round_or_none(row['avg_price']),
        'avg_occupancy_pct': _round_or_none(row['avg_occupancy_pct'])
      }
      for _, row in room_group.iterrows()
    ]

    room_type_metric_chart = [
      {
        'label': row['room_type'],
        'avg_price': _round_or_none(row['avg_price']),
        'avg_occupancy_pct': _round_or_none(row['avg_occupancy_pct']),
        'listing_count': int(row['listing_count'])
      }
      for _, row in room_group.iterrows()
    ]

    monthly = (
      filtered.dropna(subset=['last_review_month'])
      .groupby('last_review_month')
      .agg(
        listing_count=('id', 'count'),
        avg_reviews_per_month=('reviews_pm_num', 'mean'),
        avg_reviews_l30d=('reviews_l30d_num', 'mean'),
        avg_occupancy_pct=('occupancy_num', 'mean')
      )
      .reset_index()
      .sort_values('last_review_month')
    )
    if len(monthly) > 18:
      monthly = monthly.tail(18)

    time_series = [
      {
        'month': str(row['last_review_month']),
        'listing_count': int(row['listing_count']),
        'avg_reviews_per_month': _round_or_none(row['avg_reviews_per_month']),
        'avg_reviews_l30d': _round_or_none(row['avg_reviews_l30d']),
        'avg_occupancy_pct': _round_or_none(row['avg_occupancy_pct'])
      }
      for _, row in monthly.iterrows()
    ]

    occ = filtered['occupancy_num'].dropna()
    if not occ.empty:
      bins = [0, 20, 40, 60, 80, 100]
      labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
      bands = pd.cut(occ.clip(lower=0, upper=100), bins=bins, labels=labels, include_lowest=True)
      band_counts = bands.value_counts().reindex(labels, fill_value=0)
      occupancy_band_chart = [
        {'band': label, 'count': int(band_counts[label])}
        for label in labels
      ]
    else:
      occupancy_band_chart = []

  return {
    'applied_filters': {
      'city': city,
      'room_type': room_type,
      'property_type': property_type,
      'superhost': superhost,
      'price_level': price_level,
      'min_accommodates': min_accommodates,
      'min_bedrooms': min_bedrooms,
      'min_beds': min_beds
    },
    'metrics': metrics,
    'time_series': time_series,
    'city_breakdown': city_breakdown,
    'room_type_breakdown': room_type_breakdown,
    'room_type_metric_chart': room_type_metric_chart,
    'occupancy_band_chart': occupancy_band_chart,
    'options': options
  }
