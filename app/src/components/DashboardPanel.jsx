import React, { useEffect, useMemo, useState } from 'react';
import axios from 'axios';

const DEFAULT_FILTERS = {
  city: 'all',
  geo_area: 'all',
  room_type: 'all',
  property_type: 'all',
  superhost: 'all',
  price_level: 'all',
  min_accommodates: 1,
  min_bedrooms: 0,
  min_beds: 0,
};

function MetricCard({ label, value, prefix = '', suffix = '' }) {
  const hasValue = value !== null && value !== undefined;
  const displayValue = hasValue ? `${prefix}${value}${suffix}` : 'Unavailable';
  return (
    <div className={`metric-card ${hasValue ? '' : 'metric-card-unavailable'}`}>
      <div className="metric-card-head">
        <p className="metric-label">{label}</p>
      </div>
      <p className={`metric-value ${hasValue ? '' : 'metric-value-empty'}`}>
        {displayValue}
      </p>
    </div>
  );
}

function DashboardPanel() {
  const [filters, setFilters] = useState(DEFAULT_FILTERS);
  const [filtersOpen, setFiltersOpen] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [data, setData] = useState(null);

  const queryParams = useMemo(
    () => ({
      ...filters,
    }),
    [filters]
  );

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      setLoading(true);
      setError('');
      try {
        const res = await axios.get('/dashboard', { params: queryParams });
        if (!cancelled) setData(res.data);
      } catch (err) {
        console.error(err);
        if (!cancelled) setError('Failed to load dashboard data.');
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    load();
    return () => {
      cancelled = true;
    };
  }, [queryParams]);

  const options = data?.options || {
    cities: [],
    room_types: [],
    property_types: [],
  };
  const metrics = data?.metrics || {};
  const priceDataAvailable = metrics.price_data_available !== false;
  const filterSummary = [];
  if (filters.city !== 'all') filterSummary.push(`City: ${filters.city}`);
  if (filters.geo_area !== 'all')
    filterSummary.push(`Area: ${filters.geo_area}`);
  if (filters.room_type !== 'all')
    filterSummary.push(`Room: ${filters.room_type}`);
  if (filters.property_type !== 'all')
    filterSummary.push(`Property: ${filters.property_type}`);
  if (filters.superhost !== 'all')
    filterSummary.push(`Superhost: ${filters.superhost}`);
  if (filters.price_level !== 'all')
    filterSummary.push(`Price level: ${filters.price_level}`);
  if (filters.min_accommodates > 1)
    filterSummary.push(`Min guests: ${filters.min_accommodates}`);
  if (filters.min_bedrooms > 0)
    filterSummary.push(`Min bedrooms: ${filters.min_bedrooms}`);
  if (filters.min_beds > 0) filterSummary.push(`Min beds: ${filters.min_beds}`);

  const cities = data?.city_breakdown || [];
  const roomTypes = data?.room_type_breakdown || [];
  const timeSeries = data?.time_series || [];
  const occupancyBands = data?.occupancy_band_chart || [];
  const roomTypeMetricChart = data?.room_type_metric_chart || [];
  const geoAreaMetricChart = data?.geo_area_metric_chart || [];
  const mapData = data?.map_data || { cities: [], by_city: {} };
  const selectedMapCity =
    filters.city !== 'all' ? filters.city : mapData.cities?.[0] || null;
  const activeCityMap = selectedMapCity
    ? mapData.by_city?.[selectedMapCity]
    : null;
  const portugalOverview = mapData?.portugal_overview || {
    bbox: null,
    points: [],
  };
  const maxListings = Math.max(
    ...timeSeries.map((item) => item.listing_count),
    1
  );
  const maxOccBandCount = Math.max(
    ...occupancyBands.map((item) => item.count),
    1
  );
  const maxGeoListings = Math.max(
    ...geoAreaMetricChart.map((item) => item.listing_count),
    1
  );

  const onFilterChange = (key, value) => {
    setFilters((prev) => ({
      ...prev,
      [key]: key === 'min_accommodates' ? Number(value || 1) : value,
    }));
  };

  return (
    <section className="dashboard-panel rounded-3xl border border-sand bg-white/80 p-6 shadow-soft backdrop-blur">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <p className="text-sm uppercase tracking-[0.3em] text-ink-muted">
            Market dashboard
          </p>
          <h2 className="mt-2 text-3xl font-semibold">
            Portugal listing analytics
          </h2>
          <p className="mt-3 max-w-2xl text-sm text-ink-subtle">
            Explore supply and demand across Portugal with filters for city,
            neighborhood, room type and price tier. Keep an eye on occupancy,
            availability and reviews when price values are not available.
          </p>
        </div>
        <button
          type="button"
          className="pill"
          onClick={() => setFilters(DEFAULT_FILTERS)}
        >
          Reset filters
        </button>
      </div>

      <div className="dashboard-filters-header mt-6">
        <div>
          <p className="text-sm uppercase tracking-[0.3em] text-ink-muted">
            Filters
          </p>
          <p className="mt-1 text-sm text-ink-subtle">
            {activeFilterCount > 0
              ? `${activeFilterCount} active filter${activeFilterCount > 1 ? 's' : ''}`
              : 'No filters applied'}
          </p>
        </div>
        <button
          type="button"
          className="pill"
          onClick={() => setFiltersOpen((prev) => !prev)}
        >
          {filtersOpen ? 'Hide filters' : 'Show filters'}
        </button>
      </div>
      {filtersOpen && (
        <div className="dashboard-filters mt-4">
          <label className="filter-field">
            <span>City</span>
            <select
              value={filters.city}
              onChange={(e) => onFilterChange('city', e.target.value)}
            >
              <option value="all">All</option>
              {options.cities.map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
            </select>
          </label>

          <label className="filter-field">
            <span>Geo Area</span>
            <select
              value={filters.geo_area}
              onChange={(e) => onFilterChange('geo_area', e.target.value)}
            >
              <option value="all">All</option>
              {(options.geo_areas || []).map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
            </select>
          </label>

          <label className="filter-field">
            <span>Room Type</span>
            <select
              value={filters.room_type}
              onChange={(e) => onFilterChange('room_type', e.target.value)}
            >
              <option value="all">All</option>
              {options.room_types.map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
            </select>
          </label>

          <label className="filter-field">
            <span>Property Type</span>
            <select
              value={filters.property_type}
              onChange={(e) => onFilterChange('property_type', e.target.value)}
            >
              <option value="all">All</option>
              {options.property_types.map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
            </select>
          </label>

          <label className="filter-field">
            <span>Superhost</span>
            <select
              value={filters.superhost}
              onChange={(e) => onFilterChange('superhost', e.target.value)}
            >
              <option value="all">All</option>
              <option value="yes">Yes</option>
              <option value="no">No</option>
            </select>
          </label>

          <label className="filter-field">
            <span>Price Level</span>
            <select
              value={filters.price_level}
              onChange={(e) => onFilterChange('price_level', e.target.value)}
            >
              <option value="all">All</option>
              <option value="low">Low</option>
              <option value="medium">Medium</option>
              <option value="high">High</option>
            </select>
          </label>

          <label className="filter-field">
            <span>Min Accommodates</span>
            <input
              type="number"
              min="1"
              value={filters.min_accommodates}
              onChange={(e) =>
                onFilterChange('min_accommodates', e.target.value)
              }
            />
          </label>

          <label className="filter-field">
            <span>Min Bedrooms</span>
            <input
              type="number"
              min="0"
              value={filters.min_bedrooms}
              onChange={(e) => onFilterChange('min_bedrooms', e.target.value)}
            />
          </label>

          <label className="filter-field">
            <span>Min Beds</span>
            <input
              type="number"
              min="0"
              value={filters.min_beds}
              onChange={(e) => onFilterChange('min_beds', e.target.value)}
            />
          </label>
        </div>
      )}

      <div
        className={`dashboard-banner mt-6 ${priceDataAvailable ? 'banner-success' : 'banner-warning'}`}
      >
        <p className="banner-heading">
          {priceDataAvailable
            ? 'Price metrics are available for this dataset.'
            : 'Price metrics are currently unavailable.'}
        </p>
        <p className="banner-copy">
          {priceDataAvailable
            ? 'Refine the dashboard with filters to compare average price across segments.'
            : 'The current dataset lacks reliable nightly price values. Use occupancy and availability as the primary signals instead.'}
        </p>
      </div>
      {filterSummary.length > 0 && (
        <div className="filter-summary mt-4">
          Active filters: {filterSummary.join(' · ')}
        </div>
      )}
      <div className="table-card mt-6">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h3 className="text-lg font-semibold">Quick start</h3>
            <p className="row-sub mt-1">
              Use one of these quick actions to explore the most useful listing
              segments faster.
            </p>
          </div>
          <button
            type="button"
            className="pill"
            onClick={() => setFilters(DEFAULT_FILTERS)}
          >
            Reset filters
          </button>
        </div>
        <div className="mt-4 grid gap-3 sm:grid-cols-2">
          {quickActions.map((item) => (
            <button
              key={item.label}
              type="button"
              className="row-item quick-action"
              onClick={item.action}
            >
              <div>
                <p className="row-title">{item.label}</p>
                <p className="row-sub">{item.description}</p>
              </div>
            </button>
          ))}
        </div>
      </div>
      {error && <p className="mt-4 text-sm text-rose-500">{error}</p>}
      {loading && (
        <p className="mt-4 text-sm text-ink-muted">Loading dashboard...</p>
      )}

      <div className="metric-grid mt-6">
        <MetricCard label="Listings" value={metrics.listing_count} />
        <MetricCard label="Avg Price" value={metrics.avg_price} prefix="$" />
        <MetricCard
          label="Median Price"
          value={metrics.median_price}
          prefix="$"
        />
        <MetricCard
          label="Avg Occupancy"
          value={metrics.avg_occupancy_pct}
          suffix="%"
        />
        <MetricCard
          label="Avg Availability"
          value={metrics.avg_availability_days}
          suffix=" days"
        />
        <MetricCard
          label="Avg Revenue (365d)"
          value={metrics.avg_revenue_l365d}
          prefix="$"
        />
        <MetricCard
          label="Reviews / Month"
          value={metrics.avg_reviews_per_month}
        />
        <MetricCard label="Avg Rating" value={metrics.avg_rating} />
        <MetricCard
          label="Superhost Share"
          value={metrics.superhost_share_pct}
          suffix="%"
        />
        <MetricCard label="Avg Bedrooms" value={metrics.avg_bedrooms} />
        <MetricCard label="Avg Beds" value={metrics.avg_beds} />
      </div>

      <div className="mt-8 grid gap-6 lg:grid-cols-2">
        <div className="table-card lg:col-span-2">
          <h3 className="text-lg font-semibold">Portugal Listing Coverage</h3>
          <p className="row-sub mt-1">
            Overview of listing coordinates across Portugal. Darker clusters
            indicate denser markets.
          </p>
          {portugalOverview?.bbox && portugalOverview?.points?.length > 0 ? (
            <div className="map-wrap map-wrap-compact mt-3">
              <svg viewBox="0 0 860 420" className="geo-map">
                {portugalOverview.points.map((point, idx) => {
                  const b = portugalOverview.bbox;
                  const width = b.max_lon - b.min_lon || 1;
                  const height = b.max_lat - b.min_lat || 1;
                  const x = ((point.lon - b.min_lon) / width) * 840 + 10;
                  const y = 410 - ((point.lat - b.min_lat) / height) * 400;
                  return (
                    <circle
                      key={`${point.city}-${idx}`}
                      cx={x}
                      cy={y}
                      r="1.2"
                      className="geo-point"
                    />
                  );
                })}
              </svg>
            </div>
          ) : (
            <p className="text-sm text-ink-muted mt-3">
              No coordinate data available for overview map.
            </p>
          )}
        </div>

        <div className="table-card lg:col-span-2">
          <h3 className="text-lg font-semibold">
            Neighborhood Map {selectedMapCity ? `(${selectedMapCity})` : ''}
          </h3>
          <p className="row-sub mt-1">
            Click an area to filter metrics for that polygon-matched
            neighborhood.
          </p>
          {filters.city === 'all' && selectedMapCity && (
            <p className="text-sm text-ink-muted mt-2">
              Showing map preview for {selectedMapCity}. Click an area to focus
              that city and neighborhood.
            </p>
          )}
          {activeCityMap?.bbox && activeCityMap?.features?.length > 0 ? (
            <div className="map-wrap map-wrap-compact mt-3">
              <svg viewBox="0 0 900 460" className="geo-map">
                {activeCityMap.features.map((feature) => {
                  const ring = feature.ring || [];
                  const b = activeCityMap.bbox;
                  const width = b.max_lon - b.min_lon || 1;
                  const height = b.max_lat - b.min_lat || 1;
                  const points = ring
                    .map(([lon, lat]) => {
                      const x = ((lon - b.min_lon) / width) * 880 + 10;
                      const y = 450 - ((lat - b.min_lat) / height) * 440;
                      return `${x},${y}`;
                    })
                    .join(' ');
                  const areaMetrics = geoAreaMetricChart.find(
                    (row) => row.name === feature.name
                  );
                  const intensity = areaMetrics
                    ? Math.max(0.15, areaMetrics.listing_count / maxGeoListings)
                    : 0.08;
                  const selected = filters.geo_area === feature.name;
                  return (
                    <polygon
                      key={feature.name}
                      points={points}
                      className={`geo-area ${selected ? 'geo-area-selected' : ''}`}
                      style={{ opacity: intensity }}
                      onClick={() => {
                        if (filters.city === 'all' && selectedMapCity) {
                          onFilterChange('city', selectedMapCity);
                        }
                        onFilterChange(
                          'geo_area',
                          selected ? 'all' : feature.name
                        );
                      }}
                    >
                      <title>
                        {feature.name}
                        {areaMetrics
                          ? ` • ${areaMetrics.listing_count} listings • $${areaMetrics.avg_price ?? 'N/A'} avg`
                          : ''}
                      </title>
                    </polygon>
                  );
                })}
              </svg>
            </div>
          ) : (
            <p className="text-sm text-ink-muted mt-4">
              No GeoJSON map data available for this city in `data/raw`.
            </p>
          )}
        </div>

        <div className="table-card">
          <h3 className="text-lg font-semibold">
            Occupancy Distribution Across Listings
          </h3>
          <div className="mt-4 space-y-2">
            {occupancyBands.length === 0 && (
              <p className="text-sm text-ink-muted">
                No occupancy data for current filters.
              </p>
            )}
            {occupancyBands.map((row) => (
              <div key={row.band} className="hbar-row">
                <div className="hbar-label">{row.band}</div>
                <div className="hbar-track">
                  <div
                    className="hbar-fill"
                    style={{
                      width: `${Math.max((row.count / maxOccBandCount) * 100, row.count > 0 ? 6 : 0)}%`,
                    }}
                  />
                </div>
                <div className="hbar-value">{row.count}</div>
              </div>
            ))}
          </div>
        </div>

        <div className="table-card">
          <h3 className="text-lg font-semibold">
            Room Type Pricing and Occupancy Snapshot
          </h3>
          <div className="mt-3 space-y-2">
            {roomTypeMetricChart.length === 0 && (
              <p className="text-sm text-ink-muted">
                No room type data for current filters.
              </p>
            )}
            {roomTypeMetricChart.map((row) => (
              <div key={`${row.label}-metric`} className="row-item">
                <div>
                  <p className="row-title">{row.label}</p>
                  <p className="row-sub">{row.listing_count} listings</p>
                </div>
                <div className="row-right">
                  <p>${row.avg_price ?? 'N/A'}</p>
                  <p className="row-sub">
                    {row.avg_occupancy_pct ?? 'N/A'}% occ
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="table-card">
          <h3 className="text-lg font-semibold">Geo Area Snapshot</h3>
          <div className="mt-3 space-y-2">
            {geoAreaMetricChart.length === 0 && (
              <p className="text-sm text-ink-muted">
                No area-level metrics for current filters.
              </p>
            )}
            {geoAreaMetricChart.slice(0, 12).map((row) => (
              <div key={row.name} className="row-item">
                <div>
                  <p className="row-title">{row.name}</p>
                  <p className="row-sub">{row.listing_count} listings</p>
                </div>
                <div className="row-right">
                  <p>${row.avg_price ?? 'N/A'}</p>
                  <p className="row-sub">
                    {row.avg_occupancy_pct ?? 'N/A'}% occ
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="table-card lg:col-span-2">
          <h3 className="text-lg font-semibold">
            Monthly Listing Activity Trend
          </h3>
          <p className="row-sub mt-1">
            Based on each listing&apos;s `last_review` month, filtered by your
            current selection.
          </p>
          <div className="trend-chart mt-4">
            {timeSeries.length === 0 && (
              <p className="text-sm text-ink-muted">
                No trend data for current filters.
              </p>
            )}
            {timeSeries.map((point) => (
              <div key={point.month} className="trend-bar-wrap">
                <div
                  className="trend-bar"
                  style={{
                    height: `${Math.max((point.listing_count / maxListings) * 140, 8)}px`,
                  }}
                  title={`${point.month}: ${point.listing_count} listings`}
                />
                <p className="trend-value">{point.listing_count}</p>
                <p className="trend-label">{point.month.slice(2)}</p>
              </div>
            ))}
          </div>
          {timeSeries.length > 0 && (
            <div className="mt-4 grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
              {timeSeries.slice(-6).map((point) => (
                <div key={`${point.month}-detail`} className="row-item">
                  <div>
                    <p className="row-title">{point.month}</p>
                    <p className="row-sub">
                      {point.listing_count} active listings
                    </p>
                  </div>
                  <div className="row-right">
                    <p>{point.avg_occupancy_pct ?? 'N/A'}% occ</p>
                    <p className="row-sub">
                      {point.avg_reviews_l30d ?? 'N/A'} rev/30d
                    </p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="table-card">
          <h3 className="text-lg font-semibold">
            Top Cities by Demand and Price
          </h3>
          <div className="mt-3 tile-grid">
            {cities.length === 0 && (
              <p className="text-sm text-ink-muted">
                No data for current filters.
              </p>
            )}
            {cities.map((row) => (
              <div key={row.city} className="analytic-tile">
                <div className="analytic-tile-head">
                  <p className="row-title">{row.city}</p>
                </div>
                <p className="row-sub">{row.listing_count} listings</p>
                <p className="analytic-kpi">${row.avg_price ?? 'N/A'}</p>
                <p className="row-sub">Avg nightly price</p>
                <p className="analytic-detail">
                  {row.avg_occupancy_pct ?? 'N/A'}% occupancy
                </p>
                <p className="analytic-detail">
                  {row.avg_reviews_per_month ?? 'N/A'} reviews/mo
                </p>
              </div>
            ))}
          </div>
        </div>

        <div className="table-card">
          <h3 className="text-lg font-semibold">Room Mix Overview</h3>
          <div className="mt-3 tile-grid">
            {roomTypes.length === 0 && (
              <p className="text-sm text-ink-muted">
                No data for current filters.
              </p>
            )}
            {roomTypes.map((row) => (
              <div key={row.room_type} className="analytic-tile">
                <div className="analytic-tile-head">
                  <p className="row-title">{row.room_type}</p>
                </div>
                <p className="row-sub">{row.listing_count} listings</p>
                <p className="analytic-kpi">${row.avg_price ?? 'N/A'}</p>
                <p className="row-sub">Avg nightly price</p>
                <p className="analytic-detail">
                  {row.avg_occupancy_pct ?? 'N/A'}% occupancy
                </p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}

export default DashboardPanel;
