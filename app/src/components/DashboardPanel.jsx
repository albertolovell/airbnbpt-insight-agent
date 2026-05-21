import React, { useEffect, useMemo, useState } from 'react';
import axios from 'axios';

const DEFAULT_FILTERS = {
  city: 'all',
  room_type: 'all',
  property_type: 'all',
  superhost: 'all',
  price_level: 'all',
  min_accommodates: 1,
  min_bedrooms: 0,
  min_beds: 0
};

function MetricCard({ label, value, prefix = '', suffix = '' }) {
  return (
    <div className="metric-card">
      <p className="metric-label">{label}</p>
      <p className="metric-value">
        {value === null || value === undefined ? 'N/A' : `${prefix}${value}${suffix}`}
      </p>
    </div>
  );
}

function DashboardPanel() {
  const [filters, setFilters] = useState(DEFAULT_FILTERS);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [data, setData] = useState(null);

  const queryParams = useMemo(
    () => ({
      ...filters
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

  const options = data?.options || { cities: [], room_types: [], property_types: [] };
  const metrics = data?.metrics || {};
  const cities = data?.city_breakdown || [];
  const roomTypes = data?.room_type_breakdown || [];
  const timeSeries = data?.time_series || [];
  const occupancyBands = data?.occupancy_band_chart || [];
  const roomTypeMetricChart = data?.room_type_metric_chart || [];
  const maxListings = Math.max(...timeSeries.map((item) => item.listing_count), 1);
  const maxOccBandCount = Math.max(...occupancyBands.map((item) => item.count), 1);

  const onFilterChange = (key, value) => {
    setFilters((prev) => ({
      ...prev,
      [key]: key === 'min_accommodates' ? Number(value || 1) : value
    }));
  };

  return (
    <section className="dashboard-panel rounded-3xl border border-sand bg-white/80 p-6 shadow-soft backdrop-blur">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <p className="text-sm uppercase tracking-[0.3em] text-ink-muted">Market dashboard</p>
          <h2 className="mt-2 text-3xl font-semibold">Portugal listing analytics</h2>
        </div>
        <button type="button" className="pill" onClick={() => setFilters(DEFAULT_FILTERS)}>
          Reset filters
        </button>
      </div>

      <div className="dashboard-filters mt-6">
        <label className="filter-field">
          <span>City</span>
          <select value={filters.city} onChange={(e) => onFilterChange('city', e.target.value)}>
            <option value="all">All</option>
            {options.cities.map((item) => (
              <option key={item} value={item}>
                {item}
              </option>
            ))}
          </select>
        </label>

        <label className="filter-field">
          <span>Room Type</span>
          <select value={filters.room_type} onChange={(e) => onFilterChange('room_type', e.target.value)}>
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
          <select value={filters.property_type} onChange={(e) => onFilterChange('property_type', e.target.value)}>
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
          <select value={filters.superhost} onChange={(e) => onFilterChange('superhost', e.target.value)}>
            <option value="all">All</option>
            <option value="yes">Yes</option>
            <option value="no">No</option>
          </select>
        </label>

        <label className="filter-field">
          <span>Price Level</span>
          <select value={filters.price_level} onChange={(e) => onFilterChange('price_level', e.target.value)}>
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
            onChange={(e) => onFilterChange('min_accommodates', e.target.value)}
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

      {error && <p className="mt-4 text-sm text-rose-500">{error}</p>}
      {loading && <p className="mt-4 text-sm text-ink-muted">Loading dashboard...</p>}

      <div className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <MetricCard label="Listings" value={metrics.listing_count} />
        <MetricCard label="Avg Price" value={metrics.avg_price} prefix="$" />
        <MetricCard label="Median Price" value={metrics.median_price} prefix="$" />
        <MetricCard label="Avg Occupancy" value={metrics.avg_occupancy_pct} suffix="%" />
        <MetricCard label="Avg Availability" value={metrics.avg_availability_days} suffix=" days" />
        <MetricCard label="Avg Revenue (365d)" value={metrics.avg_revenue_l365d} prefix="$" />
        <MetricCard label="Reviews / Month" value={metrics.avg_reviews_per_month} />
        <MetricCard label="Avg Rating" value={metrics.avg_rating} />
        <MetricCard label="Superhost Share" value={metrics.superhost_share_pct} suffix="%" />
        <MetricCard label="Avg Bedrooms" value={metrics.avg_bedrooms} />
        <MetricCard label="Avg Beds" value={metrics.avg_beds} />
      </div>

      <div className="mt-8 grid gap-6 lg:grid-cols-2">
        <div className="table-card">
          <h3 className="text-lg font-semibold">Occupancy distribution</h3>
          <div className="mt-4 space-y-2">
            {occupancyBands.length === 0 && <p className="text-sm text-ink-muted">No occupancy data for current filters.</p>}
            {occupancyBands.map((row) => (
              <div key={row.band} className="hbar-row">
                <div className="hbar-label">{row.band}</div>
                <div className="hbar-track">
                  <div className="hbar-fill" style={{ width: `${Math.max((row.count / maxOccBandCount) * 100, row.count > 0 ? 6 : 0)}%` }} />
                </div>
                <div className="hbar-value">{row.count}</div>
              </div>
            ))}
          </div>
        </div>

        <div className="table-card">
          <h3 className="text-lg font-semibold">Room type performance</h3>
          <div className="mt-3 space-y-2">
            {roomTypeMetricChart.length === 0 && <p className="text-sm text-ink-muted">No room type data for current filters.</p>}
            {roomTypeMetricChart.map((row) => (
              <div key={`${row.label}-metric`} className="row-item">
                <div>
                  <p className="row-title">{row.label}</p>
                  <p className="row-sub">{row.listing_count} listings</p>
                </div>
                <div className="row-right">
                  <p>${row.avg_price ?? 'N/A'}</p>
                  <p className="row-sub">{row.avg_occupancy_pct ?? 'N/A'}% occ</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="table-card lg:col-span-2">
          <h3 className="text-lg font-semibold">Time series: listing activity by month</h3>
          <p className="row-sub mt-1">Based on each listing&apos;s `last_review` month, filtered by your current selection.</p>
          <div className="trend-chart mt-4">
            {timeSeries.length === 0 && <p className="text-sm text-ink-muted">No trend data for current filters.</p>}
            {timeSeries.map((point) => (
              <div key={point.month} className="trend-bar-wrap">
                <div
                  className="trend-bar"
                  style={{ height: `${Math.max((point.listing_count / maxListings) * 140, 8)}px` }}
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
                    <p className="row-sub">{point.listing_count} active listings</p>
                  </div>
                  <div className="row-right">
                    <p>{point.avg_occupancy_pct ?? 'N/A'}% occ</p>
                    <p className="row-sub">{point.avg_reviews_l30d ?? 'N/A'} rev/30d</p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="table-card">
          <h3 className="text-lg font-semibold">City breakdown</h3>
          <div className="mt-3 space-y-2">
            {cities.length === 0 && <p className="text-sm text-ink-muted">No data for current filters.</p>}
            {cities.map((row) => (
              <div key={row.city} className="row-item">
                <div>
                  <p className="row-title">{row.city}</p>
                  <p className="row-sub">
                    {row.listing_count} listings • ${row.avg_price ?? 'N/A'} avg price
                  </p>
                </div>
                <div className="row-right">
                  <p>{row.avg_occupancy_pct ?? 'N/A'}%</p>
                  <p className="row-sub">{row.avg_reviews_per_month ?? 'N/A'} rev/mo</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="table-card">
          <h3 className="text-lg font-semibold">Room type split</h3>
          <div className="mt-3 space-y-2">
            {roomTypes.length === 0 && <p className="text-sm text-ink-muted">No data for current filters.</p>}
            {roomTypes.map((row) => (
              <div key={row.room_type} className="row-item">
                <div>
                  <p className="row-title">{row.room_type}</p>
                  <p className="row-sub">
                    {row.listing_count} listings • ${row.avg_price ?? 'N/A'} avg price
                  </p>
                </div>
                <div className="row-right">
                  <p>{row.avg_occupancy_pct ?? 'N/A'}%</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}

export default DashboardPanel;
