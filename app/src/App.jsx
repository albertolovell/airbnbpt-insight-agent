import React, { useEffect, useRef, useState } from 'react';
import axios from 'axios';
import DashboardPanel from './components/DashboardPanel';

const STARTER_PROMPTS = [
  'Best neighborhoods in Lisbon for a quiet stay',
  'Listings with pool in Porto and good reviews',
  'Affordable places near the coast with amenities',
  'Top 3 amenities mentioned in Lisbon reviews'
];

function App() {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content:
        "Welcome to Airbnb Portugal. Ask about listings, neighborhoods, amenities, or price levels and I'll search reviews + the knowledge graph."
    }
  ]);
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [darkMode, setDarkMode] = useState(true);
  const [updateModalOpen, setUpdateModalOpen] = useState(false);
  const [updateStatus, setUpdateStatus] = useState({ status: 'idle', message: '' });
  const [lastSuccessAt, setLastSuccessAt] = useState(null);
  const chatEndRef = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [messages, loading]);

  useEffect(() => {
    if (!updateModalOpen) return undefined;
    if (updateStatus.status !== 'pending') return undefined;

    const intervalId = setInterval(async () => {
      try {
        const res = await axios.get('/update-listings/status');
        const payload = res.data || {};
        setUpdateStatus({
          status: payload.status || 'idle',
          message: payload.message || ''
        });
        if (payload.last_success_at) {
          setLastSuccessAt(payload.last_success_at);
        }
      } catch (err) {
        console.error(err);
        setUpdateStatus({
          status: 'error',
          message: 'Unable to fetch update status.'
        });
      }
    }, 2500);

    return () => clearInterval(intervalId);
  }, [updateModalOpen, updateStatus.status]);

  useEffect(() => {
    const loadStatus = async () => {
      try {
        const res = await axios.get('/update-listings/status');
        const payload = res.data || {};
        if (payload.last_success_at) {
          setLastSuccessAt(payload.last_success_at);
        }
      } catch (err) {
        console.error(err);
      }
    };
    loadStatus();
  }, []);

  const sendMessage = async (content) => {
    if (!content.trim() || loading) return;
    setError('');
    setQuery('');
    setMessages((prev) => [...prev, { role: 'user', content }]);
    setLoading(true);
    try {
      const res = await axios.post('/ask', { query: content });
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: res.data?.answer || 'No response received.' }
      ]);
    } catch (err) {
      console.error(err);
      setError('Unable to reach the agent. Is the API running on localhost:8000?');
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: 'Sorry, I could not reach the agent right now.' }
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    sendMessage(query);
  };

  const handleUpdateListings = async () => {
    setUpdateModalOpen(true);
    setUpdateStatus({ status: 'pending', message: 'database update pending' });
    try {
      const res = await axios.post('/update-listings');
      const payload = res.data || {};
      setUpdateStatus({
        status: payload.status || 'pending',
        message: payload.message || 'database update pending'
      });
      if (payload.last_success_at) {
        setLastSuccessAt(payload.last_success_at);
      }
    } catch (err) {
      console.error(err);
      setUpdateStatus({
        status: 'error',
        message: 'Unable to start listings update.'
      });
    }
  };

  const lastUpdatedLabel = lastSuccessAt
    ? new Date(lastSuccessAt).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' })
    : 'never';

  return (
    <div className={`min-h-screen bg-shell text-ink relative overflow-hidden ${darkMode ? 'theme-dark' : ''}`}>
      <div className="hero-glow" aria-hidden="true" />
      <header className="relative z-10 mx-auto flex w-full max-w-5xl items-center justify-between px-5 py-5 md:px-10 md:py-6">
        <div className="flex items-center gap-3">
          <div className="logo-mark">AP</div>
          <div>
            <p className="text-sm uppercase tracking-[0.3em] text-ink-muted">Portugal</p>
            <h1 className="text-2xl font-semibold">Airbnb Atlas</h1>
          </div>
        </div>
        <div className="flex items-center gap-3 text-sm text-ink-subtle">
          <button
            className="pill"
            type="button"
            onClick={() => setDarkMode((prev) => !prev)}
            aria-pressed={darkMode}
          >
            {darkMode ? 'Light mode' : 'Dark mode'}
          </button>
          <button className="pill" type="button" onClick={handleUpdateListings}>
            Update listings
          </button>
          <span className="update-stamp" aria-live="polite">
            Last updated: {lastUpdatedLabel}
          </span>
        </div>
      </header>

      <main className="relative z-10 mx-auto grid w-full max-w-5xl grid-cols-1 gap-6 px-5 pb-12 md:px-10 lg:grid-cols-[1.2fr_0.8fr]">
        <section className="order-2 md:order-1">
          <DashboardPanel />
        </section>

        <section className="chat-panel order-1 md:order-2">
          <div className="flex items-center justify-between border-b border-sand px-6 py-4">
            <div>
              <p className="text-xs uppercase tracking-[0.3em] text-ink-muted">Ask the agent</p>
              <h3 className="text-lg font-semibold">Portugal insight chat</h3>
            </div>
            <span className="status-pill">{loading ? 'Searching' : 'Ready'}</span>
          </div>

          <div className="chat-body">
            {messages.map((message, index) => (
              <div
                key={`${message.role}-${index}`}
                className={`chat-bubble ${message.role === 'user' ? 'chat-bubble-user' : 'chat-bubble-agent'}`}
              >
                <p className="text-xs uppercase tracking-[0.3em] text-ink-muted">
                  {message.role === 'user' ? 'You' : 'Agent'}
                </p>
                <p className="mt-2 whitespace-pre-wrap text-sm">{message.content}</p>
              </div>
            ))}
            {loading && (
              <div className="chat-bubble chat-bubble-agent">
                <p className="text-xs uppercase tracking-[0.3em] text-ink-muted">Agent</p>
                <p className="mt-2 text-sm">Thinking through reviews + graph...</p>
              </div>
            )}
            <div ref={chatEndRef} />
          </div>

          <div className="chat-composer px-6 pb-5 pt-4">
            {error && <p className="mb-2 text-sm text-rose-500">{error}</p>}
            <div className="flex flex-wrap gap-2 pb-3">
              {STARTER_PROMPTS.map((prompt) => (
                <button
                  key={prompt}
                  type="button"
                  className="chip"
                  onClick={() => sendMessage(prompt)}
                >
                  {prompt}
                </button>
              ))}
            </div>
            <form onSubmit={handleSubmit} className="flex items-end gap-3">
              <textarea
                className="chat-input"
                rows="3"
                placeholder="Ask about neighborhoods, amenities, or price levels..."
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                required
              />
              <button className="send-btn" type="submit" disabled={loading}>
                Send
              </button>
            </form>
            <p className="mt-3 text-xs text-ink-muted">
              Powered by reviews in Qdrant and amenities in Neo4j.
            </p>
          </div>
        </section>
      </main>

      {updateModalOpen && (
        <div className="modal-overlay" role="dialog" aria-modal="true">
          <div className="modal-card">
            <button
              type="button"
              className="modal-close"
              onClick={() => setUpdateModalOpen(false)}
              aria-label="Close update status modal"
            >
              ×
            </button>
            <h3 className="text-xl font-semibold">Listings update</h3>
            <div className="modal-content-scroll mt-2">
              <p className="text-sm text-ink-subtle">
                {updateStatus.status === 'up_to_date' ? 'already up to date' : updateStatus.message}
              </p>
            </div>
            {updateStatus.status === 'pending' && (
              <p className="mt-2 text-xs text-ink-muted">database update pending</p>
            )}
            <div className="mt-5 flex justify-end">
              <button
                type="button"
                className="pill"
                onClick={() => setUpdateModalOpen(false)}
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
