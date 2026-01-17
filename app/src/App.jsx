import React, { useEffect, useRef, useState } from 'react';
import axios from 'axios';

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
  const [darkMode, setDarkMode] = useState(false);
  const chatEndRef = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [messages, loading]);

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

  return (
    <div className={`min-h-screen bg-shell text-ink relative overflow-hidden ${darkMode ? 'theme-dark' : ''}`}>
      <div className="hero-glow" aria-hidden="true" />
      <header className="relative z-10 mx-auto flex w-full max-w-6xl items-center justify-between px-6 py-6">
        <div className="flex items-center gap-3">
          <div className="logo-mark">AP</div>
          <div>
            <p className="text-sm uppercase tracking-[0.3em] text-ink-muted">Portugal</p>
            <h1 className="text-2xl font-semibold">Airbnb Atlas</h1>
          </div>
        </div>
        <div className="flex items-center gap-3 text-sm text-ink-subtle">
          <div className="hidden items-center gap-6 md:flex">
            <span>Lisbon</span>
            <span>Porto</span>
            <span>Coastal</span>
            <button className="pill">Explore map</button>
          </div>
          <button
            className="pill"
            type="button"
            onClick={() => setDarkMode((prev) => !prev)}
            aria-pressed={darkMode}
          >
            {darkMode ? 'Light mode' : 'Dark mode'}
          </button>
        </div>
      </header>

      <main className="relative z-10 mx-auto grid w-full max-w-6xl grid-cols-1 gap-6 px-6 pb-12 md:grid-cols-[1.1fr_0.9fr]">
        <section className="rounded-3xl border border-sand bg-white/80 p-6 shadow-soft backdrop-blur">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <p className="text-sm uppercase tracking-[0.3em] text-ink-muted">Featured stays</p>
              <h2 className="mt-2 text-3xl font-semibold">Airbnb Portugal, simplified.</h2>
            </div>
            <div className="flex gap-2">
              <span className="badge">New: Reviews + KG</span>
              <span className="badge badge-muted">Qdrant</span>
              <span className="badge badge-muted">Neo4j</span>
            </div>
          </div>

          <div className="mt-6 grid gap-4 sm:grid-cols-2">
            <article className="listing-card">
              <div className="listing-art lisboa" />
              <div className="mt-4">
                <p className="text-xs uppercase tracking-[0.3em] text-ink-muted">Lisbon</p>
                <h3 className="mt-2 text-lg font-semibold">Alfama Rooftop Hideaway</h3>
                <p className="mt-2 text-sm text-ink-subtle">
                  Walkable alleys, sunset terraces, and quiet mornings near the river.
                </p>
                <div className="mt-4 flex flex-wrap gap-2 text-xs text-ink-subtle">
                  <span className="tag">Terrace</span>
                  <span className="tag">Historic</span>
                  <span className="tag">Coffee nearby</span>
                </div>
              </div>
            </article>

            <article className="listing-card">
              <div className="listing-art porto" />
              <div className="mt-4">
                <p className="text-xs uppercase tracking-[0.3em] text-ink-muted">Porto</p>
                <h3 className="mt-2 text-lg font-semibold">Ribeira Brick Loft</h3>
                <p className="mt-2 text-sm text-ink-subtle">
                  Riverfront views with polished wood interiors and late-night vinho.
                </p>
                <div className="mt-4 flex flex-wrap gap-2 text-xs text-ink-subtle">
                  <span className="tag">River view</span>
                  <span className="tag">Loft</span>
                  <span className="tag">Local eats</span>
                </div>
              </div>
            </article>
          </div>

          <div className="mt-6 grid gap-4 sm:grid-cols-3">
            <div className="stat-card">
              <p className="text-xs uppercase tracking-[0.3em] text-ink-muted">Coverage</p>
              <h4 className="mt-2 text-2xl font-semibold">2 cities</h4>
              <p className="mt-1 text-sm text-ink-subtle">Lisbon + Porto listings</p>
            </div>
            <div className="stat-card">
              <p className="text-xs uppercase tracking-[0.3em] text-ink-muted">Signals</p>
              <h4 className="mt-2 text-2xl font-semibold">Amenities</h4>
              <p className="mt-1 text-sm text-ink-subtle">Pool, wifi, view, vibe</p>
            </div>
            <div className="stat-card">
              <p className="text-xs uppercase tracking-[0.3em] text-ink-muted">Mode</p>
              <h4 className="mt-2 text-2xl font-semibold">Conversational</h4>
              <p className="mt-1 text-sm text-ink-subtle">Ask in natural language</p>
            </div>
          </div>
        </section>

        <section className="chat-panel">
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

          <div className="px-6 pb-5 pt-4">
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
    </div>
  );
}

export default App;
