import React, {useState} from 'react';
import axios from 'axios';

function QueryForm({ setResponse, setLoading }) {
  const [query, setQuery] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const res = await axios.post('http://localhost:8000/ask', { query });
      setResponse(res.data.answer);
    } catch (err) {
      console.error(err);
      setResponse({ answer: 'failed to fetch response'})
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="w-full max-w-xl flex flex-col gap-4">
      <textarea
        className="border border-gray-300 rounded-md p-3 text-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-400 resize-none"
        rows="3"
        placeholder="Ask a question like: Find a quiet place with good reviews"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        required
      />
      <button
        type="submit"
        className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded-md transition"
      > Ask Agent </button>
    </form>
  );
}


export default QueryForm