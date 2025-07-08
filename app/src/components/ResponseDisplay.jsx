import React from 'react';

function ResponseDisplay({ response, loading }) {
  if (loading) {
    return <p className="mt-6 text-blue-600 font-medium animate-pulse">Thinking...</p>;
  }
  if (!response) return null;

  return (
    <div className="mt-6 w-full max-w-3xl bg-white border border-gray-200 rounded-md shadow p-4">
      <h2 className="text-lg font-semibold text-gray-700 mb-2">Answer:</h2>
      <p className="text-gray-800 whitespace-pre-wrap">{response.answer || response}</p>
    </div>
  );
}

export default ResponseDisplay;