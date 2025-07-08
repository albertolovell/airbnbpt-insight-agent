import React, { useState } from 'react';
import QueryForm from './components/QueryForm';
import ResponseDisplay from './components/ResponseDisplay';


function App() {
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);

  return (
    <div className="min-h-screen bg-gray-50 px-4 py-8 flex flex-col items-center">
      <h1 className="text=3xl font-bold mb-6 text-gray-800">Airbnb Insight Agent</h1>
      <QueryForm setResponse={setResponse} setLoading={setLoading} />
      <ResponseDisplay response={response} loading={loading} />
    </div>
  );
}

export default App;
