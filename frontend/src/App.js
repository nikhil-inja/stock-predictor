import React, { useState } from 'react';
import './App.css';

function App() {
  const [ticker, setTicker] = useState('');
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handlePredict = async () => {
    setLoading(true);
    setError('');
    setPrediction(null);

    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ ticker: ticker.toUpperCase() }),
      });

      const data = await response.json();

      if (response.ok) {
        setPrediction(data);
      } else {
        setError(data.error || 'An error occurred.');
      }
    } catch (err) {
      setError('Failed to connect to the server.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>ML Stock Predictor</h1>
        <div className="input-container">
          <input
            type="text"
            value={ticker}
            onChange={(e) => setTicker(e.target.value)}
            placeholder="Enter stock ticker (e.g., AAPL)"
          />
          <button onClick={handlePredict} disabled={loading}>
            {loading ? 'Predicting...' : 'Predict'}
          </button>
        </div>
        {prediction && (
          <div className="result">
            <p>Predicted Next Close for {prediction.ticker}:</p>
            <h2>${prediction.predicted_next_close}</h2>
          </div>
        )}
        {error && <p className="error">{error}</p>}
      </header>
    </div>
  );
}

export default App;