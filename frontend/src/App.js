import React, { useState } from 'react';
import StockChart from './StockChart';
import './App.css';

const API_URL = 'http://127.0.0.1:5000/predict';

function App() {
  const [ticker, setTicker] = useState('');
  const [predictionData, setPredictionData] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (event) => {
    event.preventDefault(); // Prevents the browser from reloading the page
    setLoading(true);
    setPredictionData(null);
    setError('');

    try {
      const response = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ticker: ticker }),
      });
      const data = await response.json();
      console.log('Prediction API response:', data);
      if (!response.ok) throw new Error(data.error || 'Prediction failed');
      setPredictionData(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>ML Stock Predictor</h1>

        {/* --- FIX: Using a single, clean form --- */}
        <form onSubmit={handleSubmit} className="input-container">
          <input
            type="text"
            value={ticker}
            onChange={(e) => setTicker(e.target.value.toUpperCase())}
            placeholder="Enter Stock Ticker (e.g., AAPL)"
            required
          />
          <button type="submit" disabled={loading}>
            {loading ? 'Predicting...' : 'Predict'}
          </button>
        </form>
      </header> {/* --- FIX: Properly closed header --- */}

      <main className="App-content">
        {/* --- FIX: Display error message cleanly --- */}
        {error && <div className="error"><p>{error}</p></div>}

        {/* --- FIX: Using the correct state variable 'predictionData' --- */}
        {predictionData && (
          <div className="content-container">
            <div className="result-box">
              <h2>Prediction: {predictionData.ticker}</h2>
              <p>Predicted Next Day Price: <strong>${predictionData?.predicted_next_day_price ?? 'N/A'}</strong></p>
              
              <h3>Model Inputs (Latest Data)</h3>
              <ul>
                <li>50-Day Moving Avg: {predictionData?.latest_features?.MA50}</li>
                <li>200-Day Moving Avg: {predictionData?.latest_features?.MA200}</li>
                <li>Relative Strength Index: {predictionData?.latest_features?.RSI}</li>
              </ul>
            </div>
            <div className="chart-container">
              {predictionData?.historical_prices?.values && predictionData.historical_prices.values.length > 0 ? (
                <StockChart
                  historicalData={predictionData.historical_prices}
                  prediction={predictionData.predicted_next_day_price}
                />
              ) : (
                <div style={{padding: '1rem'}}>No chart data available for this ticker.</div>
              )}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;