from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import yfinance as yf
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

model = joblib.load('model/stock_model.pkl')
scaler = joblib.load('model/scaler.pkl')

def engineer_features(data):
    """A helper function to create features for the model."""
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()
    
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    return data.dropna()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    ticker = data.get('ticker')
    
    if not ticker:
        return jsonify({'error': 'Ticker symbol not provided'}), 400
    
    try:
        stock_data = yf.download(ticker, period='1y', interval='1d')
        
        if stock_data.empty:
            return jsonify({'error': 'Could not fetch data for the ticker'}), 404
            
        featured_data = engineer_features(stock_data.copy())

        last_row = featured_data.iloc[[-1]]

        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA50', 'MA200', 'RSI']
        X_pred = last_row[features]

        X_pred_scaled = scaler.transform(X_pred)

        prediction = model.predict(X_pred_scaled)
        predicted_price = float(prediction[0])
        
        return jsonify({'ticker': ticker, 'predicted_next_close': round(predicted_price, 2)})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)