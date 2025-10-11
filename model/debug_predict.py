import joblib
import yfinance as yf
import pandas as pd
import numpy as np
import os
import traceback

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'stock_model.pkl')
scaler_path = os.path.join(BASE_DIR, 'scaler.pkl')

print('Loading model from', model_path)
model = joblib.load(model_path)
print('Loading scaler from', scaler_path)
scaler = joblib.load(scaler_path)

ticker = 'PLTR'
print('Downloading data for', ticker)
data = yf.download(ticker, period='1y', auto_adjust=True)
print('Downloaded rows:', len(data))

# Feature engineering
print('Computing features...')
data['MA50'] = data['Close'].rolling(window=50).mean()
data['MA200'] = data['Close'].rolling(window=200).mean()
delta = data['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
data['RSI'] = 100 - (100 / (1 + rs))

featured_data = data.dropna()
print('Rows after dropna:', len(featured_data))
if featured_data.empty:
    print('No featured data after dropna; aborting')
    raise SystemExit(1)

last_row = featured_data.iloc[[-1]]
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA50', 'MA200', 'RSI']
missing = [f for f in features if f not in last_row.columns]
print('Missing features:', missing)
X_pred = last_row[features]
print('X_pred shape:', X_pred.shape)
print('X_pred dtypes:\n', X_pred.dtypes)
print('Any NaNs:', X_pred.isnull().any().any())

try:
    X_pred_scaled = scaler.transform(X_pred)
    print('Scaled shape:', X_pred_scaled.shape)
    pred = model.predict(X_pred_scaled)
    print('Prediction:', pred)
except Exception:
    traceback.print_exc()
    raise
