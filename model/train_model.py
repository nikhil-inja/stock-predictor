# model/train_model.py
import yfinance as yf
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib
import warnings

warnings.filterwarnings('ignore')

# --- 1. Data Acquisition ---
print("Fetching data for PLTR...")
ticker = 'PLTR'
start_date = '2020-01-01'
end_date = '2023-12-31'
data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)

# --- 2. Feature Engineering ---
print("Engineering features...")
data['MA50'] = data['Close'].rolling(window=50).mean()
data['MA200'] = data['Close'].rolling(window=200).mean()

delta = data['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
data['RSI'] = 100 - (100 / (1 + rs))

# --- 3. Data Preparation ---
print("Preparing data...")
# Predicting the next day's 'Close' price
data['Next_Close'] = data['Close'].shift(-1)
data = data.dropna()

features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA50', 'MA200', 'RSI']
X = data[features]
y = data['Next_Close']

# Chronological split
split_index = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 4. Model Training ---
print("Training XGBoost model...")
model = xgb.XGBRegressor(objective='reg:squarederror',
                         n_estimators=1000,
                         learning_rate=0.05,
                         early_stopping_rounds=50,
                         n_jobs=-1)

model.fit(X_train_scaled, y_train,
          eval_set=[(X_test_scaled, y_test)],
          verbose=False)

# --- 5. Evaluation ---
predictions = model.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f"Model trained with RMSE: ${rmse:.2f}")

# --- 6. Save Artifacts ---
joblib.dump(model, "model/stock_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
print("Model and scaler have been saved successfully to the 'model/' directory.")