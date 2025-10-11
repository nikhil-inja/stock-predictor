from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import yfinance as yf
import pandas as pd
import numpy as np
import os
import traceback
import json

app = Flask(__name__)
CORS(app)

# --- Load Model and Scaler (Correctly pathed) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'model', 'stock_model.pkl')
scaler_path = os.path.join(BASE_DIR, 'model', 'scaler.pkl')

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Load optional metadata about the artifacts (features, trained ticker)
metadata = {}
metadata_path = os.path.join(BASE_DIR, 'model', 'metadata.json')
if os.path.exists(metadata_path):
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    except Exception:
        metadata = {}

# --- FIX: Consolidated single helper function ---
def get_prediction_data(ticker):
    """Fetches and engineers all data needed for prediction and the API response."""
    data = yf.download(ticker, period='1y', auto_adjust=True)
    if data.empty:
        return None

    # --- Feature Engineering ---
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Drop NaNs after all calculations to get a clean dataset
    featured_data = data.dropna()
    if featured_data.empty:
        return None

    return featured_data

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    ticker = data.get('ticker')
    if not ticker:
        return jsonify({'error': 'Ticker symbol not provided'}), 400
    # If metadata contains trained ticker, ensure user isn't asking for a different ticker
    trained_ticker = metadata.get('ticker')
    if trained_ticker and trained_ticker.upper() != ticker.upper():
        return jsonify({'error': f"Model was trained on '{trained_ticker}' - please retrain for '{ticker}' or use the matching model."}), 400
    
    try:
        # 1. Get the featured data using the helper function
        featured_data = get_prediction_data(ticker)
        if featured_data is None:
            return jsonify({'error': f'Could not fetch or process data for {ticker}.'}), 404
            
        # 2. Prepare the last row for prediction
        last_row = featured_data.iloc[[-1]]

        # --- FIX: Use the SAME 8 features the model was trained on ---
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA50', 'MA200', 'RSI']
        # Defensive: ensure all required features are present
        missing = [f for f in features if f not in last_row.columns]
        if missing:
            return jsonify({'error': f'Missing required feature(s): {missing}'}), 400

        X_pred = last_row[features]

        # Defensive: check for NaNs before scaling
        if X_pred.isnull().any().any():
            return jsonify({'error': 'Not enough historical data to compute required features (NaNs present).'}), 400

        # 3. Scale and predict
        # Diagnostic: print shapes and dtypes to help debug mismatches
        try:
            print('\n--- Predict diagnostics ---')
            print('X_pred shape:', X_pred.shape)
            print('X_pred dtypes:\n', X_pred.dtypes)
            if hasattr(scaler, 'n_features_in_'):
                print('scaler.n_features_in_ =', scaler.n_features_in_)
            if hasattr(model, 'n_features_in_'):
                print('model.n_features_in_ =', model.n_features_in_)

            X_pred_scaled = scaler.transform(X_pred)
            prediction = model.predict(X_pred_scaled)
        except Exception:
            # Print full traceback to server console for debugging and return 500
            traceback.print_exc()
            return jsonify({'error': 'Error during scaling/prediction. See server logs for details.'}), 500
        predicted_price = float(prediction[0])

        # 4. Prepare data for the API response
        latest_features_for_display = {
            "MA50": round(last_row['MA50'].values[0], 2),
            "MA200": round(last_row['MA200'].values[0], 2),
            "RSI": round(last_row['RSI'].values[0], 2)
        }
        
        historical_chart_data = featured_data['Close'].tail(90)

        # Defensive: ensure historical_chart_data is a Series (not a DataFrame)
        # (sometimes pandas operations can return a single-column DataFrame)
        if isinstance(historical_chart_data, pd.DataFrame):
            historical_chart_data = historical_chart_data.squeeze()

        # If it's still not a Series, coerce to one (flatten values)
        if not isinstance(historical_chart_data, pd.Series):
            historical_chart_data = pd.Series(historical_chart_data.values.flatten())

        # Defensive: safe conversion to lists for JSON
        if historical_chart_data.empty:
            labels = []
            values = []
        else:
            labels = historical_chart_data.index.strftime('%Y-%m-%d').tolist()
            values = historical_chart_data.round(2).tolist()

        # --- FIX: Return the JSON structure the frontend expects ---
        return jsonify({
            'ticker': ticker.upper(),
            'predicted_next_day_price': round(predicted_price, 2),
            'latest_features': latest_features_for_display,
            'historical_prices': {
                'labels': labels,
                'values': values
            }
        })
        
    except Exception as e:
        # Print full traceback so you can inspect the server console and find the root cause
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/predict_debug', methods=['POST'])
def predict_debug():
    """Return diagnostic information about the inputs, scaler and model for a given ticker."""
    data = request.get_json()
    ticker = data.get('ticker')
    if not ticker:
        return jsonify({'error': 'Ticker symbol not provided'}), 400

    try:
        featured_data = get_prediction_data(ticker)
        if featured_data is None:
            return jsonify({'error': f'Could not fetch or process data for {ticker}.'}), 404

        last_row = featured_data.iloc[[-1]]
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA50', 'MA200', 'RSI']
        missing = [f for f in features if f not in last_row.columns]
        if missing:
            return jsonify({'error': f'Missing required feature(s): {missing}'}), 400

        X_pred = last_row[features]
        if X_pred.isnull().any().any():
            return jsonify({'error': 'Not enough historical data to compute required features (NaNs present).'}), 400

        # Build serializable output
        out = {'ticker': ticker}
        # include metadata if present
        if metadata:
            out['metadata'] = metadata
        # original features (convert numpy/pandas types)
        serial_X = {}
        for k, v in X_pred.iloc[0].items():
            # convert numpy types to python
            try:
                serial_X[str(k)] = float(v) if np.isscalar(v) else (v.tolist() if hasattr(v, 'tolist') else str(v))
            except Exception:
                serial_X[str(k)] = str(v)
        out['X_pred'] = serial_X

        # scaler diagnostics
        try:
            X_pred_scaled = scaler.transform(X_pred)
            out['X_pred_scaled'] = [float(x) for x in X_pred_scaled.flatten().tolist()]
        except Exception as e:
            out['scale_error'] = str(e)

        if hasattr(scaler, 'mean_'):
            out['scaler_mean'] = [float(x) for x in scaler.mean_.tolist()]
        if hasattr(scaler, 'scale_'):
            out['scaler_scale'] = [float(x) for x in scaler.scale_.tolist()]
        if hasattr(scaler, 'n_features_in_'):
            out['scaler_n_features_in'] = int(scaler.n_features_in_)

        # model diagnostics and prediction
        try:
            pred = model.predict(X_pred_scaled)
            out['prediction_raw'] = float(pred[0])
        except Exception as e:
            out['predict_error'] = str(e)

        if hasattr(model, 'n_features_in_'):
            out['model_n_features_in'] = int(model.n_features_in_)
        # model params (short)
        try:
            out['model_params'] = {k: v for k, v in model.get_params().items() if k in ['n_estimators','learning_rate','objective']}
        except Exception:
            out['model_params'] = {}

        return jsonify(out)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)