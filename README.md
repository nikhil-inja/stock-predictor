# ML Stock Predictor

A full-stack machine learning application that predicts next-day stock prices using XGBoost regression. The application features a React frontend with interactive charts and a Flask backend that serves trained ML models.

## 🏗️ Architecture

This is a full-stack application with:
- **Backend**: Flask API server with XGBoost ML model
- **Frontend**: React.js web application with Chart.js visualizations
- **ML Pipeline**: Python-based training and prediction system

## 🚀 Features

- **Real-time Stock Predictions**: Enter any stock ticker to get next-day price predictions
- **Interactive Charts**: Visualize 90-day historical data with prediction overlay
- **Technical Indicators**: Model uses multiple technical indicators including:
  - 50-day and 200-day moving averages
  - Relative Strength Index (RSI)
  - OHLCV (Open, High, Low, Close, Volume) data
- **Model Insights**: View the latest technical indicators used for prediction
- **Error Handling**: Comprehensive error handling for invalid tickers or data issues

## 🛠️ Tech Stack

### Backend
- **Flask**: Web framework for the API server
- **XGBoost**: Gradient boosting machine learning model
- **scikit-learn**: Data preprocessing and scaling
- **yfinance**: Yahoo Finance API for stock data
- **pandas/numpy**: Data manipulation and analysis
- **joblib**: Model serialization

### Frontend
- **React.js**: Frontend framework
- **Chart.js**: Interactive data visualization
- **react-chartjs-2**: React wrapper for Chart.js

### Deployment
- **Gunicorn**: WSGI server for production deployment
- **Heroku**: Cloud platform (configured via Procfile)

## 📁 Project Structure

```
stock-predictor/
├── app.py                 # Flask backend server
├── requirements.txt       # Python dependencies
├── Procfile              # Heroku deployment configuration
├── model/
│   ├── train_model.py    # ML model training script
│   ├── stock_model.pkl   # Trained XGBoost model
│   ├── scaler.pkl        # Feature scaler
│   └── metadata.json     # Model metadata
└── frontend/
    ├── package.json      # Node.js dependencies
    ├── src/
    │   ├── App.js        # Main React component
    │   └── StockChart.js # Chart visualization component
    └── public/           # Static assets
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Node.js 14+
- npm or yarn

### Backend Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd stock-predictor
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the model** (optional - pre-trained model included)
   ```bash
   cd model
   python train_model.py [TICKER]  # e.g., python train_model.py AAPL
   ```

5. **Start the Flask server**
   ```bash
   python app.py
   ```
   The API will be available at `http://localhost:5000`

### Frontend Setup

1. **Navigate to frontend directory**
   ```bash
   cd frontend
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Start the development server**
   ```bash
   npm start
   ```
   The app will open at `http://localhost:3000`

## 🔧 API Endpoints

### POST `/predict`
Predicts the next day's stock price for a given ticker.

**Request:**
```json
{
  "ticker": "AAPL"
}
```

**Response:**
```json
{
  "ticker": "AAPL",
  "predicted_next_day_price": 185.42,
  "latest_features": {
    "MA50": 180.25,
    "MA200": 175.80,
    "RSI": 65.3
  },
  "historical_prices": {
    "labels": ["2024-01-01", "2024-01-02", ...],
    "values": [175.20, 176.50, ...]
  }
}
```

### POST `/predict_debug`
Returns diagnostic information about model inputs and processing.

## 🎯 How It Works

1. **Data Collection**: Fetches 1 year of historical stock data using yfinance
2. **Feature Engineering**: Calculates technical indicators (MA50, MA200, RSI)
3. **Data Preprocessing**: Scales features using StandardScaler
4. **Prediction**: Uses trained XGBoost model to predict next day's closing price
5. **Visualization**: Frontend displays historical data and prediction on interactive chart

## 🤖 Machine Learning Model

- **Algorithm**: XGBoost Regressor
- **Features**: 8 technical indicators (OHLCV + MA50 + MA200 + RSI)
- **Training Period**: 2020-2023 data (configurable)
- **Validation**: 80/20 chronological split
- **Metrics**: RMSE evaluation

## 🚀 Deployment

### Heroku Deployment

1. **Install Heroku CLI**
2. **Login and create app**
   ```bash
   heroku login
   heroku create your-app-name
   ```

3. **Deploy**
   ```bash
   git add .
   git commit -m "Deploy stock predictor"
   git push heroku main
   ```

The Procfile is configured to run the Flask app with Gunicorn.

## 📊 Model Performance

The model uses XGBoost with the following configuration:
- **Learning Rate**: 0.05
- **N Estimators**: 1000
- **Early Stopping**: 50 rounds
- **Objective**: reg:squarederror

Performance metrics are displayed during training and can be found in the server logs.

## ⚠️ Important Notes

- **Model Specificity**: The current model is trained on a specific ticker. To predict other stocks, retrain the model with their data
- **Data Limitations**: Predictions are based on historical patterns and may not account for unexpected market events
- **Disclaimer**: This is for educational purposes only. Do not use for actual trading decisions

## 🔍 Troubleshooting

### Common Issues

1. **Model not found error**: Ensure `stock_model.pkl` and `scaler.pkl` exist in the `model/` directory
2. **CORS errors**: Make sure Flask-CORS is installed and configured
3. **Chart not displaying**: Check that Chart.js dependencies are properly installed
4. **Invalid ticker**: Verify the stock symbol exists on Yahoo Finance

### Debug Mode

Use the `/predict_debug` endpoint to inspect model inputs and identify issues with feature scaling or data processing.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

**Disclaimer**: This application is for educational and research purposes only. Stock market predictions are inherently uncertain and should not be used as the sole basis for investment decisions.
