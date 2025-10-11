import joblib
import numpy as np
import os

BASE = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE, 'stock_model.pkl'))
scaler = joblib.load(os.path.join(BASE, 'scaler.pkl'))

# synthetic features roughly matching AAPL magnitudes
x = np.array([[254.86, 255.92, 253.11, 254.63, 37666900, 230.09, 221.64, 82.77]])
print('raw x:', x)

try:
    xs = scaler.transform(x)
    print('scaled x:', xs)
    pred = model.predict(xs)
    print('raw pred:', pred)
except Exception as e:
    print('error during predict:', e)
    raise
