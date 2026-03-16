import yfinance as yf

# download Apple stock data
data = yf.download("AAPL", start="2015-01-01", end="2026-03-16")

# show first 5 rows
print(data.head())

# save dataset as CSV
data.to_csv("apple_stock.csv")

print("Dataset downloaded successfully!")


#THE PROCESSING PART
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# load dataset
data = pd.read_csv("apple_stock.csv")

# convert Close column to numeric
data["Close"] = pd.to_numeric(data["Close"], errors="coerce")

# remove missing values
data = data.dropna()

# take only Close price
close_prices = data["Close"].values.reshape(-1,1)

# scale data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(close_prices)

print("Scaled Data Sample:")
print(scaled_data[:5])


#THE SCALING PART 
import numpy as np

X = []
y = []

window_size = 60

for i in range(window_size, len(scaled_data)):
    X.append(scaled_data[i-window_size:i, 0])
    y.append(scaled_data[i, 0])

X = np.array(X)
y = np.array(y)

# reshape for LSTM
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

train_size = int(len(X) * 0.8)

X_train = X[:train_size]
X_test = X[train_size:]

y_train = y[:train_size]
y_test = y[train_size:]


#THE LSTM PART 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# build LSTM model
model = Sequential()

model.add(LSTM(units=50, return_sequences=False, input_shape=(X.shape[1],1)))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

print("Model created successfully")

model.fit(X_train, y_train, epochs=15, batch_size=32)

print("Training completed")

#THE PREDECTION PART
predictions = model.predict(X_test)

predictions = scaler.inverse_transform(predictions)
actual_prices = scaler.inverse_transform(y_test.reshape(-1,1))

#PLOTTING GRAPH
import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(go.Scatter(
    y=actual_prices.flatten(),
    mode='lines',
    name='Actual Price'
))

fig.add_trace(go.Scatter(
    y=predictions.flatten(),
    mode='lines',
    name='Predicted Price'
))

fig.update_layout(
    title="Apple Stock Price Prediction",
    xaxis_title="Time",
    yaxis_title="Stock Price"
)

fig.show()

from sklearn.metrics import mean_squared_error
import numpy as np

rmse = np.sqrt(mean_squared_error(actual_prices, predictions))
print("RMSE:", rmse)