import math
from typing import List
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn import preprocessing
import matplotlib.pyplot as plt
import tensorflow as tf
from keras._tf_keras.keras.layers import Dense, LSTM
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.callbacks import History

def download_stockprices(ticker="TSLA", period="10y", interval="1d"):
    data = yf.Ticker(ticker).history(period=period, interval=interval, actions=False)
    data.to_csv(f"{ticker}_historical_data.csv")

def get_stockprices_from_csv(ticker: str = "TSLA") -> pd.DataFrame:
    return pd.DataFrame(pd.read_csv(f"{ticker}_historical_data.csv"))

# ================================================================================

data = get_stockprices_from_csv()
data.set_index('Date', inplace=True)
data = data.astype(float)

data.drop(axis=1, columns=['High', 'Low', 'Volume', 'Open'], inplace=True)
data.dropna(inplace=True)

train_ratio = 0.8
train_size = int(len(data) * train_ratio)
train_data, test_data = data[:train_size], data[train_size:]

scaler = preprocessing.MinMaxScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

# LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

seq_length = 50
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

model = Sequential()
model.add(LSTM(units=128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(units=64, return_sequences=False))
model.add(Dense(units=1))
model.compile(loss='mean_squared_error', optimizer='adam')

history : History = model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=2)

# plt.plot(history.history['loss'])
# plt.title('Model Loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.show()

trainScore = model.evaluate(X_train, y_train, verbose=0)
print('Train score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
testScore = model.evaluate(X_test, y_test, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))

# Evaluating loss
loss = model.evaluate(X_test, y_test)
print(f"Loss: {loss}")

# Evaluating model
example = test_data[-seq_length:]
example = example.reshape((1, seq_length, 1))
prediction = model.predict(example)
print(f"Scaled predicted value: {prediction[0][0]}")
predicted_value = scaler.inverse_transform(prediction)
print(f"Predicted Value (Original Scale): {predicted_value[0][0]}")

# Predictions for future data
seed: List[float]  = test_data[-seq_length:].flatten().tolist()

num_predictions = 10

future_predictions = []
for i in range(num_predictions):
    x_input = np.array(seed[-seq_length:]).reshape((1, seq_length, 1))
    yhat = model.predict(x_input, verbose=0)[0][0]
    future_predictions.append(yhat)
    seed.append(yhat)

print("Future predictions (scaled):", future_predictions)

future_predictions_original = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
print("Future predictions (original scale):", future_predictions_original.flatten())

plt.plot(range(len(future_predictions_original)), future_predictions_original, label='Future Predictions')
plt.xlabel('Time Steps')
plt.ylabel('Stock Price')
plt.title('Future Stock Price Predictions')
plt.legend()
plt.show()