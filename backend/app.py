from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

app = Flask(__name__)
CORS(app)

def get_stock_data(stock_symbol):
    data = yf.download(stock_symbol, start='2010-01-01', end='2023-08-06')
    data = data[['Close']]
    return data

def train_lstm_model(data):
    dataset = data.values
    training_data_len = int(np.ceil(len(dataset) * 0.8))
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    train_data = scaled_data[0:int(training_data_len), :]
    x_train = []
    y_train = []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=1)
    return model, scaler, training_data_len

def make_predictions(model, scaler, training_data_len, data):
    dataset = data.values
    scaled_data = scaler.transform(dataset)
    test_data = scaled_data[training_data_len - 60:, :]
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    return predictions, y_test

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        data = request.get_json()
        stock_symbol = data.get('stock_symbol', 'AAPL')
        # Your prediction logic here
        return jsonify({"message": f"Prediction for {stock_symbol}"})
    else:
        return jsonify({"message": "Use POST method with a stock symbol to get predictions."})


if __name__ == '__main__':
    app.run(debug=True)
