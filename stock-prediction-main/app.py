from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import yfinance as yf
from flask import Flask, render_template, request
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Function to get historical trading data
def get_historical_data(stock_symbol, start_date, end_date):
    try:
        data = yf.download(stock_symbol, start=start_date, end=end_date)
        if not data.empty:
            return data
    except Exception as e:
        print(f"Error fetching data: {e}")
    return None

# Load your trained model (update with the correct model path)
model = load_model(r'C:\Users\ashis\OneDrive\Desktop\stocknew2\stock_predict_new1.keras')

# Route for home page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        stock_symbol = request.form['stock_symbol']

        # Set the date range for fetching data
        start_date = '2010-01-01'  # Fetch from 2010 to ensure enough data
        end_date = datetime.today().strftime('%Y-%m-%d')

        # Fetch historical data
        data = get_historical_data(stock_symbol, start_date, end_date)
        if data is None or data.empty:
            return render_template('index.html', error="Invalid stock symbol or no data found.", zip=zip)

        # Prepare data for training and testing
        data_train = pd.DataFrame(data['Close'][0: int(len(data) * 0.80)])
        data_test = pd.DataFrame(data['Close'][int(len(data) * 0.80): len(data)])
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_train_scaled = scaler.fit_transform(data_train)

        # Create training sequences
        x_train, y_train = [], []
        for i in range(100, data_train_scaled.shape[0]):
            x_train.append(data_train_scaled[i - 100:i])
            y_train.append(data_train_scaled[i, 0])
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        # Prepare last 100 days of data from the test set for prediction
        past_100_days = data_test.tail(100)
        final_data = past_100_days['Close'].values.reshape(-1, 1)
        final_data_scaled = scaler.transform(final_data)

        # Predict the next 5 days
        predictions = []
        input_data = final_data_scaled
        for _ in range(5):
            input_data = np.array(input_data).reshape(1, 100, 1)
            pred = model.predict(input_data)
            predictions.append(pred[0, 0])
            input_data = np.append(input_data[0, 1:], pred[0, 0])
            input_data = input_data.reshape(100, 1)

        # Rescale predictions back to original scale
        predictions = np.array(predictions).reshape(-1, 1)
        predictions_rescaled = scaler.inverse_transform(predictions)

        # Create future dates for the next 5 days
        last_date = data.index[-1]
        future_dates = [last_date + timedelta(days=i) for i in range(1, 6)]

        # Plot past trend and future prediction using Plotly
        past_trace = go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Past Prices')
        future_trace = go.Scatter(x=future_dates, y=predictions_rescaled.flatten(), mode='lines+markers',
                                  name='Predicted Prices (Next 5 Days)', line=dict(color='red'))

        layout = go.Layout(
            title=f'{stock_symbol} Stock Price: Past Trend and Future Prediction',
            xaxis={'title': 'Date'},
            yaxis={'title': 'Price'},
            hovermode='x'
        )

        fig = go.Figure(data=[past_trace, future_trace], layout=layout)
        graph1 = fig.to_html(full_html=False)

        # Second graph: Detailed past 1 month with predictions
        past_1_month = data.tail(30)
        past_trace_month = go.Scatter(x=past_1_month.index, y=past_1_month['Close'], mode='lines', name='Past Month Prices')
        future_trace_month = go.Scatter(x=future_dates, y=predictions_rescaled.flatten(), mode='lines+markers',
                                        name='Predicted Prices (Next 5 Days)', line=dict(color='green'))

        layout_month = go.Layout(
            title=f'{stock_symbol} Stock Price: Past Month and Future Prediction',
            xaxis={'title': 'Date'},
            yaxis={'title': 'Price'},
            hovermode='x'
        )

        fig_month = go.Figure(data=[past_trace_month, future_trace_month], layout=layout_month)
        graph2 = fig_month.to_html(full_html=False)

        # Get the current stock price (latest closing price) and round to 3 decimals
        current_price = round(data['Close'].iloc[-1], 3)

        # Create future dates for the next 5 days in 'dd-mm-yyyy' format
        last_date = data.index[-1]
        future_dates = [(last_date + timedelta(days=i)).strftime('%d-%m-%Y') for i in range(1, 6)]

        # Round the predictions to 3 decimal places
        predictions_rescaled = [round(pred[0], 3) for pred in predictions_rescaled]

        # Return template with graphs and predictions
        return render_template('index.html', stock_symbol=stock_symbol, graph1=graph1, graph2=graph2, current_price=current_price,predictions=predictions_rescaled, future_dates=future_dates,
                               zip=zip)

    return render_template('index.html', zip=zip)

if __name__ == '__main__':
    app.run(debug=True)
