"""
stock_predictors

This module provides tools for predicting stock prices using machine learning and for
customizing predictions based on option trading strategies.

Classes:
    - StockPricePredictor: Basic stock price predictor using a machine learning model.
    - StockCustomPredictor: Extended predictor with support for options trading parameters.

Typical usage:
    from stock_predictors import StockPricePredictor, StockCustomPredictor

    basic_predictor = StockPricePredictor(symbol="AAPL", prediction_days=30)
    custom_predictor = StockCustomPredictor(
        symbol="AAPL",
        formula="lstm",
        option_type="call",
        strike_value=150,
        expiration_days=45
    )
"""

import os, joblib
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from alpha_vantage.timeseries import TimeSeries
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

class StockPricePredictor:
    """
        A basic stock price predictor using a machine learning model.

        This class sets up the environment to predict future stock prices based on
        historical data using a scalable preprocessing pipeline.

        Attributes:
            symbol (str): The stock symbol to be analyzed.
            prediction_days (int): Number of days to predict into the future.
            model (Any): Placeholder for the trained prediction model (e.g., LSTM).
            scaler (MinMaxScaler): Scaler for normalizing input data for the model.
    """
    def __init__(self, symbol, prediction_days=60):
        """
        Initializes the stock price predictor with symbol and prediction settings.

        Args:
            symbol (str): The stock ticker symbol (e.g., "AAPL").
            prediction_days (int, optional): Number of days to predict into the future (default is 60).
        """
        self.symbol = symbol
        self.prediction_days = prediction_days
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def fetch_data(self, start_date='2020-01-01'):
        """Fetch historical stock data using yfinance"""
        # start_date = date.today() - timedelta(weeks=104)
        # print("Start Date", start_date)
        stock = yf.Ticker(self.symbol)
        # df = stock.history(start=start_date)
        df = stock.history(period='5y')
        return df
    
    def prepare_data(self, data):
        """Prepare data for LSTM model"""
        # Scale the data
        scaled_data = self.scaler.fit_transform(data['Close'].values.reshape(-1, 1))
        
        x_train, y_train = [], []
        
        for x in range(self.prediction_days, len(scaled_data)):
            x_train.append(scaled_data[x-self.prediction_days:x, 0])
            y_train.append(scaled_data[x, 0])
            
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        
        return x_train, y_train
    
    def build_model(self, input_shape):
        """Build LSTM model"""
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))
        
        # Use mixed precision training for better GPU performance
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        
        # Configure optimizer with loss scaling for mixed precision training
        optimizer = tf.keras.optimizers.Adam()
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        return model
    
    def train(self, epochs=25, batch_size=32, save_path='models'):
        """Train the model and save it"""
        # Fetch and prepare data
        data = self.fetch_data()
        x_train, y_train = self.prepare_data(data)
        
        # Build and train model
        self.model = self.build_model((x_train.shape[1], 1))
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
        
        # Make predictions on training data for evaluation
        predictions = self.model.predict(x_train)
        
        # Inverse scale predictions and actual values
        predictions = self.scaler.inverse_transform(predictions)
        actual = self.scaler.inverse_transform(y_train.reshape(-1, 1))

        # Evaluation metrics
        mse = mean_squared_error(actual, predictions)
        mae = mean_absolute_error(actual, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(actual, predictions)

        print("\nModel Evaluation on Training Data:")
        print(f"  Mean Absolute Error (MAE): {mae:.4f}")
        print(f"  Mean Squared Error (MSE): {mse:.4f}")
        print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"  R² Score: {r2:.4f}")

        # Save the model and scaler
        self.save_model(save_path)
        return self.model
        
    def save_model(self, save_path='models'):
        """Save the model and scaler"""
        os.makedirs(save_path, exist_ok=True)
        
        # Save Keras model
        self.model.save(f'{save_path}/{self.symbol}_predictor.h5')
        
        # Save scaler using joblib
        joblib.dump(self.scaler, f'{save_path}/{self.symbol}_scaler.pkl')
        
    def load_model(self, load_path='models'):
        """Load the saved model and scaler"""
        # Load Keras model
        self.model = tf.keras.models.load_model(f'{load_path}/{self.symbol}_predictor.h5')
        
        # Load scaler
        self.scaler = joblib.load(f'{load_path}/{self.symbol}_scaler.pkl')
    
    def predict_next_day(self):
        """Predict the next day's price"""
        # Fetch the most recent data
        data = self.fetch_data()
        
        # Prepare the data for prediction
        scaled_data = self.scaler.transform(data['Close'].values.reshape(-1, 1))
        x_test = []
        x_test.append(scaled_data[-self.prediction_days:, 0])
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        
        # Make prediction
        pred = self.model.predict(x_test)
        pred = self.scaler.inverse_transform(pred)
        
        return pred[0][0]
    
    def predict_future_prices(self, days=1):
        """
        Predict the next 'days' number of prices (e.g., 1, 7, 15)
        """
        # Fetch and scale the data
        data = self.fetch_data()
        scaled_data = self.scaler.transform(data['Close'].values.reshape(-1, 1))
        
        # Prepare the last sequence used for prediction
        last_sequence = scaled_data[-self.prediction_days:].reshape(1, self.prediction_days, 1)
        
        predictions = []

        for _ in range(days):
            # Predict the next price
            pred = self.model.predict(last_sequence)
            predictions.append(pred[0][0])

            # Append prediction to the sequence and slide window
            new_input = np.append(last_sequence[0, 1:, 0], pred[0][0])
            last_sequence = new_input.reshape(1, self.prediction_days, 1)

        # Inverse transform predictions back to original scale
        predictions = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

        return predictions.flatten()

class StockCustomPredictor:
    """
        A custom predictor for forecasting stock prices and evaluating options strategies.

        This class is designed to customize stock prediction based on options trading strategies,
        such as call or put options, and allows flexible model selection through the formula parameter.

        Attributes:
            stock_symbol (str): Ticker symbol of the stock to be analyzed.
            formula (str): The predictive formula or algorithm to be used (e.g., "linear_regression", "lstm").
            option_type (str): Type of the option ("call" or "put").
            strike_value (float): The strike price of the option contract.
            expiration_days (int): Number of days until the option expires.
            days_to_predict (int): Number of future days to predict (default is 60).
            start_date (str): Historical data start date in "YYYY-MM-DD" format (overridden to "2024-01-01").
            end_date (str): Historical data end date in "YYYY-MM-DD" format (overridden to "2025-01-01").
    """
    def __init__(self, symbol, formula, option_type, strike_value, expiration_days, days_to_predict=60, start_date="2022-01-01", end_date="2024-12-31"):
        """
            Initializes the StockCustomPredictor with stock and option configuration.

            Args:
                symbol (str): Ticker symbol of the stock to analyze.
                formula (str): Name of the prediction algorithm or model.
                option_type (str): Type of the option ("call" or "put").
                strike_value (float): Strike price for the option.
                expiration_days (int): Days until option expiration.
                days_to_predict (int, optional): Number of days into the future to predict. Default is 60.
                start_date (str, optional): Start date for historical data analysis (ignored, overridden internally).
                end_date (str, optional): End date for historical data analysis (ignored, overridden internally).

            Note:
                The start_date and end_date parameters are hardcoded inside the class to:
                - start_date = "2024-01-01"
                - end_date = "2025-01-01"
        """
        self.stock_symbol = symbol
        self.days_to_predict = days_to_predict
        self.start_date =  start_date
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
        end_date_plus_one = end_date_obj + timedelta(days=1)
        self.end_date = end_date_plus_one.strftime("%Y-%m-%d")
        self.formula = formula
        self.option_type = option_type
        self.strike_value = strike_value
        self.expiration_days = expiration_days
        self.model = None
        self.stock_data = self.fetch_data()
        self.future_price = None
        self.greeks = None
        
    def fetch_data(self):
        """
            Fetch historical stock data using yfinance
            :Parameters:
                tickers : str, list
                    List of tickers to download
                start: str
                    Download start date string (YYYY-MM-DD) or _datetime, inclusive.
                    Default is 99 years ago
                    E.g. for start="2020-01-01", the first data point will be on "2020-01-01"
                end: str
                    Download end date string (YYYY-MM-DD) or _datetime, exclusive.
                    Default is now
                    E.g. for end="2023-01-01", the last data point will be on "2022-12-31"
        """
        # start_date = date.today() - timedelta(weeks=104)
        # print("Start Date", start_date)
        
        return yf.download(self.stock_symbol, start=self.start_date, end=self.end_date)

    def calculate_black_scholes(self):
        stock_returns = np.log(self.stock_data['Close'] / self.stock_data['Close'].shift(1))
        volatility = (stock_returns.std() * np.sqrt(252)).item()
        delta = (stock_returns.mean() * 252).item()
        sigma = volatility
        theta = -0.5 * sigma ** 2
        vega = delta * sigma

        greeks = {
            "Delta": delta,
            "Theta": theta,
            "Vega": vega,
            "Sigma": sigma
        }

        if self.days_to_predict is None:
            return greeks

        df = self.stock_data.reset_index()
        df['Date'] = pd.to_datetime(df['Date'])
        df['Days'] = (df['Date'] - df['Date'].min()).dt.days
        X = df[['Days']]
        y = df[['Close']]

        n_samples = len(X)
        if n_samples <= 1:
            raise ValueError("Not enough data samples for prediction.")

        test_size = 0.2 if n_samples > 5 else 1 / n_samples

        X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=test_size, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)

        future_day = df['Days'].iloc[-1] + self.days_to_predict
        future_price = model.predict([[future_day]])
        return future_price[0][0], greeks

    def calculate_jump_dufuison(self):
        stock_returns = np.log(self.stock_data['Close'] / self.stock_data['Close'].shift(1))
        volatility = (stock_returns.std() * np.sqrt(252)).item()
        delta = (stock_returns.mean() * 252).item()
        sigma = volatility
        theta = -0.5 * sigma ** 2
        vega = delta * sigma

        greeks = {
            "Delta": delta,
            "Theta": theta,
            "Vega": vega,
            "Sigma": sigma
        }

        S0 = self.stock_data['Close'].iloc[-1]
        mu = self.stock_data['Close'].pct_change().mean().item()
        sigma = self.stock_data['Close'].pct_change().std().item()
        lamb = 0.01
        m = 0.2
        v = np.std(m)
        T = self.days_to_predict
        dt = 1

        N = int(T/dt)
        price_path = np.zeros(N)
        price_path[0] = S0

        for t in range(1, N):
            W = np.sqrt(dt) * np.random.normal()
            Poisson = np.random.poisson(lamb * dt)
            price_path[t] = price_path[t-1] + mu*price_path[t-1]*dt + sigma*price_path[t-1]*W + price_path[t-1]*Poisson*(np.exp(m + v*np.random.normal() - 0.5*v**2) - 1)

        predicted_price = np.mean(price_path)
        return predicted_price, greeks

    def calculate_geometric_brownian(self):
        stock_returns = np.log(self.stock_data['Close'] / self.stock_data['Close'].shift(1))
        volatility = (stock_returns.std() * np.sqrt(252)).item()
        delta = (stock_returns.mean() * 252).item()
        sigma = volatility
        theta = -0.5 * sigma ** 2
        vega = delta * sigma

        greeks = {
            "Delta": delta,
            "Theta": theta,
            "Vega": vega,
            "Sigma": sigma
        }

        S0 = self.stock_data['Close'].iloc[-1]
        mu = self.stock_data['Close'].pct_change().mean().item()
        sigma = self.stock_data['Close'].pct_change().std().item()

        T = self.days_to_predict
        dt = 1
        n_simulations = 1000

        N = int(T/dt)
        price_paths = np.zeros((n_simulations, N))
        price_paths[:, 0] = S0

        for t in range(1, N):
            random_shock = np.sqrt(dt) * np.random.randn(n_simulations)
            price_paths[:, t] = price_paths[:, t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * random_shock)
            
        predicted_price = np.mean(price_paths[:, -1])
        return predicted_price, greeks

    def calculate_stochastic_volatility(self):
        stock_returns = np.log(self.stock_data['Close'] / self.stock_data['Close'].shift(1)).dropna()
        
        volatility = stock_returns.std().item() * np.sqrt(252)
        delta = stock_returns.mean().item() * 252
        sigma = volatility
        theta = -0.5 * sigma ** 2
        vega = delta * sigma

        greeks = {
            "Delta": delta,
            "Theta": theta,
            "Vega": vega,
            "Sigma": sigma
        }

        S0 = self.stock_data['Close'].iloc[-1].item()
        mu = self.stock_data['Close'].pct_change().mean().item()
        dt = 1
        rho = -0.5
        kappa = 2.0
        N = int(self.days_to_predict / dt)
        v0 = 0.04
        n_simulations = 1000
        sqrt_dt = np.sqrt(dt)

        S = np.zeros((n_simulations, N))
        v = np.zeros((n_simulations, N))
        S[:, 0] = S0
        v[:, 0] = v0

        for t in range(1, N):
            dW1 = sqrt_dt * np.random.randn(n_simulations)
            dW2 = sqrt_dt * (rho * dW1 + np.sqrt(1 - rho ** 2) * np.random.randn(n_simulations))
            v[:, t] = np.abs(v[:, t-1] + kappa * (theta - np.maximum(v[:, t-1], 0)) * dt +
                            sigma * np.sqrt(np.maximum(v[:, t-1], 0)) * dW1)
            S[:, t] = S[:, t-1] * np.exp((mu - 0.5 * np.maximum(v[:, t-1], 0)) * dt +
                                        np.sqrt(np.maximum(v[:, t-1], 0)) * dW2)

        predicted_price = np.mean(S[:, -1])
        return predicted_price, greeks

    def show_future_prediction(self):
        return f"In {self.days_to_predict} days, the predicted price for {self.stock_symbol} is ${self.future_price:.2f}"

    def display_greek_info_gui(self):
        greek_symbols = {
            "Delta": "\u0394",
            "Theta": "\u03F4",
            "Vega": "\u03F5",
            "Sigma": "\u03A3"
        }
        alphadata = {}
        for i, (symbol_name, symbol_value) in enumerate(self.greeks.items()):
            alphadata[symbol_name] = f"{symbol_name} ({greek_symbols[symbol_name]}): {symbol_value:.2f}"
        return alphadata
        
    def check_stock_status(self):
        try:
            stock_value = self.stock_data['Close'].iloc[-1].item()
            status = "Invalid option type"  # Define 'status' beforehand
            if self.option_type.lower() == "call":
                if stock_value > self.strike_value:
                    status = "In-the-money"
                elif abs(stock_value - self.strike_value) <= 1:  # Slightly below
                    status = "Near-the-money"
                else:  # Significantly below
                    status = "Out-of-the-money"
            elif self.option_type.lower() == "put":
                if stock_value <= self.strike_value:  # Here's the change
                    status = "In-the-money"
                elif stock_value - self.strike_value < 2:  # And here
                    status = "At-the-money"
                else:  # And here
                    status = "Out-of-the-money"
            return f"The option is {status}.\nCurrent Stock Value: ${stock_value:.2f}\nStrike Value: ${self.strike_value:.2f}"
        except Exception as e:
            print(f"Error: Failed to check option status: {e}")
    
    def predict_data(self):
        if self.formula=='Black Scholes':
            self.future_price, self.greeks = self.calculate_black_scholes()
        elif self.formula=='Jump Diffusion':
            self.future_price, self.greeks = self.calculate_jump_dufuison()
        elif self.formula=='Monte Carlo':
            self.future_price, self.greeks = self.calculate_geometric_brownian()
        elif self.formula=='Stochastic Volatility':
            self.future_price, self.greeks = self.calculate_stochastic_volatility()
        else:
            return {"Formula Selection is not Correct": "Option Are: Black Scholes, Jump Diffusion, Monte Carlo, Stochastic Volatility"}
        future_prediction = self.show_future_prediction()
        option_status = self.check_stock_status()
        alphadata = self.display_greek_info_gui()
        close_json = eval(self.stock_data[('Close', self.stock_symbol.upper())].to_json(date_format='iso', indent=2))
        formatted_json = {str(pd.to_datetime(k).date()): v for k, v in close_json.items()}
        return {"Stock_Data": formatted_json, "Stock_Symbol": self.stock_symbol, "Predicted_Amount": self.future_price, "Future_Prediction": future_prediction, "Option Status": option_status, "Greeks": alphadata}

    def plot_stock_data(self):
        """
        Plots the closing stock price from a DataFrame.

        Args:
            stock_data (pd.DataFrame): DataFrame containing a 'Close' column and a datetime index.
            stock_symbol (str): Optional stock symbol for the chart title.
        """
        # print(plt.style.available)
        # plt.style.use('seaborn-v0_8-darkgrid')  # Set plot style
        # fig, ax = plt.subplots(figsize=(10, 5))
        plt.style.use('seaborn-v0_8-colorblind')
        # Create a large, high-res figure
        fig, ax = plt.subplots(figsize=(14, 7))

        # Plot closing prices
        self.stock_data['Close'].plot(
            ax=ax,
            label='Close Price',
            marker='o',
            markersize=3,
            linestyle='-',
            linewidth=1.5
        )

        # Configure chart
        ax.set_title(f"Stock Price History: {self.stock_symbol}", fontsize=16)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Price", fontsize=12)
        ax.legend()
        ax.grid(True)

        # Draw a red vertical line at the last date
        ax.axvline(x=self.stock_data.index[-1], color='r', linestyle='--', label='Last Date')
        save_path = "chart.png"
        plt.tight_layout()
        fig.savefig(save_path, dpi=300)  # High-res output
        print(f"🔥 Gorgeous chart saved to: {save_path}")

        # plt.show()
