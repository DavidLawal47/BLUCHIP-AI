from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback
import os, json
from stock_predictor import StockPricePredictor, StockCustomPredictor
import yfinance as yf

app = Flask(__name__)
CORS(app)

# Initialize predictor
os.makedirs("models", exist_ok=True)
predictor = StockPricePredictor('AAPL')  # Default to AAPL
predictor.load_model('models')  # Load pre-trained model

@app.route('/api/model/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("data from api", data)
        symbol = data.get('symbol', 'AAPL')
        days = int(data.get('days', 1))
        
        # Update symbol if different from current
        if symbol != predictor.symbol:
            predictor.symbol = symbol

        # Check IF Symbol in Models
        models = [model.lower() for model in os.listdir("models")] 
        if f"{symbol}_predictor.h5".lower() not in models:
            predictor.train(epochs=25)

        # Get prediction
        futuret_day_price = predictor.predict_future_prices(days=days)

        # Get historical data for chart
        stock = yf.Ticker(symbol)
        hist = stock.history(period='5y')
        historical_data = hist['Close'].tolist()
        dates = hist.index.strftime('%Y-%m-%d').tolist()
        
        return jsonify({
            'symbol': symbol,
            f'prediction': [float(price) for price in futuret_day_price],
            'historical_data': historical_data,
            'dates': dates,
            'forecast_period': days,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model/train', methods=['POST'])
def train():
    try:
        data = request.get_json()
        symbol = data.get('symbol', 'AAPL')
        epochs = int(data.get('epochs', 25))
        
        # Update symbol if different
        if symbol != predictor.symbol:
            predictor.symbol = symbol
        
        # Train model
        predictor.train(epochs=epochs)
        
        return jsonify({'message': f'Model trained successfully for {symbol}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/custommodel', methods=['POST'])
def custom_model():
    try:
        data = request.get_json()
        symbol = data.get('symbol', 'AAPL')
        days_to_predict = int(data.get('days', 1))
        start_date = data.get('start_date', '2025-01-01')
        end_date = data.get('end_date', '2025-05-01')
        formula = data.get('formula', 'Monte Carlo')
        option_type = data.get('option_type', 'Put')
        strike_value = int(data.get('strike_value', 88))
        expiration_days = int(data.get('expiration_days', 55))
        stock_predictor = StockCustomPredictor(symbol=symbol, days_to_predict=days_to_predict, formula=formula, start_date=start_date, end_date=end_date, option_type=option_type, strike_value=strike_value, expiration_days=expiration_days)
        response = stock_predictor.predict_data()
        # stock_predictor.plot_stock_data()
        # print(response)
        return jsonify({
                "Message": "Success",
                "Response":  response
                }), 200
    except Exception as e:
        return jsonify({'Error': str(e)}), 500
        traceback.print_exc()

@app.route('/api/symbols', methods=['GET'])
def get_symbols():
    with open('symbols.json', 'r+') as fp:
        return json.load(fp)
    return None

if __name__ == '__main__':
    app.run(debug=True, port=5000, use_reloader=False)