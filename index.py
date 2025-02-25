from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from model import train_universal_model, predict_stock

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/predict', methods=['GET'])
def predict():
    stock_code = request.args.get('stock_code')
    result, trend = predict_stock(stock_code)
    if result is None:
        return jsonify({'error': trend}), 400
    return jsonify({"trend": trend, "predictions": result.reset_index().to_dict(orient='records')})

if __name__ == '__main__':
    if not os.path.exists("universal_lstm_model.h5"):
        train_universal_model()
    app.run(host='0.0.0.0', port=5800, debug=True)