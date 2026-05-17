import os
from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)
# Load the model once when the app starts
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
@app.route('/predict', methods=['POST'])
def predict():
    # Expect JSON with a 'features' array
    data = request.json
    features = np.array(data['features']).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(features)
    
    # Return result as JSON
    return jsonify({'prediction': prediction.tolist()})
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})
if __name__ == '__main__':
    # Read port from environment variable, default to 8000
    port = int(os.getenv('PORT', 8000))
    app.run(host='0.0.0.0', port=port)