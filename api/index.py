import pickle as pk
import os
from flask import Flask, request, jsonify, render_template

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'templates'))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(BASE_DIR, 'model.pkl'), 'rb') as f:
    model = pk.load(f)
with open(os.path.join(BASE_DIR, 'scaler.pkl'), 'rb') as f:
    scaler = pk.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    review = data.get('review', '')
    if not review.strip():
        return jsonify({'error': 'Please enter a review'}), 400
    review_scaled = scaler.transform([review]).toarray()
    result = model.predict(review_scaled)
    sentiment = 'Positive' if result[0] == 1 else 'Negative'
    emoji = '😊' if result[0] == 1 else '😞'
    return jsonify({'sentiment': sentiment, 'emoji': emoji})

if __name__ == '__main__':
    app.run(debug=True)