import pickle
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load model and vectorizer
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    news = data.get('news', '')
    
    if not news or len(news.strip()) < 10:
        return jsonify({'error': 'Please enter at least 10 characters'}), 400
    
    try:
        # Vectorize the input
        news_vec = vectorizer.transform([news])
        
        # Predict
        prediction = model.predict(news_vec)[0]
        probability = model.predict_proba(news_vec)[0]
        
        fake_prob = probability[0]  # Probability of being fake
        real_prob = probability[1]  # Probability of being real
        
        if fake_prob >= 0.5:
            result = "🚨 FAKE NEWS"
            confidence = round(fake_prob * 100, 2)
        else:
            result = "✅ REAL NEWS"
            confidence = round(real_prob * 100, 2)
        
        return jsonify({
            'result': result,
            'confidence': confidence,
            'fake_probability': round(fake_prob * 100, 2),
            'real_probability': round(real_prob * 100, 2)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
