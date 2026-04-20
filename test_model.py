import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load model and vectorizer
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Political news headlines (what your model is trained on)
test_articles = [
    "Trump announces new campaign strategy for 2024 election",
    "Biden signs executive order on climate change",
    "Congress debates healthcare reform bill",
    "Senate votes on new tax legislation",
    "Democratic party holds national convention",
    "Republican candidate wins Iowa caucuses",
    "President meets with world leaders at summit",
    "Parliament passes immigration law",
    "Election results show record voter turnout",
    "Government announces new economic stimulus package"
]

print("="*60)
print("TESTING FAKE NEWS DETECTOR ON POLITICAL NEWS")
print("="*60 + "\n")

for article in test_articles:
    # Vectorize
    news_vec = vectorizer.transform([article])
    
    # Predict
    prediction = model.predict(news_vec)[0]
    probability = model.predict_proba(news_vec)[0]
    
    fake_prob = probability[0]
    real_prob = probability[1]
    
    result = "🚨 FAKE" if fake_prob >= 0.5 else "✅ REAL"
    confidence = max(fake_prob, real_prob) * 100
    
    print(f"📰 {article}")
    print(f"   Result: {result} ({confidence:.2f}% confidence)")
    print(f"   Fake: {fake_prob*100:.2f}% | Real: {real_prob*100:.2f}%\n")

