import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load data
data = pd.read_csv('newsdata.csv')

print(f"Total articles: {len(data)}")

# BALANCED SAMPLE
print("Creating balanced dataset...")
fake_data = data[data['label'] == 0].sample(n=8000, random_state=42)
real_data = data[data['label'] == 1].sample(n=8000, random_state=42)
data = pd.concat([fake_data, real_data], ignore_index=True)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Balanced dataset: {len(data)} articles")

# Combine title and text
data['combined_text'] = data['title'] + ' ' + data['text']

X = data['combined_text']
y = data['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training: {len(X_train)}, Test: {len(X_test)}")

# TF-IDF - REMOVE sensational words from features
print("\nCreating features...")
vectorizer = TfidfVectorizer(
    max_features=2000,
    stop_words='english',
    min_df=5,
    max_df=0.7,
    ngram_range=(1, 1)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print(f"Features: {X_train_vec.shape[1]}")

# Train simple LogisticRegression
print("\nTraining LogisticRegression...")
model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight='balanced',
    solver='liblinear',
    C=0.5
)
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)
y_pred_proba = model.predict_proba(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print("\nDetailed Report:")
print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))

# Save model and vectorizer
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("\n✅ Model saved!")
