import pandas as pd

# Load data
data = pd.read_csv('newsdata.csv')

print("=== DATA QUALITY CHECK ===\n")

# Check for missing values
print("Missing values:")
print(data.isnull().sum())
print()

# Check label distribution
print("Label distribution:")
print(f"Fake (0): {(data['label'] == 0).sum()}")
print(f"Real (1): {(data['label'] == 1).sum()}")
print()

# Check sample fake news
print("=== SAMPLE FAKE NEWS ===")
fake_samples = data[data['label'] == 0].head(3)
for idx, row in fake_samples.iterrows():
    print(f"\nTitle: {row['title'][:80]}")
    print(f"Text: {row['text'][:150]}")
print()

# Check sample real news
print("=== SAMPLE REAL NEWS ===")
real_samples = data[data['label'] == 1].head(3)
for idx, row in real_samples.iterrows():
    print(f"\nTitle: {row['title'][:80]}")
    print(f"Text: {row['text'][:150]}")
