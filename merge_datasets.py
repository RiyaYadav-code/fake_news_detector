import pandas as pd

# Read both files
fake_df = pd.read_csv('Fake.csv')
true_df = pd.read_csv('True.csv')

print("Fake.csv shape:", fake_df.shape)
print("True.csv shape:", true_df.shape)

# Add label column
fake_df['label'] = 0  # 0 = Fake
true_df['label'] = 1  # 1 = Real

# Combine both datasets
data = pd.concat([fake_df, true_df], ignore_index=True)

# Shuffle the data
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Save as single CSV
data.to_csv('newsdata.csv', index=False)

print(f"\n✅ newsdata.csv created successfully!")
print(f"Total articles: {len(data)}")
print(f"Fake: {(data['label'] == 0).sum()}")
print(f"Real: {(data['label'] == 1).sum()}")
print(f"Columns: {list(data.columns)}")
