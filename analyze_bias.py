import pandas as pd

data = pd.read_csv('newsdata.csv')

# Create combined text
data['combined_text'] = data['title'] + ' ' + data['text']

# Check what subjects are in fake vs real
print("=== FAKE NEWS SUBJECTS ===")
fake_subjects = data[data['label'] == 0]['subject'].value_counts().head(10)
print(fake_subjects)

print("\n=== REAL NEWS SUBJECTS ===")
real_subjects = data[data['label'] == 1]['subject'].value_counts().head(10)
print(real_subjects)

# Check average text length
print("\n=== TEXT LENGTH ===")
data['text_length'] = data['text'].str.len()
print(f"Fake news avg length: {data[data['label'] == 0]['text_length'].mean():.0f}")
print(f"Real news avg length: {data[data['label'] == 1]['text_length'].mean():.0f}")

# Check for specific keywords
print("\n=== KEYWORD ANALYSIS ===")
fake_data = data[data['label'] == 0]['combined_text'].str.lower()
real_data = data[data['label'] == 1]['combined_text'].str.lower()

keywords = ['trump', 'obama', 'clinton', 'democrat', 'republican', 'breaking', 'exclusive', 'shocking']
for keyword in keywords:
    fake_count = (fake_data.str.contains(keyword)).sum()
    real_count = (real_data.str.contains(keyword)).sum()
    print(f"{keyword}: Fake={fake_count}, Real={real_count}")
