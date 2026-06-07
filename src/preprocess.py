import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Download stopwords (only needed once)
nltk.download('stopwords')

# Load data
df = pd.read_csv('data/IMDB Dataset.csv')
print("Loaded:", df.shape)

def clean_text(text):
    # Remove HTML tags like <br />
    text = re.sub(r'<.*?>', '', text)
    
    # Remove everything that is NOT a letter or space
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Split into individual words
    words = text.split()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words]
    
    # Join words back into a single string
    return ' '.join(words)

print("Cleaning text... (this takes ~1 min)")
df['clean_review'] = df['review'].apply(clean_text)

print("Done! Sample clean review:")
print(df['clean_review'][0])

# Convert sentiment to binary: positive=1, negative=0
df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
print(df['label'].value_counts())

# TF-IDF: convert clean text into numbers
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_review'])
y = df['label'].values

print("TF-IDF matrix shape:", X.shape)

# Save cleaned dataframe
df.to_csv('data/cleaned_imdb.csv', index=False)

# Save vectorizer for later use
with open('data/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# Save X and y
import numpy as np
np.save('data/X.npy', X.toarray())
np.save('data/y.npy', y)

print("All saved! ✅")