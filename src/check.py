import pandas as pd

# Load the dataset
df = pd.read_csv('data/IMDB Dataset.csv')

# Check the first 5 rows
print(df.head())

# Check how many positive vs negative reviews
print(df['sentiment'].value_counts())

# Check shape — should be 50,000 rows
print("Shape:", df.shape)