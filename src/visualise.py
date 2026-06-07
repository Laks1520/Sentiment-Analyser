import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud

# Load data
X = np.load('data/X.npy')
y = np.load('data/y.npy')
df = pd.read_csv('data/cleaned_imdb.csv')

# Load models
with open('data/nb_model.pkl', 'rb') as f:
    nb_model = pickle.load(f)

with open('data/lr_model.pkl', 'rb') as f:
    lr_model = pickle.load(f)

# Recreate same split using same random_state=42
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Get predictions
nb_pred = nb_model.predict(X_test)
lr_pred = lr_model.predict(X_test)

nb_acc = accuracy_score(y_test, nb_pred)
lr_acc = accuracy_score(y_test, lr_pred)

print(f"NB: {nb_acc*100:.2f}%  |  LR: {lr_acc*100:.2f}%")

plt.figure(figsize=(7, 5))

models = ['Naive Bayes', 'Logistic Regression']
accuracies = [nb_acc * 100, lr_acc * 100]
colors = ['#4C72B0', '#55A868']

bars = plt.bar(models, accuracies, color=colors, width=0.4)

# Add accuracy value on top of each bar
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.3,
             f'{acc:.2f}%',
             ha='center', fontsize=12, fontweight='bold')

plt.ylim(80, 95)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('Model Comparison — Sentiment Analyser', fontsize=14)
plt.tight_layout()
plt.savefig('outputs/model_comparison.png', dpi=150)
plt.show()
print("Model comparison chart saved! ✅")

# Separate positive and negative reviews
positive_words = ' '.join(df[df['sentiment'] == 'positive']['clean_review'])
negative_words = ' '.join(df[df['sentiment'] == 'negative']['clean_review'])

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Positive word cloud
wc_pos = WordCloud(width=600, height=400,
                   background_color='white',
                   colormap='Greens').generate(positive_words)
axes[0].imshow(wc_pos, interpolation='bilinear')
axes[0].axis('off')
axes[0].set_title('Most Common Words — Positive Reviews', fontsize=13)

# Negative word cloud
wc_neg = WordCloud(width=600, height=400,
                   background_color='white',
                   colormap='Reds').generate(negative_words)
axes[1].imshow(wc_neg, interpolation='bilinear')
axes[1].axis('off')
axes[1].set_title('Most Common Words — Negative Reviews', fontsize=13)

plt.tight_layout()
plt.savefig('outputs/wordclouds.png', dpi=150)
plt.show()
print("Word clouds saved! ✅")

plt.figure(figsize=(6, 6))

labels = ['Positive', 'Negative']
sizes = [25000, 25000]
colors = ['#55A868', '#C44E52']
explode = (0.05, 0.05)

plt.pie(sizes, labels=labels, colors=colors,
        autopct='%1.1f%%', explode=explode,
        shadow=True, startangle=90)

plt.title('Dataset — Sentiment Distribution', fontsize=14)
plt.tight_layout()
plt.savefig('outputs/sentiment_distribution.png', dpi=150)
plt.show()
print("Pie chart saved! ✅")

