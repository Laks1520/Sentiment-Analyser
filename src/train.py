import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load saved data from Day 2
X = np.load('data/X.npy')
y = np.load('data/y.npy')

print("Data loaded!")
print("X shape:", X.shape)
print("y shape:", y.shape)
# Split: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])
print("\nTraining Naive Bayes...")
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# Predict on test set
nb_predictions = nb_model.predict(X_test)

# Evaluate
nb_accuracy = accuracy_score(y_test, nb_predictions)
print(f"Naive Bayes Accuracy: {nb_accuracy * 100:.2f}%")
print(classification_report(y_test, nb_predictions, target_names=['Negative', 'Positive']))

print("\nTraining Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

# Predict
lr_predictions = lr_model.predict(X_test)

# Evaluate
lr_accuracy = accuracy_score(y_test, lr_predictions)
print(f"Logistic Regression Accuracy: {lr_accuracy * 100:.2f}%")
print(classification_report(y_test, lr_predictions, target_names=['Negative', 'Positive']))


fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Naive Bayes confusion matrix
cm_nb = confusion_matrix(y_test, nb_predictions)
sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'],
            ax=axes[0])
axes[0].set_title('Naive Bayes — Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

# Logistic Regression confusion matrix
cm_lr = confusion_matrix(y_test, lr_predictions)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'],
            ax=axes[1])
axes[1].set_title('Logistic Regression — Confusion Matrix')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.savefig('outputs/confusion_matrices.png')
plt.show()
print("Confusion matrix saved to outputs/!")

# Save models for Day 4
with open('data/nb_model.pkl', 'wb') as f:
    pickle.dump(nb_model, f)

with open('data/lr_model.pkl', 'wb') as f:
    pickle.dump(lr_model, f)

print("\nBoth models saved! ✅")
print(f"\nFinal Comparison:")
print(f"Naive Bayes:          {nb_accuracy * 100:.2f}%")
print(f"Logistic Regression:  {lr_accuracy * 100:.2f}%")

