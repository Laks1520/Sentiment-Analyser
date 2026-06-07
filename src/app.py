from flask import Flask, render_template, request
import pickle
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

app = Flask(__name__, template_folder='../templates')

# Load model and vectorizer
with open('data/lr_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('data/tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Same clean function from Day 2
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    confidence = None
    review = ''

    if request.method == 'POST':
        review = request.form['review']
        cleaned = clean_text(review)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        probability = model.predict_proba(vectorized)[0]

        if prediction == 1:
            result = 'POSITIVE'
            confidence = round(probability[1] * 100, 2)
        else:
            result = 'NEGATIVE'
            confidence = round(probability[0] * 100, 2)

    return render_template('index.html',
                           result=result,
                           confidence=confidence,
                           review=review)

if __name__ == '__main__':
    app.run(debug=False)