import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("Mental-Health-Twitter.csv")


# Text cleaning
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'@\w+|#', '', text)
    text = re.sub(r"[^a-z\s]", '', text)
    return text


df['clean_text'] = df['post_text'].apply(clean_text)

# Word Frequency Plot
all_words = ' '.join(df['clean_text'])
word_counts = Counter(all_words.split())
top_words = word_counts.most_common(20)
words, freqs = zip(*top_words)

plt.figure(figsize=(10, 5))
sns.barplot(x=list(freqs), y=list(words), palette="coolwarm")
plt.title("Top 20 Frequent Words")
plt.xlabel("Frequency")
plt.ylabel("Words")
plt.tight_layout()
plt.savefig("top_words.png")
plt.close()

# WordCloud
wc = WordCloud(width=800, height=400, background_color='white').generate(all_words)
plt.figure(figsize=(10, 5))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud of Tweets")
plt.tight_layout()
plt.savefig("wordcloud.png")
plt.close()

# Features and labels
X = df['clean_text']
y = df['label']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "SVC": SVC(),
    "Random Forest": RandomForestClassifier()
}

# Hyperparameters
params = {
    "SVC": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
    "Random Forest": {"n_estimators": [100, 200], "max_depth": [None, 10]}
}

# Train and evaluate
for name, model in models.items():
    print(f"\n=== {name} ===")
    if name in params:
        grid = GridSearchCV(model, params[name], cv=3, scoring="f1", verbose=0)
        grid.fit(X_train_tfidf, y_train)
        best_model = grid.best_estimator_
        print("Best Parameters:", grid.best_params_)
    else:
        best_model = model.fit(X_train_tfidf, y_train)

    y_pred = best_model.predict(X_test_tfidf)
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
    plt.title(f"Confusion Matrix: {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"conf_matrix_{name.replace(' ', '_')}.png")
    plt.close()
