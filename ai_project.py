# -- coding: utf-8 --
"""AI_Project.py"""

# Import necessary libraries
import kagglehub
import os
import pandas as pd
import re
import spacy
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup

# Dataset Loading
path = kagglehub.dataset_download("snehaanbhawal/resume-dataset")
print("Path to dataset files:", path)

print("Dataset files:", os.listdir(path))

csv_file = os.path.join(path, "Resume/Resume.csv")
df = pd.read_csv(csv_file)

print(df.head())

# Dataset Cleaning
nltk.download("stopwords")
nlp = spacy.load("en_core_web_sm")

df.dropna(subset=["Resume_str", "Category"], inplace=True)

def remove_html(text):
    return BeautifulSoup(text, "html.parser").get_text()

df["Resume_str"] = df["Resume_str"].apply(remove_html)

def clean_text(text):
    text = text.lower()  # lowercase conversion
    text = re.sub(r"\d+", "", text)  # remove numbers
    text = re.sub(r"[^\w\s]", "", text)  # remove special characters
    text = re.sub(r"\s+", " ", text).strip()  # remove extra spaces
    return text

df["Resume_str"] = df["Resume_str"].apply(clean_text)

stop_words = set(stopwords.words("english"))

def remove_stopwords(text):
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

df["Resume_str"] = df["Resume_str"].apply(remove_stopwords)

def lemmatize_text(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])

df["Resume_str"] = df["Resume_str"].apply(lemmatize_text)

df.drop_duplicates(subset=["Resume_str"], inplace=True)
df.to_csv("cleaned_resume_dataset.csv", index=False)

print(df.head())

# Dataset Augmentation
import random
from nltk.corpus import wordnet
from transformers import pipeline

nltk.download('wordnet')
nltk.download('omw-1.4')

paraphrase = pipeline("text2text-generation", model="humarin/chatgpt_paraphraser_on_T5_base")

def synonym_replacement(sentence, n=1):
    words = sentence.split()
    if not words:
        return sentence
    new_words = words.copy()
    for _ in range(n):
        word = random.choice(words)
        synonyms = wordnet.synsets(word)
        if synonyms:
            synonym = synonyms[0].lemmas()[0].name()
            new_words[new_words.index(word)] = synonym
    return ' '.join(new_words)

df["synonym_replacement"] = df["Resume_str"].apply(lambda x: synonym_replacement(x))
df.to_csv("augmented_resume_dataset.csv", index=False)

print(df.head())

# Vectorization (Word2Vec and TF-IDF)
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer

# Word2Vec Model
word2vec_model = Word2Vec(sentences=df["synonym_replacement"], vector_size=100, window=5, min_count=1, workers=4)

def get_word2vec_vector(tokens):
    vectors = [word2vec_model.wv[word] for word in tokens if word in word2vec_model.wv]
    return sum(vectors) / len(vectors) if vectors else [0] * 100

df["word2vec_vector"] = df["synonym_replacement"].apply(get_word2vec_vector)
print(df.head())

# TF-IDF Vectorizer
df["processed_text_tf-idf"] = df["synonym_replacement"].apply(lambda x: " ".join(x))

tfidf_vectorizer = TfidfVectorizer(max_features=5000)
tfidf_vectors = tfidf_vectorizer.fit_transform(df["synonym_replacement"])

tfidf_df = pd.DataFrame(tfidf_vectors.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

df = pd.concat([df, tfidf_df], axis=1)
print(df.head())

df.to_csv("vectorized_resume_dataset.csv", index=False)

# KNN Model Pretesting (For reference, KNN is already done)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier

# Prepare the data
X = df.drop(columns=['Category', 'word2vec_vector'])
y = df['Category']

scaler = StandardScaler()
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Encode categorical features
for column in X.select_dtypes(include=['object']).columns:
    X[column] = encoder.fit_transform(X[column])

# Impute missing values and scale features
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)
X_scaled = scaler.fit_transform(X)

# Train and test KNN
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
knn = KNeighborsClassifier(n_neighbors=5, metric='cosine')
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"KNN Accuracy: {accuracy:.4f}")
print("KNN Classification Report:\n", classification_report(y_test, y_pred))

# Testing Other ML Models (Logistic Regression, Decision Tree, Random Forest, SVM)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Logistic Regression
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)
accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
print(f"Logistic Regression Accuracy: {accuracy_logreg:.4f}")
print("Logistic Regression Classification Report:\n", classification_report(y_test, y_pred_logreg))

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(f"Decision Tree Accuracy: {accuracy_dt:.4f}")
print("Decision Tree Classification Report:\n", classification_report(y_test, y_pred_dt))

# Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {accuracy_rf:.4f}")
print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf))

# Support Vector Machine
svm = SVC()
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"SVM Accuracy: {accuracy_svm:.4f}")
print("SVM Classification Report:\n", classification_report(y_test, y_pred_svm))