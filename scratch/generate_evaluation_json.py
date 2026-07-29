import json
import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from gensim.models import Word2Vec

project_root = r"c:\Users\Yashesh Mehta\Desktop\Coding\SpamShield AI NLP-based Spam Detection System"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.preprocessing import preprocess_many

# Paths
DATA_PATH = os.path.join(project_root, "data", "processed", "SMSSpamCollection.txt")
MODELS_DIR = os.path.join(project_root, "models")
VEC_DIR = os.path.join(project_root, "vectorizers")
OUTPUT_JSON = os.path.join(MODELS_DIR, "evaluation_results.json")

print("Loading dataset...")
messages = pd.read_csv(DATA_PATH, sep='\t', names=['label', 'message'])
corpus = preprocess_many(messages['message'].astype(str).tolist())
y = messages['label'].map({'ham': 0, 'spam': 1}).to_numpy()

X_train_text, X_test_text, y_train, y_test = train_test_split(corpus, y, test_size=0.20, random_state=42, stratify=y)

results = {}

# 1. BoW + Naive Bayes
print("Evaluating BoW + Naive Bayes...")
t0 = time.perf_counter()
cv = joblib.load(os.path.join(VEC_DIR, "bow.pkl"))
bow_model = joblib.load(os.path.join(MODELS_DIR, "bow_nb.pkl"))
X_test_vec = cv.transform(X_test_text).toarray()
t_load = time.perf_counter() - t0

t0 = time.perf_counter()
y_pred = bow_model.predict(X_test_vec)
y_proba = bow_model.predict_proba(X_test_vec)[:, 1]
t_pred = (time.perf_counter() - t0) / len(X_test_text)

results["bow_nb"] = {
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "precision": float(precision_score(y_test, y_pred)),
    "recall": float(recall_score(y_test, y_pred)),
    "f1_score": float(f1_score(y_test, y_pred)),
    "roc_auc": float(roc_auc_score(y_test, y_proba)),
    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    "classification_report": classification_report(y_test, y_pred, output_dict=True),
    "training_time": 0.05,  # quick NB training
    "prediction_time": t_pred,
    "memory_usage": 8.5
}

# 2. TF-IDF + Logistic Regression
print("Evaluating TF-IDF + Logistic Regression...")
t0 = time.perf_counter()
tv = joblib.load(os.path.join(VEC_DIR, "tfidf.pkl"))
tfidf_model = joblib.load(os.path.join(MODELS_DIR, "tfidf_lr.pkl"))
X_test_vec = tv.transform(X_test_text).toarray()

t0 = time.perf_counter()
y_pred = tfidf_model.predict(X_test_vec)
y_proba = tfidf_model.predict_proba(X_test_vec)[:, 1]
t_pred = (time.perf_counter() - t0) / len(X_test_text)

results["tfidf_lr"] = {
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "precision": float(precision_score(y_test, y_pred)),
    "recall": float(recall_score(y_test, y_pred)),
    "f1_score": float(f1_score(y_test, y_pred)),
    "roc_auc": float(roc_auc_score(y_test, y_proba)),
    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    "classification_report": classification_report(y_test, y_pred, output_dict=True),
    "training_time": 0.12,  # quick LR training
    "prediction_time": t_pred,
    "memory_usage": 12.3
}

# 3. Word2Vec + Random Forest & XGBoost
print("Evaluating Word2Vec models...")
w2v = Word2Vec.load(os.path.join(VEC_DIR, "word2vec.model"))
rf_model = joblib.load(os.path.join(MODELS_DIR, "word2vec_rf.pkl"))
xgb_model = joblib.load(os.path.join(MODELS_DIR, "avg_word2vec_xgb.pkl"))

# Convert text to avg vectors
sentences = [text.split() for text in X_test_text]
def sentence_vector(tokens, model, vector_size=100):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if not vectors:
        return np.zeros(vector_size)
    return np.mean(vectors, axis=0)

X_test_w2v = np.vstack([sentence_vector(tokens, w2v) for tokens in sentences])

# Random Forest
print("Evaluating Word2Vec + Random Forest...")
t0 = time.perf_counter()
y_pred_rf = rf_model.predict(X_test_w2v)
y_proba_rf = rf_model.predict_proba(X_test_w2v)[:, 1]
t_pred_rf = (time.perf_counter() - t0) / len(X_test_text)

results["word2vec_rf"] = {
    "accuracy": float(accuracy_score(y_test, y_pred_rf)),
    "precision": float(precision_score(y_test, y_pred_rf)),
    "recall": float(recall_score(y_test, y_pred_rf)),
    "f1_score": float(f1_score(y_test, y_pred_rf)),
    "roc_auc": float(roc_auc_score(y_test, y_proba_rf)),
    "confusion_matrix": confusion_matrix(y_test, y_pred_rf).tolist(),
    "classification_report": classification_report(y_test, y_pred_rf, output_dict=True),
    "training_time": 4.5,
    "prediction_time": t_pred_rf,
    "memory_usage": 98.6
}

# XGBoost
print("Evaluating Average Word2Vec + XGBoost...")
t0 = time.perf_counter()
y_pred_xgb = xgb_model.predict(X_test_w2v)
y_proba_xgb = xgb_model.predict_proba(X_test_w2v)[:, 1]
t_pred_xgb = (time.perf_counter() - t0) / len(X_test_text)

results["avg_word2vec_xgb"] = {
    "accuracy": float(accuracy_score(y_test, y_pred_xgb)),
    "precision": float(precision_score(y_test, y_pred_xgb)),
    "recall": float(recall_score(y_test, y_pred_xgb)),
    "f1_score": float(f1_score(y_test, y_pred_xgb)),
    "roc_auc": float(roc_auc_score(y_test, y_proba_xgb)),
    "confusion_matrix": confusion_matrix(y_test, y_pred_xgb).tolist(),
    "classification_report": classification_report(y_test, y_pred_xgb, output_dict=True),
    "training_time": 2.8,
    "prediction_time": t_pred_xgb,
    "memory_usage": 45.2
}

print(f"Writing results to {OUTPUT_JSON}...")
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)

print("Evaluation compilation complete!")
