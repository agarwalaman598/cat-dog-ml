import numpy as np
import joblib

import os
os.makedirs("models", exist_ok=True)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

# ----------------------------------------------------
# LOAD DATA (reuse preprocessing output)
# ----------------------------------------------------
# If you saved X, y earlier, load them.
# Otherwise, import preprocessing code and regenerate.
# Here we assume X and y are saved as .npy files.

X = np.load("X.npy")
y = np.load("y.npy")

# ----------------------------------------------------
# TRAIN-TEST SPLIT
# ----------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)

# ----------------------------------------------------
# 1. LOGISTIC REGRESSION
# ----------------------------------------------------
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)
print("\nLogistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print(confusion_matrix(y_test, y_pred_lr))

# ----------------------------------------------------
# 2. SUPPORT VECTOR MACHINE
# ----------------------------------------------------
svm = SVC(kernel="rbf", gamma="scale")
svm.fit(X_train, y_train)

y_pred_svm = svm.predict(X_test)
print("\nSVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print(confusion_matrix(y_test, y_pred_svm))

# ----------------------------------------------------
# 3. RANDOM FOREST
# ----------------------------------------------------
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
print("\nRandom Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))

# ----------------------------------------------------
# 4. K-MEANS (UNSUPERVISED)
# ----------------------------------------------------
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_train)

# Map clusters to labels using majority voting
clusters = kmeans.labels_
label_map = {}

for cluster in [0, 1]:
    true_labels = y_train[clusters == cluster]
    label_map[cluster] = np.bincount(true_labels).argmax()

# Predict on test set
test_clusters = kmeans.predict(X_test)
y_pred_km = np.array([label_map[c] for c in test_clusters])

print("\nK-Means Accuracy:", accuracy_score(y_test, y_pred_km))
print(confusion_matrix(y_test, y_pred_km))

# ----------------------------------------------------
# SAVE MODELS
# ----------------------------------------------------
joblib.dump(lr, "models/logistic_regression.pkl")
joblib.dump(svm, "models/svm.pkl")
joblib.dump(rf, "models/random_forest.pkl")
joblib.dump(kmeans, "models/kmeans.pkl")

print("\nAll models saved successfully.")
