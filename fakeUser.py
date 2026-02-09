# ----------------------------------------------
# 1. IMPORT LIBRARIES
# ----------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_curve, auc,
    classification_report, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# ----------------------------------------------
# 2. LOAD DATASET
# ----------------------------------------------
df = pd.read_csv("/content/mock_user_disclosure_dataset.csv")

X = df.drop("target", axis=1)
y = df["target"]

# ----------------------------------------------
# 3. SPLIT DATA
# ----------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# ----------------------------------------------
# 4. SCALE FEATURES
# ----------------------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ----------------------------------------------
# 5. DEFINE MODELS
# ----------------------------------------------
models = {
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM": SVC(probability=True, kernel="rbf"),
    "Logistic Regression": LogisticRegression(max_iter=500),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB()
}

accuracies = {}
roc_data = {}

# ----------------------------------------------
# 6. TRAIN & EVALUATE MODELS
# ----------------------------------------------
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies[name] = acc

    # ROC curve
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_data[name] = (fpr, tpr, auc(fpr, tpr))

    print(f"\n================ {name} ================")
    print("Accuracy:", round(acc * 100, 2), "%")
    print(classification_report(y_test, y_pred))

    # CONFUSION MATRIX
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# ----------------------------------------------
# 7. ACCURACY COMPARISON GRAPH
# ----------------------------------------------
plt.figure(figsize=(8,5))
plt.bar(accuracies.keys(), accuracies.values())
plt.ylim(0, 1)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.show()

# ----------------------------------------------
# 8. ROC CURVE COMPARISON
# ----------------------------------------------
plt.figure(figsize=(8,6))
for name, (fpr, tpr, roc_auc) in roc_data.items():
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

plt.plot([0,1], [0,1], "k--")
plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
