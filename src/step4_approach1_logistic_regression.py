import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

print("\n===== APPROACH 1 : PANDAS + LOGISTIC REGRESSION =====\n")

# Load engineered dataset
df = pd.read_csv("data/ecommerce_features.csv")

X = df.drop("purchase", axis=1)
y = df["purchase"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scale features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
start_time = time.time()

model = LogisticRegression(max_iter=1000)

model.fit(X_train_scaled, y_train)

end_time = time.time()

# Predictions
y_pred = model.predict(X_test_scaled)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))
print("Training Time:", round(end_time - start_time, 2), "seconds")

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Save metrics
results = pd.DataFrame({
    "Approach": ["Approach 1"],
    "Model": ["Logistic Regression"],
    "Accuracy": [accuracy],
    "Precision": [precision],
    "Recall": [recall],
    "F1 Score": [f1],
    "Training Time": [round(end_time - start_time, 2)]
})

results.to_csv("outputs/comparison/approach1_results.csv", index=False)

print("\nResults Saved -> outputs/comparison/approach1_results.csv")