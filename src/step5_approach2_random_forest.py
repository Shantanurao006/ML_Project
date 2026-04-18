import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

print("\n===== APPROACH 2 : RANDOM FOREST =====\n")

# Load dataset
df = pd.read_csv("data/ecommerce_features.csv")

X = df.drop("purchase", axis=1)
y = df["purchase"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train Random Forest
start_time = time.time()

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

end_time = time.time()

# Predictions
y_pred = model.predict(X_test)

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

# Feature Importance
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nTop Features:\n")
print(feature_importance)

feature_importance.to_csv(
    "outputs/comparison/approach2_feature_importance.csv",
    index=False
)

# Save results
results = pd.DataFrame({
    "Approach": ["Approach 2"],
    "Model": ["Random Forest"],
    "Accuracy": [accuracy],
    "Precision": [precision],
    "Recall": [recall],
    "F1 Score": [f1],
    "Training Time": [round(end_time - start_time, 2)]
})

results.to_csv("outputs/comparison/approach2_results.csv", index=False)

print("\nResults Saved -> outputs/comparison/approach2_results.csv")