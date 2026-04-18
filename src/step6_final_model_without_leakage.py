import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("\n===== APPROACH 3 : FINAL MODEL WITHOUT DATA LEAKAGE =====\n")

# Load data
df = pd.read_csv("data/ecommerce_features.csv")

# Remove leakage feature
X = df.drop(["purchase", "event_type"], axis=1)
y = df["purchase"]

print("Features Used:")
print(X.columns.tolist())

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train
start = time.time()

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

end = time.time()

# Predict
y_pred = model.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nAccuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))
print("Training Time:", round(end - start, 2), "seconds")

# Importance
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance:\n")
print(importance_df)

# Save comparison
results = pd.DataFrame({
    "Approach": ["Approach 3"],
    "Model": ["Random Forest Without Leakage"],
    "Accuracy": [accuracy],
    "Precision": [precision],
    "Recall": [recall],
    "F1 Score": [f1],
    "Training Time": [round(end - start, 2)]
})

results.to_csv("outputs/comparison/approach3_results.csv", index=False)
importance_df.to_csv("outputs/comparison/approach3_feature_importance.csv", index=False)

print("\nSaved final results to outputs/comparison/")