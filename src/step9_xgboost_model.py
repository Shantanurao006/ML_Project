import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

print("\n===== APPROACH 4 : XGBOOST =====\n")

# Load dataset
df = pd.read_csv("data/ecommerce_features.csv")

# Remove leakage feature
X = df.drop(["purchase", "event_type"], axis=1)
y = df["purchase"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Handle class imbalance
scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

# Train XGBoost
start_time = time.time()

model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

end_time = time.time()

# Predict
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

# Save results
results = pd.DataFrame({
    "Approach": ["Approach 4"],
    "Model": ["XGBoost"],
    "Accuracy": [accuracy],
    "Precision": [precision],
    "Recall": [recall],
    "F1 Score": [f1],
    "Training Time": [round(end_time - start_time, 2)]
})

results.to_csv(
    "outputs/comparison/approach4_results.csv",
    index=False
)

print("\nSaved -> outputs/comparison/approach4_results.csv")