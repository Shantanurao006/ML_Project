import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

print("\n===== STEP 10 : SAVE FINAL MODEL =====\n")

# Load data
df = pd.read_csv("data/ecommerce_features.csv")

# Use final selected features
X = df.drop(["purchase", "event_type"], axis=1)
y = df["purchase"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Final selected model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Create output folder
os.makedirs("outputs/models", exist_ok=True)

# Save model
joblib.dump(model, "outputs/models/final_model.pkl")

print("Saved -> outputs/models/final_model.pkl")