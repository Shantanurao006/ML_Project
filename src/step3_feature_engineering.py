import pandas as pd
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore")

print("\n===== STEP 3 : FEATURE ENGINEERING =====\n")

# Load dataset
df = pd.read_csv("data/ecommerce_step1.csv")

# Convert event_time to datetime
df["event_time"] = pd.to_datetime(df["event_time"])

# Create time features
df["year"] = df["event_time"].dt.year
df["month"] = df["event_time"].dt.month
df["day"] = df["event_time"].dt.day
df["hour"] = df["event_time"].dt.hour
df["weekday"] = df["event_time"].dt.weekday

# Fill missing values
df["brand"] = df["brand"].fillna("unknown")
df["category_code"] = df["category_code"].fillna("unknown")

# Keep top 20 brands only
top_brands = df["brand"].value_counts().head(20).index
df["brand"] = df["brand"].apply(
    lambda x: x if x in top_brands else "other"
)

# Keep top 20 categories only
top_categories = df["category_code"].value_counts().head(20).index
df["category_code"] = df["category_code"].apply(
    lambda x: x if x in top_categories else "other"
)

# Encode categorical columns
label_encoders = {}

for col in ["event_type", "brand", "category_code"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Select final features
features = [
    "event_type",
    "brand",
    "category_code",
    "price",
    "month",
    "day",
    "hour",
    "weekday"
]

target = "purchase"

X = df[features]
y = df[target]

print("Feature Columns:")
print(features)

print("\nFeature Dataset Shape:", X.shape)
print("Target Shape:", y.shape)

# Save processed dataset
processed_df = pd.concat([X, y], axis=1)
processed_df.to_csv("data/ecommerce_features.csv", index=False)

print("\nFeature Engineering Completed Successfully")
print("Saved: data/ecommerce_features.csv")