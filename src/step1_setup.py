import pandas as pd
import warnings

warnings.filterwarnings("ignore")

print("\n===== STEP 1 : DATASET LOADING =====\n")

# Load only first 500000 rows for initial development
df = pd.read_csv(
    "data/ecommerce.csv",
    nrows=500000
)

print("Dataset Shape:", df.shape)

print("\nColumns:\n")
print(df.columns.tolist())

print("\nFirst 5 Rows:\n")
print(df.head())

print("\nMissing Values:\n")
print(df.isnull().sum())

# Create target column
df["purchase"] = df["event_type"].apply(
    lambda x: 1 if str(x).lower() == "purchase" else 0
)

print("\nPurchase Distribution:\n")
print(df["purchase"].value_counts())

# Save smaller working dataset
df.to_csv("data/ecommerce_step1.csv", index=False)

print("\nStep 1 Completed Successfully")