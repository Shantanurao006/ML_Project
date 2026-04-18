import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# Load processed dataset
df = pd.read_csv("data/ecommerce_step1.csv")

print("\n===== STEP 2 : EXPLORATORY DATA ANALYSIS =====\n")

print("Dataset Shape:", df.shape)

# Convert event_time to datetime
df["event_time"] = pd.to_datetime(df["event_time"])

# Create extra time-based features
df["year"] = df["event_time"].dt.year
df["month"] = df["event_time"].dt.month
df["day"] = df["event_time"].dt.day
df["hour"] = df["event_time"].dt.hour

# Create output folder if not exists
import os
os.makedirs("outputs/charts", exist_ok=True)

# --------------------------------------------
# 1. Purchase Distribution
# --------------------------------------------
plt.figure(figsize=(6,4))
sns.countplot(x="purchase", data=df)
plt.title("Purchase Distribution")
plt.savefig("outputs/charts/purchase_distribution.png")
plt.close()

# --------------------------------------------
# 2. Event Type Distribution
# --------------------------------------------
plt.figure(figsize=(8,5))
sns.countplot(x="event_type", data=df, order=df["event_type"].value_counts().index)
plt.title("Event Type Distribution")
plt.xticks(rotation=45)
plt.savefig("outputs/charts/event_type_distribution.png")
plt.close()

# --------------------------------------------
# 3. Top 10 Brands
# --------------------------------------------
top_brands = df["brand"].value_counts().head(10)

plt.figure(figsize=(10,5))
sns.barplot(x=top_brands.index, y=top_brands.values)
plt.title("Top 10 Brands")
plt.xticks(rotation=45)
plt.savefig("outputs/charts/top_brands.png")
plt.close()

# --------------------------------------------
# 4. Top 10 Categories
# --------------------------------------------
top_categories = df["category_code"].value_counts().head(10)

plt.figure(figsize=(12,6))
sns.barplot(x=top_categories.index, y=top_categories.values)
plt.title("Top 10 Product Categories")
plt.xticks(rotation=75)
plt.savefig("outputs/charts/top_categories.png")
plt.close()

# --------------------------------------------
# 5. Price Distribution
# --------------------------------------------
plt.figure(figsize=(8,5))
sns.histplot(df["price"], bins=50)
plt.title("Price Distribution")
plt.savefig("outputs/charts/price_distribution.png")
plt.close()

# --------------------------------------------
# 6. Purchase by Hour
# --------------------------------------------
hourly_purchase = df[df["purchase"] == 1]["hour"].value_counts().sort_index()

plt.figure(figsize=(10,5))
sns.lineplot(x=hourly_purchase.index, y=hourly_purchase.values)
plt.title("Purchases by Hour")
plt.xlabel("Hour")
plt.ylabel("Number of Purchases")
plt.savefig("outputs/charts/purchase_by_hour.png")
plt.close()

# --------------------------------------------
# 7. Correlation Heatmap
# --------------------------------------------
numeric_df = df[["price", "purchase", "month", "day", "hour"]]

plt.figure(figsize=(8,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("outputs/charts/correlation_heatmap.png")
plt.close()

print("EDA Charts Saved in outputs/charts/")
print("\nTop 10 Brands:\n")
print(top_brands)

print("\nTop 10 Categories:\n")
print(top_categories)