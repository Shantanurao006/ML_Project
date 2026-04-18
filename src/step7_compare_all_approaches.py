import pandas as pd
import matplotlib.pyplot as plt
import os

print("\n===== FINAL COMPARISON OF ALL APPROACHES =====\n")

# Load all results
a1 = pd.read_csv("outputs/comparison/approach1_results.csv")
a2 = pd.read_csv("outputs/comparison/approach2_results.csv")
a3 = pd.read_csv("outputs/comparison/approach3_results.csv")
a4 = pd.read_csv("outputs/comparison/approach4_results.csv")

# Combine
results = pd.concat([a1, a2, a3, a4], ignore_index=True)

print(results)

# Save final table
results.to_csv("outputs/comparison/final_comparison.csv", index=False)

os.makedirs("outputs/comparison", exist_ok=True)

# Accuracy
plt.figure(figsize=(10,5))
plt.bar(results["Model"], results["Accuracy"])
plt.title("Accuracy Comparison")
plt.ylabel("Accuracy")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("outputs/comparison/accuracy_comparison.png")
plt.close()

# Recall
plt.figure(figsize=(10,5))
plt.bar(results["Model"], results["Recall"])
plt.title("Recall Comparison")
plt.ylabel("Recall")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("outputs/comparison/recall_comparison.png")
plt.close()

# F1
plt.figure(figsize=(10,5))
plt.bar(results["Model"], results["F1 Score"])
plt.title("F1 Score Comparison")
plt.ylabel("F1 Score")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("outputs/comparison/f1_comparison.png")
plt.close()

# Training Time
plt.figure(figsize=(10,5))
plt.bar(results["Model"], results["Training Time"])
plt.title("Training Time Comparison")
plt.ylabel("Seconds")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("outputs/comparison/time_comparison.png")
plt.close()

print("\nUpdated comparison files saved in outputs/comparison/")