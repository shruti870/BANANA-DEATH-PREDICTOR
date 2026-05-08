import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "data/raw/banana_dataset.csv"
SAVE_DIR  = "results/eda"
os.makedirs(SAVE_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="YlOrBr")
df = pd.read_csv(DATA_PATH)
FEATURE_COLS = [c for c in df.columns if c not in ("image_id", "days_to_death")]

print(df.describe().round(3))

plt.figure(figsize=(8, 4))
sns.histplot(df["days_to_death"], bins=20, color="#f5a623", edgecolor="black", kde=True)
plt.title("Distribution of Days to Banana Death")
plt.savefig(f"{SAVE_DIR}/target_distribution.png", dpi=150, bbox_inches="tight")
plt.close()

plt.figure(figsize=(10, 8))
corr = df[FEATURE_COLS + ["days_to_death"]].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="YlOrBr", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()

fig, axes = plt.subplots(2, 5, figsize=(18, 7))
axes = axes.flatten()
for i, col in enumerate(FEATURE_COLS):
    axes[i].scatter(df[col], df["days_to_death"], alpha=0.3, color="#f5a623")
    axes[i].set_xlabel(col, fontsize=9)
    axes[i].set_ylabel("Days to Death", fontsize=9)
plt.suptitle("Features vs Days to Death", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/feature_scatters.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"EDA plots saved to {SAVE_DIR}/")