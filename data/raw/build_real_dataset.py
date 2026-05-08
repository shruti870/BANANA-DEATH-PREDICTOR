import os
import pandas as pd
import sys
sys.path.append("src")
from feature_extraction import extract_features_from_image

LABEL_MAP = {
    "unripe":   12,
    "ripe":      6,
    "overripe":  2,
    "rotten":    0,
}

IMAGE_DIR = "data/raw/real_images"
rows = []
total = 0

for label, days in LABEL_MAP.items():
    folder = os.path.join(IMAGE_DIR, label)
    files  = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    print(f"\nProcessing {label}: {len(files)} images...")
    for fname in files:
        path = os.path.join(folder, fname)
        try:
            feats = extract_features_from_image(path)
            feats["image_id"]      = fname
            feats["days_to_death"] = days
            rows.append(feats)
            total += 1
            if total % 100 == 0:
                print(f"  Done {total} images so far...")
        except Exception as e:
            print(f"  SKIP {fname}: {e}")

df = pd.DataFrame(rows)
df.to_csv("data/raw/banana_dataset.csv", index=False)
print(f"\nDataset saved: {len(df)} images")
print(df["days_to_death"].value_counts())