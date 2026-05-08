import argparse
import joblib
import pandas as pd
import numpy as np
import os
import sys
sys.path.append("src")

MODEL_PATH    = "models/best_model.pkl"
FEATURES_PATH = "models/feature_cols.pkl"

def load_model():
    model    = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    return model, features

def predict_from_csv(csv_path: str, n_samples: int = 10):
    model, features = load_model()
    df = pd.read_csv(csv_path)
    sample = df.sample(n=n_samples, random_state=0)
    X = sample[features]
    preds = np.clip(np.round(model.predict(X), 1), 0, 14)
    print(f"\n{'='*55}")
    print(f"  {'Image ID':<20} {'Actual':>8} {'Predicted':>10} {'Diff':>8}")
    print(f"{'='*55}")
    for row, pred in zip(sample.itertuples(), preds):
        actual = row.days_to_death
        diff   = pred - actual
        sign   = "+" if diff >= 0 else ""
        print(f"  {row.image_id:<20} {actual:>8.1f} {pred:>10.1f} {sign}{diff:>7.1f}")
    print(f"{'='*55}\n")

def predict_from_image(image_path: str):
    from feature_extraction import extract_features_from_image
    model, features = load_model()
    feat_dict = extract_features_from_image(image_path)
    X = pd.DataFrame([feat_dict])[features]
    pred = float(np.clip(model.predict(X)[0], 0, 14))
    print(f"\nImage     : {image_path}")
    print(f"Prediction: {pred:.1f} days until banana death")
    if pred < 1:
        print("Status    : Already dead / rotten")
    elif pred < 3:
        print("Status    : Use immediately!")
    elif pred < 7:
        print("Status    : Eat within a few days")
    else:
        print("Status    : Still fresh")
    return pred

def predict_all_folders():
    from feature_extraction import extract_features_from_image
    model, features = load_model()

    FOLDERS = {
        "unripe":   "data/raw/real_images/unripe",
        "ripe":     "data/raw/real_images/ripe",
        "overripe": "data/raw/real_images/overripe",
        "rotten":   "data/raw/real_images/rotten",
    }

    print(f"\n{'='*65}")
    print(f"  {'Folder':<12} {'Image':<30} {'Predicted':>10}")
    print(f"{'='*65}")

    for label, folder in FOLDERS.items():
        if not os.path.exists(folder):
            print(f"  {label:<12} FOLDER NOT FOUND")
            continue
        files = [f for f in os.listdir(folder)
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if not files:
            print(f"  {label:<12} NO IMAGES FOUND")
            continue
        fname = files[0]
        path  = os.path.join(folder, fname)
        try:
            feat_dict = extract_features_from_image(path)
            X = pd.DataFrame([feat_dict])[features]
            pred = float(np.clip(model.predict(X)[0], 0, 14))
            if pred < 1:
                status = "Rotten"
            elif pred < 3:
                status = "Use immediately!"
            elif pred < 7:
                status = "Eat soon"
            else:
                status = "Still fresh"
            print(f"  {label:<12} {fname[:28]:<30} {pred:>6.1f} days  ->  {status}")
        except Exception as e:
            print(f"  {label:<12} ERROR: {e}")

    print(f"{'='*65}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",     type=str)
    parser.add_argument("--image",   type=str)
    parser.add_argument("--folders", action="store_true",
                        help="Test one image from each folder")
    args = parser.parse_args()

    if args.csv:
        predict_from_csv(args.csv)
    elif args.image:
        predict_from_image(args.image)
    elif args.folders:
        predict_all_folders()
    else:
        predict_all_folders()