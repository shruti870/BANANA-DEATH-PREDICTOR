import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH   = "data/raw/banana_dataset.csv"
MODEL_DIR   = "models"
RESULTS_DIR = "results"

FEATURE_COLS = [
    "yellow_ratio", "brown_ratio", "green_ratio", "black_ratio",
    "texture_roughness", "mean_brightness", "std_brightness",
    "edge_density", "hue_mean", "saturation_mean",
]
TARGET_COL = "days_to_death"

def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    return X, y

def evaluate(model, X_test, y_test, name="Model"):
    preds = model.predict(X_test)
    mae   = mean_absolute_error(y_test, preds)
    rmse  = np.sqrt(mean_squared_error(y_test, preds))
    r2    = r2_score(y_test, preds)
    print(f"\n{'='*40}")
    print(f"  {name}")
    print(f"  MAE  : {mae:.4f} days")
    print(f"  RMSE : {rmse:.4f} days")
    print(f"  R2   : {r2:.4f}")
    print(f"{'='*40}")
    return {"model": name, "MAE": mae, "RMSE": rmse, "R2": r2, "preds": preds}

def plot_results(y_test, results_list):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    best = min(results_list, key=lambda x: x["MAE"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Banana Death Predictor — Results", fontsize=15, fontweight="bold")

    ax = axes[0]
    ax.scatter(y_test, best["preds"], alpha=0.6, color="#f5a623", edgecolors="black", linewidths=0.4)
    lims = [0, 15]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual Days to Death")
    ax.set_ylabel("Predicted Days to Death")
    ax.set_title(f"Actual vs Predicted ({best['model']})\nR2={best['R2']:.3f}, MAE={best['MAE']:.3f}")
    ax.legend()
    ax.set_xlim(lims); ax.set_ylim(lims)

    ax2 = axes[1]
    names  = [r["model"] for r in results_list]
    maes   = [r["MAE"]   for r in results_list]
    colors = ["#f5a623" if n == best["model"] else "#ccc" for n in names]
    bars = ax2.barh(names, maes, color=colors, edgecolor="black", linewidth=0.5)
    ax2.set_xlabel("Mean Absolute Error (days)")
    ax2.set_title("Model Comparison (MAE lower = better)")
    for bar, val in zip(bars, maes):
        ax2.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                 f"{val:.3f}", va="center", fontsize=10)
    ax2.set_xlim(0, max(maes) * 1.25)

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/model_comparison.png", dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {RESULTS_DIR}/model_comparison.png")
    plt.close()

def plot_feature_importance(model, feature_names):
    gbr = model.named_steps["regressor"]
    if not hasattr(gbr, "feature_importances_"):
        return
    imp = gbr.feature_importances_
    idx = np.argsort(imp)
    plt.figure(figsize=(8, 5))
    plt.barh([feature_names[i] for i in idx], imp[idx], color="#f5a623", edgecolor="black", linewidth=0.5)
    plt.xlabel("Feature Importance")
    plt.title("Feature Importance — Gradient Boosting")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/feature_importance.png", dpi=150, bbox_inches="tight")
    print(f"Feature importance saved to {RESULTS_DIR}/feature_importance.png")
    plt.close()

def main():
    print("Loading data...")
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    models = {
        "Ridge Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=1.0)),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", GradientBoostingRegressor(
                n_estimators=300, learning_rate=0.05,
                max_depth=4, subsample=0.8, random_state=42
            )),
        ]),
    }

    results = []
    trained = {}
    for name, pipe in models.items():
        print(f"\nTraining {name}...")
        pipe.fit(X_train, y_train)
        res = evaluate(pipe, X_test, y_test, name=name)
        results.append(res)
        trained[name] = pipe

    best_name  = min(results, key=lambda x: x["MAE"])["model"]
    best_model = trained[best_name]

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model,   f"{MODEL_DIR}/best_model.pkl")
    joblib.dump(FEATURE_COLS, f"{MODEL_DIR}/feature_cols.pkl")
    print(f"\nBest model: {best_name} saved to {MODEL_DIR}/best_model.pkl")

    plot_results(y_test, results)
    plot_feature_importance(best_model, FEATURE_COLS)

    metrics_df = pd.DataFrame([{k: v for k, v in r.items() if k != "preds"} for r in results])
    metrics_df.to_csv(f"{RESULTS_DIR}/metrics.csv", index=False)
    print(f"Metrics saved to {RESULTS_DIR}/metrics.csv")

if __name__ == "__main__":
    main()