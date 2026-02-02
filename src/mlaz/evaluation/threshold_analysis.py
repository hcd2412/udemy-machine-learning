from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from mlaz.data.io import load_csv
from mlaz.data.splits import split_features_target
from mlaz.data.splits import train_test_split_xy


def _latest_model_path(models_dir: Path, prefix: str) -> Path:
    candidates = sorted(models_dir.glob(f"{prefix}_*.joblib"))
    if not candidates:
        raise FileNotFoundError(f"No model files found with prefix: {prefix} in {models_dir}")
    return candidates[-1]


def compute_curve(y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        rows.append(
            {
                "threshold": float(t),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]

    # Use your existing dataset path
    data_path = project_root / "data/raw/classification/social_network_ads.csv"
    df = load_csv(str(data_path))

    X, y = split_features_target(
        df,
        features=["Age", "EstimatedSalary"],
        target="Purchased",
    )

    X_train, X_test, y_train, y_test = train_test_split_xy(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    # Load the latest trained logistic regression model
    models_dir = project_root / "exports/models"
    model_path = _latest_model_path(models_dir, "logistic_regression_social_network_ads")
    model = joblib.load(model_path)

    # Probabilities (note: this is on the FULL dataset, just for threshold intuition)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_true = y_test.to_numpy().astype(int)

    thresholds = np.linspace(0.05, 0.95, 19)
    curve = compute_curve(y_true, y_prob, thresholds)

    best_idx = int(curve["f1"].idxmax())
    best_t = float(curve.loc[best_idx, "threshold"])

    print("Loaded model:", model_path.name)
    print("Best F1 threshold:", best_t)
    print(curve.to_string(index=False))

    # Plot
    plt.plot(curve["threshold"], curve["precision"], label="precision")
    plt.plot(curve["threshold"], curve["recall"], label="recall")
    plt.plot(curve["threshold"], curve["f1"], label="f1")
    plt.axvline(best_t, linestyle="--", label=f"best_f1={best_t:.2f}")
    plt.xlabel("threshold")
    plt.ylabel("score")
    plt.legend()
    plt.title("Threshold tuning (test set)")
    plt.show()

    # Confusion matrix at best threshold
    y_pred_best = (y_prob >= best_t).astype(int)
    cm = confusion_matrix(y_true, y_pred_best)
    print("Confusion matrix @ best F1 threshold:\n", cm)


if __name__ == "__main__":
    main()
