from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import joblib
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from mlaz.data.io import load_csv
from mlaz.data.splits import split_features_target, train_test_split_xy


def _classification_metrics(y_true, y_pred, y_prob) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


def train_from_config(config_path: str) -> None:
    project_root = Path(__file__).resolve().parents[4]

    config_path_p = Path(config_path)
    if not config_path_p.is_absolute():
        config_path_p = (project_root / config_path_p).resolve()

    with open(config_path_p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Load data
    data_path = (project_root / cfg["data"]["path"]).resolve()
    df = load_csv(str(data_path))

    X, y = split_features_target(
        df,
        features=cfg["data"]["features"],
        target=cfg["data"]["target"],
    )

    X_train, X_test, y_train, y_test = train_test_split_xy(
        X,
        y,
        test_size=float(cfg["data"]["test_size"]),
        random_state=int(cfg["experiment"]["random_state"]),
        stratify=y,
    )

    steps = []
    if cfg.get("preprocess", {}).get("scale_features", False):
        steps.append(("scaler", StandardScaler()))

    steps.append(
        (
            "clf",
            LogisticRegression(
                random_state=int(cfg["experiment"]["random_state"]),
                **cfg["model"]["params"],
            ),
        )
    )

    pipeline = Pipeline(steps)
    pipeline.fit(X_train, y_train)

    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)
    y_test_prob = pipeline.predict_proba(X_test)[:, 1]

    train_metrics = _classification_metrics(
        y_train, y_train_pred, pipeline.predict_proba(X_train)[:, 1]
    )
    test_metrics = _classification_metrics(y_test, y_test_pred, y_test_prob)

    # Save model
    model_dir = project_root / "exports" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{cfg['experiment']['name']}_{run_id}.joblib"
    joblib.dump(pipeline, model_path)
    print("Saved model:", model_path)

    # Save metrics
    metrics_dir = project_root / "exports" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": cfg["experiment"]["name"],
        "run_id": run_id,
        "data_path": cfg["data"]["path"],
        "model": cfg["model"],
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }
    metrics_path = metrics_dir / f"{cfg['experiment']['name']}_{run_id}.json"
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("Saved metrics:", metrics_path)

    print("Train metrics:", train_metrics)
    print("Test metrics:", test_metrics)

    # Confusion matrix (console)
    cm = confusion_matrix(y_test, y_test_pred)
    print("Confusion matrix:\n", cm)


if __name__ == "__main__":
    train_from_config("configs/classification/logistic_regression.yaml")
