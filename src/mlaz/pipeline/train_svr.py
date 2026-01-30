from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from mlaz.data.io import load_csv
from mlaz.data.splits import split_features_target, train_test_split_xy
from mlaz.evaluation.metrics import regression_metrics


def train_from_config(config_path: str) -> None:
    project_root = Path(__file__).resolve().parents[3]

    # Resolve config path
    config_path_p = Path(config_path)
    if not config_path_p.is_absolute():
        config_path_p = (project_root / config_path_p).resolve()

    with open(config_path_p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        print("Loaded model config:", cfg["model"])

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Load data
    data_path = (project_root / cfg["data"]["path"]).resolve()
    df = load_csv(str(data_path))

    # Split features / target
    X, y = split_features_target(
        df,
        features=cfg["data"]["features"],
        target=cfg["data"]["target"],
    )

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split_xy(
        X,
        y,
        test_size=float(cfg["data"]["test_size"]),
        random_state=int(cfg["experiment"]["random_state"]),
    )

    # SVR requires scaling
    svr = SVR(
        kernel=cfg["model"]["kernel"],
        C=float(cfg["model"]["C"]),
        epsilon=float(cfg["model"]["epsilon"]),
        gamma=cfg["model"]["gamma"],
    )

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("svr", svr),
        ]
    )

    pipe.fit(X_train, y_train)

    # Predictions
    y_train_pred = pipe.predict(X_train)
    y_test_pred = pipe.predict(X_test)

    # Metrics
    train_metrics = regression_metrics(y_train, y_train_pred)
    test_metrics = regression_metrics(y_test, y_test_pred)

    # Persist model
    model_dir = project_root / "exports" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"{cfg['experiment']['name']}_{run_id}.joblib"
    joblib.dump(pipe, model_path)
    print("Saved model:", model_path)

    # Persist metrics
    out_dir = project_root / "exports" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "experiment": cfg["experiment"]["name"],
        "run_id": run_id,
        "data_path": cfg["data"]["path"],
        "model": cfg["model"],
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }

    out_path = out_dir / f"{cfg['experiment']['name']}_{run_id}.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("Saved metrics:", out_path)

    print("Train metrics:", train_metrics)
    print("Test metrics:", test_metrics)

    # --- Plot: Actual vs SVR curve ---
    feature_name = cfg["data"]["features"][0]  # e.g., "Level"

    # Build a smooth grid WITH feature name to avoid sklearn warning
    x_min = float(X[feature_name].min())
    x_max = float(X[feature_name].max())
    X_grid = pd.DataFrame(
        {feature_name: np.linspace(x_min, x_max, 200)}
    )

    y_grid_pred = pipe.predict(X_grid)

    plt.figure(figsize=(8, 5))

    # Actual data
    plt.scatter(X[feature_name], y, color="tab:blue", alpha=0.8, s=50, zorder=3, label="Actual salary")

    # SVR prediction (smooth curve)
    plt.plot(X_grid[feature_name], y_grid_pred, color="tab:red", linewidth=2.0, zorder=2, label="SVR prediction")

    plt.title("SVR: Level vs Salary")
    plt.xlabel(feature_name)
    plt.ylabel(cfg["data"]["target"])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    train_from_config("configs/regression/svr.yaml")
