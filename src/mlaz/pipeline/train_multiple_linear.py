from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from mlaz.data.io import load_csv
from mlaz.data.splits import split_features_target, train_test_split_xy
from mlaz.evaluation.metrics import regression_metrics


def train_from_config(config_path: str) -> None:
    project_root = Path(__file__).resolve().parents[3]

    # Resolve config path
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = (project_root / config_path).resolve()

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Load data
    data_path = (project_root / cfg["data"]["path"]).resolve()
    df = load_csv(str(data_path))

    # Build X/y
    all_features = cfg["data"]["numeric_features"] + cfg["data"]["categorical_features"]
    X, y = split_features_target(df, features=all_features, target=cfg["data"]["target"])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split_xy(
        X,
        y,
        test_size=cfg["data"]["test_size"],
        random_state=cfg["experiment"]["random_state"],
    )

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", cfg["data"]["numeric_features"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cfg["data"]["categorical_features"]),
        ],
        remainder="drop",
    )

    # Model
    model = LinearRegression(**cfg["model"]["params"])

    # Full pipeline
    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    # Train
    pipe.fit(X_train, y_train)

    # Predictions + metrics
    y_train_pred = pipe.predict(X_train)
    y_test_pred = pipe.predict(X_test)

    train_metrics = regression_metrics(y_train, y_train_pred)
    test_metrics = regression_metrics(y_test, y_test_pred)

    # Save model artifact
    model_dir = project_root / "exports" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{cfg['experiment']['name']}_{run_id}.joblib"
    joblib.dump(pipe, model_path)

    # Save metrics artifact
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

    # Console output (temporary)
    print("Saved model:", model_path)
    print("Saved metrics:", metrics_path)
    print("Train metrics:", train_metrics)
    print("Test metrics:", test_metrics)


if __name__ == "__main__":
    train_from_config("configs/regression/multiple_linear.yaml")
