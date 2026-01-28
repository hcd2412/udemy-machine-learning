from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import yaml
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from mlaz.data.io import load_csv
from mlaz.data.splits import split_features_target, train_test_split_xy
from mlaz.evaluation.metrics import regression_metrics


def train_from_config(config_path: str) -> None:
    project_root = Path(__file__).resolve().parents[3]

    # Resolve config path (supports running from anywhere)
    config_path_p = Path(config_path)
    if not config_path_p.is_absolute():
        config_path_p = (project_root / config_path_p).resolve()

    with open(config_path_p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

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

    degree = int(cfg["model"]["degree"])
    include_bias = bool(cfg["model"].get("include_bias", False))

    # Pipeline: poly features -> linear regression
    pipe = Pipeline(
        steps=[
            ("poly", PolynomialFeatures(degree=degree, include_bias=include_bias)),
            ("model", LinearRegression()),
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
        "model": {
            "type": cfg["model"]["type"],
            "degree": degree,
            "include_bias": include_bias,
        },
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }

    out_path = out_dir / f"{cfg['experiment']['name']}_{run_id}.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("Saved metrics:", out_path)

    print("Train metrics:", train_metrics)
    print("Test metrics:", test_metrics)


if __name__ == "__main__":
    train_from_config("configs/regression/polynomial.yaml")
