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
from mlaz.export.c_header import write_mlr_coeffs_header


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

    # Extract feature names after preprocessing (numeric + one-hot categories)
    pre = pipe.named_steps["preprocess"]
    cat_ohe = pre.named_transformers_["cat"]
    cat_feature_names = list(cat_ohe.get_feature_names_out(cfg["data"]["categorical_features"]))

    feature_names = cfg["data"]["numeric_features"] + cat_feature_names

    # Extract coefficients from the linear model (must match feature order)
    lr = pipe.named_steps["model"]
    coefs = [float(c) for c in lr.coef_.ravel()]
    intercept = float(lr.intercept_)

    # Export embedded C header
    deploy_cfg = cfg.get("deploy", {})
    embedded_dir = deploy_cfg.get("embedded_dir", "deploy/embedded/mlr_startups")
    header_guard = deploy_cfg.get("header_guard", "MLR_STARTUPS_MODEL_COEFFS_AUTOGEN_H")

    header_path = project_root / embedded_dir / "model_coeffs_autogen.h"
    write_mlr_coeffs_header(
        out_path=header_path,
        header_guard=header_guard,
        intercept=intercept,
        feature_names=feature_names,
        coefs=coefs,
    )
    print("Saved C header:", header_path)

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
