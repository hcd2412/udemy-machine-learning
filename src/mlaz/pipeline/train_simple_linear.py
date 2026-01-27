from __future__ import annotations

import yaml
import joblib
import json
from pathlib import Path
from datetime import datetime, timezone

from sklearn.linear_model import LinearRegression

from mlaz.data.io import load_csv
from mlaz.data.splits import split_features_target, train_test_split_xy
from mlaz.evaluation.metrics import regression_metrics
from mlaz.export.c_header import write_slr_coeffs_header


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

    # Split features / target
    X, y = split_features_target(
        df,
        features=cfg["data"]["features"],
        target=cfg["data"]["target"],
    )

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split_xy(
        X,
        y,
        test_size=cfg["data"]["test_size"],
        random_state=cfg["experiment"]["random_state"],
    )

    # Train model
    model = LinearRegression(**cfg["model"]["params"])
    model.fit(X_train, y_train)

    model_dir = project_root / "exports" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"{cfg['experiment']['name']}_{run_id}.joblib"
    joblib.dump(model, model_path)
    print("Saved model:", model_path)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Metrics
    train_metrics = regression_metrics(y_train, y_train_pred)
    test_metrics = regression_metrics(y_test, y_test_pred)

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

    # Temporary console output
    print("Intercept:", model.intercept_)
    print("Coefficients:", model.coef_)
    print("Train metrics:", train_metrics)
    print("Test metrics:", test_metrics)

    write_slr_coeffs_header(
        out_path=project_root / "deploy" / "embedded" / "slr_salary" / "model_coeffs_autogen.h",
        header_guard="SLR_SALARY_MODEL_COEFFS_AUTOGEN_H",
        intercept=float(model.intercept_),
        coef=float(model.coef_[0]),
    )
    print("Saved C header:", project_root / "deploy" / "embedded" / "slr_salary" / "model_coeffs_autogen.h")

if __name__ == "__main__":
    train_from_config("configs/regression/simple_linear.yaml")
