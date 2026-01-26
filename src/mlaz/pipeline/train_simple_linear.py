from __future__ import annotations

import yaml
from pathlib import Path

from sklearn.linear_model import LinearRegression

from mlaz.data.io import load_csv
from mlaz.data.splits import split_features_target, train_test_split_xy


def train_from_config(config_path: str) -> None:
    # Load config
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

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

    # Basic sanity output (temporary)
    print("Intercept:", model.intercept_)
    print("Coefficients:", model.coef_)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[3]
    config_file = project_root / "configs" / "regression" / "simple_linear.yaml"
    train_from_config(str(config_file))
