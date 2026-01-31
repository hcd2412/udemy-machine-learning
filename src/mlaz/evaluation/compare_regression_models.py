from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


def _load_metrics_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _flatten_record(rec: Dict[str, Any], source_file: str) -> Dict[str, Any]:
    # Expecting your pipeline payload structure:
    # {
    #   "experiment": "...",
    #   "run_id": "...",
    #   "data_path": "...",
    #   "model": {...},
    #   "train_metrics": {...},
    #   "test_metrics": {...}
    # }
    out: Dict[str, Any] = {
        "source_file": source_file,
        "experiment": rec.get("experiment"),
        "run_id": rec.get("run_id"),
        "data_path": rec.get("data_path"),
        "model_type": (rec.get("model") or {}).get("type"),
    }

    tr = rec.get("train_metrics") or {}
    te = rec.get("test_metrics") or {}
    for k, v in tr.items():
        out[f"train_{k}"] = v
    for k, v in te.items():
        out[f"test_{k}"] = v

    return out


def build_comparison_table(metrics_dir: str = "exports/metrics") -> pd.DataFrame:
    project_root = Path(__file__).resolve().parents[3]
    mdir = (project_root / metrics_dir).resolve()

    if not mdir.exists():
        raise FileNotFoundError(f"Metrics directory not found: {mdir}")

    rows: List[Dict[str, Any]] = []
    for p in sorted(mdir.glob("*.json")):
        rec = _load_metrics_json(p)
        rows.append(_flatten_record(rec, source_file=p.name))

    if not rows:
        raise RuntimeError(f"No metrics JSON files found in: {mdir}")

    df = pd.DataFrame(rows)

    # Drop exact duplicate rows (same metrics repeated)
    df = df.drop_duplicates()

    # Keep only the best run per (data_path, model_type)
    if "test_rmse" in df.columns:
        df = df.sort_values(by=["data_path", "model_type", "test_rmse"], ascending=[True, True, True])
        df = df.groupby(["data_path", "model_type"], as_index=False).head(1)
        # Re-sort for display: best overall per dataset
        df = df.sort_values(by=["data_path", "test_rmse"], ascending=[True, True])

    # Default sort: best test RMSE first (lower is better)
    if "test_rmse" in df.columns:
        df = df.sort_values(by=["data_path", "test_rmse"], ascending=[True, True])

    return df


def main() -> None:
    df = build_comparison_table()

    # Print a clean view
    cols = [
        "data_path",
        "experiment",
        "model_type",
        "run_id",
        "test_rmse",
        "test_mae",
        "test_r2",
    ]
    cols = [c for c in cols if c in df.columns]

    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
