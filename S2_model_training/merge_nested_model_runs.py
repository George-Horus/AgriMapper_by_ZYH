"""Merge model tables produced by the base and Agrimapper environments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main", type=Path, required=True)
    parser.add_argument("--optional", type=Path, required=True)
    args = parser.parse_args()

    specs = {
        "fold_metrics.csv": ["Model", "Fold"],
        "oof_predictions.csv": ["Model", "Fold", "candidate_row"],
        "model_summary.csv": ["Model"],
        "model_status.csv": ["Model"],
    }
    for filename, keys in specs.items():
        left = pd.read_csv(args.main / filename)
        right = pd.read_csv(args.optional / filename)
        merged = pd.concat([left, right], ignore_index=True).drop_duplicates(keys, keep="last")
        sort_keys = ["Model"] if filename in {"model_summary.csv", "model_status.csv"} else keys
        merged.sort_values(sort_keys).to_csv(args.main / filename, index=False)

    metadata_path = args.main / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    optional_metadata = json.loads((args.optional / "run_metadata.json").read_text(encoding="utf-8"))
    statuses = pd.read_csv(args.main / "model_status.csv").to_dict(orient="records")
    metadata["status"] = statuses
    metadata["merged_model_environments"] = {
        "base_python": [row["Model"] for row in statuses if row["Model"] not in {"AutoGluon", "TPOT", "XGBoost"}],
        "Agrimapper_conda_environment": ["AutoGluon", "TPOT", "XGBoost"],
    }
    metadata["optional_run_metadata"] = optional_metadata
    metadata["note"] = "Model groups were executed in two environments; fold assignments and nested feature selection were identical."
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Merged {len(statuses)} model statuses into {args.main.resolve()}")


if __name__ == "__main__":
    main()
