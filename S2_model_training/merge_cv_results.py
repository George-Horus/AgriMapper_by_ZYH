"""Merge compatible five-fold CV runs created by run_model_comparison.py.

This utility is used when model families are run separately to stay within a
job time limit.  It verifies that every input used the identical data audit,
feature list, and shared outer-fold assignment before producing one formal
result directory.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

try:
    from .cv_pipeline import _build_summary, _write_excel
except ImportError:  # Allow `python merge_cv_results.py` from this directory.
    from cv_pipeline import _build_summary, _write_excel


REQUIRED_FILES = (
    "input_data_audit.csv",
    "feature_columns.csv",
    "fold_assignments.csv",
    "fold_metrics.csv",
    "oof_predictions.csv",
    "model_status.csv",
    "run_metadata.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge compatible completed five-fold CV result directories.")
    parser.add_argument("--inputs", nargs="+", type=Path, required=True, help="Completed CV result directories.")
    parser.add_argument("--output", type=Path, required=True, help="New directory for the merged results.")
    return parser.parse_args()


def read_metadata(directory: Path) -> dict:
    return json.loads((directory / "run_metadata.json").read_text(encoding="utf-8"))


def require_completed_directory(directory: Path) -> None:
    missing = [name for name in REQUIRED_FILES if not (directory / name).exists()]
    if missing:
        raise FileNotFoundError(f"{directory} is incomplete; missing: {', '.join(missing)}")


def assert_equal(left: pd.DataFrame, right: pd.DataFrame, filename: str, directory: Path) -> None:
    if not left.equals(right):
        raise ValueError(f"{filename} in {directory} does not match the first input directory.")


def main() -> None:
    args = parse_args()
    source_dirs = [directory.resolve() for directory in args.inputs]
    if len(set(source_dirs)) != len(source_dirs):
        raise ValueError("Each input directory must be provided once.")
    for directory in source_dirs:
        require_completed_directory(directory)

    audits = [pd.read_csv(directory / "input_data_audit.csv") for directory in source_dirs]
    features = [pd.read_csv(directory / "feature_columns.csv") for directory in source_dirs]
    assignments = [pd.read_csv(directory / "fold_assignments.csv") for directory in source_dirs]
    metadata = [read_metadata(directory) for directory in source_dirs]
    for index, directory in enumerate(source_dirs[1:], start=1):
        assert_equal(audits[0], audits[index], "input_data_audit.csv", directory)
        assert_equal(features[0], features[index], "feature_columns.csv", directory)
        assert_equal(assignments[0], assignments[index], "fold_assignments.csv", directory)
        for field in ("data_path", "target_column", "n_rows_input", "n_rows_included", "n_features", "n_splits", "random_state", "splitter"):
            if metadata[index].get(field) != metadata[0].get(field):
                raise ValueError(f"Metadata field '{field}' in {directory} does not match the first input directory.")

    source_fold_metrics = [pd.read_csv(directory / "fold_metrics.csv") for directory in source_dirs]
    source_model_sets = [set(frame["Model"]) for frame in source_fold_metrics]
    for index, model_set in enumerate(source_model_sets):
        for other_set in source_model_sets[index + 1:]:
            overlap = model_set & other_set
            if overlap:
                raise ValueError(f"Models appear in more than one input directory: {', '.join(sorted(overlap))}")

    fold_metrics = pd.concat(source_fold_metrics, ignore_index=True)
    oof_predictions = pd.concat([pd.read_csv(directory / "oof_predictions.csv") for directory in source_dirs], ignore_index=True)
    status = pd.concat([pd.read_csv(directory / "model_status.csv") for directory in source_dirs], ignore_index=True)
    if status["Model"].duplicated().any():
        raise ValueError("A model status appears in more than one input directory.")

    expected_folds = int(metadata[0]["n_splits"])
    expected_rows = int(metadata[0]["n_rows_included"])
    fold_counts = fold_metrics.groupby("Model")["Fold"].nunique()
    oof_counts = oof_predictions.groupby("Model").size()
    if not (fold_counts == expected_folds).all():
        raise ValueError("At least one model does not have a complete set of outer-fold metrics.")
    if not (oof_counts == expected_rows).all():
        raise ValueError("At least one model does not have one OOF prediction per included sample.")

    summary = _build_summary(fold_metrics)
    output_dir = args.output.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    audits[0].to_csv(output_dir / "input_data_audit.csv", index=False)
    features[0].to_csv(output_dir / "feature_columns.csv", index=False)
    assignments[0].to_csv(output_dir / "fold_assignments.csv", index=False)
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    oof_predictions.to_csv(output_dir / "oof_predictions.csv", index=False)
    summary.to_csv(output_dir / "model_summary.csv", index=False)
    status.to_csv(output_dir / "model_status.csv", index=False)

    merged_metadata = dict(metadata[0])
    merged_metadata.update(
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "merged_from": [str(directory) for directory in source_dirs],
            "requested_models": status["Model"].tolist(),
            "completed_models": summary["Model"].tolist(),
            "merge_validation": "Input audit, feature columns, outer-fold assignments, and core CV metadata matched across all source runs.",
        }
    )
    (output_dir / "run_metadata.json").write_text(json.dumps(merged_metadata, indent=2), encoding="utf-8")
    workbook = _write_excel(output_dir, summary, fold_metrics, oof_predictions, assignments[0], status, merged_metadata)
    print(f"Merged {len(summary)} models into: {output_dir}")
    print(f"Workbook: {workbook}")


if __name__ == "__main__":
    main()
