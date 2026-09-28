"""Prepare the All_features_N_filtered_240 workbook for nested CV."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_excel(args.input)
    if "N" not in frame.columns:
        raise ValueError("Input workbook must contain an N target column")
    feature_columns = [column for column in frame.columns if column != "N"]
    if len(frame) != 240 or len(feature_columns) <= 12:
        raise ValueError(f"Expected 240 rows and >12 candidate features, got {frame.shape}")

    output = frame.copy()
    output.insert(0, "source_file", args.input.name)
    output.insert(1, "source_excel_row", np.arange(len(output), dtype=int) + 2)
    output.insert(2, "No", np.arange(len(output), dtype=int) + 1)
    output = output[["source_file", "source_excel_row", "No", "N", *feature_columns]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_workbook": str(args.input.resolve()),
        "source_workbook_sha256": sha256(args.input),
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "n_rows": len(output),
        "n_candidate_features": len(feature_columns),
        "selection_basis": "All predictor columns in All_features_N_filtered_240.xlsx; no outcome-dependent feature selection before nested CV",
    }
    args.output.with_name(args.output.stem + "_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
