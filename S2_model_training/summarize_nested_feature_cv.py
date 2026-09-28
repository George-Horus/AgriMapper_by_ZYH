"""Validate and summarize a completed nested-feature CV run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def metric(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {"R2": r2_score(y_true, y_pred), "RMSE": np.sqrt(mse), "MSE": mse,
            "MAE": mean_absolute_error(y_true, y_pred)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    d = args.results
    candidate = pd.read_csv(d / "candidate_features.csv")
    assignments = pd.read_csv(d / "fold_assignments.csv")
    features = pd.read_csv(d / "fold_features.csv")
    folds = pd.read_csv(d / "fold_metrics.csv")
    oof = pd.read_csv(d / "oof_predictions.csv")
    summary = pd.read_csv(d / "model_summary.csv")
    status = pd.read_csv(d / "model_status.csv")

    checks = []
    checks.append({"check": "candidate_rows", "passed": len(candidate) == 240, "actual": len(candidate), "expected": 240})
    checks.append({"check": "candidate_features", "passed": len(candidate.columns) - 4 == 196,
                   "actual": len(candidate.columns) - 4, "expected": 196})
    checks.append({"check": "all_models_completed", "passed": bool((status.Status == "completed").all()) and len(status) == 15,
                   "actual": status.Status.tolist(), "expected": "15 completed models"})
    checks.append({"check": "fold_metrics_rows", "passed": len(folds) == 75, "actual": len(folds), "expected": 75})
    checks.append({"check": "oof_rows", "passed": len(oof) == 3600, "actual": len(oof), "expected": 3600})
    selected = {int(row.fold): int(row.selected_count) for row in features.itertuples()}
    checks.append({"check": "outer_fold_sizes", "passed": assignments.groupby("outer_fold").size().eq(48).all(),
                   "actual": assignments.groupby("outer_fold").size().to_dict(), "expected": "48 per fold"})
    for model, group in oof.groupby("Model"):
        checks.append({"check": f"{model}_oof_rows", "passed": len(group) == 240 and group.candidate_row.nunique() == 240,
                       "actual": len(group), "expected": 240})
        model_folds = folds[folds.Model == model]
        for _, row in model_folds.iterrows():
            part = group[group.Fold == row.Fold]
            actual = metric(part.y_true, part.y_pred)
            for name in ("R2", "RMSE", "MSE", "MAE"):
                checks.append({"check": f"{model}_fold_{int(row.Fold)}_{name}",
                               "passed": bool(np.isclose(actual[name], row[name], rtol=1e-8, atol=1e-8)),
                               "actual": float(actual[name]), "expected": float(row[name])})
    failed = [check for check in checks if not check["passed"]]
    if failed:
        raise AssertionError(f"{len(failed)} audit checks failed: {failed[:2]}")

    audit = {"results_dir": str(d.resolve()), "n_rows": len(candidate), "n_candidate_features": 196,
             "models": sorted(oof.Model.unique()), "checks_passed": len(checks),
             "outer_cv": "5-fold KFold, shuffle=True, random_state=42",
             "inner_cv": "3-fold KFold inside each outer training fold",
             "selection": "training-only median imputation, |Pearson r| > 0.2, RF-RFE, inner RF RMSE feature-count choice",
             "limitations": ["The 240-row workbook is a preselected case table.",
                             "Random KFold is not a grouped plot/date or independent growth-stage holdout."]}
    (d / "nested_feature_cv_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    with pd.ExcelWriter(d / "nested_feature_cv_summary.xlsx", engine="openpyxl") as writer:
        summary.sort_values("R2_mean", ascending=False).to_excel(writer, sheet_name="model_summary", index=False)
        folds.to_excel(writer, sheet_name="fold_metrics", index=False)
        features.to_excel(writer, sheet_name="fold_features", index=False)
        pd.DataFrame(checks).to_excel(writer, sheet_name="audit_checks", index=False)
    lines = ["# Nested Feature-Selection CV Summary", "", "Input: `All_features_N_filtered_240.xlsx` (240 rows, 196 candidate features).", "",
             "Feature selection was recomputed inside each outer training fold. The outer validation fold was not used for Pearson screening, RFE, feature-count selection, imputation, or model fitting.", "",
             "| Model | R2 mean +/- SD | RMSE mean +/- SD | MAE mean +/- SD |", "|---|---:|---:|---:|"]
    for row in summary.sort_values("R2_mean", ascending=False).itertuples():
        lines.append(f"| {row.Model} | {row.R2_mean:.3f} +/- {row.R2_sd:.3f} | {row.RMSE_mean:.3f} +/- {row.RMSE_sd:.3f} | {row.MAE_mean:.3f} +/- {row.MAE_sd:.3f} |")
    lines += ["", f"{len(checks)} audit checks passed.", "", "Limitations: the 240 rows were selected before this run; random KFold does not provide an independent plot/date or growth-stage holdout."]
    (d / "nested_feature_cv_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(checks)} audit checks passed; wrote summary to {d.resolve()}")


if __name__ == "__main__":
    main()
