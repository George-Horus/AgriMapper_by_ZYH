"""Leakage-controlled case-study validation from the original field tables and rasters.

This is a separate experiment: it never uses the preselected 12-feature/240-row
workbook. Run ``--prepare`` once, then run model groups independently if needed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import warnings
import sys
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold

try:
    from .cv_pipeline import (
        CVConfig, OptionalDependencyError, _build_summary, _metric_record,
        _package_versions, model_registry, resolve_model_names,
    )
except ImportError:
    from cv_pipeline import (
        CVConfig, OptionalDependencyError, _build_summary, _metric_record,
        _package_versions, model_registry, resolve_model_names,
    )

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw_data"
DEFAULT_OUTPUT = ROOT / "validation_results_r3" / "nested_feature_cv"
DATES = ("20250307", "20250317", "20250331", "20250421")
BAND_NAMES = ("GREEN", "NIR", "RED", "RedEdge")
AUDIT_COLUMNS = ("source_file", "source_excel_row", "No")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_candidates(raw_dir: Path, output_dir: Path, window_size: int = 10) -> Path:
    """Compute target-independent spectral features for all original records."""
    import pyproj
    import rasterio
    from rasterio.windows import Window
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from S1_preprocessing.VMID_1 import VMID

    if window_size < 1:
        raise ValueError("window_size must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    sources = []
    half = window_size // 2
    for date in DATES:
        workbook = raw_dir / f"{date}.xlsx"
        raster = raw_dir / f"{date}_multispectral.tif"
        if not workbook.is_file() or not raster.is_file():
            raise FileNotFoundError(f"Missing source pair: {workbook}, {raster}")
        field = pd.read_excel(workbook)
        if not {"lon", "lat", "N"}.issubset(field.columns):
            raise ValueError(f"Missing lon/lat/N in {workbook}")
        bands = []
        with rasterio.open(raster) as image:
            if image.count != len(BAND_NAMES):
                raise ValueError(f"Expected four bands in {raster}; got {image.count}")
            transformer = pyproj.Transformer.from_crs("EPSG:4326", image.crs, always_xy=True)
            for lon, lat in field[["lon", "lat"]].itertuples(index=False, name=None):
                if pd.isna(lon) or pd.isna(lat):
                    bands.append([np.nan] * image.count)
                    continue
                x, y = transformer.transform(float(lon), float(lat))
                if not (image.bounds.left <= x < image.bounds.right and image.bounds.bottom < y <= image.bounds.top):
                    bands.append([np.nan] * image.count)
                    continue
                row, col = image.index(x, y)
                row0 = max(0, row - half)
                col0 = max(0, col - half)
                row1 = min(image.height, row + half + 1)
                col1 = min(image.width, col + half + 1)
                pixels = image.read(window=Window(col0, row0, col1 - col0, row1 - row0), masked=True)
                means = np.ma.mean(pixels, axis=(1, 2)).filled(np.nan)
                bands.append(np.asarray(means, dtype=float))
        spectral = pd.DataFrame(bands, columns=BAND_NAMES)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            indices = VMID(
                red=spectral["RED"], green=spectral["GREEN"],
                red_edge=spectral["RedEdge"], nir=spectral["NIR"],
            ).export_all()
        features = pd.concat([spectral, indices], axis=1)
        if features.columns.duplicated().any():
            features = features.loc[:, ~features.columns.duplicated()]
        features = features.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        audit = pd.DataFrame({
            "source_file": workbook.name,
            "source_excel_row": np.arange(len(field)) + 2,
            "No": field["No"] if "No" in field else np.arange(len(field)) + 1,
            "N": pd.to_numeric(field["N"], errors="coerce"),
        })
        frames.append(pd.concat([audit.reset_index(drop=True), features.reset_index(drop=True)], axis=1))
        sources.append({
            "field_file": str(workbook.resolve()), "field_sha256": _sha256(workbook),
            "raster_file": str(raster.resolve()), "raster_bytes": raster.stat().st_size,
            "raster_mtime_ns": raster.stat().st_mtime_ns,
            "n_rows": len(field), "raster_crs": str(image.crs),
        })
        print(f"Prepared {date}: {len(field)} rows, {features.shape[1]} candidate features", flush=True)
    combined = pd.concat(frames, ignore_index=True)
    candidate_path = output_dir / "candidate_features.csv"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        combined.to_csv(candidate_path, index=False)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_file": str(candidate_path.resolve()),
        "candidate_sha256": _sha256(candidate_path),
        "window_size_pixels": window_size,
        "band_order": BAND_NAMES,
        "n_rows": len(combined),
        "n_candidate_features": len(combined.columns) - len(AUDIT_COLUMNS) - 1,
        "source_files": sources,
        "note": "No outcome-dependent feature or row selection is performed during extraction.",
    }
    (output_dir / "candidate_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return candidate_path


def _ranked_features(x: pd.DataFrame, y: pd.Series, threshold: float, count: int, seed: int):
    """Fit imputation, Pearson screening and RF-RFE on this training slice only."""
    valid_columns = [name for name in x if x[name].notna().any()]
    if not valid_columns:
        raise ValueError("No observed candidate feature in training data")
    imputer = SimpleImputer(strategy="median")
    filled = pd.DataFrame(imputer.fit_transform(x[valid_columns]), columns=valid_columns, index=x.index)
    correlations = filled.corrwith(y).abs().fillna(0).sort_values(ascending=False)
    passing = correlations[correlations > threshold].index.tolist()
    if not passing:
        raise ValueError(f"No training-only feature exceeds |r| > {threshold}")
    selected_count = min(count, len(passing))
    if len(passing) > selected_count:
        selector = RFE(
            RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1),
            n_features_to_select=selected_count, step=3,
        )
        selector.fit(filled[passing], y)
        chosen = [name for name, keep in zip(passing, selector.support_) if keep]
    else:
        chosen = passing
    return valid_columns, imputer, chosen, correlations, len(passing)


def _transform(x: pd.DataFrame, valid_columns: list[str], imputer: SimpleImputer, selected: list[str]) -> pd.DataFrame:
    filled = pd.DataFrame(imputer.transform(x[valid_columns]), columns=valid_columns, index=x.index)
    return filled[selected]


def _choose_feature_count(x: pd.DataFrame, y: pd.Series, threshold: float, counts: list[int], seed: int):
    """Inner CV includes the entire selection procedure; outer labels remain untouched."""
    inner = KFold(n_splits=3, shuffle=True, random_state=seed)
    records = []
    for count in counts:
        for inner_fold, (train, valid) in enumerate(inner.split(x), start=1):
            train_x, train_y = x.iloc[train], y.iloc[train]
            valid_x, valid_y = x.iloc[valid], y.iloc[valid]
            columns, imputer, chosen, _, passing = _ranked_features(
                train_x, train_y, threshold, count, seed + inner_fold,
            )
            model = RandomForestRegressor(n_estimators=100, random_state=seed + inner_fold, n_jobs=-1)
            model.fit(_transform(train_x, columns, imputer, chosen), train_y)
            predicted = model.predict(_transform(valid_x, columns, imputer, chosen))
            records.append({
                "requested_count": count, "inner_fold": inner_fold,
                "actual_count": len(chosen), "n_passing_threshold": passing,
                "RMSE": _metric_record(valid_y.to_numpy(), predicted)["RMSE"],
            })
    trace = pd.DataFrame(records)
    means = trace.groupby("requested_count")["RMSE"].mean()
    chosen_count = int(min(counts, key=lambda count: (means[count], count)))
    return chosen_count, trace


def _read_candidates(path: Path):
    raw = pd.read_csv(path)
    needed = {*AUDIT_COLUMNS, "N"}
    if not needed.issubset(raw.columns):
        raise ValueError(f"Candidate file must contain {sorted(needed)}")
    columns = [name for name in raw if name not in needed]
    if len(columns) <= 12:
        raise ValueError("Candidate table has <=12 columns; a preselected feature table cannot be used")
    x = raw[columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    y = pd.to_numeric(raw["N"], errors="coerce")
    included = y.notna() & x.notna().any(axis=1)
    audit = raw[list(AUDIT_COLUMNS)].copy()
    audit["included"] = included
    audit["reason"] = np.where(included, "", "invalid target or no observed feature")
    return x.loc[included].reset_index(drop=True), y.loc[included].reset_index(drop=True), audit, raw.loc[included, list(AUDIT_COLUMNS)].reset_index(drop=True)


def run_experiment(candidate_path: Path, output_dir: Path, models: list[str], threshold: float,
                   counts: list[int], seed: int, autogluon_seconds: int, tpot_minutes: int,
                   torch_epochs: int) -> None:
    if not 0 <= threshold < 1 or not counts or min(counts) < 1:
        raise ValueError("Invalid threshold or feature-count grid")
    output_dir.mkdir(parents=True, exist_ok=True)
    x, y, audit, provenance = _read_candidates(candidate_path)
    audit.to_csv(output_dir / "input_audit.csv", index=False)
    splitter = KFold(n_splits=5, shuffle=True, random_state=seed)
    splits = list(splitter.split(x))
    fold_selection = []
    inner_traces = []
    fold_features = {}
    assignments = np.zeros(len(x), dtype=int)
    for fold, (train, valid) in enumerate(splits, start=1):
        requested, trace = _choose_feature_count(x.iloc[train], y.iloc[train], threshold, counts, seed + fold)
        columns, imputer, chosen, correlations, passing = _ranked_features(
            x.iloc[train], y.iloc[train], threshold, requested, seed + fold,
        )
        fold_features[fold] = (columns, imputer, chosen)
        assignments[valid] = fold
        trace.insert(0, "outer_fold", fold)
        inner_traces.append(trace)
        fold_selection.append({
            "fold": fold, "requested_count": requested, "selected_count": len(chosen),
            "n_passing_threshold": passing, "selected_features": json.dumps(chosen),
            "train_index_hash": hashlib.sha256(np.asarray(train, dtype=np.int64).tobytes()).hexdigest(),
            "validation_index_hash": hashlib.sha256(np.asarray(valid, dtype=np.int64).tobytes()).hexdigest(),
        })
        pd.DataFrame({"feature": correlations.index, "abs_pearson_r_training": correlations.to_numpy(),
                      "above_threshold": correlations.to_numpy() > threshold,
                      "selected": correlations.index.isin(chosen)}).to_csv(
            output_dir / f"fold_{fold:02d}_selection.csv", index=False,
        )
        print(f"Fold {fold}: {len(train)} train, {len(valid)} validation, {len(chosen)} selected: {chosen}", flush=True)
    provenance = provenance.copy()
    provenance.insert(0, "candidate_row", np.arange(len(provenance)))
    provenance["outer_fold"] = assignments
    provenance.to_csv(output_dir / "fold_assignments.csv", index=False)
    pd.DataFrame(fold_selection).to_csv(output_dir / "fold_features.csv", index=False)
    pd.concat(inner_traces, ignore_index=True).to_csv(output_dir / "inner_cv_selection.csv", index=False)

    config = CVConfig(
        data_path=candidate_path, output_dir=output_dir, random_state=seed,
        autogluon_time_limit=autogluon_seconds, tpot_max_time_mins=tpot_minutes,
        torch_epochs=torch_epochs,
    )
    registry = model_registry()
    metric_rows, predictions, status_rows = [], [], []
    for name in resolve_model_names(models):
        spec = registry[name]
        local_metrics, local_predictions = [], []
        status, detail = "completed", ""
        for fold, (train, valid) in enumerate(splits, start=1):
            try:
                columns, imputer, chosen = fold_features[fold]
                train_x = _transform(x.iloc[train], columns, imputer, chosen)
                valid_x = _transform(x.iloc[valid], columns, imputer, chosen)
                estimator = spec.builder(seed + fold, len(chosen), output_dir / "model_artifacts" / name / f"fold_{fold:02d}", config)
                estimator.fit(train_x, y.iloc[train])
                predicted = np.asarray(estimator.predict(valid_x), dtype=float).ravel()
                if len(predicted) != len(valid) or not np.isfinite(predicted).all():
                    raise ValueError("Invalid predictions")
                local_metrics.append({"Model": name, "Type": spec.family, "Fold": fold,
                                      "n_train": len(train), "n_validation": len(valid),
                                      "n_features": len(chosen), **_metric_record(y.iloc[valid].to_numpy(), predicted)})
                local_predictions.append(pd.DataFrame({
                    "Model": name, "Fold": fold, "candidate_row": valid,
                    "source_file": provenance.iloc[valid]["source_file"].to_numpy(),
                    "source_excel_row": provenance.iloc[valid]["source_excel_row"].to_numpy(),
                    "y_true": y.iloc[valid].to_numpy(), "y_pred": predicted,
                }))
                print(f"{name} fold {fold}: R2={local_metrics[-1]['R2']:.3f}", flush=True)
            except OptionalDependencyError as exc:
                status, detail = "skipped", str(exc)
                break
            except Exception as exc:
                status, detail = "failed", f"{type(exc).__name__}: {exc}"
                error_dir = output_dir / "model_artifacts" / name
                error_dir.mkdir(parents=True, exist_ok=True)
                (error_dir / f"fold_{fold:02d}_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
                break
        if status == "completed":
            metric_rows.extend(local_metrics)
            predictions.extend(local_predictions)
        status_rows.append({"Model": name, "Status": status, "Detail": detail})
        pd.DataFrame(metric_rows).to_csv(output_dir / "fold_metrics.csv", index=False)
        pd.DataFrame(status_rows).to_csv(output_dir / "model_status.csv", index=False)
        if predictions:
            pd.concat(predictions, ignore_index=True).to_csv(output_dir / "oof_predictions.csv", index=False)
        if metric_rows:
            _build_summary(pd.DataFrame(metric_rows)).to_csv(output_dir / "model_summary.csv", index=False)
        print(f"{name}: {status} {detail}", flush=True)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_path": str(candidate_path.resolve()), "candidate_sha256": _sha256(candidate_path),
        "n_candidate_rows": len(audit), "n_included_rows": len(x), "n_candidate_features": x.shape[1],
        "outer_cv": "KFold(n_splits=5, shuffle=True, random_state=seed)",
        "inner_cv": "KFold(n_splits=3, shuffle=True, random_state=seed+outer_fold)",
        "selection": "training-only median imputation, |Pearson r| threshold, RF-RFE(step=3); inner RF RMSE chooses feature count",
        "threshold": threshold, "feature_count_grid": counts, "seed": seed,
        "model_config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "package_versions": _package_versions(), "status": status_rows,
        "limitations": "Random folds are not independent growth-stage/year validation; raster extraction settings may differ from the historical analysis.",
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help="Rebuild candidates from the four original tables and rasters")
    parser.add_argument("--prepare-only", action="store_true", help="Stop after rebuilding the candidate table")
    parser.add_argument("--raw-dir", type=Path, default=RAW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--models", default="all", help="all, ml, dl, automl or comma-separated model names")
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--counts", default="6,9,12,15")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--autogluon-seconds", type=int, default=600)
    parser.add_argument("--tpot-minutes", type=int, default=10)
    parser.add_argument("--torch-epochs", type=int, default=300)
    args = parser.parse_args()
    candidate = args.candidate or args.output / "candidate_features.csv"
    if args.prepare or args.prepare_only:
        candidate = prepare_candidates(args.raw_dir, args.output, args.window_size)
    if args.prepare_only:
        return
    if not candidate.is_file():
        parser.error("Candidate table missing; run with --prepare first")
    run_experiment(candidate, args.output, args.models.split(","), args.threshold,
                   [int(item) for item in args.counts.split(",")], args.seed,
                   args.autogluon_seconds, args.tpot_minutes, args.torch_epochs)


if __name__ == "__main__":
    main()
