"""Command-line entry point for five-fold modelling on an S1 feature table."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .cv_pipeline import CVConfig, DEFAULT_DATA_PATH, DEFAULT_OUTPUT_DIR, run_cross_validation
except ImportError:  # Allow `python run_model_comparison.py` from this directory.
    from cv_pipeline import CVConfig, DEFAULT_DATA_PATH, DEFAULT_OUTPUT_DIR, run_cross_validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run five-fold model comparison on a target-plus-fixed-feature workbook. "
            "Pearson filtering and RFE must already have been completed in S1."
        )
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Excel workbook with target plus predictors.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--target", default="N", help="Target column name.")
    parser.add_argument(
        "--models",
        default="all",
        help="Comma-separated model names or groups: all, automl, ml, dl.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Fixed random seed for the shared KFold partition.")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds.")
    parser.add_argument("--autogluon-time-limit", type=int, default=600, help="Seconds per AutoGluon outer fold.")
    parser.add_argument("--tpot-generations", type=int, default=10, help="TPOT generations per outer fold.")
    parser.add_argument("--tpot-population-size", type=int, default=20, help="TPOT population size per outer fold.")
    parser.add_argument(
        "--tpot-max-time-minutes",
        type=int,
        default=10,
        help="TPOT >= 1.0 search budget per outer fold; ignored by TPOT 0.x.",
    )
    parser.add_argument(
        "--tpot-max-eval-minutes",
        type=int,
        default=2,
        help="Maximum evaluation time per TPOT candidate for TPOT >= 1.0.",
    )
    parser.add_argument("--torch-epochs", type=int, default=300, help="Epochs for each DL outer-fold fit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.folds != 5:
        raise ValueError("This revision is configured for the manuscript's five-fold protocol; use --folds 5.")
    config = CVConfig(
        data_path=args.data,
        output_dir=args.output,
        target_column=args.target,
        n_splits=args.folds,
        random_state=args.seed,
        autogluon_time_limit=args.autogluon_time_limit,
        tpot_generations=args.tpot_generations,
        tpot_population_size=args.tpot_population_size,
        tpot_max_time_mins=args.tpot_max_time_minutes,
        tpot_max_eval_time_mins=args.tpot_max_eval_minutes,
        torch_epochs=args.torch_epochs,
    )
    results = run_cross_validation(config, args.models.split(","))
    print("\nFive-fold summary (mean +/- sample SD):")
    for _, row in results["summary"].iterrows():
        print(
            f"{row['Model']:12s} "
            f"R2={row['R2_mean']:.3f}+/-{row['R2_sd']:.3f}, "
            f"RMSE={row['RMSE_mean']:.3f}+/-{row['RMSE_sd']:.3f}, "
            f"MAE={row['MAE_mean']:.3f}+/-{row['MAE_sd']:.3f}"
        )
    print(f"\nWorkbook: {results['workbook']}")
    print(f"Next: python plot_cv_results.py --results {results['output_dir']}")


if __name__ == "__main__":
    main()
