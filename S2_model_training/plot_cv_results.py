"""Create manuscript-ready plots and scatter workbooks from five-fold OOF results."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

# This script writes files only and must also run on headless conda environments.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =============================================================================
# Central plotting configuration
# Edit the values in this section to change the appearance of all exported plots.
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results" / "original_240_corrected_plsr_tuned_final"

FONT_NAME = "Times New Roman"
DEFAULT_DPI = 300
DEFAULT_SCATTER_COLUMNS = 3

SCATTER_PANEL_WIDTH = 5.6
SCATTER_PANEL_HEIGHT = 5.6
SCATTER_MARKER_SIZE = 48
SCATTER_MARKER_LINEWIDTH = 1.3
SCATTER_MARKER_ALPHA = 0.82
SCATTER_MARKER_COLOR = "#0F4761"
ONE_TO_ONE_LINE_COLOR = "0.45"
REGRESSION_LINE_COLOR = "#C00000"
REFERENCE_LINE_WIDTH = 1.3
SCATTER_TITLE_FONT_SIZE = 25
SCATTER_TITLE_PAD = 10
SCATTER_AXIS_LABEL_FONT_SIZE = 25
SCATTER_TICK_FONT_SIZE = 25
SCATTER_ANNOTATION_FONT_SIZE = 25
SCATTER_EQUATION_FONT_SIZE = 25
SCATTER_AXIS_PADDING_RATIO = 0.06
SCATTER_MIN_AXIS_PADDING = 0.1
SCATTER_ANNOTATION_X = 0.04
SCATTER_ANNOTATION_Y = 0.96
SCATTER_EQUATION_X = 0.96
SCATTER_EQUATION_Y = 0.05

RANKING_FIGURE_WIDTH = 8.2
RANKING_MIN_HEIGHT = 4.5
RANKING_HEIGHT_PER_MODEL = 0.52
RANKING_EXTRA_HEIGHT = 1.8
RANKING_AXIS_LABEL_FONT_SIZE = 12
RANKING_TICK_FONT_SIZE = 11
RANKING_BAR_ERROR_CAP_SIZE = 4
RANKING_GRID_COLOR = "0.9"
RANKING_GRID_LINE_WIDTH = 0.9
MODEL_FAMILY_COLORS = {"AutoML": "#0F4761", "ML": "#4C78A8", "DL": "#B279A2"}

SCATTER_FIGURE_NAME = "oof_scatter_matrix.png"
RANKING_FIGURE_NAME = "model_ranking_cv_r2.png"

plt.rcParams.update({"font.family": FONT_NAME, "axes.unicode_minus": False})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot OOF predictions produced by run_model_comparison.py.")
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="CV result directory (default: Modelling/results/original_240_corrected_plsr_tuned_final).",
    )
    parser.add_argument(
        "--columns", type=int, default=DEFAULT_SCATTER_COLUMNS, help="Columns in the scatter matrix."
    )
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI, help="Output image resolution.")
    return parser.parse_args()


def load_results(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_path = results_dir / "model_summary.csv"
    oof_path = results_dir / "oof_predictions.csv"
    folds_path = results_dir / "fold_metrics.csv"
    missing = [str(path) for path in [summary_path, oof_path, folds_path] if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing CV output file(s): " + ", ".join(missing))
    return pd.read_csv(summary_path), pd.read_csv(oof_path), pd.read_csv(folds_path)


def oof_metrics(frame: pd.DataFrame) -> dict[str, float]:
    y_true = frame["y_true"].to_numpy(dtype=float)
    y_pred = frame["y_pred"].to_numpy(dtype=float)
    mse = mean_squared_error(y_true, y_pred)
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def axis_limits(oof: pd.DataFrame) -> tuple[float, float]:
    values = np.concatenate([oof["y_true"].to_numpy(dtype=float), oof["y_pred"].to_numpy(dtype=float)])
    lower, upper = float(np.nanmin(values)), float(np.nanmax(values))
    padding = max((upper - lower) * SCATTER_AXIS_PADDING_RATIO, SCATTER_MIN_AXIS_PADDING)
    return lower - padding, upper + padding


def draw_scatter(ax, frame: pd.DataFrame, summary_row: pd.Series, show_x: bool, show_y: bool) -> None:
    y_true = frame["y_true"].to_numpy(dtype=float)
    y_pred = frame["y_pred"].to_numpy(dtype=float)
    lo, hi = axis_limits(frame)
    slope, intercept = np.polyfit(y_true, y_pred, deg=1)

    ax.scatter(
        y_true,
        y_pred,
        facecolors="none",
        edgecolors=SCATTER_MARKER_COLOR,
        s=SCATTER_MARKER_SIZE,
        linewidth=SCATTER_MARKER_LINEWIDTH,
        alpha=SCATTER_MARKER_ALPHA,
    )
    ax.plot(
        [lo, hi], [lo, hi], color=ONE_TO_ONE_LINE_COLOR, linestyle="--", linewidth=REFERENCE_LINE_WIDTH, label="1:1"
    )
    ax.plot(
        [lo, hi], slope * np.array([lo, hi]) + intercept,
        color=REGRESSION_LINE_COLOR,
        linestyle="--",
        linewidth=REFERENCE_LINE_WIDTH,
    )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(direction="in", top=True, right=True, labelsize=SCATTER_TICK_FONT_SIZE)

    if show_x:
        ax.set_xlabel("Observed PNC (g kg$^{-1}$)", fontsize=SCATTER_AXIS_LABEL_FONT_SIZE)
    else:
        ax.set_xticklabels([])
    if show_y:
        ax.set_ylabel("Predicted PNC (g kg$^{-1}$)", fontsize=SCATTER_AXIS_LABEL_FONT_SIZE)
    else:
        ax.set_yticklabels([])

    text = (
        f"R$^2$ = {summary_row['R2_mean']:.3f} +/- {summary_row['R2_sd']:.3f}\n"
        f"RMSE = {summary_row['RMSE_mean']:.3f} +/- {summary_row['RMSE_sd']:.3f}"
    )
    ax.text(
        SCATTER_ANNOTATION_X,
        SCATTER_ANNOTATION_Y,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=SCATTER_ANNOTATION_FONT_SIZE,
    )
    ax.text(
        SCATTER_EQUATION_X,
        SCATTER_EQUATION_Y,
        f"y = {slope:.2f}x + {intercept:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=SCATTER_EQUATION_FONT_SIZE,
    )


def plot_scatter_matrix(summary: pd.DataFrame, oof: pd.DataFrame, output_dir: Path, columns: int, dpi: int) -> Path:
    ordered = summary.sort_values("R2_mean", ascending=False).reset_index(drop=True)
    rows = math.ceil(len(ordered) / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(SCATTER_PANEL_WIDTH * columns, SCATTER_PANEL_HEIGHT * rows),
        squeeze=False,
    )

    for index, row in ordered.iterrows():
        axis = axes[index // columns][index % columns]
        frame = oof.loc[oof["Model"] == row["Model"]]
        draw_scatter(axis, frame, row, show_x=index // columns == rows - 1, show_y=index % columns == 0)
        axis.set_title(
            f"{row['Model']} ({row['Type']})",
            fontsize=SCATTER_TITLE_FONT_SIZE,
            pad=SCATTER_TITLE_PAD,
        )

    for index in range(len(ordered), rows * columns):
        axes[index // columns][index % columns].axis("off")

    fig.tight_layout()
    path = output_dir / SCATTER_FIGURE_NAME
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_model_ranking(summary: pd.DataFrame, output_dir: Path, dpi: int) -> Path:
    ordered = summary.sort_values("R2_mean", ascending=True)
    figure_height = max(
        RANKING_MIN_HEIGHT,
        RANKING_HEIGHT_PER_MODEL * len(ordered) + RANKING_EXTRA_HEIGHT,
    )
    fig, axis = plt.subplots(figsize=(RANKING_FIGURE_WIDTH, figure_height))
    colors = [MODEL_FAMILY_COLORS[family] for family in ordered["Type"]]
    axis.barh(
        ordered["Model"],
        ordered["R2_mean"],
        xerr=ordered["R2_sd"],
        color=colors,
        capsize=RANKING_BAR_ERROR_CAP_SIZE,
    )
    axis.set_xlabel("Five-fold cross-validated R$^2$ (mean +/- SD)", fontsize=RANKING_AXIS_LABEL_FONT_SIZE)
    axis.set_ylabel("Model", fontsize=RANKING_AXIS_LABEL_FONT_SIZE)
    axis.tick_params(direction="in", labelsize=RANKING_TICK_FONT_SIZE)
    axis.grid(axis="x", color=RANKING_GRID_COLOR, linewidth=RANKING_GRID_LINE_WIDTH)
    axis.set_axisbelow(True)
    fig.tight_layout()
    path = output_dir / RANKING_FIGURE_NAME
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def export_plot_workbooks(summary: pd.DataFrame, fold_metrics: pd.DataFrame, oof: pd.DataFrame, output_dir: Path) -> list[Path]:
    exported: list[Path] = []
    summary.sort_values("R2_mean", ascending=False).to_excel(output_dir / "Metrics_by_model_sorted.xlsx", index=False)
    exported.append(output_dir / "Metrics_by_model_sorted.xlsx")

    for family, filename in [("AutoML", "AutoML_per_model_oof.xlsx"), ("ML", "ML_per_model_oof.xlsx"), ("DL", "DL_per_model_oof.xlsx")]:
        family_data = oof.loc[oof["Type"] == family]
        if family_data.empty:
            continue
        path = output_dir / filename
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for model, group in family_data.groupby("Model", sort=False):
                group.sort_values(["fold", "source_row"]).to_excel(writer, sheet_name=str(model)[:31], index=False)
            fold_metrics.loc[fold_metrics["Type"] == family].to_excel(writer, sheet_name="Fold_metrics", index=False)
        exported.append(path)
    return exported


def main() -> None:
    args = parse_args()
    summary, oof, fold_metrics = load_results(args.results)
    figure_dir = args.results / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    matrix = plot_scatter_matrix(summary, oof, figure_dir, args.columns, args.dpi)
    ranking = plot_model_ranking(summary, figure_dir, args.dpi)
    workbooks = export_plot_workbooks(summary, fold_metrics, oof, args.results)
    print(f"Saved OOF scatter matrix: {matrix}")
    print(f"Saved CV ranking plot: {ranking}")
    for workbook in workbooks:
        print(f"Saved workbook: {workbook}")


if __name__ == "__main__":
    main()
