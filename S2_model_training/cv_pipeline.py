"""Shared five-fold modelling pipeline for AgriMapper.

Feature engineering is deliberately outside this module.  S1 performs the
Pearson filtering and RFE once and exports a fixed modelling table.  This
module only reads that table (target column plus selected predictors), creates
the common K-fold assignments, fits the requested models, and evaluates each
held-out fold.  It never recalculates correlations, RFE, or any other
target-dependent feature selection.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import platform
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = ROOT / "data" / "best_features.xlsx"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "five_fold_cv"


@dataclass(frozen=True)
class CVConfig:
    """Configuration recorded with every run."""

    data_path: Path = DEFAULT_DATA_PATH
    output_dir: Path = DEFAULT_OUTPUT_DIR
    target_column: str = "N"
    n_splits: int = 5
    random_state: int = 42
    autogluon_time_limit: int | None = 600
    tpot_generations: int = 10
    tpot_population_size: int = 20
    tpot_max_time_mins: int = 10
    tpot_max_eval_time_mins: int = 2
    torch_epochs: int = 600
    torch_learning_rate: float = 1e-3
    dl_inner_splits: int = 3


class OptionalDependencyError(RuntimeError):
    """Raised when a requested optional modelling package is unavailable."""


class ELMRegressor(BaseEstimator, RegressorMixin):
    """Small deterministic extreme-learning-machine regressor."""

    def __init__(self, n_hidden: int = 100, alpha: float = 1e-6, random_state: int = 42):
        self.n_hidden = n_hidden
        self.alpha = alpha
        self.random_state = random_state

    def fit(self, x, y):
        x_array = np.asarray(x, dtype=float)
        y_array = np.asarray(y, dtype=float).ravel()
        rng = np.random.default_rng(self.random_state)
        self.weights_ = rng.normal(0.0, 1.0, size=(x_array.shape[1], self.n_hidden))
        self.bias_ = rng.normal(0.0, 1.0, size=self.n_hidden)
        hidden = np.maximum(0.0, x_array @ self.weights_ + self.bias_)
        self.output_model_ = Ridge(alpha=self.alpha).fit(hidden, y_array)
        return self

    def predict(self, x):
        x_array = np.asarray(x, dtype=float)
        hidden = np.maximum(0.0, x_array @ self.weights_ + self.bias_)
        return self.output_model_.predict(hidden)


class TorchTabularRegressor(BaseEstimator, RegressorMixin):
    """Deterministic PyTorch regressors for the four reported DL architectures."""

    def __init__(
        self,
        architecture: str,
        random_state: int = 42,
        epochs: int = 300,
        learning_rate: float = 1e-3,
        autoencoder_weight: float = 0.05,
        hidden_width: int = 32,
        dropout: float = 0.0,
        weight_decay: float = 0.0,
    ):
        self.architecture = architecture
        self.random_state = random_state
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.autoencoder_weight = autoencoder_weight
        self.hidden_width = hidden_width
        self.dropout = dropout
        self.weight_decay = weight_decay

    def _import_torch(self):
        try:
            import torch
            from torch import nn
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise OptionalDependencyError(
                "PyTorch is required for DNN, MLP, CNN, and Autoencoder models."
            ) from exc
        return torch, nn

    def _build_network(self, n_features: int, nn):
        medium_width = max(8, self.hidden_width // 2)
        dropout = nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity()
        if self.architecture == "DNN":
            return nn.Sequential(
                nn.Linear(n_features, self.hidden_width), nn.ReLU(), dropout,
                nn.Linear(self.hidden_width, medium_width), nn.ReLU(),
                nn.Linear(medium_width, 1),
            )
        if self.architecture == "MLP":
            return nn.Sequential(
                nn.Linear(n_features, self.hidden_width), nn.ReLU(), dropout,
                nn.Linear(self.hidden_width, 1),
            )
        if self.architecture == "CNN":
            channels = max(8, self.hidden_width // 2)
            head_width = self.hidden_width
            dropout_layer = dropout

            class ConvRegressor(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.features = nn.Sequential(
                        nn.Conv1d(1, channels, kernel_size=3, padding=1), nn.ReLU(),
                        nn.Conv1d(channels, channels, kernel_size=3, padding=1), nn.ReLU(),
                        nn.Flatten(),
                    )
                    self.head = nn.Sequential(
                        nn.Linear(channels * n_features, head_width), nn.ReLU(), dropout_layer,
                        nn.Linear(head_width, 1),
                    )

                def forward(self, values):
                    return self.head(self.features(values.unsqueeze(1)))

            return ConvRegressor()
        if self.architecture == "Autoencoder":
            latent_size = max(2, min(medium_width, n_features // 2))
            encoder_width = self.hidden_width
            decoder_width = self.hidden_width
            head_width = medium_width
            dropout_layer = dropout

            class AutoencoderRegressor(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.encoder = nn.Sequential(
                        nn.Linear(n_features, encoder_width), nn.ReLU(), dropout_layer,
                        nn.Linear(encoder_width, latent_size), nn.ReLU(),
                    )
                    self.decoder = nn.Sequential(
                        nn.Linear(latent_size, decoder_width), nn.ReLU(),
                        nn.Linear(decoder_width, n_features),
                    )
                    self.head = nn.Sequential(nn.Linear(latent_size, head_width), nn.ReLU(), nn.Linear(head_width, 1))

                def forward(self, values):
                    latent = self.encoder(values)
                    return self.head(latent), self.decoder(latent)

            return AutoencoderRegressor()
        raise ValueError(f"Unsupported torch architecture: {self.architecture}")

    def fit(self, x, y):
        torch, nn = self._import_torch()
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        torch.use_deterministic_algorithms(True, warn_only=True)

        x_tensor = torch.as_tensor(np.asarray(x, dtype=np.float32))
        y_tensor = torch.as_tensor(np.asarray(y, dtype=np.float32).reshape(-1, 1))
        self.network_ = self._build_network(x_tensor.shape[1], nn)
        optimizer = torch.optim.Adam(
            self.network_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        mse_loss = nn.MSELoss()

        self.network_.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            if self.architecture == "Autoencoder":
                prediction, reconstruction = self.network_(x_tensor)
                loss = mse_loss(prediction, y_tensor) + self.autoencoder_weight * mse_loss(
                    reconstruction, x_tensor
                )
            else:
                prediction = self.network_(x_tensor)
                loss = mse_loss(prediction, y_tensor)
            loss.backward()
            optimizer.step()
        return self

    def predict(self, x):
        torch, _ = self._import_torch()
        values = torch.as_tensor(np.asarray(x, dtype=np.float32))
        self.network_.eval()
        with torch.no_grad():
            output = self.network_(values)
            if self.architecture == "Autoencoder":
                output = output[0]
        return output.detach().cpu().numpy().ravel()


class AutoGluonRegressor(BaseEstimator, RegressorMixin):
    """Thin optional adapter that keeps every outer validation fold untouched."""

    def __init__(self, model_path: Path, random_state: int, time_limit: int | None):
        self.model_path = Path(model_path)
        self.random_state = random_state
        self.time_limit = time_limit

    def fit(self, x, y):
        try:
            from autogluon.tabular import TabularPredictor
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise OptionalDependencyError("AutoGluon is not installed.") from exc

        # Some AutoGluon versions do not expose random_seed in Predictor.fit.
        # Seed the libraries visible to the estimator before every outer fold.
        import random

        random.seed(self.random_state)
        np.random.seed(self.random_state)
        try:  # pragma: no cover - optional framework within AutoGluon
            import torch

            torch.manual_seed(self.random_state)
        except ImportError:
            pass

        train_data = pd.DataFrame(x).copy()
        self.feature_names_ = [str(column) for column in train_data.columns]
        train_data.columns = self.feature_names_
        self.target_name_ = "__target__"
        train_data[self.target_name_] = np.asarray(y, dtype=float)
        self.model_path.mkdir(parents=True, exist_ok=True)
        self.predictor_ = TabularPredictor(
            label=self.target_name_, path=str(self.model_path), verbosity=0
        )
        kwargs = {"presets": "medium_quality", "time_limit": self.time_limit}
        parameters = inspect.signature(self.predictor_.fit).parameters
        if "random_seed" in parameters:
            kwargs["random_seed"] = self.random_state
        self.predictor_.fit(train_data, **{key: value for key, value in kwargs.items() if value is not None})
        return self

    def predict(self, x):
        values = pd.DataFrame(x).copy()
        values.columns = self.feature_names_
        return self.predictor_.predict(values).to_numpy()


class TPOTRegressorAdapter(BaseEstimator, RegressorMixin):
    """Optional TPOT adapter with an inner CV restricted to each outer training fold."""

    def __init__(
        self,
        random_state: int,
        generations: int,
        population_size: int,
        max_time_mins: int,
        max_eval_time_mins: int,
    ):
        self.random_state = random_state
        self.generations = generations
        self.population_size = population_size
        self.max_time_mins = max_time_mins
        self.max_eval_time_mins = max_eval_time_mins

    def fit(self, x, y):
        try:
            from tpot import TPOTRegressor
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise OptionalDependencyError("TPOT is not installed.") from exc
        candidate = {
            "generations": self.generations,
            "population_size": self.population_size,
            "cv": 3,
            "random_state": self.random_state,
            "n_jobs": 1,
            "verbosity": 0,
            "disable_update_check": True,
            # TPOT >= 1.0 uses a time-bounded evolutionary search instead of
            # the generations/population controls of TPOT 0.x.
            "search_space": "linear",
            "max_time_mins": self.max_time_mins,
            "max_eval_time_mins": self.max_eval_time_mins,
        }
        signature = inspect.signature(TPOTRegressor).parameters
        kwargs = {key: value for key, value in candidate.items() if key in signature}
        self.model_ = TPOTRegressor(**kwargs)
        self.model_.fit(x, np.asarray(y).ravel())
        return self

    def predict(self, x):
        return self.model_.predict(x)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    builder: Callable[[int, int, Path, CVConfig], BaseEstimator]


def _scaled(estimator: BaseEstimator) -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("model", estimator)])


def _scaled_target(estimator: BaseEstimator) -> TransformedTargetRegressor:
    """Scale the response using only each fit's training subset."""

    return TransformedTargetRegressor(regressor=estimator, transformer=StandardScaler())


def _inner_tuned_estimator(
    estimator: BaseEstimator,
    parameter_grid: dict[str, list],
    seed: int,
    config: CVConfig,
) -> GridSearchCV:
    """Select hyperparameters within an outer-fold training subset only."""

    inner_cv = KFold(
        n_splits=config.dl_inner_splits,
        shuffle=True,
        random_state=seed,
    )
    return GridSearchCV(
        estimator=estimator,
        param_grid=parameter_grid,
        scoring="r2",
        cv=inner_cv,
        n_jobs=1,
        refit=True,
        error_score="raise",
    )


def model_registry() -> dict[str, ModelSpec]:
    """Return the 15 model names used in the manuscript comparison."""

    def mlr(seed, n_features, fold_dir, config):
        return LinearRegression()

    def plsr(seed, n_features, fold_dir, config):
        # Select a reduced latent space within each outer-fold training set.
        # Capping components below the full feature rank prevents PLSR from
        # collapsing to the ordinary least-squares solution used by MLR.
        maximum_components = min(9, n_features)
        return _inner_tuned_estimator(
            _scaled(PLSRegression()),
            {"model__n_components": list(range(1, maximum_components + 1))},
            seed,
            config,
        )

    def knn(seed, n_features, fold_dir, config):
        return _scaled(KNeighborsRegressor(n_neighbors=5, weights="distance"))

    def svr(seed, n_features, fold_dir, config):
        return _scaled(SVR())

    def rf(seed, n_features, fold_dir, config):
        return RandomForestRegressor(n_estimators=500, random_state=seed, n_jobs=-1)

    def gbdt(seed, n_features, fold_dir, config):
        return GradientBoostingRegressor(random_state=seed)

    def xgboost(seed, n_features, fold_dir, config):
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise OptionalDependencyError("XGBoost is not installed.") from exc
        return XGBRegressor(
            random_state=seed,
            n_estimators=300,
            objective="reg:squarederror",
            n_jobs=1,
        )

    def elm(seed, n_features, fold_dir, config):
        return _scaled(ELMRegressor(random_state=seed))

    def bpnn(seed, n_features, fold_dir, config):
        estimator = _scaled_target(
            _scaled(
                MLPRegressor(
                    activation="relu",
                    solver="adam",
                    random_state=seed,
                    max_iter=1500,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=50,
                )
            )
        )
        return _inner_tuned_estimator(
            estimator,
            {
                "regressor__model__hidden_layer_sizes": [(32,), (64,), (64, 32)],
                "regressor__model__alpha": [1e-4, 1e-3],
                "regressor__model__learning_rate_init": [1e-3],
            },
            seed,
            config,
        )

    def torch_builder(architecture: str):
        def builder(seed, n_features, fold_dir, config):
            estimator = _scaled_target(
                _scaled(
                    TorchTabularRegressor(
                        architecture=architecture,
                        random_state=seed,
                        epochs=config.torch_epochs,
                        learning_rate=config.torch_learning_rate,
                    )
                )
            )
            if architecture not in {"CNN", "Autoencoder"}:
                return estimator
            return _inner_tuned_estimator(
                estimator,
                {
                    "regressor__model__hidden_width": [16, 32],
                    "regressor__model__learning_rate": [1e-3, 3e-3],
                    "regressor__model__weight_decay": [0.0, 1e-4],
                },
                seed,
                config,
            )

        return builder

    def autogluon(seed, n_features, fold_dir, config):
        return AutoGluonRegressor(
            model_path=fold_dir / "autogluon_predictor",
            random_state=seed,
            time_limit=config.autogluon_time_limit,
        )

    def tpot(seed, n_features, fold_dir, config):
        return TPOTRegressorAdapter(
            random_state=seed,
            generations=config.tpot_generations,
            population_size=config.tpot_population_size,
            max_time_mins=config.tpot_max_time_mins,
            max_eval_time_mins=config.tpot_max_eval_time_mins,
        )

    return {
        "AutoGluon": ModelSpec("AutoGluon", "AutoML", autogluon),
        "TPOT": ModelSpec("TPOT", "AutoML", tpot),
        "MLR": ModelSpec("MLR", "ML", mlr),
        "PLSR": ModelSpec("PLSR", "ML", plsr),
        "KNN": ModelSpec("KNN", "ML", knn),
        "SVR": ModelSpec("SVR", "ML", svr),
        "RF": ModelSpec("RF", "ML", rf),
        "GBDT": ModelSpec("GBDT", "ML", gbdt),
        "XGBoost": ModelSpec("XGBoost", "ML", xgboost),
        "ELM": ModelSpec("ELM", "ML", elm),
        "BPNN": ModelSpec("BPNN", "ML", bpnn),
        "DNN": ModelSpec("DNN", "DL", torch_builder("DNN")),
        "MLP": ModelSpec("MLP", "DL", torch_builder("MLP")),
        "CNN": ModelSpec("CNN", "DL", torch_builder("CNN")),
        "Autoencoder": ModelSpec("Autoencoder", "DL", torch_builder("Autoencoder")),
    }


MODEL_GROUPS = {
    "all": list(model_registry().keys()),
    "automl": ["AutoGluon", "TPOT"],
    "ml": ["MLR", "PLSR", "KNN", "SVR", "RF", "GBDT", "XGBoost", "ELM", "BPNN"],
    "dl": ["DNN", "MLP", "CNN", "Autoencoder"],
}


def resolve_model_names(requested: Iterable[str]) -> list[str]:
    registry = model_registry()
    canonical_names = {name.lower(): name for name in registry}
    result: list[str] = []
    for item in requested:
        key = item.strip()
        if not key:
            continue
        group_key = key.lower()
        names = MODEL_GROUPS.get(group_key, [canonical_names.get(group_key, key)])
        for name in names:
            if name not in registry:
                valid = ", ".join(registry)
                raise ValueError(f"Unknown model '{name}'. Available models: {valid}")
            if name not in result:
                result.append(name)
    if not result:
        raise ValueError("At least one model must be requested.")
    return result


def _metric_record(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": rmse,
        "MSE": mse,
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def _hash_indices(indices: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()


def _package_versions() -> dict[str, str]:
    packages = ["numpy", "pandas", "scikit-learn", "openpyxl", "matplotlib", "torch", "xgboost", "autogluon", "tpot"]
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def load_dataset(config: CVConfig) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Read the fixed S1 modelling table and retain the source-row audit trail.

    The first configured column is the target and every remaining column is
    treated as a predictor already selected by S1.  Keeping this function
    limited to schema validation and numeric conversion is what separates the
    feature-engineering stage from the modelling stage.
    """

    if not config.data_path.exists():
        raise FileNotFoundError(f"Input workbook not found: {config.data_path}")
    raw = pd.read_excel(config.data_path)
    if config.target_column not in raw.columns:
        available = ", ".join(str(column) for column in raw.columns)
        raise ValueError(f"Target column '{config.target_column}' was not found. Available columns: {available}")

    feature_columns = [column for column in raw.columns if column != config.target_column]
    if not feature_columns:
        raise ValueError("The input workbook contains no predictor columns.")
    numeric = raw[[config.target_column, *feature_columns]].apply(pd.to_numeric, errors="coerce")
    valid = numeric.notna().all(axis=1)
    if valid.sum() < config.n_splits:
        raise ValueError("Too few complete rows for the requested number of folds.")

    audit = pd.DataFrame(
        {
            "source_row": raw.index + 2,
            "included": valid.to_numpy(),
            "exclusion_reason": np.where(valid, "", "missing or non-numeric target/predictor value"),
        }
    )
    included = numeric.loc[valid].copy()
    included.index = raw.index[valid]
    x = included[feature_columns]
    y = included[config.target_column]
    return x, y, audit


def make_fold_assignments(x: pd.DataFrame, config: CVConfig) -> tuple[pd.DataFrame, list[tuple[np.ndarray, np.ndarray]]]:
    """Generate one shared shuffled five-fold partition for every requested model."""

    splitter = KFold(n_splits=config.n_splits, shuffle=True, random_state=config.random_state)
    assignments = np.full(len(x), -1, dtype=int)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    positions = np.arange(len(x))
    for fold, (train_pos, valid_pos) in enumerate(splitter.split(positions), start=1):
        assignments[valid_pos] = fold
        splits.append((train_pos, valid_pos))

    if (assignments < 1).any():
        raise RuntimeError("At least one observation was not assigned to a validation fold.")
    frame = pd.DataFrame(
        {
            "source_row": x.index.to_numpy() + 2,
            "fold": assignments,
        }
    )
    return frame, splits


def _write_excel(output_dir: Path, summary, fold_metrics, oof_predictions, assignments, status, metadata):
    workbook = output_dir / "cv_results.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        fold_metrics.to_excel(writer, sheet_name="Fold_metrics", index=False)
        oof_predictions.to_excel(writer, sheet_name="OOF_predictions", index=False)
        assignments.to_excel(writer, sheet_name="Fold_assignments", index=False)
        status.to_excel(writer, sheet_name="Model_status", index=False)
        pd.DataFrame([metadata]).to_excel(writer, sheet_name="Run_metadata", index=False)
    return workbook


def _build_summary(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (model, family), group in fold_metrics.groupby(["Model", "Type"], sort=False):
        record = {"Model": model, "Type": family, "Folds": int(len(group))}
        for metric in ["R2", "RMSE", "MSE", "MAE"]:
            record[f"{metric}_mean"] = float(group[metric].mean())
            record[f"{metric}_sd"] = float(group[metric].std(ddof=1))
        records.append(record)
    return pd.DataFrame(records).sort_values("R2_mean", ascending=False, ignore_index=True)


def run_cross_validation(config: CVConfig, requested_models: Iterable[str]) -> dict[str, Path | pd.DataFrame]:
    """Run all requested models and write CV outputs that can be audited and plotted."""

    requested = resolve_model_names(requested_models)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = config.output_dir / "model_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    x, y, input_audit = load_dataset(config)
    assignments, splits = make_fold_assignments(x, config)
    registry = model_registry()

    input_audit.to_csv(config.output_dir / "input_data_audit.csv", index=False)
    assignments.to_csv(config.output_dir / "fold_assignments.csv", index=False)
    pd.DataFrame({"feature": x.columns}).to_csv(config.output_dir / "feature_columns.csv", index=False)

    fold_records: list[dict] = []
    prediction_records: list[pd.DataFrame] = []
    status_records: list[dict] = []

    for model_name in requested:
        spec = registry[model_name]
        model_predictions: list[pd.DataFrame] = []
        model_metrics: list[dict] = []
        status = "completed"
        detail = ""

        try:
            for fold, (train_pos, valid_pos) in enumerate(splits, start=1):
                x_train = x.iloc[train_pos]
                y_train = y.iloc[train_pos]
                x_valid = x.iloc[valid_pos]
                y_valid = y.iloc[valid_pos]
                fold_dir = artifact_dir / model_name / f"fold_{fold:02d}"
                fold_seed = config.random_state + fold
                estimator = spec.builder(fold_seed, x.shape[1], fold_dir, config)
                estimator.fit(x_train, y_train)
                prediction = np.asarray(estimator.predict(x_valid), dtype=float).ravel()
                if len(prediction) != len(y_valid):
                    raise RuntimeError(
                        f"{model_name} fold {fold} returned {len(prediction)} predictions for {len(y_valid)} samples."
                    )
                if not np.isfinite(prediction).all():
                    raise RuntimeError(f"{model_name} fold {fold} produced non-finite predictions.")

                metrics = _metric_record(y_valid.to_numpy(), prediction)
                model_metrics.append(
                    {
                        "Model": model_name,
                        "Type": spec.family,
                        "Fold": fold,
                        "n_train": int(len(train_pos)),
                        "n_validation": int(len(valid_pos)),
                        "train_index_hash": _hash_indices(train_pos),
                        "validation_index_hash": _hash_indices(valid_pos),
                        **metrics,
                    }
                )
                model_predictions.append(
                    pd.DataFrame(
                        {
                            "Model": model_name,
                            "Type": spec.family,
                            "source_row": x_valid.index.to_numpy() + 2,
                            "fold": fold,
                            "y_true": y_valid.to_numpy(dtype=float),
                            "y_pred": prediction,
                        }
                    )
                )
        except OptionalDependencyError as exc:
            status = "skipped"
            detail = str(exc)
        except Exception as exc:  # pragma: no cover - protects long comparison runs
            status = "failed"
            detail = f"{type(exc).__name__}: {exc}"
            (artifact_dir / model_name).mkdir(parents=True, exist_ok=True)
            (artifact_dir / model_name / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")

        status_records.append({"Model": model_name, "Type": spec.family, "Status": status, "Detail": detail})
        if status == "completed":
            fold_records.extend(model_metrics)
            prediction_records.extend(model_predictions)
            print(f"Completed {model_name}: {len(model_metrics)} folds, {sum(len(frame) for frame in model_predictions)} OOF predictions")
        else:
            print(f"{status.title()} {model_name}: {detail}")

    if not fold_records:
        raise RuntimeError("No model completed. Install the requested optional packages or choose available models.")

    fold_metrics = pd.DataFrame(fold_records)
    oof_predictions = pd.concat(prediction_records, ignore_index=True)
    summary = _build_summary(fold_metrics)
    status = pd.DataFrame(status_records)

    fold_metrics.to_csv(config.output_dir / "fold_metrics.csv", index=False)
    oof_predictions.to_csv(config.output_dir / "oof_predictions.csv", index=False)
    summary.to_csv(config.output_dir / "model_summary.csv", index=False)
    status.to_csv(config.output_dir / "model_status.csv", index=False)

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "data_path": str(config.data_path.resolve()),
        "target_column": config.target_column,
        "n_rows_input": int(len(input_audit)),
        "n_rows_included": int(len(x)),
        "n_features": int(x.shape[1]),
        "feature_columns": list(map(str, x.columns)),
        "n_splits": config.n_splits,
        "random_state": config.random_state,
        "splitter": "KFold(shuffle=True, random_state=random_state)",
        "stratification": "not applied; the continuous target is not class-stratified",
        "external_holdout": "not used",
        "outlier_removal": "not applied",
        "feature_engineering": "completed before S2 using S1 Pearson filtering and RFE",
        "feature_selection": "fixed before cross-validation; Pearson/RFE are not repeated in S2",
        "requested_models": requested,
        "completed_models": summary["Model"].tolist(),
        "package_versions": _package_versions(),
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
    }
    (config.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    workbook = _write_excel(config.output_dir, summary, fold_metrics, oof_predictions, assignments, status, metadata)

    return {
        "summary": summary,
        "fold_metrics": fold_metrics,
        "oof_predictions": oof_predictions,
        "output_dir": config.output_dir,
        "workbook": workbook,
    }
