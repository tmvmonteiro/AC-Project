from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from feature_engineering import (
    TARGET_COLUMN,
    build_ranking_dataset,
    get_feature_sets,
)


DEFAULT_START_YEAR = 1995
DEFAULT_WINDOW_SIZE = 9
RANDOM_STATE = 42

def _make_estimators() -> Dict[str, Any]:
    """Create fresh estimator instances for a single training fold."""
    random_forest = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    gradient_boosting = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=2,
        loss="squared_error",
        random_state=RANDOM_STATE,
    )
    linear_regression = LinearRegression()

    ensemble = VotingRegressor(
        estimators=[
            ("random_forest", clone(random_forest)),
            ("gradient_boosting", clone(gradient_boosting)),
            ("linear_regression", clone(linear_regression)),
        ],
        weights=None,
    )

    return {
        "MeanBaseline": DummyRegressor(strategy="mean"),
        "LinearRegression": linear_regression,
        "RandomForest": random_forest,
        "GradientBoosting": gradient_boosting,
        "Ensemble": ensemble,
    }


def _make_pipeline(estimator: Any) -> Pipeline:
    """Fit imputation only on the training fold, then fit the estimator."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", estimator),
        ]
    )

def _prepare_ranking_data(
    data: Union[pd.DataFrame, Mapping[str, pd.DataFrame]],
    start_year: int,
    target_col: str,
) -> pd.DataFrame:
    """Accept either an engineered DataFrame or cleaned datasets dictionary."""
    if isinstance(data, pd.DataFrame):
        ranking_data = data.copy()
    elif isinstance(data, Mapping):
        if "Teams" not in data:
            raise KeyError(
                "The datasets dictionary must contain a 'Teams' DataFrame. "
                "Pass cleaned_datasets or an engineered ranking DataFrame."
            )
        ranking_data = build_ranking_dataset(
            data["Teams"], start_year=start_year, verbose=False
        )
    else:
        raise TypeError(
            "data must be an engineered pandas DataFrame or a dictionary "
            "containing the cleaned 'Teams' DataFrame."
        )

    if target_col == "rank":
        raise ValueError(
            "Do not use Teams.rank as the target: it is a divisional rank. "
            f"Use '{TARGET_COLUMN}' (points percentage) instead."
        )

    required = {"year", "tmID", target_col}
    missing = sorted(required.difference(ranking_data.columns))
    if missing:
        raise ValueError(
            "The ranking dataset is missing required column(s): " + ", ".join(missing)
        )

    ranking_data = ranking_data.copy()
    ranking_data["year"] = pd.to_numeric(ranking_data["year"], errors="coerce")
    ranking_data[target_col] = pd.to_numeric(
        ranking_data[target_col], errors="coerce"
    )
    ranking_data = ranking_data.dropna(subset=["year", "tmID", target_col]).copy()

    non_integer_year = ranking_data["year"] % 1 != 0
    if non_integer_year.any():
        raise ValueError("The 'year' column must contain integer season identifiers.")

    ranking_data["year"] = ranking_data["year"].astype(int)
    ranking_data = ranking_data[ranking_data["year"] >= int(start_year)].copy()
    ranking_data["tmID"] = ranking_data["tmID"].astype(str).str.strip()
    ranking_data = ranking_data[ranking_data["tmID"].ne("")].copy()

    if target_col == TARGET_COLUMN:
        invalid_target = ~ranking_data[target_col].between(0.0, 1.0)
        if invalid_target.any():
            examples = ranking_data.loc[invalid_target, ["tmID", "year", target_col]].head(5)
            raise ValueError(
                f"'{target_col}' must be between 0 and 1. Check the target calculation. "
                f"Examples: {examples.to_dict('records')}"
            )

    duplicated = ranking_data.duplicated(["tmID", "year"], keep=False)
    if duplicated.any():
        examples = ranking_data.loc[duplicated, ["tmID", "year"]].head(8)
        raise ValueError(
            "The ranking dataset has duplicate team-season rows. Examples: "
            f"{examples.to_dict('records')}"
        )

    if ranking_data.empty:
        raise ValueError("No usable team-season rows remain after input validation.")

    return ranking_data.sort_values(["year", "tmID"]).reset_index(drop=True)


def _spearman_correlation(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Calculate within-season Spearman correlation, safely handling ties."""
    actual_series = pd.Series(np.asarray(actual, dtype=float))
    predicted_series = pd.Series(np.asarray(predicted, dtype=float))
    if len(actual_series) < 2:
        return float("nan")

    actual_ranks = actual_series.rank(method="average")
    predicted_ranks = predicted_series.rank(method="average")
    if actual_ranks.nunique() < 2 or predicted_ranks.nunique() < 2:
        return float("nan")
    return float(actual_ranks.corr(predicted_ranks, method="pearson"))


def _rank_mae(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Mean absolute error between actual and predicted league positions."""
    actual_ranks = pd.Series(np.asarray(actual, dtype=float)).rank(
        ascending=False, method="average"
    )
    predicted_ranks = pd.Series(np.asarray(predicted, dtype=float)).rank(
        ascending=False, method="average"
    )
    return float(np.mean(np.abs(actual_ranks.to_numpy() - predicted_ranks.to_numpy())))


def _calculate_metrics(
    actual: Sequence[float],
    predicted: Sequence[float],
    team_ids: Sequence[str],
    train_size: int,
    features_used: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Return regression and ranking metrics plus the values needed downstream."""
    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)

    return {
        "mse": float(mean_squared_error(actual_array, predicted_array)),
        "mae": float(mean_absolute_error(actual_array, predicted_array)),
        "spearman": _spearman_correlation(actual_array, predicted_array),
        "rank_mae": _rank_mae(actual_array, predicted_array),
        "predictions": predicted_array,
        "actuals": actual_array,
        "team_ids": np.asarray(team_ids, dtype=str),
        "n_train": int(train_size),
        "n_test": int(len(actual_array)),
        "features_used": list(features_used or []),
    }

def _baseline_predictions(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    target_col: str,
) -> Dict[str, np.ndarray]:
    """Generate last-season and two-season baselines without future information.

    For teams with no usable history, the fallback is the mean target observed
    in the training period. The two-season baseline falls back to last season
    first, then to the training mean.
    """
    training_mean = float(train_data[target_col].mean())

    if "PtsPct_Lag1" in test_data.columns:
        last_season = pd.to_numeric(
            test_data["PtsPct_Lag1"], errors="coerce"
        ).fillna(training_mean)
    else:
        last_season = pd.Series(training_mean, index=test_data.index, dtype=float)

    if "PtsPct_2Y_Avg" in test_data.columns:
        two_season = pd.to_numeric(
            test_data["PtsPct_2Y_Avg"], errors="coerce"
        )
    else:
        two_season = pd.Series(np.nan, index=test_data.index, dtype=float)

    two_season = two_season.fillna(last_season).fillna(training_mean)

    return {
        "LastSeasonBaseline": last_season.to_numpy(dtype=float),
        "TwoSeasonAverageBaseline": two_season.to_numpy(dtype=float),
    }

def walk_forward_train_test(
    datasets: Union[pd.DataFrame, Mapping[str, pd.DataFrame]],
    target_col: str = TARGET_COLUMN,
    window_size: int = DEFAULT_WINDOW_SIZE,
    window_type: str = "expanding",
    start_year: int = DEFAULT_START_YEAR,
    feature_sets: Optional[Mapping[str, Sequence[str]]] = None,
    verbose: bool = True,
) -> Dict[int, Dict[str, Dict[str, Dict[str, Any]]]]:
    """Run one-season-ahead walk-forward validation.

    Parameters
    ----------
    datasets:
        Either the engineered output of ``build_ranking_dataset`` or the
        cleaned datasets dictionary containing ``datasets['Teams']``.
    target_col:
        Must be ``target_PtsPct`` (or the target name used in a custom
        engineered DataFrame). ``rank`` is explicitly rejected.
    window_size:
        Number of prior target seasons required before the first test season.
        For a sliding window, it is also the number of seasons retained for
        every subsequent training fold. Default: 9.
    window_type:
        ``'expanding'`` uses all prior seasons after the initial window;
        ``'sliding'`` keeps only the latest ``window_size`` seasons.
    start_year:
        Used only when a cleaned datasets dictionary is passed. Historical
        seasons before this year remain available as lagged feature sources.
    feature_sets:
        Optional mapping of feature-set names to column names. By default,
        uses the three configurations in ``feature_engineering.py``.
    verbose:
        Print fold-level progress and metrics.

    Returns
    -------
    dict
        ``results[test_year][feature_set][model_name]`` contains ``mse``,
        ``mae``, ``spearman``, ``rank_mae``, predictions, actuals, team IDs,
        and fold metadata. This structure is compatible with the summary
        function in ``test_ranking.py`` which reads ``mse`` and ``mae``.
    """
    if not isinstance(window_size, int) or window_size < 2:
        raise ValueError("window_size must be an integer of at least 2 seasons.")
    if window_type not in {"expanding", "sliding"}:
        raise ValueError("window_type must be either 'expanding' or 'sliding'.")

    ranking_data = _prepare_ranking_data(datasets, start_year, target_col)

    if feature_sets is None:
        feature_sets = get_feature_sets()
    else:
        feature_sets = {name: list(columns) for name, columns in feature_sets.items()}

    if not feature_sets:
        raise ValueError("At least one feature set must be provided.")

    validated_feature_sets: Dict[str, List[str]] = {}
    for set_name, columns in feature_sets.items():
        available = [column for column in columns if column in ranking_data.columns]
        if not available:
            warnings.warn(
                f"Feature set '{set_name}' has no available columns and will be skipped.",
                RuntimeWarning,
            )
            continue
        validated_feature_sets[set_name] = available

    if not validated_feature_sets:
        raise ValueError("None of the requested feature sets has usable columns.")

    years = sorted(ranking_data["year"].unique().tolist())
    if len(years) <= window_size:
        raise ValueError(
            f"Only {len(years)} target seasons are available, but at least "
            f"{window_size + 1} are required for a {window_size}-season initial "
            "training window and one test season. Check start_year and the data."
        )

    if verbose:
        print("=== Walk-Forward Training & Testing ===")
        print(f"Target: {target_col}")
        print(f"Seasons: {years[0]}–{years[-1]}")
        print(f"Validation: {window_type} window ({window_size} initial training seasons)")
        print(f"Feature sets: {', '.join(validated_feature_sets)}")

    all_results: Dict[int, Dict[str, Dict[str, Dict[str, Any]]]] = {}

    for year_index in range(window_size, len(years)):
        test_year = years[year_index]

        if window_type == "expanding":
            train_years = years[:year_index]
        else:
            train_years = years[year_index - window_size : year_index]

        train_data = ranking_data[ranking_data["year"].isin(train_years)].copy()
        test_data = ranking_data[ranking_data["year"].eq(test_year)].copy()

        train_data = train_data.dropna(subset=[target_col])
        test_data = test_data.dropna(subset=[target_col])
        if train_data.empty or test_data.empty:
            if verbose:
                print(f"Skipping {test_year}: empty training or test fold.")
            continue

        y_train = train_data[target_col].astype(float)
        y_test = test_data[target_col].astype(float)
        team_ids = test_data["tmID"].astype(str).to_numpy()

        if verbose:
            print(
                f"\n--- Train: {train_years[0]}–{train_years[-1]} "
                f"({len(train_data)} team-seasons) | "
                f"Test: {test_year} ({len(test_data)} teams) ---"
            )

        all_results[test_year] = {}

        baseline_predictions = _baseline_predictions(train_data, test_data, target_col)

        for feature_set_name, requested_features in validated_feature_sets.items():
            fold_results: Dict[str, Dict[str, Any]] = {}

            usable_features = []
            for column in requested_features:
                train_values = pd.to_numeric(train_data[column], errors="coerce")
                if train_values.notna().any():
                    usable_features.append(column)


            for baseline_name, predictions in baseline_predictions.items():
                fold_results[baseline_name] = _calculate_metrics(
                    y_test,
                    predictions,
                    team_ids,
                    train_size=len(train_data),
                    features_used=[],
                )

            if usable_features:
                X_train = train_data[usable_features].apply(
                    pd.to_numeric, errors="coerce"
                )
                X_test = test_data[usable_features].apply(
                    pd.to_numeric, errors="coerce"
                )

                for model_name, estimator in _make_estimators().items():
                    pipeline = _make_pipeline(estimator)
                    try:
                        pipeline.fit(X_train, y_train)
                        predictions = pipeline.predict(X_test)
                    except Exception as exc:
                        raise RuntimeError(
                            f"Model '{model_name}' failed for feature set "
                            f"'{feature_set_name}' in test season {test_year}: {exc}"
                        ) from exc

                    fold_results[model_name] = _calculate_metrics(
                        y_test,
                        predictions,
                        team_ids,
                        train_size=len(train_data),
                        features_used=usable_features,
                    )
            else:
                warnings.warn(
                    f"No feature has observed training values for '{feature_set_name}' "
                    f"before {test_year}; model estimates were skipped. Baselines remain available.",
                    RuntimeWarning,
                )

            all_results[test_year][feature_set_name] = fold_results

            if verbose:
                print(f"\nFeatures: {feature_set_name} {usable_features}")
                for model_name, metrics in fold_results.items():
                    spearman = metrics["spearman"]
                    spearman_text = f"{spearman:.3f}" if np.isfinite(spearman) else "n/a"
                    print(
                        f"{model_name:<27} "
                        f"MSE: {metrics['mse']:.5f} | "
                        f"MAE: {metrics['mae']:.5f} | "
                        f"Spearman: {spearman_text} | "
                        f"Rank MAE: {metrics['rank_mae']:.2f}"
                    )

    if not all_results:
        raise RuntimeError(
            "Walk-forward validation produced no results. Check the number of "
            "available seasons and the selected window size."
        )

    return all_results


if __name__ == "__main__":
    print(
        "This module is designed to be imported by main.py. Build the cleaned "
        "Teams table, then call walk_forward_train_test(...)."
    )