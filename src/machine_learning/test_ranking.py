from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence
import warnings

import numpy as np
import pandas as pd


BASELINE_MODEL = "TwoSeasonAverageBaseline"
DEFAULT_PERIOD_SPLIT_YEAR = 2005

METRIC_COLUMNS = ["mse", "mae", "spearman", "rank_mae"]


def _as_float(value: Any) -> float:
    """Convert a metric to float, returning NaN for missing/non-numeric values."""
    try:
        if value is None:
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _results_to_frame(
    walk_forward_results: Mapping,
) -> pd.DataFrame:
    """Flatten the nested walk-forward results into one row per model/fold."""
    if not isinstance(walk_forward_results, Mapping) or not walk_forward_results:
        raise ValueError("walk_forward_results must be a non-empty results dictionary.")

    rows = []
    for test_year, feature_results in walk_forward_results.items():
        try:
            year = int(test_year)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid test-season key: {test_year!r}") from exc

        if not isinstance(feature_results, Mapping):
            warnings.warn(
                f"Skipping malformed results for season {test_year}: expected a mapping.",
                RuntimeWarning,
            )
            continue

        for feature_set, model_results in feature_results.items():
            if not isinstance(model_results, Mapping):
                warnings.warn(
                    f"Skipping malformed feature-set results for {year}/{feature_set}.",
                    RuntimeWarning,
                )
                continue

            for model_name, metrics in model_results.items():
                if not isinstance(metrics, Mapping):
                    warnings.warn(
                        f"Skipping malformed model results for {year}/{feature_set}/{model_name}.",
                        RuntimeWarning,
                    )
                    continue

                row = {
                    "test_year": year,
                    "feature_set": str(feature_set),
                    "model": str(model_name),
                }
                for metric in METRIC_COLUMNS:
                    row[metric] = _as_float(metrics.get(metric))

                row["n_test"] = _as_float(metrics.get("n_test"))
                row["n_train"] = _as_float(metrics.get("n_train"))
                rows.append(row)

    if not rows:
        raise ValueError("No valid model metrics were found in walk_forward_results.")

    frame = pd.DataFrame(rows)
    frame = frame.sort_values(["test_year", "feature_set", "model"]).reset_index(drop=True)
    return frame


def _add_baseline_deltas(
    overall: pd.DataFrame,
    by_season: pd.DataFrame,
    baseline_model: str = BASELINE_MODEL,
) -> pd.DataFrame:
    """Add paired mean improvements relative to the two-season baseline.

    Positive Spearman delta is better. Positive MAE/rank-MAE delta is also
    better because those errors are reduced relative to the baseline.
    """
    if overall.empty or by_season.empty:
        overall["spearman_delta_vs_2y"] = np.nan
        overall["mae_improvement_vs_2y"] = np.nan
        overall["rank_mae_improvement_vs_2y"] = np.nan
        return overall

    baseline = by_season.loc[
        by_season["model"].eq(baseline_model),
        ["test_year", "feature_set", "spearman", "mae", "rank_mae"],
    ].rename(
        columns={
            "spearman": "baseline_spearman",
            "mae": "baseline_mae",
            "rank_mae": "baseline_rank_mae",
        }
    )

    paired = by_season.merge(
        baseline,
        on=["test_year", "feature_set"],
        how="left",
        validate="many_to_one",
    )
    paired["spearman_delta"] = paired["spearman"] - paired["baseline_spearman"]
    paired["mae_improvement"] = paired["baseline_mae"] - paired["mae"]
    paired["rank_mae_improvement"] = (
        paired["baseline_rank_mae"] - paired["rank_mae"]
    )

    deltas = (
        paired.groupby(["feature_set", "model"], as_index=False)
        .agg(
            spearman_delta_vs_2y=("spearman_delta", "mean"),
            mae_improvement_vs_2y=("mae_improvement", "mean"),
            rank_mae_improvement_vs_2y=("rank_mae_improvement", "mean"),
        )
    )
    return overall.merge(deltas, on=["feature_set", "model"], how="left")


def _aggregate_metrics(
    frame: pd.DataFrame,
    group_columns: Sequence[str],
) -> pd.DataFrame:
    """Aggregate fold metrics without pooling team-level observations."""
    grouped = frame.groupby(list(group_columns), dropna=False)
    summary = grouped.agg(
        n_seasons=("test_year", "nunique"),
        mean_spearman=("spearman", "mean"),
        median_spearman=("spearman", "median"),
        std_spearman=("spearman", "std"),
        mean_mae=("mae", "mean"),
        mean_mse=("mse", "mean"),
        mean_rank_mae=("rank_mae", "mean"),
        mean_test_teams=("n_test", "mean"),
    ).reset_index()
    return summary


def build_prediction_table(
    walk_forward_results: Mapping,
) -> pd.DataFrame:
    """Build team-level predictions with actual and predicted league positions.

    The returned table contains one row per test team, feature set and model.
    Actual and predicted positions are ranked in descending points percentage.
    Ties receive average positions, matching the rank convention used by the
    Spearman calculation in ``train_ranking.py``.
    """
    if not isinstance(walk_forward_results, Mapping) or not walk_forward_results:
        raise ValueError("walk_forward_results must be a non-empty results dictionary.")

    output = []
    for test_year, feature_results in walk_forward_results.items():
        year = int(test_year)
        if not isinstance(feature_results, Mapping):
            continue

        for feature_set, model_results in feature_results.items():
            if not isinstance(model_results, Mapping):
                continue

            for model_name, metrics in model_results.items():
                if not isinstance(metrics, Mapping):
                    continue

                required = ["predictions", "actuals", "team_ids"]
                if any(key not in metrics for key in required):
                    continue

                predictions = np.asarray(metrics["predictions"], dtype=float).reshape(-1)
                actuals = np.asarray(metrics["actuals"], dtype=float).reshape(-1)
                team_ids = np.asarray(metrics["team_ids"], dtype=str).reshape(-1)

                if not (len(predictions) == len(actuals) == len(team_ids)):
                    warnings.warn(
                        "Skipping team-level output with mismatched array lengths: "
                        f"{year}/{feature_set}/{model_name}.",
                        RuntimeWarning,
                    )
                    continue
                if len(predictions) == 0:
                    continue

                actual_positions = pd.Series(actuals).rank(
                    ascending=False, method="average"
                ).to_numpy()
                predicted_positions = pd.Series(predictions).rank(
                    ascending=False, method="average"
                ).to_numpy()

                for idx in range(len(team_ids)):
                    output.append(
                        {
                            "year": year,
                            "feature_set": str(feature_set),
                            "model": str(model_name),
                            "tmID": team_ids[idx],
                            "actual_PtsPct": actuals[idx],
                            "predicted_PtsPct": predictions[idx],
                            "actual_rank": actual_positions[idx],
                            "predicted_rank": predicted_positions[idx],
                            "absolute_rank_error": abs(
                                actual_positions[idx] - predicted_positions[idx]
                            ),
                        }
                    )

    if not output:
        raise ValueError(
            "No team-level predictions could be extracted. Check that the training "
            "results contain predictions, actuals and team_ids."
        )

    return pd.DataFrame(output).sort_values(
        ["year", "feature_set", "model", "actual_rank", "tmID"]
    ).reset_index(drop=True)


def summarize_walk_forward_results(
    walk_forward_results: Mapping,
    *,
    period_split_year: int = DEFAULT_PERIOD_SPLIT_YEAR,
    output_dir: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Summarize walk-forward model performance and optionally export CSV files.

    Parameters
    ----------
    walk_forward_results:
        Nested results dictionary returned by ``walk_forward_train_test``.
    period_split_year:
        First year of the later period. The default separates 1995–2004 from
        2005 onwards, following the notebook's structural-break discussion.
    output_dir:
        Optional folder for CSV reports. If supplied, the folder is created
        and the overall, period, season and team-level prediction tables are
        saved there.
    verbose:
        Print the overall ranking and the pre-/post-2005 comparison.

    Returns
    -------
    dict[str, pandas.DataFrame]
        ``overall``, ``by_period``, ``by_season`` and ``predictions`` tables.
        The overall table is ranked primarily by mean Spearman (descending).
    """
    fold_metrics = _results_to_frame(walk_forward_results)

    overall = _aggregate_metrics(fold_metrics, ["feature_set", "model"])
    by_season = fold_metrics.copy()

    by_season["period"] = np.where(
        by_season["test_year"] < int(period_split_year),
        f"Before {period_split_year}",
        f"{period_split_year} onward",
    )
    by_period = _aggregate_metrics(
        by_season,
        ["period", "feature_set", "model"],
    )

    overall = _add_baseline_deltas(overall, by_season)
    overall = overall.sort_values(
        ["mean_spearman", "mean_rank_mae", "mean_mae"],
        ascending=[False, True, True],
        na_position="last",
    ).reset_index(drop=True)

    by_period = by_period.sort_values(
        ["period", "mean_spearman", "mean_rank_mae", "mean_mae"],
        ascending=[True, False, True, True],
        na_position="last",
    ).reset_index(drop=True)

    predictions = build_prediction_table(walk_forward_results)

    if verbose:
        print("\n=== Overall Walk-Forward Performance ===")
        display_columns = [
            "feature_set",
            "model",
            "n_seasons",
            "mean_spearman",
            "std_spearman",
            "mean_mae",
            "mean_mse",
            "mean_rank_mae",
            "spearman_delta_vs_2y",
            "mae_improvement_vs_2y",
        ]
        print(
            overall[display_columns].to_string(
                index=False,
                formatters={
                    "mean_spearman": "{:.3f}".format,
                    "std_spearman": "{:.3f}".format,
                    "mean_mae": "{:.4f}".format,
                    "mean_mse": "{:.5f}".format,
                    "mean_rank_mae": "{:.2f}".format,
                    "spearman_delta_vs_2y": "{:+.3f}".format,
                    "mae_improvement_vs_2y": "{:+.4f}".format,
                },
            )
        )

        print(f"\n=== Performance by Period (split at {period_split_year}) ===")
        print(
            by_period[
                [
                    "period",
                    "feature_set",
                    "model",
                    "n_seasons",
                    "mean_spearman",
                    "mean_mae",
                    "mean_rank_mae",
                ]
            ].to_string(
                index=False,
                formatters={
                    "mean_spearman": "{:.3f}".format,
                    "mean_mae": "{:.4f}".format,
                    "mean_rank_mae": "{:.2f}".format,
                },
            )
        )

        print(
            "\nInterpretation: higher Spearman is better; lower MAE, MSE and "
            "Rank MAE are better. Positive deltas versus the two-season "
            "baseline indicate improvement."
        )

    summary = {
        "overall": overall,
        "by_period": by_period,
        "by_season": by_season,
        "predictions": predictions,
    }

    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        overall.to_csv(destination / "ranking_summary_overall.csv", index=False)
        by_period.to_csv(destination / "ranking_summary_by_period.csv", index=False)
        by_season.to_csv(destination / "ranking_summary_by_season.csv", index=False)
        predictions.to_csv(destination / "ranking_team_predictions.csv", index=False)
        if verbose:
            print(f"\nCSV reports saved to: {destination.resolve()}")

    return summary


if __name__ == "__main__":
    print(
        "Import summarize_walk_forward_results(results) after running "
        "walk_forward_train_test() in train_ranking.py."
    )
