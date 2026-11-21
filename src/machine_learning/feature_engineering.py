from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


TARGET_COLUMN = "target_PtsPct"

ID_COLUMNS = ["tmID", "year"]

LAST_SEASON_FEATURES = ["PtsPct_Lag1"]
TWO_SEASON_FEATURES = ["PtsPct_2Y_Avg"]

EXTENDED_FEATURES = [
    "PtsPct_2Y_Avg",
    "PtsPct_Trend",
    "GoalDiffPerGame_Lag1",
    "GoalsAgainstPerGame_Lag1",
    "PenaltyMinutesPerGame_Lag1",
]

FEATURE_SETS: Dict[str, List[str]] = {
    "LastSeason": LAST_SEASON_FEATURES.copy(),
    "TwoSeasonAverage": TWO_SEASON_FEATURES.copy(),
    "Extended": EXTENDED_FEATURES.copy(),
}

_REQUIRED_COLUMNS = ["year", "tmID", "G", "Pts"]
_OPTIONAL_NUMERIC_COLUMNS = ["GF", "GA", "PIM"]


def _numeric_column(df: pd.DataFrame, column: str) -> pd.Series:
    """Return a numeric column, or an all-NaN series if it is unavailable."""
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce").astype("float64")


def _prepare_team_seasons(teams_df: pd.DataFrame) -> pd.DataFrame:
    """Validate and calculate season-level metrics from the cleaned Teams table."""
    if not isinstance(teams_df, pd.DataFrame):
        raise TypeError("teams_df must be a pandas DataFrame.")

    missing = [column for column in _REQUIRED_COLUMNS if column not in teams_df.columns]
    if missing:
        raise ValueError(
            "The Teams DataFrame is missing required columns: " + ", ".join(missing)
        )

    df = teams_df.copy()

    df["tmID"] = df["tmID"].astype("string").str.strip()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["G"] = pd.to_numeric(df["G"], errors="coerce")
    df["Pts"] = pd.to_numeric(df["Pts"], errors="coerce")

    non_integer_years = df["year"].notna() & (df["year"] % 1 != 0)
    if non_integer_years.any():
        examples = df.loc[non_integer_years, "year"].head(5).tolist()
        raise ValueError(f"'year' contains non-integer values, for example: {examples}")

    df = df.dropna(subset=["year", "tmID", "G", "Pts"]).copy()
    df = df[df["tmID"].ne("") & df["G"].gt(0)].copy()
    df["year"] = df["year"].astype("int64")

    for column in _OPTIONAL_NUMERIC_COLUMNS:
        df[column] = _numeric_column(df, column)

    duplicate_mask = df.duplicated(subset=["tmID", "year"], keep=False)
    if duplicate_mask.any():
        examples = (
            df.loc[duplicate_mask, ["tmID", "year"]]
            .drop_duplicates()
            .head(10)
            .to_dict("records")
        )
        raise ValueError(
            "Teams contains more than one record for a team-season. "
            f"Resolve these duplicates before feature engineering. Examples: {examples}"
        )

    df["PtsPct"] = df["Pts"] / (2.0 * df["G"])

    df["GoalDiffPerGame"] = (df["GF"] - df["GA"]) / df["G"]
    df["GoalsAgainstPerGame"] = df["GA"] / df["G"]
    df["PenaltyMinutesPerGame"] = df["PIM"] / df["G"]

    invalid_pct = df["PtsPct"].notna() & ~df["PtsPct"].between(0, 1)
    if invalid_pct.any():
        examples = df.loc[invalid_pct, ["tmID", "year", "Pts", "G", "PtsPct"]].head(5)
        raise ValueError(
            "Calculated PtsPct falls outside [0, 1]. Check Pts and G. "
            f"Examples: {examples.to_dict('records')}"
        )

    return df[
        [
            "tmID",
            "year",
            "PtsPct",
            "GoalDiffPerGame",
            "GoalsAgainstPerGame",
            "PenaltyMinutesPerGame",
        ]
    ].sort_values(["tmID", "year"]).reset_index(drop=True)


def build_ranking_dataset(
    teams_df: pd.DataFrame,
    start_year: int = 1995,
    require_two_seasons: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Build a leakage-safe, supervised dataset for next-season prediction.

    Each output row describes a team in target season ``year``. For that row:

    * ``PtsPct_Lag1`` and the three per-game statistics come from year - 1.
    * ``PtsPct_Lag2`` comes from year - 2.
    * ``PtsPct_2Y_Avg`` is the unweighted average of the previous two seasons.
    * ``PtsPct_Trend`` is previous season minus the season before it.
    * ``target_PtsPct`` is the actual points percentage in the target season.

    Exact year joins are used: a team's record from two seasons ago is not
    treated as its record from last season when an intervening season is absent.

    Parameters
    ----------
    teams_df:
        Cleaned ``Teams`` DataFrame (for example, ``cleaned_datasets["Teams"]``).
    start_year:
        First target season to include. The default follows the notebook's
        decision to focus the modelling period on 1995 onwards. Historical
        seasons before this year are still used to build lagged features.
    require_two_seasons:
        If True, retain only rows with points-percentage history for both
        preceding seasons (useful for a strict complete-history experiment).
        If False (default), retain all target team-seasons and leave unavailable
        historical features as NaN. This avoids silently excluding expansion or
        renamed teams; the training pipeline must impute using training-fold
        statistics only.
    verbose:
        Print a short summary of rows retained and excluded.

    Returns
    -------
    pandas.DataFrame
        A DataFrame sorted by target ``year`` and ``tmID``. It contains the ID
        columns, engineered features, and ``target_PtsPct``. The original
        ``rank`` column is deliberately not used.
    """
    if not isinstance(start_year, (int, np.integer)):
        raise TypeError("start_year must be an integer.")

    season_metrics = _prepare_team_seasons(teams_df)

    target = season_metrics[["tmID", "year", "PtsPct"]].rename(
        columns={"PtsPct": TARGET_COLUMN}
    )
    target = target[target["year"] >= int(start_year)].copy()

    lag1 = season_metrics[
        [
            "tmID",
            "year",
            "PtsPct",
            "GoalDiffPerGame",
            "GoalsAgainstPerGame",
            "PenaltyMinutesPerGame",
        ]
    ].copy()
    lag1["year"] = lag1["year"] + 1
    lag1 = lag1.rename(
        columns={
            "PtsPct": "PtsPct_Lag1",
            "GoalDiffPerGame": "GoalDiffPerGame_Lag1",
            "GoalsAgainstPerGame": "GoalsAgainstPerGame_Lag1",
            "PenaltyMinutesPerGame": "PenaltyMinutesPerGame_Lag1",
        }
    )

    lag2 = season_metrics[["tmID", "year", "PtsPct"]].copy()
    lag2["year"] = lag2["year"] + 2
    lag2 = lag2.rename(columns={"PtsPct": "PtsPct_Lag2"})

    result = target.merge(
        lag1,
        on=["tmID", "year"],
        how="left",
        validate="one_to_one",
    ).merge(
        lag2,
        on=["tmID", "year"],
        how="left",
        validate="one_to_one",
    )

    # In 2005, the 2004 season is absent (NHL lockout). For derived features,
    # use the last available season instead of leaving every team with the
    # same all-NaN inputs. Keep PtsPct_Lag1 itself missing to preserve the
    # exact-year feature semantics.
    lag1_fallback = result["PtsPct_Lag1"].fillna(result["PtsPct_Lag2"])
    result["PtsPct_2Y_Avg"] = (
        lag1_fallback + result["PtsPct_Lag2"]
    ) / 2.0
    result["PtsPct_Trend"] = lag1_fallback - result["PtsPct_Lag2"]

    rows_before_history_filter = len(result)
    incomplete_history = result[["PtsPct_Lag1", "PtsPct_Lag2"]].isna().any(axis=1)
    incomplete_history_count = int(incomplete_history.sum())
    if require_two_seasons:
        result = result.loc[~incomplete_history].copy()

    result = result.sort_values(["year", "tmID"]).reset_index(drop=True)

    output_columns = [
        "tmID",
        "year",
        "target_PtsPct",
        "PtsPct_Lag1",
        "PtsPct_Lag2",
        "PtsPct_2Y_Avg",
        "PtsPct_Trend",
        "GoalDiffPerGame_Lag1",
        "GoalsAgainstPerGame_Lag1",
        "PenaltyMinutesPerGame_Lag1",
    ]
    result = result[output_columns]

    if verbose:
        excluded = rows_before_history_filter - len(result)
        if result.empty:
            print(
                "Feature engineering produced 0 rows. Check that Teams contains "
                "at least two historical seasons before start_year."
            )
        else:
            print("=== Feature Engineering Summary ===")
            print(f"Target seasons: {result['year'].min()}–{result['year'].max()}")
            print(f"Team-season examples created: {len(result):,}")
            if require_two_seasons:
                print(f"Rows excluded for missing two-season history: {excluded:,}")
            else:
                print(f"Rows with incomplete two-season history (kept): {incomplete_history_count:,}")
                print("Note: impute missing features inside each training fold, not before walk-forward validation.")
            print(f"Teams represented: {result['tmID'].nunique():,}")
            missing_optional = [
                feature
                for feature in [
                    "GoalDiffPerGame_Lag1",
                    "GoalsAgainstPerGame_Lag1",
                    "PenaltyMinutesPerGame_Lag1",
                ]
                if result[feature].isna().all()
            ]
            if missing_optional:
                print(
                    "Warning: no values available for optional feature(s): "
                    + ", ".join(missing_optional)
                )

    return result


def get_feature_sets() -> Dict[str, List[str]]:
    """Return a fresh copy of the predefined feature configurations."""
    return {name: columns.copy() for name, columns in FEATURE_SETS.items()}


if __name__ == "__main__":
    print(
        "Import build_ranking_dataset from feature_engineering.py and pass "
        "cleaned_datasets['Teams']."
    )
