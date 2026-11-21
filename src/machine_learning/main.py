from __future__ import annotations

import argparse
import inspect
import os
import sys
import traceback
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from feature_engineering import TARGET_COLUMN, build_ranking_dataset, get_feature_sets
from train_ranking import walk_forward_train_test
from test_ranking import summarize_walk_forward_results


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_START_YEAR = 1995
DEFAULT_WINDOW_SIZE = 9
DEFAULT_PERIOD_SPLIT_YEAR = 2005


class Terminal:

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = bool(
            enabled and sys.stdout.isatty() and "NO_COLOR" not in os.environ
        )

    def paint(self, text: str, code: str) -> str:
        if not self.enabled:
            return text
        return f"\033[{code}m{text}\033[0m"

    def title(self, text: str) -> None:
        line = "=" * 76
        print(self.paint(line, "36;1"))
        print(self.paint(f"  {text}", "36;1"))
        print(self.paint(line, "36;1"))

    def section(self, number: str, text: str) -> None:
        print()
        print(self.paint(f"[{number}] {text}", "34;1"))
        print(self.paint("-" * 76, "34"))

    def success(self, text: str) -> None:
        print(self.paint(f"[OK] {text}", "32;1"))

    def warning(self, text: str) -> None:
        print(self.paint(f"[WARNING] {text}", "33;1"))

    def error(self, text: str) -> None:
        print(self.paint(f"[ERROR] {text}", "31;1"))


TERM = Terminal()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the NHL team ranking prediction pipeline."
    )

    parser.add_argument(
        "--data-dir",
        default=None,
        help="Directory containing CSV files (default: <project>/data).",
    )

    parser.add_argument(
        "--results-dir",
        default=None,
        help="Directory where results will be saved (default: <project>/results).",
    )

    parser.add_argument(
        "--start-year",
        type=int,
        default=DEFAULT_START_YEAR,
        help=f"First season to predict (default: {DEFAULT_START_YEAR}).",
    )

    parser.add_argument(
        "--window-size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help=f"Initial number of training seasons (default: {DEFAULT_WINDOW_SIZE}).",
    )

    parser.add_argument(
        "--window-type",
        choices=("expanding", "sliding"),
        default="expanding",
        help="Walk-forward window type (default: expanding).",
    )

    parser.add_argument(
        "--period-split-year",
        type=int,
        default=DEFAULT_PERIOD_SPLIT_YEAR,
        help=f"First season of the second evaluation period (default: {DEFAULT_PERIOD_SPLIT_YEAR}).",
    )

    parser.add_argument(
        "--require-two-seasons",
        action="store_true",
        help="Exclude teams without complete history for the previous two seasons.",
    )

    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable terminal colors.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Display the full traceback when an error occurs.",
    )

    return parser


def _resolve_path(value: Optional[str], default: Path) -> Path:
    if value is None:
        return default.resolve()

    path = Path(value).expanduser()

    if not path.is_absolute():
        path = Path.cwd() / path

    return path.resolve()


def load_data(folder_path: Path) -> Dict[str, pd.DataFrame]:

    if not folder_path.exists():
        raise FileNotFoundError(f"Data directory does not exist: {folder_path}")

    if not folder_path.is_dir():
        raise NotADirectoryError(f"Data path is not a directory: {folder_path}")

    csv_files = sorted(
        path for path in folder_path.iterdir()
        if path.is_file() and path.suffix.lower() == ".csv"
    )

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {folder_path}")

    datasets: Dict[str, pd.DataFrame] = {}
    read_errors = []

    for file_path in csv_files:
        table_name = file_path.stem

        try:
            datasets[table_name] = pd.read_csv(
                file_path,
                encoding="utf-8-sig"
            )
        except Exception as exc:
            read_errors.append(f"{file_path.name}: {exc}")

    if read_errors:
        details = "\n  - ".join(read_errors)
        raise RuntimeError(
            "Failed to read the following CSV files:\n  - " + details
        )

    return datasets


def run_cleaning(
    datasets: Dict[str, pd.DataFrame]
) -> Dict[str, pd.DataFrame]:

    try:
        from cleaning import clean_datasets
    except ImportError as exc:
        raise ImportError(
            "Could not import clean_datasets. Make sure cleaning.py "
            "is in the same directory as main.py."
        ) from exc

    parameters = inspect.signature(clean_datasets).parameters

    if "verbose" in parameters:
        cleaned = clean_datasets(datasets, verbose=False)
    else:
        cleaned = clean_datasets(datasets)

    if not isinstance(cleaned, dict):
        raise TypeError(
            "clean_datasets() must return a dictionary of DataFrames."
        )

    if "Teams" not in cleaned:
        raise KeyError(
            "The cleaning process did not return the 'Teams' table. "
            "Check TABLES_TO_KEEP and the CSV filenames."
        )

    if not isinstance(cleaned["Teams"], pd.DataFrame) or cleaned["Teams"].empty:
        raise ValueError(
            "The 'Teams' table is empty or is not a valid DataFrame."
        )

    return cleaned


def _print_cleaning_summary(
    loaded: Dict[str, pd.DataFrame],
    cleaned: Dict[str, pd.DataFrame]
) -> None:

    print(f"CSV tables loaded       : {len(loaded)}")
    print(f"Tables after cleaning   : {len(cleaned)}")
    print()

    print(f"{'Table':<20}{'Before':>12}{'After':>12}{'Removed':>14}")
    print("-" * 58)

    for name in sorted(cleaned):
        before = len(loaded[name]) if name in loaded else 0
        after = len(cleaned[name])
        removed = before - after

        print(
            f"{name:<20}{before:>12,}{after:>12,}{removed:>14,}"
        )


def _print_final_evaluation(
    summary: dict,
    period_split_year: int
) -> None:

    overall = summary["overall"]
    by_period = summary["by_period"]

    TERM.section("6/6", "OVERALL RESULTS — BEST CONFIGURATIONS")

    columns = [
        "feature_set",
        "model",
        "n_seasons",
        "mean_spearman",
        "mean_mae",
        "mean_rank_mae",
        "spearman_delta_vs_2y",
    ]

    view = overall[columns].head(12).copy()

    view = view.rename(
        columns={
            "feature_set": "Features",
            "model": "Model",
            "n_seasons": "Seasons",
            "mean_spearman": "Spearman",
            "mean_mae": "MAE",
            "mean_rank_mae": "Rank MAE",
            "spearman_delta_vs_2y": "Delta vs 2Y",
        }
    )

    print(
        view.to_string(
            index=False,
            formatters={
                "Spearman": lambda value: (
                    "n/a" if pd.isna(value) else f"{value:.3f}"
                ),
                "MAE": lambda value: (
                    "n/a" if pd.isna(value) else f"{value:.4f}"
                ),
                "Rank MAE": lambda value: (
                    "n/a" if pd.isna(value) else f"{value:.2f}"
                ),
                "Delta vs 2Y": lambda value: (
                    "n/a" if pd.isna(value) else f"{value:+.3f}"
                ),
            },
        )
    )

    baseline_names = {
        "MeanBaseline",
        "LastSeasonBaseline",
        "TwoSeasonAverageBaseline",
    }

    machine_learning = overall[
        ~overall["model"].isin(baseline_names)
    ]

    if not machine_learning.empty:

        best = machine_learning.iloc[0]

        TERM.section("HIGHLIGHT", "BEST MACHINE LEARNING MODEL")

        print(f"Model               : {best['model']}")
        print(f"Feature set         : {best['feature_set']}")
        print(f"Mean Spearman       : {best['mean_spearman']:.3f}")
        print(f"Mean MAE            : {best['mean_mae']:.4f}")
        print(
            f"Mean ranking error  : "
            f"{best['mean_rank_mae']:.2f} positions"
        )

        delta = best.get(
            "spearman_delta_vs_2y",
            float("nan")
        )

        if pd.notna(delta):
            comparison = (
                "better" if delta > 0
                else "worse" if delta < 0
                else "equal"
            )

            print(
                f"Compared with 2Y baseline: "
                f"{delta:+.3f} Spearman ({comparison})"
            )

    two_year = overall[
        overall["model"].eq("TwoSeasonAverageBaseline")
    ]

    if not two_year.empty:

        baseline_score = two_year["mean_spearman"].mean()

        print()
        print(
            f"Two-season baseline (Walk-Forward): "
            f"Mean Spearman = {baseline_score:.3f}"
        )

        print(
            "Exploratory notebook reference: Spearman = 0.57"
        )

        print(
            "Note: The fairest comparison is with the 2Y baseline "
            "evaluated on the same validation seasons."
        )

    TERM.section(
        "TEMPORAL COMPARISON",
        f"BEFORE AND AFTER {period_split_year}"
    )

    if not by_period.empty:

        for period in by_period["period"].drop_duplicates().tolist():

            period_rows = by_period[
                by_period["period"].eq(period)
            ]

            print(f"\n{period}")

            print(
                period_rows[
                    [
                        "feature_set",
                        "model",
                        "n_seasons",
                        "mean_spearman",
                        "mean_mae",
                    ]
                ].head(5).to_string(
                    index=False,
                    formatters={
                        "mean_spearman": lambda value: (
                            "n/a" if pd.isna(value) else f"{value:.3f}"
                        ),
                        "mean_mae": lambda value: (
                            "n/a" if pd.isna(value) else f"{value:.4f}"
                        ),
                    },
                )
            )


def _print_output_files(results_dir: Path) -> None:

    expected_files = [
        "ranking_summary_overall.csv",
        "ranking_summary_by_period.csv",
        "ranking_summary_by_season.csv",
        "ranking_team_predictions.csv",
        "ranking_feature_dataset.csv",
    ]

    TERM.section("OUTPUT FILES", "RESULT EXPORT")

    for filename in expected_files:

        file_path = results_dir / filename

        if file_path.exists():
            TERM.success(str(file_path))
        else:
            TERM.warning(f"Not found: {file_path}")


def main(argv: Optional[list[str]] = None) -> int:

    global TERM

    args = _build_parser().parse_args(argv)

    TERM = Terminal(enabled=not args.no_color)

    data_dir = _resolve_path(
        args.data_dir,
        PROJECT_DIR / "data"
    )

    results_dir = _resolve_path(
        args.results_dir,
        PROJECT_DIR / "results"
    )

    TERM.title("NHL TEAM RANKING — MACHINE LEARNING PIPELINE")

    print(f"Project directory : {PROJECT_DIR}")
    print(f"Data directory    : {data_dir}")
    print(f"Results directory : {results_dir}")
    print(f"Target            : {TARGET_COLUMN} (Points Percentage)")
    print(
        f"Walk-forward      : {args.window_type} | "
        f"initial window = {args.window_size} seasons"
    )
    print(f"Analysis period   : since {args.start_year}")

    if args.window_size < 2:
        TERM.error("--window-size must be at least 2.")
        return 2

    if args.start_year < 1900:
        TERM.error("--start-year is invalid. Provide a valid season year.")
        return 2

    try:

        TERM.section("1/6", "DATA LOADING")

        datasets = load_data(data_dir)

        TERM.success(f"{len(datasets)} CSV files loaded.")

        print("Tables: " + ", ".join(sorted(datasets.keys())))

        TERM.section("2/6", "DATA CLEANING")

        cleaned_datasets = run_cleaning(datasets)

        _print_cleaning_summary(
            datasets,
            cleaned_datasets
        )

        TERM.success("Data cleaning completed.")

        teams = cleaned_datasets["Teams"]

        required_columns = {
            "year",
            "tmID",
            "G",
            "Pts"
        }

        missing_columns = sorted(
            required_columns.difference(teams.columns)
        )

        if missing_columns:
            raise ValueError(
                "The Teams table is missing required columns: "
                + ", ".join(missing_columns)
            )

        TERM.section("3/6", "FEATURE ENGINEERING")

        ranking_data = build_ranking_dataset(
            teams,
            start_year=args.start_year,
            require_two_seasons=args.require_two_seasons,
            verbose=True,
        )

        if ranking_data.empty:
            raise ValueError(
                "Feature engineering did not produce any training observations."
            )

        feature_sets = get_feature_sets()

        print("\nFeature configurations:")

        for name, features in feature_sets.items():
            print(f"  - {name:<20}: {', '.join(features)}")

        print(f"Target: {TARGET_COLUMN}")

        TERM.success(
            f"Modeling dataset created: {len(ranking_data):,} team-seasons."
        )

        results_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        ranking_data.to_csv(
            results_dir / "ranking_feature_dataset.csv",
            index=False
        )

        TERM.section("4/6", "WALK-FORWARD TRAINING")

        print("Running models and baselines for each test season...")
        print(
            "Missing values are handled within each training fold."
        )

        results = walk_forward_train_test(
            ranking_data,
            target_col=TARGET_COLUMN,
            window_size=args.window_size,
            window_type=args.window_type,
            start_year=args.start_year,
            verbose=False,
        )

        tested_years = sorted(results.keys())

        if not tested_years:
            raise ValueError(
                "Walk-forward training did not produce any test results."
            )

        TERM.success(
            f"Training and testing completed: {len(tested_years)} seasons "
            f"({tested_years[0]}–{tested_years[-1]})."
        )

        TERM.section("5/6", "EVALUATION AND EXPORT")

        summary = summarize_walk_forward_results(
            results,
            period_split_year=args.period_split_year,
            output_dir=str(results_dir),
            verbose=False,
        )
        
        _print_final_evaluation(
            summary,
            args.period_split_year
        )

        _print_output_files(results_dir)
        predictions_file = results_dir / "ranking_team_predictions.csv"

        if predictions_file.exists():
            predictions = pd.read_csv(predictions_file)

            print("\nAVAILABLE PREDICTIONS")
            print(predictions.columns.tolist())
            print(predictions.head(10).to_string(index=False))
        else:
            TERM.warning("The predictions file was not generated.")
            
        TERM.title("PIPELINE COMPLETED SUCCESSFULLY")

        print("Primary metric: Spearman (higher is better).")
        print("Error metrics: MAE, MSE and Rank MAE (lower is better).")
        print(
            "Check model stability across seasons before drawing conclusions."
        )   

        return 0

    except Exception as exc:

        TERM.error(f"{type(exc).__name__}: {exc}")

        print(
            "\nCheck CSV filenames, the Teams table columns "
            "and module compatibility."
        )

        if args.debug:
            print("\nFull traceback:")
            traceback.print_exc()
        else:
            print(
                "Run again with --debug to see the technical details."
            )

        return 1


if __name__ == "__main__":
    raise SystemExit(main())