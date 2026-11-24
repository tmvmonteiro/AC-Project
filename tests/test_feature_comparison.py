import ast
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor
from scipy.stats import spearmanr
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error


NOTEBOOK = Path(__file__).resolve().parents[1] / "main.ipynb"


def load_functions():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = next(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if "def compare_feature_sets(" in "".join(cell.get("source", []))
    )
    tree = ast.parse(source)
    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"run_sliding_window_pipeline", "compare_feature_sets"}
    ]
    namespace = {"np": np, "pd": pd, "SimpleImputer": SimpleImputer}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    return namespace["compare_feature_sets"]


class FeatureComparisonTests(unittest.TestCase):
    def setUp(self):
        self.compare = load_functions()
        self.df = pd.DataFrame({
            "year": [2000, 2001, 2002, 2003],
            "first": [1.0, 2.0, 3.0, 4.0],
            "second": [4.0, 3.0, 2.0, 1.0],
            "target": [1.0, 2.0, 3.0, 4.0],
        })
        self.options = {
            "feature_sets": [["first"], ["first", "second"]],
            "df": self.df,
            "start_year_train": 2000,
            "start_year_test": 2002,
            "end_year": 2003,
            "target": "target",
            "models_dict": {"model_with_underscore": DummyRegressor()},
            "baseline_dict": {"baseline": lambda train, test, target: np.zeros(len(test))},
            "eval_dict": {
                "score": lambda actual, predicted, test: float(test["year"].iloc[0] - 2000),
                "matrix": lambda actual, predicted, test: np.full((2, 2), test["year"].iloc[0] - 2000),
            },
            "use_optuna": False,
            "window_size": 2,
        }

    def test_feature_year_and_aggregate_tables(self):
        yearly, summary = self.compare(**self.options)
        self.assertEqual(len(yearly), 16)
        self.assertEqual(len(summary), 8)
        self.assertEqual(set(yearly["year"]), {2002, 2003})
        self.assertEqual(set(yearly["features"]), {("first",), ("first", "second")})
        scalar = summary.query("feature_set == 1 and model == 'baseline' and evaluation == 'score'").iloc[0]
        self.assertEqual(scalar["kind"], "baseline")
        self.assertEqual(scalar["mean"], 2.5)
        self.assertEqual(scalar["median"], 2.5)
        self.assertAlmostEqual(scalar["std"], np.sqrt(0.5))
        matrix = summary.query("feature_set == 2 and model == 'model_with_underscore' and evaluation == 'matrix'").iloc[0]
        np.testing.assert_array_equal(matrix["mean"], np.full((2, 2), 2.5))
        np.testing.assert_array_equal(matrix["median"], np.full((2, 2), 2.5))
        np.testing.assert_allclose(matrix["std"], np.full((2, 2), np.sqrt(0.5)))

    def test_missing_values_and_single_season(self):
        self.options["end_year"] = 2002
        self.options["eval_dict"] = {"score": lambda actual, predicted, test: np.nan}
        yearly, summary = self.compare(**self.options)
        self.assertTrue(yearly["value"].isna().all())
        self.assertTrue(summary[["mean", "median", "std"]].isna().all().all())

    def test_empty_folds_and_invalid_feature_sets(self):
        self.options["start_year_train"] = 2004
        self.options["start_year_test"] = 2004
        self.options["end_year"] = 2004
        with contextlib.redirect_stdout(io.StringIO()):
            yearly, summary = self.compare(**self.options)
        self.assertTrue(yearly.empty)
        self.assertTrue(summary.empty)
        self.options["feature_sets"] = [["first"], []]
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
            self.compare(**self.options)


class RankingExperimentTests(unittest.TestCase):
    def test_variants_run_and_write_separate_result_files(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cell = next(cell for cell in notebook["cells"] if cell.get("id") == "compare-feature-sets-a")
        data = pd.read_csv(NOTEBOOK.parent / "processed/team_season.csv")
        data = data[data["year"].between(1995, 1998)]
        with tempfile.TemporaryDirectory() as temp_dir:
            namespace = {
                "Path": lambda path: Path(temp_dir) / path,
                "pd": SimpleNamespace(read_csv=lambda path: data.copy()),
                "xgb": xgb,
                "lgb": lgb,
                "CatBoostRegressor": CatBoostRegressor,
                "RandomForestRegressor": RandomForestRegressor,
                "Ridge": Ridge,
                "Lasso": Lasso,
                "compare_feature_sets": load_functions(),
                "spearman_metric": lambda yt, yp, td: spearmanr(yt, yp).statistic,
                "mae_metric": lambda yt, yp, td: mean_absolute_error(yt, yp),
                "top_k_accuracy": lambda yt, yp, k: len(
                    set(yt.nsmallest(k).index) & set(pd.Series(yp, index=yt.index).nsmallest(k).index)
                ) / k,
            }
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile("".join(cell["source"]), str(NOTEBOOK), "exec"), namespace)

            folder = Path(temp_dir) / "results"
            for name in ("xgboost", "lightgbm", "catboost", "random_forest", "ridge", "lasso"):
                yearly = pd.read_csv(folder / f"{name}_feature_year_results.csv")
                summary = pd.read_csv(folder / f"{name}_feature_summary.csv")
                self.assertEqual(yearly["model"].nunique(), 3)
                self.assertEqual(set(yearly["year"]), {1996, 1997, 1998})
                self.assertEqual(set(yearly["evaluation"]), {"Spearman", "MAE", "Top-5"})
                self.assertEqual(len(summary), 9)
                self.assertTrue({"mean", "median", "std"}.issubset(summary.columns))
            self.assertEqual(len(list(folder.glob("*.csv"))), 12)


if __name__ == "__main__":
    unittest.main()
