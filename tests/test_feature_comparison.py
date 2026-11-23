import ast
import contextlib
import io
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer


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


if __name__ == "__main__":
    unittest.main()
