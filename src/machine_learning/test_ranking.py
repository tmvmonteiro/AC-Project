import numpy as np

def summarize_walk_forward_results(walk_forward_results):

    summary = {}

    for test_year, feature_results in walk_forward_results.items():

        for feature_name, models in feature_results.items():

            if feature_name not in summary:
                summary[feature_name] = {}

            for model_name, metrics in models.items():

                if model_name not in summary[feature_name]:
                    summary[feature_name][model_name] = {
                        "mse": [],
                        "mae": []
                    }

                summary[feature_name][model_name]["mse"].append(
                    metrics["mse"]
                )

                summary[feature_name][model_name]["mae"].append(
                    metrics["mae"]
                )

    print("\n=== Feature Efficiency Summary ===")

    for feature_name, models in summary.items():

        print(f"\n--- Features: {feature_name} ---")

        for model_name, metrics in models.items():

            avg_mse = np.mean(metrics["mse"])
            avg_mae = np.mean(metrics["mae"])

            print(
                f"{model_name:<18} | "
                f"Avg MSE: {avg_mse:.4f} | "
                f"Avg MAE: {avg_mae:.4f}"
            )

    return summary