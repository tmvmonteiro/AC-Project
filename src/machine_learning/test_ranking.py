def summarize_walk_forward_results(walk_forward_results):
    """
    Summarizes average performance metrics across all test years for each model.
    """
    summary = {}
    
    for test_year, models in walk_forward_results.items():
        for model_name, metrics in models.items():
            if model_name not in summary:
                summary[model_name] = {"mse_list": [], "mae_list": []}
            summary[model_name]["mse_list"].append(metrics["mse"])
            summary[model_name]["mae_list"].append(metrics["mae"])
            
    print("\n=== Overall Walk-Forward Summary Across All Test Years ===")
    for model_name, metrics in summary.items():
        avg_mse = sum(metrics["mse_list"]) / len(metrics["mse_list"])
        avg_mae = sum(metrics["mae_list"]) / len(metrics["mae_list"])
        print(f"Model: {model_name:<18} | Avg MSE: {avg_mse:.4f} | Avg MAE: {avg_mae:.4f}")
        
    return summary