import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# --------------------------------------------------
#                  FEATURES 
# --------------------------------------------------

def calculate_win_loss_ratio(df):
    """Calculate the proportion of games won."""
    total_games = df["W"] + df["L"] + df.get("T", 0)

    return np.where(
        total_games > 0,
        df["W"] / total_games,
        0.0
    )

FEATURES = {
    "WinLossRatio": calculate_win_loss_ratio,
}

def engineer_features(df):
    """
    Engineers individual features
    """
    df = df.copy()

    selected_features = FEATURES.keys()

    for feature_name in selected_features:
        if feature_name not in FEATURES:
            raise ValueError(f"Unknown feature: {feature_name}")

        df[feature_name] = FEATURES[feature_name](df)

    return df

def get_model_pipeline():
    """
    Defines a list of models and an ensemble model.
    """
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    gb = GradientBoostingRegressor(random_state=42)
    lr = LinearRegression()
    
    # Ensemble model combining the list of models using VotingRegressor
    ensemble = VotingRegressor(estimators=[
        ('rf', rf),
        ('gb', gb),
        ('lr', lr)
    ])
    
    models_dict = {
        "RandomForest": rf,
        "GradientBoosting": gb,
        "LinearRegression": lr,
        "Ensemble": ensemble
    }
    
    return models_dict

def walk_forward_train_test(datasets, target_col='rank', window_size=9):
    """
    Performs non-overlapping walk-forward validation.

    Compares model performance using Features
    """

    if 'Teams' not in datasets:
        raise ValueError("Teams dataset is required for ranking models.")

    df = datasets['Teams'].copy()
    df = engineer_features(df)

    if target_col not in df.columns or 'year' not in df.columns:
        raise ValueError("Target or year column not found.")

    df = df.dropna(subset=[target_col, 'year'])

    # Feature configurations to compare
    feature_sets = {
        "WinLossRatio": ['WinLossRatio'],
        "GoalDifferential": ['GoalDifferential'],
        "Combined": ['WinLossRatio', 'GoalDifferential']
    }

    # Keep only available features
    feature_sets = {
        name: [col for col in features if col in df.columns]
        for name, features in feature_sets.items()
    }

    models_dict = get_model_pipeline()
    years = sorted(df['year'].unique())

    if len(years) < window_size + 1:
        raise ValueError("Not enough distinct years for validation.")

    all_results = {}
    step_size = window_size + 1

    for i in range(0, len(years) - window_size, step_size):

        train_years = years[i:i + window_size]
        test_year = years[i + window_size]

        train_data = df[df['year'].isin(train_years)]
        test_data = df[df['year'] == test_year]

        print(
            f"\n--- Training: {train_years[0]}–{train_years[-1]} "
            f"| Testing: {test_year} ---"
        )

        all_results[test_year] = {}

        for feature_name, feature_cols in feature_sets.items():

            if not feature_cols:
                continue

            # Remove missing feature values
            train = train_data.dropna(subset=feature_cols + [target_col])
            test = test_data.dropna(subset=feature_cols + [target_col])

            if train.empty or test.empty:
                continue

            X_train = train[feature_cols]
            y_train = train[target_col]

            X_test = test[feature_cols]
            y_test = test[target_col]

            all_results[test_year][feature_name] = {}

            print(f"\nFeatures: {feature_name}")

            for model_name, model in models_dict.items():

                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                mse = mean_squared_error(y_test, preds)
                mae = mean_absolute_error(y_test, preds)

                all_results[test_year][feature_name][model_name] = {
                    "mse": mse,
                    "mae": mae,
                    "predictions": preds
                }

                print(
                    f"[{model_name}] "
                    f"MSE: {mse:.4f} | MAE: {mae:.4f}"
                )

    return all_results