import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

def engineer_features(df):
    """
    Engineers individual features like Win-Loss ratio and Goal Differential.
    """
    df = df.copy()
    
    # Calculate Win-Loss Ratio (handling division by zero if games played is 0)
    total_games = df['W'] + df['L'] + df.get('T', 0)
    df['WinLossRatio'] = np.where(total_games > 0, df['W'] / total_games, 0.0)
    
    # Calculate Goal Differential
    if 'GF' in df.columns and 'GA' in df.columns:
        df['GoalDifferential'] = df['GF'] - df['GA']
    else:
        df['GoalDifferential'] = 0
        
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

def walk_forward_train_test(datasets, target_col='rank'):
    """
    Performs walk-forward validation: trains on past years, tests on the next year, 
    and rolls forward.
    """
    if 'Teams' not in datasets:
        raise ValueError("Teams dataset is required for ranking models.")
    
    df = datasets['Teams'].copy()
    df = engineer_features(df)
    
    feature_cols = ['WinLossRatio', 'GoalDifferential', 'W', 'L', 'GF', 'GA']
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    if target_col not in df.columns or 'year' not in df.columns:
        raise ValueError(f"Target column '{target_col}' or 'year' column not found in Teams dataset.")
    
    # Clean missing values for features, target, and year
    df = df.dropna(subset=feature_cols + [target_col, 'year'])
    
    years = sorted(df['year'].unique())
    if len(years) < 2:
        raise ValueError("Not enough distinct years for walk-forward validation.")
    
    models_dict = get_model_pipeline()
    all_results = {}
    
    # Walk-forward loop: Train on years up to year t, test on year t+1
    for i in range(len(years) - 1):
        train_years = years[:i+1]
        test_year = years[i+1]
        
        train_data = df[df['year'].isin(train_years)]
        test_data = df[df['year'] == test_year]
        
        if len(test_data) == 0:
            continue
            
        X_train = train_data[feature_cols]
        y_train = train_data[target_col]
        X_test = test_data[feature_cols]
        y_test = test_data[target_col]
        
        print(f"\n--- Walk-Forward Step: Train Years {train_years[0]}–{train_years[-1]} | Test Year: {test_year} ---")
        
        year_results = {}
        for name, model in models_dict.items():
            # Train model
            model.fit(X_train, y_train)
            
            # Predict
            preds = model.predict(X_test)
            
            mse = mean_squared_error(y_test, preds)
            mae = mean_absolute_error(y_test, preds)
            
            year_results[name] = {
                "model": model,
                "mse": mse,
                "mae": mae,
                "predictions": preds
            }
            print(f"[{name}] Test MSE: {mse:.4f} | Test MAE: {mae:.4f}")
            
        all_results[test_year] = year_results
        
    return all_results