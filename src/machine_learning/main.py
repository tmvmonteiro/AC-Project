import os
import pandas as pd
from cleaning import clean_datasets
from train_ranking import walk_forward_train_test
from test_ranking import summarize_walk_forward_results

def load_data(folder_path="data/"):
    """
    Loads all CSV files from the specified folder into a dictionary of DataFrames.
    """
    datasets = {}
    if not os.path.exists(folder_path):
        print(f"Directory '{folder_path}' does not exist.")
        return datasets

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            table_name = file_name[:-4]
            datasets[table_name] = pd.read_csv(file_path)
            
    return datasets

def main():
    print("=== Starting Machine Learning Pipeline ===")
    
    # Load Data
    print("Loading datasets from 'data/'...")
    datasets = load_data("data/")
    print(f"Successfully loaded {len(datasets)} datasets.")
    
    # Clean Data
    print("Running data cleaning pipeline...")
    cleaned_datasets = clean_datasets(datasets)
    print("=== Data Cleaning Complete: All datasets successfully processed. ===")

    print("=== Starting Walk-Forward Training & Testing Phase ===")
    try:
        results = walk_forward_train_test(cleaned_datasets, target_col='rank')
        summarize_walk_forward_results(results)
    except Exception as e:
        print(f"Error during training/testing: {e}")
        return
    
    print("=== Ending Machine Learning Pipeline) ===")
    
    return cleaned_datasets

if __name__ == "__main__":
    main()