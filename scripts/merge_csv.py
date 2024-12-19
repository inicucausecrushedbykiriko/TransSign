import pandas as pd
import os

def merge_csv_files(input_dirs, output_file):
    """
    Merges all CSV files from specified directories into a single CSV file.

    Args:
        input_dirs (list): List of directories containing CSV files.
        output_file (str): Path to save the merged CSV file.
    """
    all_data = []
    for input_dir in input_dirs:
        for csv_file in os.listdir(input_dir):
            if csv_file.endswith('.csv'):
                file_path = os.path.join(input_dir, csv_file)
                data = pd.read_csv(file_path)
                all_data.append(data)

    merged_data = pd.concat(all_data, ignore_index=True)
    merged_data.to_csv(output_file, index=False)
    print(f"Merged dataset saved to {output_file}")

if __name__ == "__main__":
    input_dirs = ['./data/features/asl_features', './data/features/csl_features']
    output_file = './data/dataset.csv'
    merge_csv_files(input_dirs, output_file)
