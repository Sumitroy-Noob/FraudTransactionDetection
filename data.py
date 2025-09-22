import pandas as pd
import os

pkl_dir = 'D:\data'  # Replace with the actual path

all_dataframes = []
for filename in os.listdir(pkl_dir):
    if filename.endswith('.pkl'):
        filepath = os.path.join(pkl_dir, filename)
        df = pd.read_pickle(filepath)
        all_dataframes.append(df)

combined_df = pd.concat(all_dataframes, ignore_index=True)
output_csv_path = 'data.csv'  # Desired name for the output CSV
combined_df.to_csv(output_csv_path, index=False)