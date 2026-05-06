import pandas as pd
import glob
import os

def clean_data():
    raw_dir = 'data/raw'
    processed_dir = 'data/processed'

    # List all CSV files
    all_files = glob.glob(os.path.join(raw_dir, '*.csv'))

    # Files to exclude (totals)
    exclude_files = ['classes_boroughs_total.csv', 'professions_total.csv']

    district_files = [f for f in all_files if os.path.basename(f) not in exclude_files]

    standard_columns = [
        'Section_Class',
        'ID',
        'Description',
        'Heads_of_Families',
        'Wives',
        'Children_under_15',
        'Young_Persons_15_to_20',
        'Unmarried_Men_and_Widowers',
        'Total_Persons',
        'A',
        'B',
        'C',
        'D',
        'E',
        'F',
        'G',
        'H',
        'Total_Classes'
    ]

    dfs = []

    for file in district_files:
        district_name = os.path.basename(file).replace('.csv', '').replace('-', '_').title()

        # Read the file
        df = pd.read_csv(file)

        # Override columns to unify them
        # Note: All district files have 18 columns
        if len(df.columns) == 18:
            df.columns = standard_columns

        # Add District column
        df['District'] = district_name

        # Replace NaNs with 0
        df = df.fillna(0)

        dfs.append(df)

    # Merge all districts
    merged_df = pd.concat(dfs, ignore_index=True)

    # Save to processed
    os.makedirs(processed_dir, exist_ok=True)
    merged_df.to_csv(os.path.join(processed_dir, 'merged_london_data.csv'), index=False)

if __name__ == '__main__':
    clean_data()
