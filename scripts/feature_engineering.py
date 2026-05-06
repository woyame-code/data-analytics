import pandas as pd
import numpy as np
import os

def engineer_features():
    input_file = 'data/processed/merged_london_data.csv'
    output_file = 'data/processed/final_analysis_data.csv'

    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    df = pd.read_csv(input_file)

    # Calculate poverty_rate: (A + B + C + D) / Total_Persons
    # Avoid division by zero
    total_persons = df['Total_Persons'].replace(0, np.nan)
    df['poverty_rate'] = (df['A'] + df['B'] + df['C'] + df['D']) / total_persons

    # Calculate kids_per_household: Children_under_15 / Heads_of_Families
    heads_of_families = df['Heads_of_Families'].replace(0, np.nan)
    df['kids_per_household'] = df['Children_under_15'] / heads_of_families

    # Calculate unmarried_ratio: Unmarried_Men_and_Widowers / Total_Persons
    df['unmarried_ratio'] = df['Unmarried_Men_and_Widowers'] / total_persons

    # Fill any NaNs created by division by zero back with 0, or leave as is if preferred.
    # To match standard behavior for ratios, we can fill with 0
    df['poverty_rate'] = df['poverty_rate'].fillna(0)
    df['kids_per_household'] = df['kids_per_household'].fillna(0)
    df['unmarried_ratio'] = df['unmarried_ratio'].fillna(0)

    # Save the final dataset
    df.to_csv(output_file, index=False)
    print(f"Feature engineering complete. Saved to {output_file}")

if __name__ == '__main__':
    engineer_features()
