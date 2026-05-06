import pandas as pd
from scipy.stats import kruskal
import os

def run_group_comparison():
    input_file = 'data/processed/final_analysis_data.csv'
    stats_file = 'results/statistics_summary.txt'

    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    df = pd.read_csv(input_file)

    # We want to compare poverty_rate across Section_Class
    # Create a list of arrays for each group
    groups = [group['poverty_rate'].values for name, group in df.groupby('Section_Class')]

    # Perform Kruskal-Wallis H-test
    stat, p_value = kruskal(*groups)

    # Append the results to the statistics summary
    with open(stats_file, 'a') as f:
        f.write("\n\nKruskal-Wallis Test on poverty_rate across Section_Class\n")
        f.write("------------------------------------------------------\n")
        f.write(f"Test Statistic: {stat:.4f}\n")
        f.write(f"p-value: {p_value:.4e}\n")

        alpha = 0.05
        if p_value < alpha:
            f.write("Conclusion: There is a significant difference between sections (reject H0)\n")
        else:
            f.write("Conclusion: No significant difference between sections (fail to reject H0)\n")

    print("Group comparison completed. Results appended to results/statistics_summary.txt.")

if __name__ == '__main__':
    run_group_comparison()
