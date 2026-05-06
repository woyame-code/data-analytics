import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro
import os

def run_eda_and_tests():
    input_file = 'data/processed/final_analysis_data.csv'
    plots_dir = 'results/plots'
    stats_file = 'results/statistics_summary.txt'

    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    df = pd.read_csv(input_file)

    # Create output directories
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(os.path.dirname(stats_file), exist_ok=True)

    # 1. Boxplot for poverty_rate by Section_Class
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Section_Class', y='poverty_rate', data=df)
    plt.title('Poverty Rate by Section/Class')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'boxplot_poverty_rate.png'))
    plt.close()

    # 2. Correlation Heatmap (only numeric columns)
    numeric_df = df.select_dtypes(include=['number'])
    plt.figure(figsize=(12, 10))
    corr_matrix = numeric_df.corr()
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'correlation_heatmap.png'))
    plt.close()

    # 3. Shapiro-Wilk Test on poverty_rate
    # Filter out NaNs if any, though feature engineering filled them with 0
    poverty_rates = df['poverty_rate'].dropna()
    stat, p_value = shapiro(poverty_rates)

    # Save the test result
    with open(stats_file, 'w') as f:
        f.write("Shapiro-Wilk Test for Normality on poverty_rate\n")
        f.write("----------------------------------------------\n")
        f.write(f"Test Statistic: {stat:.4f}\n")
        f.write(f"p-value: {p_value:.4e}\n")

        alpha = 0.05
        if p_value > alpha:
            f.write("Conclusion: The sample looks Gaussian (fail to reject H0)\n")
        else:
            f.write("Conclusion: The sample does not look Gaussian (reject H0)\n")

    print("EDA and tests completed. Outputs saved to results/ directory.")

if __name__ == '__main__':
    run_eda_and_tests()
