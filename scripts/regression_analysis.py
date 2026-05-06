import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_regression():
    input_file = 'data/processed/final_analysis_data.csv'
    plots_dir = 'results/plots'
    model_output_file = 'results/model_output.csv'

    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    df = pd.read_csv(input_file)

    # Patsy intercepts 'C' as the column 'C' instead of the Categorical wrapper if column 'C' exists.
    # We drop the class count columns to avoid this conflict.
    columns_to_drop = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    # Ensure directories exist
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(os.path.dirname(model_output_file), exist_ok=True)

    import numpy as np

    df['log_poverty_rate'] = np.log1p(df['poverty_rate'])

    # Formulas
    formula_base = 'poverty_rate ~ kids_per_household + unmarried_ratio + C(Section_Class)'
    formula_transformed = 'log_poverty_rate ~ kids_per_household + unmarried_ratio + C(Section_Class)'

    # Fit both models with robust standard errors (HC3)
    model_base = smf.ols(formula=formula_base, data=df)
    results_base = model_base.fit(cov_type='HC3')

    model_transformed = smf.ols(formula=formula_transformed, data=df)
    results_transformed = model_transformed.fit(cov_type='HC3')

    # Calculate residuals
    residuals_base = results_base.resid
    residuals_transformed = results_transformed.resid

    # 1. Shapiro-Wilk test on residuals
    from scipy.stats import shapiro
    stat_base, p_value_base = shapiro(residuals_base)
    stat_trans, p_value_trans = shapiro(residuals_transformed)

    stats_file = 'results/statistics_summary.txt'
    with open(stats_file, 'a') as f:
        f.write("\n\nShapiro-Wilk Test for Normality on Regression Residuals (Base Model)\n")
        f.write("-------------------------------------------------------\n")
        f.write(f"Test Statistic: {stat_base:.4f}\n")
        f.write(f"p-value: {p_value_base:.4e}\n")

        alpha = 0.05
        if p_value_base > alpha:
            f.write("Conclusion: The residuals look Gaussian (fail to reject H0)\n")
        else:
            f.write("Conclusion: The residuals do not look Gaussian (reject H0)\n")

        f.write("\n\nShapiro-Wilk Test for Normality on Regression Residuals (Transformed Model)\n")
        f.write("-------------------------------------------------------\n")
        f.write(f"Test Statistic: {stat_trans:.4f}\n")
        f.write(f"p-value: {p_value_trans:.4e}\n")

        if p_value_trans > alpha:
            f.write("Conclusion: The residuals look Gaussian (fail to reject H0)\n")
        else:
            f.write("Conclusion: The residuals do not look Gaussian (reject H0)\n")

        f.write("\n\nModel Comparison\n")
        f.write("----------------\n")
        if p_value_trans > p_value_base:
            f.write("The transformed model (log1p) better fulfills the statistical prerequisites (normality of residuals).\n")
        else:
            f.write("The base model better fulfills the statistical prerequisites (normality of residuals).\n")

    # 2. Q-Q Plots
    import statsmodels.api as sm

    # Base model Q-Q plot
    plt.figure(figsize=(8, 8))
    sm.qqplot(residuals_base, line='45', fit=True)
    plt.title('Q-Q Plot of Regression Residuals (Base Model)')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'base_model_qqplot.png'))
    plt.close('all')

    # Transformed model Q-Q plot
    plt.figure(figsize=(8, 8))
    sm.qqplot(residuals_transformed, line='45', fit=True)
    plt.title('Q-Q Plot of Regression Residuals (Transformed Model)')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'transformed_model_qqplot.png'))
    plt.close('all')

    # 3. Output Final Regression Report
    final_report_file = 'results/final_regression_report.txt'
    with open(final_report_file, 'w') as f:
        if p_value_trans > p_value_base:
            f.write(results_transformed.summary().as_text())
        else:
            f.write(results_base.summary().as_text())

    print("Regression analysis completed. Outputs saved.")

if __name__ == '__main__':
    run_regression()
