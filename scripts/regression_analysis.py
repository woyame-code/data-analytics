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

    # Ensure directories exist
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(os.path.dirname(model_output_file), exist_ok=True)

    # Formula: poverty_rate ~ kids_per_household + unmarried_ratio + Section_Class
    # The column is named 'Section_Class' instead of 'Section'
    formula = 'poverty_rate ~ kids_per_household + unmarried_ratio + Section_Class'

    # Fit the multiple linear regression model
    model = smf.ols(formula=formula, data=df)
    results = model.fit()

    # 1. Save coefficients table and summary
    # We can write the summary as text to a CSV for simplicity, or extract the tables.
    # The prompt asks for "Koeffizienten-Tabelle und die Modell-Zusammenfassung (summary) als CSV oder Textdatei"
    # We will save the summary as text in a CSV named file.
    with open(model_output_file, 'w') as f:
        f.write(results.summary().as_csv())

    # 2. Residual Plot
    # Calculate residuals and fitted values
    residuals = results.resid
    fitted = results.fittedvalues

    plt.figure(figsize=(10, 6))
    sns.residplot(x=fitted, y=residuals, lowess=True,
                  scatter_kws={'alpha': 0.5}, line_kws={'color': 'red', 'lw': 2})
    plt.title('Residuals vs Fitted Values')
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.axhline(0, color='black', linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'regression_residuals.png'))
    plt.close()

    # 3. Shapiro-Wilk test on residuals
    from scipy.stats import shapiro
    stat, p_value = shapiro(residuals)
    stats_file = 'results/statistics_summary.txt'
    with open(stats_file, 'a') as f:
        f.write("\n\nShapiro-Wilk Test for Normality on Regression Residuals\n")
        f.write("-------------------------------------------------------\n")
        f.write(f"Test Statistic: {stat:.4f}\n")
        f.write(f"p-value: {p_value:.4e}\n")

        alpha = 0.05
        if p_value > alpha:
            f.write("Conclusion: The residuals look Gaussian (fail to reject H0)\n")
        else:
            f.write("Conclusion: The residuals do not look Gaussian (reject H0)\n")

    # 4. Q-Q Plot of the residuals
    import statsmodels.api as sm
    plt.figure(figsize=(8, 8))
    sm.qqplot(residuals, line='45', fit=True)
    plt.title('Q-Q Plot of Regression Residuals')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'regression_qqplot.png'))
    plt.close('all')

    print("Regression analysis completed. Outputs saved.")

if __name__ == '__main__':
    run_regression()
