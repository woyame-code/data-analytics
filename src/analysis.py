import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from factor_analyzer import FactorAnalyzer
import pingouin as pg
import seaborn as sns
import matplotlib.pyplot as plt
import pyreadstat
import os

def load_and_split():
    shortlist_df = pd.read_csv('data/exploration_shortlist_100.csv')
    valid_ids = shortlist_df['Variable_ID'].tolist()

    print(f"Loading {len(valid_ids)} variables from allbus_2023.dta...")

    try:
        df, meta = pyreadstat.read_dta('data/allbus_2023.dta', usecols=valid_ids)
    except FileNotFoundError:
        print("data/allbus_2023.dta not found. Please ensure it is present.")
        return None, None, shortlist_df

    print(f"Loaded dataframe with shape: {df.shape}")

    # Clean: map (-1 to -10) to NaN
    for col in df.select_dtypes(include=[np.number]).columns:
        df.loc[(df[col] >= -10) & (df[col] <= -1), col] = np.nan

    print("Cleaned negative values (mapped to NaN).")

    # Split: 30% exploratory, 70% confirmatory
    exploratory_set, confirmatory_set = train_test_split(df, test_size=0.7, random_state=42)
    print(f"Split data: Exploratory ({exploratory_set.shape[0]} rows), Confirmatory ({confirmatory_set.shape[0]} rows)")

    return exploratory_set, confirmatory_set, shortlist_df

def missingness_audit_and_impute(df):
    print("\n--- Missingness Audit & Imputation ---")
    missing_percentages = df.isnull().mean() * 100

    high_missing = missing_percentages[missing_percentages > 30]
    low_missing = missing_percentages[(missing_percentages <= 30) & (missing_percentages > 0)]

    if not high_missing.empty:
        print("Variables with > 30% missing data (Will be excluded from models):")
        for var, pct in high_missing.sort_values(ascending=False).items():
            print(f"  {var}: {pct:.1f}%")

    print(f"\nImputing {len(low_missing)} variables with < 30% missing data using IterativeImputer...")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cols_to_impute = [col for col in numeric_cols if col in low_missing.index or col not in high_missing.index]

    # We impute the columns that have <30% missingness (or 0 missingness for context)
    imputer = IterativeImputer(random_state=42, max_iter=10)
    imputed_data = imputer.fit_transform(df[cols_to_impute])

    # Create new df with imputed values for <30% missing columns, and keep >30% as is (or ignore them later)
    df_imputed = df.copy()
    df_imputed[cols_to_impute] = imputed_data

    return df_imputed, list(high_missing.index)

if __name__ == "__main__":
    pass

def construct_measurement(df, shortlist_df, high_missing_cols):
    print("\n--- Construct Measurement (Factor Analysis) ---")

    # 1. Dynamically identify Minority Attitude variables
    keywords = ['ZUZUG', 'AUSLAENDER', 'ASYLBEWERB', 'JUDEN', 'ISLAM', 'MIGRANTEN']
    minority_vars = []
    for _, row in shortlist_df.iterrows():
        desc = str(row['Description']).upper()
        if any(kw in desc for kw in keywords):
            minority_vars.append(row['Variable_ID'])

    if 'mg03' not in minority_vars and 'mg03' in shortlist_df['Variable_ID'].values:
        minority_vars.append('mg03')

    # Keep only variables that exist and aren't in high_missing_cols
    existing_vars = [var for var in minority_vars if var in df.columns and var not in high_missing_cols]

    print(f"Identified {len(existing_vars)} valid 'Minority Attitude' variables for EFA.")

    efa_df = df[existing_vars].copy()

    # Calculate Cronbach's Alpha
    # pingouin.cronbach_alpha requires a dataframe where rows are subjects and columns are items
    alpha, ci = pg.cronbach_alpha(data=efa_df)
    print(f"Cronbach's Alpha for the {len(existing_vars)} items: {alpha:.3f} (95% CI: {ci})")

    # Perform Exploratory Factor Analysis
    fa = FactorAnalyzer(n_factors=1, rotation=None)
    fa.fit(efa_df)

    # Get factor scores
    acceptance_score = fa.transform(efa_df)
    df['Acceptance_Score'] = acceptance_score

    print("Added 'Acceptance_Score' (primary latent factor) to the dataframe.")
    return df, existing_vars

def bivariate_associations(df, high_missing_cols):
    print("\n--- Methodologically Sound Bivariate Associations (Spearman) ---")

    if 'id03' not in df.columns or 'id03' in high_missing_cols:
        print("Variable 'id03' not found or has too many missing values.")
        return

    numeric_df = df.select_dtypes(include=[np.number]).drop(columns=high_missing_cols, errors='ignore')

    # Calculate Spearman rank-order correlations with id03
    correlations = numeric_df.corr(method='spearman')['id03'].drop('id03').dropna()

    correlations = correlations.sort_values(ascending=False)

    print("\nTop 15 Positively Correlated Variables with id03:")
    for var, corr in correlations.head(15).items():
        print(f"  {var}: {corr:.3f}")

    print("\nTop 15 Negatively Correlated Variables with id03:")
    for var, corr in correlations.tail(15).sort_values(ascending=True).items():
        print(f"  {var}: {corr:.3f}")

def surprise_ranking_algorithmic(df, shortlist_df, minority_vars, high_missing_cols):
    print("\n--- Algorithmic Feature Selection (Surprise Ranking) ---")

    if 'Acceptance_Score' not in df.columns:
        print("Acceptance_Score not found.")
        return

    numeric_df = df.select_dtypes(include=[np.number]).drop(columns=high_missing_cols, errors='ignore')

    # Exclude the minority variables used to construct the score, and the score itself
    exclude_vars = set(minority_vars + ['Acceptance_Score'])

    X = numeric_df.drop(columns=[col for col in exclude_vars if col in numeric_df.columns])
    y = df['Acceptance_Score']

    # Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    top_20 = importances.head(20)

    print("\nTop 20 Most Important Variables driving 'Acceptance Score':")
    for var, imp in top_20.items():
        desc = shortlist_df[shortlist_df['Variable_ID'] == var]['Description'].values
        desc_str = desc[0] if len(desc) > 0 else "Unknown"
        print(f"  {var}: {imp:.4f} - {desc_str}")

    # Plotting
    plt.figure(figsize=(10, 8))
    sns.barplot(x=top_20.values, y=top_20.index, palette='viridis', hue=top_20.index, legend=False)
    plt.title("Top 20 Variables Predicting 'Acceptance Score' (Random Forest)")
    plt.xlabel("Feature Importance")
    plt.ylabel("Variable")
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("Saved feature importance chart to 'feature_importance.png'.")

if __name__ == "__main__":
    exploratory_df, confirmatory_df, shortlist_df = load_and_split()

    if exploratory_df is not None:
        # 1. Impute missing data
        df_imputed, high_missing_cols = missingness_audit_and_impute(exploratory_df)

        # 2. Factor Analysis
        df_scored, minority_vars = construct_measurement(df_imputed, shortlist_df, high_missing_cols)

        # 3. Bivariate Associations
        bivariate_associations(df_scored, high_missing_cols)

        # 4. Algorithmic Feature Selection
        surprise_ranking_algorithmic(df_scored, shortlist_df, minority_vars, high_missing_cols)
