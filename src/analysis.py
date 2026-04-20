import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt
import pyreadstat
import os

def load_and_clean():
    shortlist_df = pd.read_csv('data/exploration_shortlist_100.csv')
    valid_ids = shortlist_df['Variable_ID'].tolist()

    print(f"Loading {len(valid_ids)} variables from allbus_2023.dta...")

    try:
        df, meta = pyreadstat.read_dta('data/allbus_2023.dta', usecols=valid_ids)
    except FileNotFoundError:
        print("data/allbus_2023.dta not found. Please ensure it is present.")
        return None, shortlist_df

    print(f"Loaded dataframe with shape: {df.shape}")

    for col in df.select_dtypes(include=[np.number]).columns:
        df.loc[(df[col] >= -10) & (df[col] <= -1), col] = np.nan

    print("Cleaned negative values (mapped to NaN).")
    return df, shortlist_df

def missingness_audit(df):
    print("\n--- Missingness Audit ---")
    missing_percentages = df.isnull().mean() * 100

    high_missing = missing_percentages[missing_percentages > 30]
    if not high_missing.empty:
        print("Variables with > 30% missing data (Possible Splits):")
        for var, pct in high_missing.sort_values(ascending=False).items():
            print(f"  {var}: {pct:.1f}%")
    else:
        print("No variables have > 30% missing data.")
    return missing_percentages

def extract_minority_attitude_score(df, shortlist_df):
    print("\n--- Factor Discovery (PCA) ---")

    keywords = ['ZUZUG', 'AUSLAENDER', 'ASYLBEWERB', 'JUDEN', 'ISLAM', 'MIGRANTEN']

    minority_vars = []
    for _, row in shortlist_df.iterrows():
        desc = str(row['Description']).upper()
        if any(kw in desc for kw in keywords):
            minority_vars.append(row['Variable_ID'])

    if 'mg03' not in minority_vars and 'mg03' in shortlist_df['Variable_ID'].values:
        minority_vars.append('mg03')

    print(f"Identified {len(minority_vars)} 'Minority Attitude' variables:")
    print(", ".join(minority_vars))

    existing_vars = [var for var in minority_vars if var in df.columns]

    pca_df = df[existing_vars].copy()

    imputer = SimpleImputer(strategy='mean')
    pca_df_imputed = imputer.fit_transform(pca_df)

    pca = PCA(n_components=1)
    acceptance_score = pca.fit_transform(pca_df_imputed)

    df['Acceptance_Score'] = acceptance_score
    print(f"PCA Variance Explained by the single component: {pca.explained_variance_ratio_[0]*100:.2f}%")
    print("Added 'Acceptance_Score' to the dataframe.")
    return df

def correlation_mining(df):
    print("\n--- Correlation Mining ---")

    if 'id03' not in df.columns:
        print("Variable 'id03' not found in dataframe.")
        return

    numeric_df = df.select_dtypes(include=[np.number])
    correlations = numeric_df.corr()['id03'].drop('id03').dropna()
    correlations = correlations.sort_values(ascending=False)

    plt.figure(figsize=(12, 10))
    corr_df = pd.DataFrame(correlations).rename(columns={'id03': 'Correlation with id03'})
    top_and_bottom = pd.concat([corr_df.head(20), corr_df.tail(20)])

    sns.heatmap(top_and_bottom, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Top and Bottom 20 Correlations with id03 (Subjective Status)')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    print("Saved correlation heatmap to 'correlation_heatmap.png'.")

def surprise_ranking(df, shortlist_df):
    print("\n--- Surprise Ranking ---")

    if 'Acceptance_Score' not in df.columns:
        print("Acceptance_Score not found.")
        return

    # Dynamically identify income/status variables
    keywords = ['EINKOMMEN', 'STATUS', 'BERUF', 'PRESTIGE', 'OBEN-UNTEN']

    status_vars = []
    for _, row in shortlist_df.iterrows():
        desc = str(row['Description']).upper()
        if any(kw in desc for kw in keywords) or row['Variable_ID'].lower() == 'id03':
            status_vars.append(row['Variable_ID'])

    print(f"Identified {len(status_vars)} Income/Status related variables to exclude:")
    print(", ".join(status_vars))

    numeric_df = df.select_dtypes(include=[np.number])
    correlations = numeric_df.corr()['Acceptance_Score'].drop('Acceptance_Score').dropna()

    minority_keywords = ['ZUZUG', 'AUSLAENDER', 'ASYLBEWERB', 'JUDEN', 'ISLAM', 'MIGRANTEN']
    minority_vars = []
    for _, row in shortlist_df.iterrows():
        desc = str(row['Description']).upper()
        if any(kw in desc for kw in minority_keywords) or row['Variable_ID'] == 'mg03':
            minority_vars.append(row['Variable_ID'])

    exclude_vars = set(status_vars + minority_vars)

    valid_corrs = correlations[~correlations.index.isin(exclude_vars)]

    top_surprises = valid_corrs.abs().sort_values(ascending=False).head(10)

    print("\nTop 10 Surprising Correlates with 'Acceptance Score':")
    for var, abs_corr in top_surprises.items():
        original_corr = correlations[var]
        desc = shortlist_df[shortlist_df['Variable_ID'] == var]['Description'].values[0]
        print(f"  {var}: {original_corr:.3f} (Abs: {abs_corr:.3f}) - {desc}")

if __name__ == "__main__":
    df, shortlist_df = load_and_clean()
    if df is not None:
        missingness_audit(df)
        df = extract_minority_attitude_score(df, shortlist_df)
        correlation_mining(df)
        surprise_ranking(df, shortlist_df)
