import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import squarify
import os

def create_presentation_plots():
    # Setup directories
    output_dir = 'results/plots/presentation/'
    os.makedirs(output_dir, exist_ok=True)

    # Load Data
    classes_df = pd.read_csv('data/raw/classes_boroughs_total.csv')
    professions_df = pd.read_csv('data/raw/professions_total.csv')

    # Colors (Booth-Palette)
    color_poverty = '#0a1d37' # Dark Blue/Black
    color_above = '#d69f4c' # Gold
    color_highlight = '#b53131' # Red

    # -------------------------------------------------------------
    # 1. Detailed Welfare Donut (All Classes A-H)
    # -------------------------------------------------------------
    class_totals = classes_df[['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']].sum()

    fig, ax = plt.subplots(figsize=(10, 10))
    labels = [f'Class {c}' for c in class_totals.index]

    # Sequential palette from dark (poverty) to light (wealth)
    cmap = plt.get_cmap('YlOrRd_r') # Red/Dark for poverty, Yellow/Light for wealth
    # Let's map 8 classes across the colormap
    colors = [cmap(i) for i in np.linspace(0.1, 0.9, 8)]

    wedges, texts, autotexts = ax.pie(class_totals, labels=labels, colors=colors, autopct='%1.1f%%',
                                      startangle=140, pctdistance=0.85,
                                      textprops={'fontsize': 12, 'weight': 'bold'},
                                      wedgeprops=dict(width=0.3, edgecolor='w'))

    for text in texts:
        text.set_fontsize(14)

    ax.set_title('Das Londoner Armuts-Rad (Klassen A-H)', fontsize=20, weight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'poverty_donut.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # -------------------------------------------------------------
    # 2. Die Job-Landschaft (Treemap Refined: Size = Heads, Color = Poverty Rate)
    # -------------------------------------------------------------
    job_df = professions_df.dropna(subset=['Description']).copy()
    job_df = job_df[job_df['Description'] != 'Total']

    job_df['Class'] = job_df['Class'].ffill()
    job_df['Heads of Famlies'] = pd.to_numeric(job_df['Heads of Famlies'], errors='coerce').fillna(0)

    for c in ['A', 'B', 'C', 'D', 'Total']:
        job_df[c] = pd.to_numeric(job_df[c], errors='coerce').fillna(0)

    job_df['poverty_rate'] = (job_df['A'] + job_df['B'] + job_df['C'] + job_df['D']) / job_df['Total']
    job_df['poverty_rate'] = job_df['poverty_rate'].fillna(0)

    job_df = job_df[job_df['Heads of Famlies'] > 0]
    job_data = job_df.sort_values(by='Heads of Famlies', ascending=False)

    fig, ax = plt.subplots(figsize=(16, 10))
    cmap = plt.get_cmap('RdYlBu_r') # Red for high poverty, Blue for low poverty
    mini = 0.0
    maxi = 1.0
    norm = plt.Normalize(vmin=mini, vmax=maxi)
    colors = [cmap(norm(value)) for value in job_data['poverty_rate']]

    squarify.plot(sizes=job_data['Heads of Famlies'],
                  label=job_data['Description'],
                  alpha=0.8,
                  color=colors,
                  text_kwargs={'fontsize':9, 'weight':'bold', 'wrap':True},
                  ax=ax)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.03, pad=0.04)
    cbar.set_label('Armutsrisiko (Poverty Rate)', fontsize=14, weight='bold')

    plt.title('Die Job-Landschaft nach Armutsrisiko', fontsize=24, weight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'job_landscape_treemap.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # -------------------------------------------------------------
    # 3. Children by Occupation (Bar Chart)
    # -------------------------------------------------------------
    prof_df_clean = professions_df.copy()
    prof_df_clean['Class'] = prof_df_clean['Class'].ffill()

    # Filter out empty or total rows
    prof_df_clean = prof_df_clean.dropna(subset=['Description'])
    prof_df_clean = prof_df_clean[prof_df_clean['Description'] != 'Total']

    # Calculate kids per household by main class
    prof_df_clean['Children -15'] = pd.to_numeric(prof_df_clean['Children -15'], errors='coerce').fillna(0)
    prof_df_clean['Heads of Famlies'] = pd.to_numeric(prof_df_clean['Heads of Famlies'], errors='coerce').fillna(0)

    grouped = prof_df_clean.groupby('Class').agg({
        'Children -15': 'sum',
        'Heads of Famlies': 'sum'
    }).reset_index()

    grouped['kids_per_household'] = grouped['Children -15'] / grouped['Heads of Famlies']
    grouped = grouped[grouped['Heads of Famlies'] > 0]
    grouped = grouped.sort_values(by='kids_per_household', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(grouped['Class'], grouped['kids_per_household'], color=color_above)

    for bar in bars:
        width = bar.get_width()
        ax.annotate(f'{width:.2f} Kinder',
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0),
                    textcoords="offset points",
                    ha='left', va='center', fontsize=12, weight='bold')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_xlabel('Durchschnittliche Kinder pro Haushalt', fontsize=14)

    plt.title('Kinder pro Haushalt nach Berufsgruppe', fontsize=20, weight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'living_environments_bar.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # -------------------------------------------------------------
    # 4. Die 'unsichtbaren' Frauen (Horizontal Bar Chart)
    # -------------------------------------------------------------
    # IDs in the raw file are in the 'Unnamed: 1' column which has index 1.
    id_col = professions_df.columns[1]

    prof_df_clean[id_col] = pd.to_numeric(prof_df_clean[id_col], errors='coerce')
    female_ids = [33, 34, 35, 36, 37, 38]
    female_df = prof_df_clean[prof_df_clean[id_col].isin(female_ids)].copy()

    female_df = female_df.sort_values(by='Heads of Famlies', ascending=True)

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.barh(female_df['Description'], female_df['Heads of Famlies'], color=color_highlight)

    for bar in bars:
        width = bar.get_width()
        ax.annotate(f'{int(width):,}',
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0),
                    textcoords="offset points",
                    ha='left', va='center', fontsize=12)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_xlabel('Anzahl der weiblichen Haushaltsvorstände', fontsize=14)

    plt.title('Die "unsichtbaren" Frauen (Alleinverdiener)', fontsize=20, weight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'invisible_women_bar.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # -------------------------------------------------------------
    # 5. London Poverty Comparison (Horizontal Bar Chart for all districts)
    # -------------------------------------------------------------
    districts_df = classes_df.copy()

    # Calculate poverty rate for each district
    for c in ['A', 'B', 'C', 'D', 'Total']:
        districts_df[c] = pd.to_numeric(districts_df[c], errors='coerce').fillna(0)

    districts_df['poverty_rate'] = (districts_df['A'] + districts_df['B'] + districts_df['C'] + districts_df['D']) / districts_df['Total'] * 100

    # Calculate overall London average
    total_pov = districts_df[['A', 'B', 'C', 'D']].sum().sum()
    total_pop = districts_df['Total'].sum()
    london_avg = (total_pov / total_pop) * 100 if total_pop > 0 else 0

    districts_df = districts_df.sort_values('poverty_rate', ascending=True)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Color bars based on whether they are above or below average
    bar_colors = [color_poverty if rate > london_avg else color_above for rate in districts_df['poverty_rate']]

    bars = ax.barh(districts_df['Borough'], districts_df['poverty_rate'], color=bar_colors, height=0.7)

    # Add vertical line for London average
    ax.axvline(london_avg, color=color_highlight, linestyle='--', linewidth=2, zorder=0)
    ax.text(london_avg + 0.5, len(districts_df) - 1, f'London Average: {london_avg:.1f}%',
            color=color_highlight, fontsize=12, weight='bold', va='center')

    for bar in bars:
        width = bar.get_width()
        ax.annotate(f'{width:.1f}%',
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0),
                    textcoords="offset points",
                    ha='left', va='center', fontsize=12)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_xlabel('Armutsquote (%)', fontsize=14)

    plt.title('Armutsquote pro Distrikt im Vergleich', fontsize=20, weight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'city_contrasts_gauge.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print("Presentation plots created successfully.")

if __name__ == "__main__":
    create_presentation_plots()
