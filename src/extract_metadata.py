import pyreadstat
import pandas as pd
from pathlib import Path

# 1. Setup Paths
script_dir = Path(__file__).parent
project_root = script_dir.parent
data_file = project_root / "data" / "allbus_2023.dta"  # Ensure this matches your filename
output_file = project_root / "variable_dictionary.csv"


def extract_allbus_metadata(input_path, save_path):
    # 2. Read Metadata only
    _, meta = pyreadstat.read_dta(str(input_path), metadataonly=True)

    # 3. Process into a clean Dictionary
    metadata_df = pd.DataFrame(
        list(meta.column_names_to_labels.items()),
        columns=['Variable_ID', 'Description']
    )

    # 4. Save and return
    metadata_df.to_csv(save_path, index=False)
    return metadata_df


if __name__ == "__main__":
    try:
        df_labels = extract_allbus_metadata(data_file, output_file)
        print(f"Success: {len(df_labels)} variables exported to {output_file.name}")
    except Exception as e:
        print(f"Error: {e}")