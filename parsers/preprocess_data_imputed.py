import pandas as pd
import numpy as np
import os

def load_data_dictionary(dict_path):
    print(f"Loading data dictionary from {dict_path}...")
    # Read CSV, assume 'Field' is the column name and 'Type' is the data type
    try:
        dd = pd.read_csv(dict_path)
        # Create a mapping from Field to descriptors
        type_map = {}
        for _, row in dd.iterrows():
            field = row['Field']
            dtype = row['Type']
            type_map[field] = dtype
        return type_map
    except Exception as e:
        print(f"Warning: Could not load data dictionary: {e}")
        return {}

def preprocess_data(input_file, dictionary_file, output_file):
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    initial_rows = len(df)
    initial_cols = len(df.columns)
    print(f"Initial shape: {df.shape}")
    
    # Filter out rows where 'sii' is NaN
    print("Filtering rows where 'sii' is NaN...")
    df_cleaned = df.dropna(subset=['sii'])
    rows_after_filter = len(df_cleaned)
    print(f"Rows dropped: {initial_rows - rows_after_filter}")
    
    # Identify and drop columns ending with 'Season'
    season_cols = [col for col in df_cleaned.columns if col.endswith('Season')]
    df_cleaned = df_cleaned.drop(columns=season_cols)

    # Identify and drop columns containing 'PCIAT'
    pciat_cols = [col for col in df_cleaned.columns if 'PCIAT' in col]
    df_cleaned = df_cleaned.drop(columns=pciat_cols)

    # Remove useless feature columns
    useless_cols = [
        'BIA-BIA_BMC', 'BIA-BIA_BMR', 'BIA-BIA_DEE', 
        'BIA-BIA_ECW', 'BIA-BIA_Fat', 'BIA-BIA_FFM', 
        'BIA-BIA_ICW', 'BIA-BIA_LDM', 'BIA-BIA_LST', 
        'BIA-BIA_SMM', 'BIA-BIA_TBW', 'BIA-BIA_FFMI', 'BIA-BIA_FMI']
    # Only drop if they exist
    existing_useless = [col for col in useless_cols if col in df_cleaned.columns]
    df_cleaned = df_cleaned.drop(columns=existing_useless)
    
    # Load Data Dictionary for Imputation
    type_map = load_data_dictionary(dictionary_file)
    
    print("Imputing missing values...")
    imputed_count = 0
    
    for col in df_cleaned.columns:
        if col == 'sii' or col == 'id':
            continue
            
        is_missing = df_cleaned[col].isna()
        if not is_missing.any():
            continue
            
        imputed_count += 1
        dtype_desc = type_map.get(col, None)
        
        # Determine imputation strategy
        if dtype_desc:
            # Strategy based on dictionary
            if 'categorical' in str(dtype_desc).lower() or 'str' in str(dtype_desc).lower():
                # Categorical -> New Category
                # If numeric categorical (categorical int), use -1
                # If string categorical, use 'Missing'
                if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                     df_cleaned[col] = df_cleaned[col].fillna(-1)
                else:
                     df_cleaned[col] = df_cleaned[col].fillna('Missing')
            elif 'int' in str(dtype_desc).lower() or 'float' in str(dtype_desc).lower():
                # Numerical -> Mean
                mean_val = df_cleaned[col].mean()
                df_cleaned[col] = df_cleaned[col].fillna(mean_val)
            else:
                # Fallback based on pandas type
                if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                    df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mean())
                else:
                    df_cleaned[col] = df_cleaned[col].fillna('Missing')
        else:
             # Fallback if not in dictionary
            if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mean())
            else:
                df_cleaned[col] = df_cleaned[col].fillna('Missing')

    print(f"Imputed {imputed_count} columns.")
    
    cols_after_drop = len(df_cleaned.columns)
    print(f"Columns dropped: {initial_cols - cols_after_drop}")
    print(f"Final shape: {df_cleaned.shape}")
    
    # Save to output file
    print(f"Saving cleaned data to {output_file}...")
    df_cleaned.to_csv(output_file, index=False)
    print("Done.")

if __name__ == "__main__":
    current_dir = os.getcwd()
    # Check if we are in 'parsers' or root
    if os.path.basename(current_dir) == 'parsers':
        base_dir = os.path.dirname(current_dir)
    else:
        base_dir = current_dir
        
    input_path = os.path.join(base_dir, "data", "train.csv")
    dict_path = os.path.join(base_dir, "data", "data_dictionary.csv")
    output_path = os.path.join(base_dir, "data", "train_cleaned_imputed.csv")
    
    if os.path.exists(input_path) and os.path.exists(dict_path):
        preprocess_data(input_path, dict_path, output_path)
    else:
        print(f"Error: Input file or dictionary not found.")
        print(f"Input: {input_path} (Exists: {os.path.exists(input_path)})")
        print(f"Dict: {dict_path} (Exists: {os.path.exists(dict_path)})")
