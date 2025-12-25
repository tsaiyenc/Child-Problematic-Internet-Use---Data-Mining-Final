import pandas as pd
import os

def preprocess_data(input_file, output_file):
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
    print(f"Found {len(season_cols)} columns ending with 'Season':")
    for col in season_cols:
        print(f" - {col}")
    df_cleaned = df_cleaned.drop(columns=season_cols)

    # Identify and drop columns containing 'PCIAT'
    pciat_cols = [col for col in df_cleaned.columns if 'PCIAT' in col]
    print(f"Found {len(pciat_cols)} columns containing 'PCIAT':")
    for col in pciat_cols:
        print(f" - {col}")
    df_cleaned = df_cleaned.drop(columns=pciat_cols)


    # Remove useless feature columns
    useless_cols = [
        'BIA-BIA_BMC', 'BIA-BIA_BMR', 'BIA-BIA_DEE', 
        'BIA-BIA_ECW', 'BIA-BIA_Fat', 'BIA-BIA_FFM', 
        'BIA-BIA_ICW', 'BIA-BIA_LDM', 'BIA-BIA_LST', 
        'BIA-BIA_SMM', 'BIA-BIA_TBW', 'BIA-BIA_FFMI', 'BIA-BIA_FMI']
    df_cleaned = df_cleaned.drop(columns=useless_cols)
    
    cols_after_drop = len(df_cleaned.columns)
    
    print(f"Columns dropped: {initial_cols - cols_after_drop}")
    print(f"Final shape: {df_cleaned.shape}")
    
    # Save to output file
    print(f"Saving cleaned data to {output_file}...")
    df_cleaned.to_csv(output_file, index=False)
    print("Done.")

if __name__ == "__main__":
    # Define paths relative to the project root (assuming script runs from root)
    # Adjusting for potential execution from within parsers directory or root
    
    current_dir = os.getcwd()
    # Check if we are in 'parsers' or root
    if os.path.basename(current_dir) == 'parsers':
        base_dir = os.path.dirname(current_dir)
    else:
        base_dir = current_dir
        
    input_path = os.path.join(base_dir, "data", "train.csv")
    # output_path = os.path.join(base_dir, "data", "train_cleaned.csv")
    # output_path = os.path.join(base_dir, "data", "train_noBlankSii.csv")
    output_path = os.path.join(base_dir, "data", "train_cleaned_2.csv")

    
    if os.path.exists(input_path):
        preprocess_data(input_path, output_path)
    else:
        print(f"Error: Input file not found at {input_path}")
