import pandas as pd
import os

def check_missing_values(file_path):
    print(f"Reading file: {file_path}")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return

    if 'id' in df.columns:
        df.set_index('id', inplace=True)
        print("Set 'id' as index.")
    
    # Column-wise missing values
    print("\n" + "="*50)
    print("MISSING VALUES PER COLUMN")
    print("="*50)
    col_missing = df.isnull().mean() * 100
    col_missing = col_missing.sort_values(ascending=False)
    
    # Display all columns with > 0 missing values
    missing_cols = col_missing[col_missing > 0]
    if missing_cols.empty:
         print("No missing values in any column.")
    else:
        print(f"{'Column':<35} | {'Missing %':<10}")
        print("-" * 50)
        for col, pct in missing_cols.items():
            print(f"{col:<35} | {pct:.2f}%")
            
    # Row-wise missing values
    print("\n" + "="*50)
    print("MISSING VALUES PER ROW (ID)")
    print("="*50)
    row_missing = df.isnull().mean(axis=1) * 100
    row_missing = row_missing.sort_values(ascending=False)
    
    print("Top 50 Samples with highest missing value percentage:")
    print(f"{'ID':<15} | {'Missing %':<10}")
    print("-" * 30)
    for idx, pct in row_missing.head(50).items():
        print(f"{idx:<15} | {pct:.2f}%")

    print("\nDistribution of Missing Values per Row:")
    print(row_missing.describe())

if __name__ == "__main__":
    # Assuming running from project root
    file_path = "data/train_noBlankSii.csv"
    if not os.path.exists(file_path):
        # Fallback for checking absolute path or relative to script if needed
        # But user specified data/train.csv relative to project root usually
        pass
    
    check_missing_values(file_path)
