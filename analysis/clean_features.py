import pandas as pd
import numpy as np
import argparse
import os

def clean_features(input_path, output_path, non_feature_cols=['id'], missing_threshold=0.3):
    """
    Cleans feature file:
    1. Check missing rates.
    2. Drop columns with missing rate > missing_threshold.
    3. Impute remaining missing values with mean.
    """
    print(f"Loading data from {input_path}...")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: File not found at {input_path}")
        return

    print(f"Initial shape: {df.shape}")
    
    # Identify feature columns
    feature_cols = [c for c in df.columns if c not in non_feature_cols]
    
    # 1. Check missing rates
    missing_counts = df[feature_cols].isnull().sum()
    missing_rates = missing_counts / len(df)
    
    print("\nMissing Value Rates:")
    print(missing_rates[missing_rates > 0].sort_values(ascending=False))
    
    # 2. Drop columns with excessive missing values
    cols_to_drop = missing_rates[missing_rates > missing_threshold].index.tolist()
    if cols_to_drop:
        print(f"\nDropping columns with > {missing_threshold*100}% missing values: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)
        # Update feature cols
        feature_cols = [c for c in feature_cols if c not in cols_to_drop]
    else:
        print(f"\nNo columns exceed the {missing_threshold*100}% missing threshold.")

    # 3. Impute remaining missing values
    # For numeric features, use mean.
    # Note: 'weekend_diff' might be missing if no weekend data. 'mean' is a reasonable default (neutral).
    print("\nImputing remaining missing values with column mean...")
    for col in feature_cols:
        if df[col].isnull().any():
            mean_val = df[col].mean()
            df[col] = df[col].fillna(mean_val) # Fixed DeprecationWarning in recent pandas by assigning back
            # or df.fillna({col: mean_val}, inplace=True)
            print(f"  Filled {col} (missing {missing_counts[col]}) with mean: {mean_val:.4f}")
            
    # Verification
    final_missing = df.isnull().sum().sum()
    print(f"\nFinal missing values count: {final_missing}")
    print(f"Final shape: {df.shape}")
    
    # Save
    df.to_csv(output_path, index=False)
    print(f"Saved cleaned data to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Clean advanced features CSV.")
    parser.add_argument('--input', type=str, default='advanced_features.csv', help='Input CSV file')
    parser.add_argument('--output', type=str, default='advanced_features_cleaned.csv', help='Output CSV file')
    parser.add_argument('--threshold', type=float, default=0.3, help='Threshold for dropping columns (0-1)')
    
    args = parser.parse_args()
    
    clean_features(args.input, args.output, missing_threshold=args.threshold)

if __name__ == "__main__":
    main()
