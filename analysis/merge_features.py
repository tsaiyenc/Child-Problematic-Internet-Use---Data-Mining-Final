import pandas as pd
import argparse
import os

def merge_features(train_path, feature_path, output_path):
    print(f"Loading training data from {train_path}...")
    try:
        train_df = pd.read_csv(train_path)
    except FileNotFoundError:
        print(f"Error: File not found at {train_path}")
        return

    print(f"Loading feature data from {feature_path}...")
    try:
        feature_df = pd.read_csv(feature_path)
    except FileNotFoundError:
        print(f"Error: File not found at {feature_path}")
        return

    print(f"Train shape: {train_df.shape}")
    print(f"Feature shape: {feature_df.shape}")
    
    # Merge on 'id'
    if 'id' not in train_df.columns:
        print("Error: 'id' column not found in training data")
        return
    if 'id' not in feature_df.columns:
        print("Error: 'id' column not found in feature data")
        return

    # Check for duplicate IDs which might cause explosion
    if feature_df['id'].duplicated().any():
        print("Warning: Duplicate IDs found in feature file. Keeping first.")
        feature_df = feature_df.drop_duplicates(subset=['id'], keep='first')

    merged_df = pd.merge(train_df, feature_df, on='id', how='left')
    
    print(f"Merged shape: {merged_df.shape}")
    print(f"Missing values in new features:\n{merged_df[feature_df.columns].isnull().sum()}")

    # Fill NaNs in new features if any (IDs present in train but not in features)
    # We might want to fill with mean from the features or 0, depending on nature. 
    # For now, let's just report. The user asked to "merge", handling missingness from non-matching IDs is implicit.
    # Usually better to left join and see.
    
    merged_df.to_csv(output_path, index=False)
    print(f"Saved merged data to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Merge features into training data.")
    parser.add_argument('--train', type=str, default='data/train_cleaned_imputed.csv', help='Training data CSV')
    parser.add_argument('--features', type=str, default='data/series_feature/advanced_features_cleaned.csv', help='Features CSV')
    parser.add_argument('--output', type=str, default='data/train_with_features.csv', help='Output CSV file')
    
    args = parser.parse_args()
    
    merge_features(args.train, args.features, args.output)

if __name__ == "__main__":
    main()
