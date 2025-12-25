import pandas as pd
import os

def main():
    # Define file paths
    base_dir = "/tmp3/tsaiyenc/114-1_class/DataMining/child-mind-institute-problematic-internet-use"
    input_file = os.path.join(base_dir, "outputs/regression_stacking_SoftBalancedWeight/submission_hgb.csv")
    
    # Check if file exists
    if not os.path.exists(input_file):
        print(f"Error: File not found at {input_file}")
        return

    print(f"Reading file: {input_file}")
    df = pd.read_csv(input_file)

    # Check for required column
    if 'sii_raw' not in df.columns:
        print("Error: 'sii_raw' column not found in the input file.")
        print(f"Available columns: {df.columns.tolist()}")
        return

    # Map sii_raw to 0, 1, 2, 3 using equal frequency binning (quantiles)
    # qcut will try to divide into equal sized buckets.
    # We use rank(method='first') to handle duplicate values gracefully if needed, 
    # but strictly speaking qcut handles basic quantiles. 
    # To ensure strictly equal counts even with ties, we can rank first.
    
    print("Mapping 'sii_raw' to 0, 1, 2, 3 using equal frequency...")
    
    # Using rank to ensure we can split evenly even if there are duplicate values at the boundaries
    # This is a robust way to force equal distribution
    df['sii_rank'] = df['sii_raw'].rank(method='first')
    df['sii_mapped'] = pd.qcut(df['sii_rank'], q=4, labels=[0, 1, 2, 3])
    
    # Drop the temporary rank column
    df = df.drop(columns=['sii_rank'])

    # Print results
    print("\nSample of results:")
    print(df[['id', 'sii_raw', 'sii_mapped']].head(10))
    
    print("\nValue Counts for 'sii_mapped' (should be roughly equal):")
    print(df['sii_mapped'].value_counts())

    # Optional: Save the output if needed, but user just asked to print.
    # output_file = input_file.replace(".csv", "_mapped.csv")
    # df.to_csv(output_file, index=False)
    # print(f"\nSaved mapped file to: {output_file}")

if __name__ == "__main__":
    main()
