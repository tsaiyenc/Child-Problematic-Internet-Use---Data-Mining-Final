import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

def plot_features(data_path):
    """
    Reads a data.csv file and plots specified features against 'step'.
    """
    try:
        df = pd.read_csv(data_path)
        
        # Features to plot (y-axis)
        features = ['enmo', 'anglez', 'light', 'time_of_day', 'non-wear_flag']
        # features = ['enmo', 'light', 'time_of_day', 'weekday', 'relative_date_PCIAT']
        
        # Check if features exist
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            print(f"Skipping {data_path}: Missing features {missing_features}")
            return

        # Calculate hour (0-24)
        df['hour'] = (df['time_of_day'] / 1e9) / 3600
        
        # Calculate real_time (days)
        # relative_date_PCIAT is in days (integer). time_of_day is in ns.
        # fractional day = time_of_day / 1e9 / 3600 / 24
        df['real_time'] = df['relative_date_PCIAT'] + (df['time_of_day'] / 1e9 / 3600 / 24)

        fig, axes = plt.subplots(len(features), 1, figsize=(10, 12), sharex=True)
        
        # 'real_time' is x-axis
        x = df['real_time']
        
        for i, feature in enumerate(features):
            axes[i].plot(x, df[feature], label=feature)
            axes[i].set_ylabel(feature)
            axes[i].legend(loc='upper right')
            axes[i].grid(True)
            
        axes[-1].set_xlabel('Real Time (days relative to PCIAT)')
        plt.tight_layout()
        
        # Save plot to the same directory
        output_path = os.path.join(os.path.dirname(data_path), 'features_plot.png')
        plt.savefig(output_path)
        plt.close(fig)
        # print(f"Saved plot to {output_path}")

    except Exception as e:
        print(f"Error processing {data_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Plot time series features.")
    parser.add_argument('--data_dir', type=str, default='data/series_train.parquet', help='Directory containing id folders')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of files to process')
    args = parser.parse_args()

    # Search pattern: data_dir/id=*/data.csv
    search_pattern = os.path.join(args.data_dir, 'id=*', 'data.csv')
    files = glob.glob(search_pattern)
    
    if not files:
        print(f"No files found matching {search_pattern}")
        return

    print(f"Found {len(files)} files.")
    
    if args.limit:
        files = files[:args.limit]
        print(f"Processing first {args.limit} files...")

    for file_path in tqdm(files):
        plot_features(file_path)

if __name__ == "__main__":
    main()
