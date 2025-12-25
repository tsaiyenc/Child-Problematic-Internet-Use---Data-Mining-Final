import os
import pandas as pd
import glob
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def process_single_file(csv_file):
    """
    Reads a single csv file and counts missing values.
    Returns a dictionary of missing counts and total rows.
    """
    try:
        df = pd.read_csv(csv_file)
        missing_counts = df.isnull().sum()
        total_rows = len(df)
        return {"missing": missing_counts, "rows": total_rows, "file": csv_file}
    except Exception as e:
        return {"error": str(e), "file": csv_file}

def analyze_missing_values(base_dirs, max_workers=4):
    """
    Traverses directories, reads data.csv files, and aggregates missing value stats.
    """
    all_csv_files = []
    for base_dir in base_dirs:
        print(f"Scanning directory: {base_dir}")
        user_dirs = glob.glob(os.path.join(base_dir, "id=*"))
        for user_dir in user_dirs:
             csv_path = os.path.join(user_dir, "data.csv")
             if os.path.exists(csv_path):
                 all_csv_files.append(csv_path)
        print(f"Found {len(all_csv_files)} data.csv files so far in {base_dir} and previous.")

    total_files = len(all_csv_files)
    print(f"Total files to process: {total_files}")
    
    overall_missing = pd.Series(dtype=int)
    total_rows_processed = 0
    errors = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_file, f): f for f in all_csv_files}
        
        with tqdm(total=total_files) as pbar:
            for future in as_completed(futures):
                result = future.result()
                if "error" in result:
                    errors.append(f"{result['file']}: {result['error']}")
                else:
                    if overall_missing.empty:
                        overall_missing = result["missing"]
                    else:
                        overall_missing = overall_missing.add(result["missing"], fill_value=0)
                    total_rows_processed += result["rows"]
                pbar.update(1)

    print("\n" + "="*40)
    print(f"Analysis Complete.")
    print(f"Total Files Processed: {total_files - len(errors)}")
    print(f"Total Rows Processed: {total_rows_processed}")
    if errors:
        print(f"Errors encountered: {len(errors)}")
        # for err in errors[:5]: # Print first 5 errors if any
        #     print(err)
    
    print("\nMissing Values Summary:")
    print(f"{'Column':<30} | {'Missing Count':<15} | {'Percentage':<10}")
    print("-" * 65)
    
    if total_rows_processed > 0:
        overall_missing = overall_missing.sort_values(ascending=False)
        for col, count in overall_missing.items():
            percentage = (count / total_rows_processed) * 100
            print(f"{col:<30} | {int(count):<15} | {percentage:.2f}%")
    else:
        print("No data processed.")
    print("="*40 + "\n")

if __name__ == "__main__":
    # Base directories to search (relative to script execution or absolute)
    # Assuming running from project root
    base_directories = ["data/series_train.parquet", "data/series_test.parquet"]
    
    current_dir = os.getcwd()
    abs_base_directories = [os.path.join(current_dir, d) for d in base_directories]
    
    workers = os.cpu_count() or 4
    print(f"Using {workers} workers for processing.")
    
    analyze_missing_values(abs_base_directories, max_workers=workers)
