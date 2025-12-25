import os
import pandas as pd
import glob
from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor, as_completed

def process_single_file(parquet_file):
    """
    Reads a single parquet file and saves it as CSV.
    Returns the file path if successful, or the error if failed.
    """
    try:
        # Read parquet file
        df = pd.read_parquet(parquet_file)
        
        # Construct output CSV path (in the same directory)
        output_dir = os.path.dirname(parquet_file)
        output_csv = os.path.join(output_dir, "data.csv")
        
        # Save as CSV
        df.to_csv(output_csv, index=False)
        return None
    except Exception as e:
        return f"Error processing {parquet_file}: {e}"

def process_parquet_files(base_dirs, max_workers=4):
    """
    Traverses directories, reads parquet files, and saves them as CSV using parallel processing.
    """
    all_parquet_files = []
    for base_dir in base_dirs:
        print(f"Scanning directory: {base_dir}")
        search_pattern = os.path.join(base_dir, "id=*", "part-0.parquet")
        files = glob.glob(search_pattern)
        all_parquet_files.extend(files)
        print(f"Found {len(files)} parquet files in {base_dir}")

    total_files = len(all_parquet_files)
    print(f"Total files to process: {total_files}")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Map files to futures
        futures = {executor.submit(process_single_file, f): f for f in all_parquet_files}
        
        # Progress bar
        with tqdm(total=total_files) as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result:
                    print(result)
                pbar.update(1)

if __name__ == "__main__":
    # Base directories to search
    base_directories = ["data/series_train.parquet", "data/series_test.parquet"]
    
    # Ensure paths are absolute or correct relative to execution point
    current_dir = os.getcwd()
    abs_base_directories = [os.path.join(current_dir, d) for d in base_directories]
    
    # Use number of CPUs for workers, or default to 4 if undeterminable
    workers = os.cpu_count() or 4
    print(f"Using {workers} workers for processing.")
    
    process_parquet_files(abs_base_directories, max_workers=workers)
    print("Processing complete.")
