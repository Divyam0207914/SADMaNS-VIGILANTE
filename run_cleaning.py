"""
SADMaNS — Run Cleaning
==========================================

Executes the Phase 3 dataset cleaning pipeline.
Outputs a detailed report and saves the cleaned dataset.
"""

import os
import sys
import time

from config import DATA_PATH, PROJECT_ROOT
from preprocessing.data_cleaner import clean_dataset


def _sep(char: str = "=", width: int = 70) -> str:
    return char * width


def main() -> int:
    print(_sep())
    print("  SADMaNS — PHASE 3 DATASET CLEANING")
    print(_sep())
    print()

    print(f"Starting cleaning pipeline for: {DATA_PATH}")
    start_time = time.time()
    
    try:
        df_cleaned, report = clean_dataset(DATA_PATH)
    except Exception as e:
        print(f"\n[ERROR] Cleaning pipeline failed: {e}")
        return 1
        
    process_time = time.time() - start_time
    print(f"Cleaning completed in {process_time:.2f} seconds.")
    
    # Define output path
    base_name = os.path.basename(DATA_PATH).replace(".csv", "")
    out_dir = os.path.join(PROJECT_ROOT, "data")
    
    try:
        # Prefer parquet for large datasets
        out_path = os.path.join(out_dir, f"{base_name}_cleaned.parquet")
        print(f"Saving cleaned dataset to: {out_path} ...")
        df_cleaned.to_parquet(out_path, index=False)
        print("Successfully saved as Parquet.")
    except Exception as e:
        print(f"Failed to save Parquet (missing dependency?): {e}")
        out_path = os.path.join(out_dir, f"{base_name}_cleaned.csv")
        print(f"Falling back to CSV: {out_path} ...")
        df_cleaned.to_csv(out_path, index=False)
        print("Successfully saved as CSV.")
        
    print("\n" + _sep("-"))
    print("  CLEANING REPORT")
    print(_sep("-"))
    
    print("\n[ ROWS & DUPLICATES ]")
    print(f"Rows before:          {report['rows_before']:,}")
    print(f"Duplicates removed:   {report['duplicates_removed']:,}")
    print(f"Rows after:           {report['rows_after']:,}")
    
    print("\n[ TIMESTAMP PRESERVATION ]")
    print(f"Invalid Timestamps:   {report['invalid_timestamps']}")
    print(f"Min Timestamp:        {report['timestamp_min']}")
    print(f"Max Timestamp:        {report['timestamp_max']}")
    if report['major_gaps']:
        print("Major Gaps Preserved:")
        for start, end, dur in report['major_gaps']:
            print(f"  {start} --> {end} ({dur})")
            
    print("\n[ NUMERIC COERCION ]")
    print(f"Values coerced to NaN: {report['coerced_to_nan']:,}")
    
    print("\n[ INFINITE VALUES ]")
    if not report['infinite_before']:
        print("None before cleaning.")
    else:
        print("Before cleaning:")
        for col, cnt in report['infinite_before'].items():
            print(f"  {col}: {cnt:,}")
    if not report['infinite_after']:
        print("After cleaning:        None (Converted to NaN)")
    else:
        print("After cleaning:        WARNING - Inf values remain!")
        
    print("\n[ MISSING VALUES ]")
    if not report['missing_before']:
        print("None before cleaning.")
    else:
        print("Before cleaning (raw):")
        for col, cnt in report['missing_before'].items():
            print(f"  {col}: {cnt:,}")
            
    print("Missing handled (NaN + coerced Inf -> imputed 0.0):")
    for col, cnt in report['missing_handled'].items():
        print(f"  {col}: {cnt:,}")
        
    if not report['missing_after']:
        print("After cleaning:        None")
    else:
        print("After cleaning:        WARNING - NaN values remain!")
        for col, cnt in report['missing_after'].items():
            print(f"  {col}: {cnt:,}")
            
    print("\n[ PROTOCOLS ]")
    for p, c in report['protocol_distribution'].items():
        print(f"  {p}: {c:,}")
        
    print("\n[ DESTINATION PORTS ]")
    print(f"Invalid Ports Found:  {report['invalid_ports']}")
    
    print("\n[ ATTACK DISTRIBUTION ]")
    print("Before cleaning:")
    for lbl, cnt in report['label_distribution_before'].items():
        print(f"  {lbl:<25} {cnt:>10,}")
    print("After cleaning:")
    for lbl, cnt in report['label_distribution_after'].items():
        print(f"  {lbl:<25} {cnt:>10,}")
        
    print("\n" + _sep())
    print("  ✓ PHASE 3 DATA CLEANING COMPLETE")
    print(_sep())
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
