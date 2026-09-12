"""
SADMaNS — Run Windowing
==========================================

Executes Phase 4: 1-Minute Temporal Windowing.
Reads the cleaned dataset, groups into 1-minute windows,
and saves the window index.
"""

import os
import sys
import time

import pandas as pd
from config import PROJECT_ROOT
from preprocessing.windowing import process_windowing


def _sep(char: str = "=", width: int = 70) -> str:
    return char * width


def main() -> int:
    print(_sep())
    print("  SADMaNS — PHASE 4: 1-MINUTE TEMPORAL WINDOWING")
    print(_sep())
    print()

    data_dir = os.path.join(PROJECT_ROOT, "data")
    
    # We should use the cleaned parquet if available, otherwise csv
    parquet_path = os.path.join(data_dir, "02-15-2018_cleaned.parquet")
    csv_path = os.path.join(data_dir, "02-15-2018_cleaned.csv")
    
    if os.path.isfile(parquet_path):
        clean_path = parquet_path
        print(f"Loading cleaned dataset from Parquet: {clean_path}")
        df_clean = pd.read_parquet(clean_path)
    elif os.path.isfile(csv_path):
        clean_path = csv_path
        print(f"Loading cleaned dataset from CSV: {clean_path}")
        df_clean = pd.read_csv(clean_path, low_memory=False)
        # Ensure timestamp is parsed properly from CSV
        df_clean['Timestamp'] = pd.to_datetime(df_clean['Timestamp'])
    else:
        print(f"[ERROR] Cleaned dataset not found in {data_dir}. Run Phase 3 first.")
        return 1

    start_time = time.time()
    
    try:
        # Process Windowing
        df_windowed, metadata, report = process_windowing(df_clean)
        
    except Exception as e:
        print(f"\n[ERROR] Windowing pipeline failed: {e}")
        return 1
        
    process_time = time.time() - start_time
    print(f"Windowing completed in {process_time:.2f} seconds.")
    
    # Save window index to parquet
    out_path = os.path.join(data_dir, "window_index.parquet")
    print(f"Saving window index to: {out_path} ...")
    
    try:
        metadata.to_parquet(out_path, index=False)
        print("Successfully saved window index as Parquet.")
    except Exception as e:
        print(f"Failed to save Parquet: {e}")
        out_csv = os.path.join(data_dir, "window_index.csv")
        print(f"Falling back to CSV: {out_csv} ...")
        metadata.to_csv(out_csv, index=False)
    
    # Calculate some derived stats
    expected_full_minutes = int((metadata['window_start'].max() - metadata['window_start'].min()).total_seconds() / 60) + 1
    empty_windows = expected_full_minutes - report['populated_windows']
    
    print("\n" + _sep("-"))
    print("  WINDOWING REPORT")
    print(_sep("-"))
    
    print("\n[ FLOW ASSIGNMENT ]")
    print(f"Total Cleaned Flows:   {report['total_flows']:,}")
    print(f"Flows Assigned:        {report['assigned_flows']:,}")
    print(f"Match:                 {'OK' if report['flows_match'] else 'FAILED'}")
    print(f"Chronological Order:   {'OK' if report['is_chronological'] else 'FAILED'}")
    
    print("\n[ WINDOW METRICS ]")
    print(f"Populated Windows:     {report['populated_windows']}")
    print(f"Empty/Missing Windows: {empty_windows}")
    print(f"Possible 1-Min Span:   {expected_full_minutes}")
    
    print("\n[ SEGMENTS & GAPS ]")
    print(f"Contiguous Segments:   {report['segments_count']}")
    
    if not report['gaps']:
        print("No gaps found.")
    else:
        print("Major Temporal Gaps:")
        for gap in report['gaps']:
            print(f"  {gap['prev_window_start'].strftime('%H:%M:%S')} --> {gap['next_window_start'].strftime('%H:%M:%S')} "
                  f"(Duration: {gap['duration']}, Missing Mins: {gap['missing_minutes']})")
                  
    print("\n[ ATTACK VALIDATION ]")
    if 'attack_windows' in report:
        print(f"Attack-Containing Windows: {report['attack_windows']}")
        print(f"GoldenEye Windows:         {report['goldeneye_windows']}")
        print(f"Slowloris Windows:         {report['slowloris_windows']}")
    else:
        print("No label data found.")
        
    print("\n" + _sep())
    print("  ✓ PHASE 4 1-MINUTE TEMPORAL WINDOWING COMPLETE")
    print(_sep())
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
