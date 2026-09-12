"""
SADMaNS — Dataset Validation Script
==========================================

Runs the Phase 2 dataset loading and validation pipeline.
Loads the 02-15-2018.csv dataset, normalizes columns, parses timestamps,
and prints a comprehensive validation report.

Usage:
    python validate_dataset.py
"""

import sys
import time

from config import DATA_PATH
from preprocessing.data_loader import (
    load_dataset,
    normalize_column_names,
    validate_columns,
    parse_and_sort_timestamps,
    get_dataset_summary
)


def _sep(char: str = "=", width: int = 70) -> str:
    return char * width


def main() -> int:
    print(_sep())
    print("  SADMaNS — PHASE 2 DATASET VALIDATION")
    print(_sep())
    print()

    print(f"Loading dataset from: {DATA_PATH}")
    print("Please wait, loading 359 MB CSV into memory...")
    
    start_time = time.time()
    
    try:
        df = load_dataset(DATA_PATH)
    except Exception as e:
        print(f"\n[ERROR] Failed to load dataset: {e}")
        return 1

    load_time = time.time() - start_time
    print(f"Dataset loaded in {load_time:.2f} seconds.")
    
    # 1. Normalize columns
    print("Normalizing column names...")
    normalize_column_names(df)
    
    # 2. Validate columns
    print("Validating schema...")
    try:
        validate_columns(df)
        print("Schema validation: OK (80 columns found)")
    except Exception as e:
        print(f"\n[ERROR] Schema validation failed: {e}")
        return 1
        
    # 3. Parse and sort timestamps
    print("Parsing and sorting timestamps...")
    try:
        invalid_ts_count = parse_and_sort_timestamps(df)
        print(f"Timestamp parsing: OK ({invalid_ts_count} invalid timestamps)")
    except Exception as e:
        print(f"\n[ERROR] Timestamp parsing failed: {e}")
        return 1
        
    # 4. Generate summary
    print("Analyzing dataset properties...")
    summary = get_dataset_summary(df, DATA_PATH)
    
    # 5. Print comprehensive report
    print("\n" + _sep("-"))
    print("  DATASET VALIDATION REPORT")
    print(_sep("-"))
    
    print(f"\nFile Size:        {summary.file_size_mb:.2f} MB")
    print(f"Total Rows:       {summary.row_count:,}")
    print(f"Total Columns:    {summary.column_count}")
    
    print("\n[ TIMESTAMPS ]")
    print(f"Min Timestamp:    {summary.timestamp_min}")
    print(f"Max Timestamp:    {summary.timestamp_max}")
    print(f"Invalid TS:       {summary.invalid_timestamps}")
    
    print("\n[ TEMPORAL GAPS ]")
    if not summary.major_gaps:
        print("No major gaps (> 10 mins) found.")
    else:
        for start, end, duration in summary.major_gaps:
            print(f"Gap: {start} --> {end} (Duration: {duration})")
            
    print("\n[ MISSING VALUES (NaN) ]")
    if not summary.missing_counts:
        print("None found.")
    else:
        for col, count in summary.missing_counts.items():
            print(f"{col}: {count:,} ({count/summary.row_count*100:.2f}%)")
            
    print("\n[ INFINITE VALUES (Inf) ]")
    if not summary.infinite_counts:
        print("None found.")
    else:
        for col, count in summary.infinite_counts.items():
            print(f"{col}: {count:,} ({count/summary.row_count*100:.2f}%)")
            
    print("\n[ DUPLICATES ]")
    print(f"Duplicate Rows:   {summary.duplicate_count:,} ({summary.duplicate_count/summary.row_count*100:.2f}%)")
    
    print("\n[ LABELS ]")
    for label, count in summary.labels.items():
        print(f"{label:<25} {count:>10,} ({count/summary.row_count*100:.2f}%)")
        
    print("\n[ PROTOCOLS ]")
    # Mapping known protocol numbers to names for the report
    proto_map = {0: "HOPOPT (0)", 6: "TCP (6)", 17: "UDP (17)"}
    for proto, count in summary.protocols.items():
        name = proto_map.get(proto, str(proto))
        print(f"{name:<15} {count:>10,} ({count/summary.row_count*100:.2f}%)")
        
    print("\n[ DESTINATION PORTS ]")
    print(f"Unique Dst Ports: {summary.unique_dst_ports:,}")
    
    # 6. Attack-specific validation
    print("\n[ ATTACK-SPECIFIC VALIDATION ]")
    attack_flows = df[df["Label"] != "Benign"]
    print(f"Total Attack Flows: {len(attack_flows):,}")
    
    if len(attack_flows) > 0:
        attack_ports = attack_flows["Dst Port"].value_counts()
        print("Attack Destination Ports:")
        for port, count in attack_ports.items():
            print(f"  Port {port}: {count:,} flows")
            
        attack_protos = attack_flows["Protocol"].value_counts()
        print("Attack Protocols:")
        for proto, count in attack_protos.items():
            name = proto_map.get(proto, str(proto))
            print(f"  Protocol {name}: {count:,} flows")
            
        attack_ts_min = attack_flows["Timestamp"].min()
        attack_ts_max = attack_flows["Timestamp"].max()
        print(f"Attack Time Range: {attack_ts_min} to {attack_ts_max}")
    
    print("\n" + _sep())
    print("  ✓ PHASE 2 VALIDATION COMPLETE")
    print(_sep())
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
