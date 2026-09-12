"""
SADMaNS — Dataset Cleaner
==========================================

Performs Phase 3 dataset cleaning:
1. Normalizes whitespace
2. Parses timestamps
3. Converts infinite values to NaN
4. Imputes missing rates (with 0.0)
5. Removes exact duplicates
6. Validates ranges for ports/protocols
7. Outputs cleaned dataset to Parquet
"""

import os
from typing import Any

import numpy as np
import pandas as pd

from preprocessing.data_loader import (
    load_dataset,
    normalize_column_names,
    parse_and_sort_timestamps,
    analyze_temporal_gaps,
)


def replace_infinite_values(df: pd.DataFrame) -> dict[str, int]:
    """
    Replace +inf and -inf with NaN in numeric columns.
    Returns a dictionary of columns and the number of infinite values replaced.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_counts = {}
    
    for col in numeric_cols:
        # Count inf
        is_inf = np.isinf(df[col])
        count = is_inf.sum()
        if count > 0:
            inf_counts[col] = count
            # Replace inf with NaN
            df.loc[is_inf, col] = np.nan
            
    return inf_counts


def handle_missing_values(df: pd.DataFrame) -> dict[str, int]:
    """
    Handle NaN values.
    
    Strategy: 
    For flow-derived rate features (Flow Byts/s, Flow Pkts/s) which become NaN 
    due to missing measurements or a 0 denominator (Flow Duration = 0), we impute 
    with 0.0. A flow with 0 duration represents instantaneous packet exchange where 
    a per-second rate is undefined, so a rate of 0.0 is a safe deterministic imputation 
    that does not leak future information.
    
    Returns a dictionary of columns and the number of missing values imputed.
    """
    missing = df.isna().sum()
    missing_counts = missing[missing > 0].to_dict()
    
    # We expect missing values primarily in 'Flow Byts/s' and 'Flow Pkts/s' (after inf->NaN)
    # Also 'Timestamp' could have missing values, but they are handled separately.
    for col in missing_counts.keys():
        if col != "Timestamp":
            df[col] = df[col].fillna(0.0)
            
    return missing_counts


def remove_duplicates(df: pd.DataFrame) -> int:
    """
    Remove exact duplicate rows across all columns.
    Returns the number of removed rows.
    """
    initial_count = len(df)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return initial_count - len(df)


def validate_destination_ports(df: pd.DataFrame) -> int:
    """
    Ensure Dst Port is numeric and within 0-65535.
    Returns the number of invalid ports found.
    """
    if "Dst Port" not in df.columns:
        return 0
        
    invalid_mask = (df["Dst Port"] < 0) | (df["Dst Port"] > 65535) | df["Dst Port"].isna()
    invalid_count = invalid_mask.sum()
    
    # For Phase 3, we just report them. The MVP assumes valid ports for the top-k selection.
    # If invalid, we could coerce to NaN, but 0 is a valid port. We will leave them for now 
    # if they exist, or drop them if they break the schema.
    return invalid_count


def validate_protocols(df: pd.DataFrame) -> dict[int, int]:
    """
    Verify the protocol values (expected 0, 6, 17).
    Returns a dict of the value counts.
    """
    if "Protocol" not in df.columns:
        return {}
    return df["Protocol"].value_counts().to_dict()


def normalize_labels(df: pd.DataFrame) -> dict[str, int]:
    """
    Strip whitespace from labels and return the value counts.
    """
    if "Label" in df.columns:
        df["Label"] = df["Label"].str.strip()
        return df["Label"].value_counts().to_dict()
    return {}


def coerce_numeric_columns(df: pd.DataFrame) -> int:
    """
    Ensure all traffic features are numeric, coercing errors to NaN.
    Returns the number of values that were coerced to NaN.
    """
    exclude_cols = ["Timestamp", "Label"]
    numeric_cols = [c for c in df.columns if c not in exclude_cols]
    
    coerced_total = 0
    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            before_nan = df[col].isna().sum()
            df[col] = pd.to_numeric(df[col], errors="coerce")
            after_nan = df[col].isna().sum()
            coerced_total += (after_nan - before_nan)
            
    return coerced_total


def clean_dataset(data_path: str) -> dict[str, Any]:
    """
    Main cleaning pipeline.
    Returns a dictionary report of the cleaning process.
    """
    report = {}
    
    # 1. Load dataset
    df = load_dataset(data_path)
    report["rows_before"] = len(df)
    
    # Pre-cleaning stats
    report["label_distribution_before"] = df["Label"].value_counts().to_dict() if "Label" in df.columns else {}
    report["missing_before"] = df.isna().sum()[df.isna().sum() > 0].to_dict()
    
    inf_before = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        cnt = np.isinf(df[col]).sum()
        if cnt > 0:
            inf_before[col] = cnt
    report["infinite_before"] = inf_before
    
    # 2. Normalize column names
    normalize_column_names(df)
    
    # 3. Parse and sort timestamps
    invalid_ts = parse_and_sort_timestamps(df)
    report["invalid_timestamps"] = invalid_ts
    
    # 4. Validate/coerce numeric columns
    coerced = coerce_numeric_columns(df)
    report["coerced_to_nan"] = coerced
    
    # 5. Convert inf -> NaN
    inf_replaced = replace_infinite_values(df)
    report["infinite_handled"] = inf_replaced
    
    # 6. Handle missing values (impute with 0.0)
    missing_imputed = handle_missing_values(df)
    report["missing_handled"] = missing_imputed
    
    # 7. Validate ports and protocols
    invalid_ports = validate_destination_ports(df)
    report["invalid_ports"] = invalid_ports
    
    protocols = validate_protocols(df)
    report["protocol_distribution"] = protocols
    
    # 8. Normalize labels
    labels_after = normalize_labels(df)
    report["label_distribution_after"] = labels_after
    
    # 9. Remove duplicates
    duplicates_removed = remove_duplicates(df)
    report["duplicates_removed"] = duplicates_removed
    
    # 10. Final sort and gap check
    df.sort_values("Timestamp", inplace=True, ignore_index=True)
    report["rows_after"] = len(df)
    
    ts = df["Timestamp"].dropna()
    report["timestamp_min"] = ts.min() if not ts.empty else None
    report["timestamp_max"] = ts.max() if not ts.empty else None
    report["major_gaps"] = analyze_temporal_gaps(df)
    
    # Post-cleaning stats
    report["missing_after"] = df.isna().sum()[df.isna().sum() > 0].to_dict()
    
    inf_after = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        cnt = np.isinf(df[col]).sum()
        if cnt > 0:
            inf_after[col] = cnt
    report["infinite_after"] = inf_after

    return df, report
