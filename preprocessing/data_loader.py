"""
SADMaNS — Dataset Loader
==========================================

Responsible for loading the raw CSV file safely, normalizing column names,
parsing timestamps, and providing initial dataset validation summaries.
"""

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

# Expected 80 columns in the CSE-CIC-IDS2018 dataset.
# Note: we use the normalized versions here (no leading/trailing spaces).
EXPECTED_COLUMNS = {
    "Dst Port", "Protocol", "Timestamp", "Flow Duration", "Tot Fwd Pkts",
    "Tot Bwd Pkts", "TotLen Fwd Pkts", "TotLen Bwd Pkts", "Fwd Pkt Len Max",
    "Fwd Pkt Len Min", "Fwd Pkt Len Mean", "Fwd Pkt Len Std", "Bwd Pkt Len Max",
    "Bwd Pkt Len Min", "Bwd Pkt Len Mean", "Bwd Pkt Len Std", "Flow Byts/s",
    "Flow Pkts/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Tot", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Tot", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "Fwd Header Len", "Bwd Header Len", "Fwd Pkts/s", "Bwd Pkts/s", "Pkt Len Min",
    "Pkt Len Max", "Pkt Len Mean", "Pkt Len Std", "Pkt Len Var", "FIN Flag Cnt",
    "SYN Flag Cnt", "RST Flag Cnt", "PSH Flag Cnt", "ACK Flag Cnt", "URG Flag Cnt",
    "CWE Flag Count", "ECE Flag Cnt", "Down/Up Ratio", "Pkt Size Avg",
    "Fwd Seg Size Avg", "Bwd Seg Size Avg", "Fwd Byts/b Avg", "Fwd Pkts/b Avg",
    "Fwd Blk Rate Avg", "Bwd Byts/b Avg", "Bwd Pkts/b Avg", "Bwd Blk Rate Avg",
    "Subflow Fwd Pkts", "Subflow Fwd Byts", "Subflow Bwd Pkts", "Subflow Bwd Byts",
    "Init Fwd Win Byts", "Init Bwd Win Byts", "Fwd Act Data Pkts", "Fwd Seg Size Min",
    "Active Mean", "Active Std", "Active Max", "Active Min", "Idle Mean",
    "Idle Std", "Idle Max", "Idle Min", "Label"
}


@dataclass
class DatasetSummary:
    file_path: str
    file_size_mb: float
    row_count: int
    column_count: int
    missing_counts: dict[str, int]
    infinite_counts: dict[str, int]
    duplicate_count: int
    labels: dict[str, int]
    protocols: dict[str, int]
    unique_dst_ports: int
    timestamp_min: pd.Timestamp
    timestamp_max: pd.Timestamp
    invalid_timestamps: int
    major_gaps: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timedelta]]


def normalize_column_names(df: pd.DataFrame) -> None:
    """Strip leading and trailing whitespace from column names in-place."""
    df.columns = df.columns.str.strip()


def validate_columns(df: pd.DataFrame) -> None:
    """Verify that all expected 80 columns are present."""
    actual_columns = set(df.columns)
    missing = EXPECTED_COLUMNS - actual_columns
    extra = actual_columns - EXPECTED_COLUMNS

    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    
    if len(df.columns) != 80:
        raise ValueError(f"Expected exactly 80 columns, got {len(df.columns)}")


def parse_and_sort_timestamps(df: pd.DataFrame) -> int:
    """
    Parse the Timestamp column (DD/MM/YYYY HH:MM:SS) in-place and sort the dataframe.
    Returns the number of invalid/NaT timestamps.
    """
    if "Timestamp" not in df.columns:
        raise KeyError("Timestamp column not found.")
    
    # Parse with dayfirst=True
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%d/%m/%Y %H:%M:%S", errors="coerce", dayfirst=True)
    
    invalid_count = df["Timestamp"].isna().sum()
    
    # Sort chronologically (ignoring NaT for the order, NaT goes to the end)
    df.sort_values("Timestamp", inplace=True, ignore_index=True)
    
    return invalid_count


def load_dataset(data_path: str) -> pd.DataFrame:
    """
    Load the dataset into memory.
    The 359MB CSV is loaded directly into a pandas DataFrame because
    modern development environments have sufficient RAM (e.g., >8GB)
    to handle it without chunking. This avoids the overhead of complex
    chunk management while keeping memory usage reasonable (~1.5GB RAM).
    """
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"Dataset not found at expected path: {data_path}")

    # Load everything into memory. Low memory is False to avoid mixed type warnings.
    df = pd.read_csv(data_path, low_memory=False)
    
    return df


def check_missing_values(df: pd.DataFrame) -> dict[str, int]:
    """Return a dictionary of columns with missing value counts."""
    missing = df.isna().sum()
    missing = missing[missing > 0]
    return missing.to_dict()


def check_infinite_values(df: pd.DataFrame) -> dict[str, int]:
    """Return a dictionary of columns with infinite value counts."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_counts = {}
    for col in numeric_cols:
        count = np.isinf(df[col]).sum()
        if count > 0:
            inf_counts[col] = count
    return inf_counts


def analyze_temporal_gaps(df: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timedelta]]:
    """
    Identify significant temporal gaps (> 10 minutes) between consecutive rows.
    Assumes df is already sorted by Timestamp.
    """
    ts = df["Timestamp"].dropna()
    if len(ts) < 2:
        return []

    diffs = ts.diff()
    # Find gaps larger than 10 minutes
    large_gaps_idx = diffs[diffs > pd.Timedelta(minutes=10)].index
    
    gaps = []
    for idx in large_gaps_idx:
        start_time = ts.loc[idx - 1]
        end_time = ts.loc[idx]
        duration = diffs.loc[idx]
        gaps.append((start_time, end_time, duration))
        
    return gaps


def get_dataset_summary(df: pd.DataFrame, file_path: str) -> DatasetSummary:
    """Generate a comprehensive summary of the loaded dataset."""
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    
    missing_counts = check_missing_values(df)
    infinite_counts = check_infinite_values(df)
    duplicate_count = df.duplicated().sum()
    
    labels = df["Label"].value_counts().to_dict() if "Label" in df.columns else {}
    protocols = df["Protocol"].value_counts().to_dict() if "Protocol" in df.columns else {}
    unique_ports = df["Dst Port"].nunique() if "Dst Port" in df.columns else 0
    
    ts = df["Timestamp"].dropna()
    timestamp_min = ts.min() if not ts.empty else pd.NaT
    timestamp_max = ts.max() if not ts.empty else pd.NaT
    
    invalid_timestamps = df["Timestamp"].isna().sum() if "Timestamp" in df.columns else 0
    
    major_gaps = analyze_temporal_gaps(df)
    
    return DatasetSummary(
        file_path=file_path,
        file_size_mb=file_size_mb,
        row_count=len(df),
        column_count=len(df.columns),
        missing_counts=missing_counts,
        infinite_counts=infinite_counts,
        duplicate_count=duplicate_count,
        labels=labels,
        protocols=protocols,
        unique_dst_ports=unique_ports,
        timestamp_min=timestamp_min,
        timestamp_max=timestamp_max,
        invalid_timestamps=invalid_timestamps,
        major_gaps=major_gaps
    )
