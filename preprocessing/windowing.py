"""
SADMaNS — Temporal Windowing
==========================================

Phase 4: Converts cleaned flow-level records into 
chronological 1-minute temporal windows.
"""

import pandas as pd
import numpy as np
from typing import Any, Tuple

def create_time_windows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Floors the Timestamp to the nearest minute.
    Returns the dataframe with a new 'window_start' column.
    """
    df = df.copy()
    if 'Timestamp' not in df.columns:
        raise ValueError("Timestamp column missing.")
    
    # Ensure Timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['Timestamp']):
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        
    df['window_start'] = df['Timestamp'].dt.floor('min')
    return df

def build_window_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Groups by window_start and computes metadata.
    """
    # Group by window_start
    grouped = df.groupby('window_start')
    
    # Compute basic counts
    metadata = grouped.size().reset_index(name='flow_count')
    metadata['window_end'] = metadata['window_start'] + pd.Timedelta(minutes=1)
    
    # Sort chronologically (should already be, but enforcing)
    metadata = metadata.sort_values('window_start').reset_index(drop=True)
    metadata['window_id'] = metadata.index
    
    # Attack metadata
    if 'Label' in df.columns:
        def contains_attack(labels):
            return (labels != 'Benign').any()
            
        def dominant_label(labels):
            return labels.value_counts().idxmax()
            
        def goldeneye_count(labels):
            return (labels == 'DoS attacks-GoldenEye').any()
            
        def slowloris_count(labels):
            return (labels == 'DoS attacks-Slowloris').any()
        
        attack_info = grouped['Label'].agg([
            ('contains_attack', contains_attack),
            ('dominant_label', dominant_label),
            ('has_goldeneye', goldeneye_count),
            ('has_slowloris', slowloris_count)
        ]).reset_index()
        
        metadata = pd.merge(metadata, attack_info, on='window_start')
    
    # Port metadata
    if 'Dst Port' in df.columns:
        ports_info = grouped['Dst Port'].nunique().reset_index(name='unique_dst_ports')
        metadata = pd.merge(metadata, ports_info, on='window_start')
        
    return metadata

def identify_contiguous_segments(metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies contiguous segments of windows.
    A gap of > 1 minute breaks the segment.
    """
    metadata = metadata.sort_values('window_start').reset_index(drop=True)
    
    if len(metadata) == 0:
        return metadata
        
    # Calculate difference between current and previous window start
    diffs = metadata['window_start'].diff()
    
    # A new segment starts when difference is > 1 minute (or it's the first row)
    is_new_segment = (diffs > pd.Timedelta(minutes=1))
    is_new_segment.iloc[0] = True  # First row is always a new segment
    
    # Cumulative sum of boolean gives segment ID
    metadata['segment_id'] = is_new_segment.cumsum()
    
    return metadata

def detect_temporal_gaps(metadata: pd.DataFrame) -> list[dict]:
    """
    Detect gaps between populated windows > 1 minute.
    """
    gaps = []
    if len(metadata) < 2:
        return gaps
        
    starts = metadata['window_start'].values
    ends = metadata['window_end'].values
    
    for i in range(1, len(metadata)):
        prev_end = pd.Timestamp(ends[i-1])
        curr_start = pd.Timestamp(starts[i])
        
        if curr_start > prev_end:
            gap_dur = curr_start - prev_end
            missing_mins = int(gap_dur.total_seconds() / 60)
            if missing_mins > 0:
                gaps.append({
                    'prev_window_start': pd.Timestamp(starts[i-1]),
                    'next_window_start': curr_start,
                    'duration': gap_dur,
                    'missing_minutes': missing_mins
                })
    return gaps

def validate_window_coverage(df: pd.DataFrame, metadata: pd.DataFrame) -> dict:
    """
    Validates the assignment of flows to windows.
    """
    res = {}
    total_flows = len(df)
    assigned_flows = metadata['flow_count'].sum()
    
    res['flows_match'] = (total_flows == assigned_flows)
    res['total_flows'] = total_flows
    res['assigned_flows'] = assigned_flows
    
    # Check chronologically ordered
    is_ordered = metadata['window_start'].is_monotonic_increasing
    res['is_chronological'] = is_ordered
    
    return res

def process_windowing(cleaned_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Main entry point for phase 4.
    """
    # 1. Floor timestamps
    df_windowed = create_time_windows(cleaned_df)
    
    # 2. Build metadata for populated windows
    metadata = build_window_metadata(df_windowed)
    
    # 3. Add segment IDs
    metadata = identify_contiguous_segments(metadata)
    
    # 4. Detect gaps
    gaps = detect_temporal_gaps(metadata)
    
    # 5. Validate
    validation = validate_window_coverage(df_windowed, metadata)
    
    report = {
        'populated_windows': len(metadata),
        'total_flows': validation['total_flows'],
        'assigned_flows': validation['assigned_flows'],
        'flows_match': validation['flows_match'],
        'is_chronological': validation['is_chronological'],
        'gaps': gaps,
        'segments_count': metadata['segment_id'].nunique(),
    }
    
    if 'contains_attack' in metadata.columns:
        report['attack_windows'] = metadata['contains_attack'].sum()
        report['goldeneye_windows'] = metadata['has_goldeneye'].sum()
        report['slowloris_windows'] = metadata['has_slowloris'].sum()
        
    return df_windowed, metadata, report
