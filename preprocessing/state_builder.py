"""
SADMaNS — Network State Builder
==========================================

Phase 5: Constructs the 13-dimensional global state and the 
17-dimensional node (destination port) state for each 1-minute window.
"""

import pandas as pd
import numpy as np

from config import GLOBAL_STATE_FEATURES, NODE_FEATURES

def calculate_entropy(counts: pd.Series) -> float:
    """Calculates Shannon entropy given a series of counts."""
    if len(counts) <= 1:
        return 0.0
    p = counts / counts.sum()
    return -np.sum(p * np.log(p))


def build_global_states(df: pd.DataFrame, window_metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Builds the 13-dimensional global state for each populated window.
    Only uses traffic occurring inside that window.
    """
    # Merge window_id onto flow df
    df = df.copy()
    
    # Calculate unique ports per window (or get from grouping)
    grouped = df.groupby('window_start')
    
    records = []
    
    # We iterate over the window metadata to ensure we only build states for populated windows
    for _, row in window_metadata.iterrows():
        w_start = row['window_start']
        w_id = row['window_id']
        seg_id = row['segment_id']
        
        # Get flows for this window
        w_flows = df[df['window_start'] == w_start]
        flow_count = len(w_flows)
        
        if flow_count == 0:
            continue
            
        # Calculate features exactly matching GLOBAL_STATE_FEATURES
        total_fwd_packets = w_flows['Tot Fwd Pkts'].sum()
        total_bwd_packets = w_flows['Tot Bwd Pkts'].sum()
        total_bytes = w_flows['TotLen Fwd Pkts'].sum() + w_flows['TotLen Bwd Pkts'].sum()
        
        mean_flow_duration = w_flows['Flow Duration'].mean()
        mean_pkt_len = w_flows['Pkt Len Mean'].mean()
        mean_flow_iat = w_flows['Flow IAT Mean'].mean()
        
        syn_rate = w_flows['SYN Flag Cnt'].sum() / flow_count
        rst_rate = w_flows['RST Flag Cnt'].sum() / flow_count
        psh_rate = w_flows['PSH Flag Cnt'].sum() / flow_count
        
        unique_dst_ports = w_flows['Dst Port'].nunique()
        
        port_counts = w_flows['Dst Port'].value_counts()
        port_entropy = calculate_entropy(port_counts)
        
        protocol_ratio_tcp = (w_flows['Protocol'] == 6).sum() / flow_count
        
        # Ensure exact order as GLOBAL_STATE_FEATURES
        record = {
            'window_id': w_id,
            'window_start': w_start,
            'segment_id': seg_id,
            'flow_count': flow_count,
            'total_fwd_packets': total_fwd_packets,
            'total_bwd_packets': total_bwd_packets,
            'total_bytes': total_bytes,
            'mean_flow_duration': mean_flow_duration,
            'mean_pkt_len': mean_pkt_len,
            'mean_flow_iat': mean_flow_iat,
            'syn_rate': syn_rate,
            'rst_rate': rst_rate,
            'psh_rate': psh_rate,
            'unique_dst_ports': unique_dst_ports,
            'port_entropy': port_entropy,
            'protocol_ratio_tcp': protocol_ratio_tcp
        }
        records.append(record)
        
    global_states_df = pd.DataFrame(records)
    
    # Fill any NaNs resulting from mean of empty groups (though flow_count > 0 so mostly safe)
    # But just in case:
    global_states_df.fillna(0.0, inplace=True)
    
    return global_states_df


def build_node_states(df: pd.DataFrame, window_metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Builds the 17-dimensional node state for EVERY destination port in each populated window.
    """
    records = []
    
    for _, row in window_metadata.iterrows():
        w_start = row['window_start']
        w_id = row['window_id']
        seg_id = row['segment_id']
        
        w_flows = df[df['window_start'] == w_start]
        
        if len(w_flows) == 0:
            continue
            
        # Group by Dst Port within this window
        port_groups = w_flows.groupby('Dst Port')
        
        for port, p_flows in port_groups:
            flow_count = len(p_flows)
            
            # Exact 17 features
            total_fwd_pkts = p_flows['Tot Fwd Pkts'].sum()
            total_bwd_pkts = p_flows['Tot Bwd Pkts'].sum()
            total_fwd_bytes = p_flows['TotLen Fwd Pkts'].sum()
            total_bwd_bytes = p_flows['TotLen Bwd Pkts'].sum()
            
            mean_flow_duration = p_flows['Flow Duration'].mean()
            mean_flow_pkts_per_s = p_flows['Flow Pkts/s'].mean()
            mean_flow_byts_per_s = p_flows['Flow Byts/s'].mean()
            mean_pkt_len = p_flows['Pkt Len Mean'].mean()
            mean_flow_iat = p_flows['Flow IAT Mean'].mean()
            
            syn_flag_count = p_flows['SYN Flag Cnt'].sum()
            ack_flag_count = p_flows['ACK Flag Cnt'].sum()
            rst_flag_count = p_flows['RST Flag Cnt'].sum()
            psh_flag_count = p_flows['PSH Flag Cnt'].sum()
            fin_flag_count = p_flows['FIN Flag Cnt'].sum()
            
            mean_fwd_pkts_per_s = p_flows['Fwd Pkts/s'].mean()
            mean_bwd_pkts_per_s = p_flows['Bwd Pkts/s'].mean()
            
            record = {
                'window_id': w_id,
                'window_start': w_start,
                'segment_id': seg_id,
                'dst_port': port,
                'flow_count': flow_count,
                'total_fwd_pkts': total_fwd_pkts,
                'total_bwd_pkts': total_bwd_pkts,
                'total_fwd_bytes': total_fwd_bytes,
                'total_bwd_bytes': total_bwd_bytes,
                'mean_flow_duration': mean_flow_duration,
                'mean_flow_pkts_per_s': mean_flow_pkts_per_s,
                'mean_flow_byts_per_s': mean_flow_byts_per_s,
                'mean_pkt_len': mean_pkt_len,
                'mean_flow_iat': mean_flow_iat,
                'syn_flag_count': syn_flag_count,
                'ack_flag_count': ack_flag_count,
                'rst_flag_count': rst_flag_count,
                'psh_flag_count': psh_flag_count,
                'fin_flag_count': fin_flag_count,
                'mean_fwd_pkts_per_s': mean_fwd_pkts_per_s,
                'mean_bwd_pkts_per_s': mean_bwd_pkts_per_s
            }
            records.append(record)
            
    node_states_df = pd.DataFrame(records)
    node_states_df.fillna(0.0, inplace=True)
    
    return node_states_df
