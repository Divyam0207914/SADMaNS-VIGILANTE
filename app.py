import os
import time
import numpy as np
import pandas as pd
import torch
import joblib
import streamlit as st
import plotly.graph_objects as go

from config import PROJECT_ROOT, CHECKPOINT_DIR, SCALER_PATH, GLOBAL_STATE_FEATURES, DEVICE
from models.early_warning import EarlyWarningEngine
from models.explanation_engine import ExplanationEngine

st.set_page_config(page_title="SADMaNS / VIGILANTE", layout="wide")

# ------------------------------------------------
# HELPERS
# ------------------------------------------------
FEATURE_MAPPING = {
    "flow_count": ("Network flows", "flows"),
    "total_fwd_packets": ("Forward packets", "packets"),
    "total_bwd_packets": ("Backward packets", "packets"),
    "total_bytes": ("Total traffic volume", "bytes"),
    "mean_flow_duration": ("Average flow duration", "us"),
    "mean_pkt_len": ("Average packet size", "bytes"),
    "mean_flow_iat": ("Average time between flows", "us"),
    "syn_rate": ("SYN activity", "%"),
    "rst_rate": ("RST activity", "%"),
    "psh_rate": ("PSH activity", "%"),
    "unique_dst_ports": ("Destination-port diversity", ""),
    "port_entropy": ("Port distribution diversity", ""),
    "protocol_ratio_tcp": ("TCP traffic ratio", "%")
}

def format_feature_name(raw_name):
    return FEATURE_MAPPING.get(raw_name, (raw_name, ""))[0]

def format_feature_value(raw_name, value):
    if pd.isna(value) or np.isinf(value):
        return "Unknown"
    unit = FEATURE_MAPPING.get(raw_name, ("", ""))[1]
    if unit == "%":
        return f"{value * 100:.2f} {unit}"
    elif unit:
        return f"{value:.2f} {unit}"
    else:
        return f"{value:.2f}"

def format_change(raw_name, current_val, next_val):
    if pd.isna(current_val) or pd.isna(next_val):
        return "Unknown"
    
    if abs(current_val) < 1e-6:
        if abs(next_val) > 1e-6:
            return "New / from zero"
        return "0%"
        
    pct_change = ((next_val - current_val) / abs(current_val)) * 100
    direction = "↑" if pct_change > 0 else "↓" if pct_change < 0 else ""
    return f"{direction} {abs(pct_change):.1f}%"

def get_human_warning(warning_level):
    mapping = {
        "WARNING_ONSET": "Potential Attack Onset",
        "WARNING_CONTINUING": "Attack Risk Continuing",
        "WATCH_RECOVERY": "Possible Recovery",
        "NORMAL": "Normal"
    }
    return mapping.get(warning_level, "Unknown")

def get_plain_language_summary(warning_level):
    mapping = {
        "WARNING_ONSET": "The risk model detects an elevated attack signal in the next-minute forecast while the current window is not labelled as an attack.",
        "WARNING_CONTINUING": "The model detects an elevated attack signal while attack traffic is already present in the current window.",
        "WATCH_RECOVERY": "The model's forecast moves away from attack conditions while attack traffic is currently present.",
        "NORMAL": "The model does not currently see enough risk in the forecast to classify the next window as an attack."
    }
    return mapping.get(warning_level, "No attack-risk transition is currently predicted.")

def get_short_warning_explanation(warning_level):
    mapping = {
        "WARNING_ONSET": "The current window is not labelled as an attack, but the risk model classifies the next window as attack-risk.",
        "WARNING_CONTINUING": "Attack traffic is already present and the model continues to detect attack risk.",
        "WATCH_RECOVERY": "Attack traffic is currently present, but the next-state forecast moves toward normal behaviour.",
        "NORMAL": "No attack-risk transition is currently predicted."
    }
    return mapping.get(warning_level, "")

def get_stage_description(stage):
    if stage == "IMPACT":
        return "Impact refers to behaviour associated with disrupting or degrading the availability of systems or services."
    return "Normal network operation without disruption."

import json

@st.cache_data
def load_dashboard_cache():
    t0 = time.time()
    cache_path = os.path.join(PROJECT_ROOT, "data/dashboard_cache.json")
    if not os.path.exists(cache_path):
        return None, 0.0
    with open(cache_path, 'r') as f:
        cache = json.load(f)
    load_time = time.time() - t0
    return cache, load_time

def run_dashboard():
    # ------------------------------------------------
    # HEADER
    # ------------------------------------------------
    st.title("SADMaNS / VIGILANTE")
    st.header("Predictive Network Defence")
    st.write("Forecasting how network behaviour may change in the next minute.")
    
    st.info("**DEMO MODE** | Historical CIC-IDS2018 traffic replay | Forecast horizon: 1 minute\n\n*Note: Offline dataset demonstration — Not live network traffic.*")
    
    with st.spinner("Loading dashboard cache..."):
        cache_data, load_time = load_dashboard_cache()
        
    if cache_data is None:
        st.error("Dashboard cache not found. Run:\npython precompute_dashboard.py")
        return
        
    total_seqs = cache_data['total_sequences']
    
    # ------------------------------------------------
    # SIDEBAR
    # ------------------------------------------------
    st.sidebar.title("Explore Historical Network Window")
    seq_idx = st.sidebar.slider("", 0, total_seqs - 1, 0)
    
    result = cache_data["sequences"][str(seq_idx)]
    t_timestamp = result["timestamp"]
    
    st.sidebar.write(f"**Current window:** {t_timestamp}")
    st.sidebar.write("**Forecast horizon:** 1 minute")
    
    # ------------------------------------------------
    # INFERENCE (CACHED)
    # ------------------------------------------------
    is_currently_attack = result["is_currently_attack"]
    inf_time = result.get("inf_time", 0.0)
    exp_time = result.get("exp_time", 0.0)
    
    explanation = result

    risk_pct = explanation.get("risk_probability", 0.0) * 100
    raw_w_level = explanation.get("warning_level", "NORMAL")
    human_w_level = get_human_warning(raw_w_level)
    
    # ------------------------------------------------
    # 1. WHAT IS HAPPENING NOW?
    # ------------------------------------------------
    st.header("WHAT IS HAPPENING NOW?")
    
    col1, col2, col3 = st.columns(3)
    
    curr_state_str = "🔴 Attack observed" if is_currently_attack else "🟢 Normal"
    col1.metric("CURRENT NETWORK STATE", curr_state_str)
    
    col2.metric("NEXT-MINUTE RISK", f"{risk_pct:.1f}%")
    col2.caption("Estimated probability that the next network window will contain attack traffic.")
    
    signal_color = "🟢"
    if "WARNING" in raw_w_level:
        signal_color = "🔴"
    elif "WATCH" in raw_w_level:
        signal_color = "🟠"
    
    sig_text = "Attack Risk Detected" if "WARNING" in raw_w_level else "Elevated Risk" if "WATCH" in raw_w_level else "Normal"
    col3.metric("MODEL SIGNAL", f"{signal_color} {sig_text}")
    
    # ------------------------------------------------
    # CURRENT -> NEXT VISUAL
    # ------------------------------------------------
    st.write("---")
    st.markdown(f"**CURRENT NETWORK** {curr_state_str} ➡️ **GCN + GRU** ➡️ **NEXT 1 MINUTE** {signal_color} {sig_text}")
    st.caption("VIGILANTE uses the previous 5 consecutive network states to forecast the next 1-minute network state.")
    
    # ------------------------------------------------
    # 2. WHAT DOES VIGILANTE EXPECT NEXT?
    # ------------------------------------------------
    st.header("WHAT DOES VIGILANTE EXPECT NEXT?")
    col_f1, col_f2, col_f3 = st.columns(3)
    col_f1.metric("FORECASTED RISK", f"{risk_pct:.1f}%")
    col_f2.metric("PREDICTED TRAFFIC BEHAVIOUR", explanation.get('predicted_attack_type', 'Unknown'))
    col_f3.metric("FORECAST HORIZON", "Next 1 minute")
    
    if (risk_pct >= 50.0 and explanation.get('predicted_attack_type', 'Benign') == 'Benign') or (risk_pct < 50.0 and explanation.get('predicted_attack_type', 'Benign') != 'Benign'):
        st.info("These are separate model outputs: the risk head estimates attack probability, while the type head predicts the most likely traffic class.")
    
    st.write(get_plain_language_summary(raw_w_level))
    
    st.write(f"**{human_w_level}**: {get_short_warning_explanation(raw_w_level)}")
    
    # ------------------------------------------------
    # 3. WHY IS THE MODEL CONCERNED?
    # ------------------------------------------------
    st.header("WHY IS THE MODEL CONCERNED?")
    st.write("VIGILANTE compares the network state observed now with the state it predicts for the next minute.")
    
    top_features = explanation.get("top_changed_features", [])
    if top_features:
        for f in top_features[:3]:
            raw_feat = f.get('feature', 'Unknown')
            feat_name = format_feature_name(raw_feat)
            c_val = format_feature_value(raw_feat, f.get('current_value', 0))
            n_val = format_feature_value(raw_feat, f.get('forecasted_value', 0))
            c_str = format_change(raw_feat, f.get('current_value', 0), f.get('forecasted_value', 0))
            st.write(f"**{feat_name}**: {c_val} ➡️ {n_val} ({c_str})")
            
    st.caption("These are the network signals showing the largest predicted changes. They help explain what the model is forecasting, but they should not be interpreted as direct causes of an attack.")
    
    # ------------------------------------------------
    # 4. WHAT CHANGES IN THE NEXT MINUTE?
    # ------------------------------------------------
    st.header("WHAT CHANGES IN THE NEXT MINUTE?")
    
    all_features = explanation.get("all_feature_changes", [])
    if all_features:
        table_data = []
        for f in all_features:
            raw_feat = f.get('feature', 'Unknown')
            table_data.append({
                "Network Signal": format_feature_name(raw_feat),
                "Now": format_feature_value(raw_feat, f.get('current_value', 0)),
                "Next Minute": format_feature_value(raw_feat, f.get('forecasted_value', 0)),
                "Change": format_change(raw_feat, f.get('current_value', 0), f.get('forecasted_value', 0)),
                "_abs_change": abs(f.get('signed_scaled_change', 0))
            })
            
        df_display = pd.DataFrame(table_data).sort_values('_abs_change', ascending=False).drop(columns=['_abs_change'])
        st.dataframe(df_display, width='stretch')
        
        # Chart
        st.subheader("What changes in the next minute?")
        top_5 = pd.DataFrame(all_features).reindex(pd.DataFrame(all_features)['absolute_scaled_change'].sort_values(ascending=False).index).head(5)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[format_feature_name(f) for f in top_5['feature']],
            y=top_5['relative_change'].apply(lambda x: x*100 if pd.notnull(x) and x != 'Inf' else 0),
            name='Predicted change (%)',
            marker_color='indianred'
        ))
        fig.update_layout(
            xaxis_title="Network Signal",
            yaxis_title="Predicted change (%)",
            barmode='group',
        )
        st.plotly_chart(fig, width='stretch')
        
        with st.expander("Show all network signals"):
            all_df = pd.DataFrame(all_features)
            fig_all = go.Figure()
            fig_all.add_trace(go.Bar(
                x=[format_feature_name(f) for f in all_df['feature']],
                y=all_df['signed_scaled_change'],
                name='Scaled Forecasted Delta',
                marker_color='indianred'
            ))
            fig_all.update_layout(
                title="Forecasted Feature Deviation (Scaled Space - Technical)",
                xaxis_title="Network Signal",
                yaxis_title="Standardized Deviation",
                barmode='group',
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig_all, width='stretch')
    
    # ------------------------------------------------
    # 5. SECURITY CONTEXT
    # ------------------------------------------------
    st.header("SECURITY CONTEXT")
    
    st.subheader("MODEL ASSESSMENT")
    col_ma1, col_ma2 = st.columns(2)
    col_ma1.metric("Risk Signal", f"{risk_pct:.1f}%")
    col_ma2.metric("Predicted Behaviour", explanation.get('predicted_attack_type', 'Unknown'))
    
    st.subheader("ATTACK CONTEXT")
    stage_val = explanation.get('stage', 'NORMAL')
    st.write(f"**Stage:** {stage_val}")
    
    mitre_data = explanation.get("mitre", None)
    if mitre_data:
        st.write("**MITRE ATT&CK**")
        st.write(f"{mitre_data.get('tactic', 'Unknown')}")
        st.write(f"{mitre_data.get('technique', 'Unknown')}")
        
    st.write(get_stage_description(stage_val))
        
    st.caption("Security context — heuristic / semantic enrichment. Stage and MITRE context are derived from the model's risk output; they are not independently learned predictions.")
    
    # ------------------------------------------------
    # 6. NETWORK TIMELINE
    # ------------------------------------------------
    st.header("NETWORK TIMELINE")
    st.write("How the forecast changed across the historical traffic replay.")
    
    # Pre-calculate
    recent_risks = []
    recent_times = []
    actual_attacks = []
    
    max_history = min(seq_idx + 1, 50)
    start_history = seq_idx - max_history + 1
    
    for i in range(start_history, seq_idx + 1):
        hist_res = cache_data["sequences"][str(i)]
        recent_risks.append(hist_res.get('risk_probability', 0.0) * 100)
        recent_times.append(hist_res['timestamp'])
        actual_attacks.append(hist_res['is_currently_attack'])
        
    timeline_df = pd.DataFrame({
        "Time": recent_times,
        "Forecasted Risk %": recent_risks,
        "Actual Attack at t": actual_attacks
    })
    
    fig_timeline = go.Figure()
    
    colors = ['green' if r < 10.0 else 'orange' if r < 50.0 else 'red' for r in timeline_df['Forecasted Risk %']]
    
    fig_timeline.add_trace(go.Scatter(
        x=timeline_df['Time'], 
        y=timeline_df['Forecasted Risk %'],
        mode='lines+markers',
        name='Forecasted Risk',
        marker=dict(color=colors),
        line=dict(color='gray')
    ))
    
    attack_times = timeline_df[timeline_df['Actual Attack at t'] == True]
    if not attack_times.empty:
        fig_timeline.add_trace(go.Scatter(
            x=attack_times['Time'],
            y=attack_times['Forecasted Risk %'],
            mode='markers',
            marker=dict(color='red', size=12, symbol='x'),
            name='Actual Attack observed'
        ))
        
    # Vertical line for current window
    fig_timeline.add_vline(x=str(t_timestamp), line_width=2, line_dash="dash", line_color="blue", annotation_text="Current Window")
        
    fig_timeline.update_layout(
        title="Attack Risk Over Time",
        xaxis_title="Time",
        yaxis_title="Forecasted attack risk (%)",
        yaxis_range=[0, 100]
    )
    
    st.write(f"**CURRENT WINDOW:** {t_timestamp} | **PREVIOUS 5 MINUTES:** Used by GRU | **FORECAST:** Next 1 minute")
    
    st.plotly_chart(fig_timeline, width='stretch')
    st.write("Each point represents a one-minute network window. The model forecasts risk for the following minute. Attack markers show what was actually labelled as attack traffic in the historical dataset.")
    
    # ------------------------------------------------
    # 7. HOW DOES VIGILANTE MAKE THE FORECAST?
    # ------------------------------------------------
    st.header("HOW DOES VIGILANTE MAKE THE FORECAST?")
    
    col_a, col_b, col_c = st.columns(3)
    col_a.info("NETWORK TRAFFIC\n↓\n1-MINUTE NETWORK STATE\n↓\nGCN\n*(What is related to what?)*")
    col_b.info("NETWORK REPRESENTATION\n↓\nGRU\n*(How is behaviour changing over time?)*")
    col_c.info("NEXT-MINUTE NETWORK STATE\n↓\nRISK + BEHAVIOUR + EXPLANATION")
    
    st.write("**GCN**: Looks at relationships between destination-port activity within a network window.")
    st.write("**GRU**: Looks at how network behaviour changes across consecutive windows.")
    st.write("**Forecast**: Combines these learned representations to predict the next network state.")
    
    # ------------------------------------------------
    # TECHNICAL DETAILS
    # ------------------------------------------------
    with st.expander("Technical Details"):
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.write("**User-facing Warning**")
            st.write(human_w_level)
            st.write("**Sequence Index**")
            st.write(str(seq_idx))
            st.write("**Node**")
            st.write("Destination port")
            st.write("**Node Features**")
            st.write("17")
            st.write("**Graph Embedding**")
            st.write("64 dimensions")
            st.write("**GRU Hidden Size**")
            st.write("128")
            st.write("**Forecast Horizon**")
            st.write("1 minute")
        with col_t2:
            st.write("**Technical Warning Value**")
            st.write(raw_w_level)
            st.write("**Model**")
            st.write("GCN + GRU")
            st.write("**Top-K Ports**")
            st.write("50")
            st.write("**Global Features**")
            st.write("13")
            st.write("**Combined State**")
            st.write("77 dimensions")
            st.write("**Sequence Length**")
            st.write("5 minutes")
            st.write("**Risk Threshold**")
            st.write("0.10 (actual checkpoint threshold)")
            
        st.write("---")
        st.write("**EDGE CONSTRUCTION**")
        st.write("Destination-port co-occurrence within a one-minute window")
        st.write("**EDGE WEIGHT**")
        st.write("min(flow_i, flow_j) / max(flow_i, flow_j)")
        
    with st.expander("Technical Diagnostics"):
        st.write(f"**Model Load Time:** {load_time:.2f}s")
        st.write(f"**Inference Time:** {inf_time*1000:.2f}ms")
        st.write(f"**Explanation Time:** {exp_time*1000:.2f}ms")

if __name__ == "__main__":
    run_dashboard()
