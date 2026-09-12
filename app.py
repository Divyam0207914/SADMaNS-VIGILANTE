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

st.set_page_config(page_title="SADMaNS / VIGILANTE SOC", layout="wide")

@st.cache_resource
def load_models_and_data():
    t0 = time.time()
    
    # Paths
    model_path = os.path.join(CHECKPOINT_DIR, "best_world_model.pt")
    scaler_path = os.path.join(CHECKPOINT_DIR, "state_scaler.pkl")
    seq_path = os.path.join(PROJECT_ROOT, "data/temporal_sequences.pt")
    idx_path = os.path.join(PROJECT_ROOT, "data/window_index.parquet")
    
    # Load Early Warning Engine
    engine = EarlyWarningEngine(checkpoint_path=model_path, scaler_path=scaler_path)
    
    # Load Explanation Engine
    explainer = ExplanationEngine(scaler=engine.scaler)
    
    # Load Data (for Demo Mode)
    seqs = torch.load(seq_path, weights_only=False, map_location='cpu')
    idx_df = pd.read_parquet(idx_path).set_index('window_id')
    
    load_time = time.time() - t0
    
    return engine, explainer, seqs, idx_df, load_time

def safe_float(val, fallback=0.0):
    if val is None or pd.isna(val) or np.isnan(val) or np.isinf(val):
        return fallback
    return float(val)

def run_dashboard():
    # ------------------------------------------------
    # HEADER
    # ------------------------------------------------
    st.title("SADMaNS / VIGILANTE")
    st.subheader("AI-based Network Attack Forecasting")
    st.caption("Offline dataset demonstration — Not live network traffic")
    
    with st.spinner("Loading models and dataset..."):
        engine, explainer, seqs, idx_df, load_time = load_models_and_data()
        
    st.sidebar.title("Dashboard Controls")
    st.sidebar.text(f"Model Load Time: {load_time:.2f}s")
    
    inputs = seqs['inputs']
    metadata = seqs['metadata']
    total_seqs = len(inputs)
    
    seq_idx = st.sidebar.slider("Select Sequence Index (Offline Demo)", 0, total_seqs - 1, 0)
    
    # Fetch Data for Selected Sequence
    seq_tensor = inputs[seq_idx]
    meta = metadata[seq_idx]
    
    t_timestamp = meta['input_end']
    
    # Find t in idx_df
    t_row = idx_df[idx_df['window_end'] == t_timestamp]
    if len(t_row) == 0:
        is_currently_attack = False
        t_w_id = -1
    else:
        is_currently_attack = bool(t_row.iloc[0]['contains_attack'])
        t_w_id = t_row.index[0]
        
    # Unscaled current state
    current_state_unscaled = seq_tensor[-1, 64:].numpy()
    
    # Scale current state for Early Warning Engine
    current_state_scaled = torch.tensor(engine.scaler.transform(current_state_unscaled.reshape(1, -1)), dtype=torch.float32)
    
    # ------------------------------------------------
    # INFERENCE
    # ------------------------------------------------
    t_inf = time.time()
    warning_data = engine.evaluate_sequence(
        sequence_tensor=seq_tensor,
        current_state_tensor=current_state_scaled,
        is_currently_attack=is_currently_attack,
        current_window_id=t_w_id
    )
    inf_time = time.time() - t_inf
    
    # ------------------------------------------------
    # EXPLANATION
    # ------------------------------------------------
    pred_state_scaled = np.array(warning_data['predicted_state_delta']) + current_state_scaled.numpy().flatten()
    
    explanation = explainer.generate_explanation(
        current_state_unscaled=current_state_unscaled,
        pred_state_scaled=pred_state_scaled,
        warning_data=warning_data
    )
    
    st.sidebar.text(f"Inference Time: {inf_time*1000:.2f}ms")
    st.sidebar.text(f"Explanation Time: {explanation.get('generation_time_sec', 0)*1000:.2f}ms")
    
    # ------------------------------------------------
    # CURRENT NETWORK STATUS
    # ------------------------------------------------
    st.header("CURRENT NETWORK STATUS")
    
    col1, col2, col3 = st.columns(3)
    
    curr_state_str = "Attack" if is_currently_attack else "Normal"
    curr_color = "red" if is_currently_attack else "green"
    col1.markdown(f"**Current State:** <span style='color:{curr_color}'>{curr_state_str}</span>", unsafe_allow_html=True)
    
    risk_pct = explanation.get("risk_probability", 0.0) * 100
    col2.markdown(f"**Forecasted Risk:** {risk_pct:.1f}%")
    
    w_level = explanation.get("warning_level", "NORMAL")
    w_color = "green"
    if "WARNING" in w_level:
        w_color = "red"
    elif "WATCH" in w_level:
        w_color = "orange"
        
    col3.markdown(f"**Warning:** <span style='color:{w_color}; font-weight:bold;'>{w_level}</span>", unsafe_allow_html=True)
    
    # ------------------------------------------------
    # ATTACK ASSESSMENT & MITRE
    # ------------------------------------------------
    st.header("ATTACK ASSESSMENT")
    
    assess_col1, assess_col2 = st.columns(2)
    
    with assess_col1:
        st.subheader("Model Predictions (Learned)")
        st.write(f"**Risk Probability:** {risk_pct:.2f}%")
        st.write(f"**Predicted Attack Type:** {explanation.get('predicted_attack_type', 'Unknown')}")
        st.write(f"**Warning Level:** {w_level}")
        
    with assess_col2:
        st.subheader("Semantic Context (Heuristic/Semantic)")
        st.write(f"**Stage (Heuristic):** {explanation.get('stage', 'UNKNOWN')}")
        
        mitre_data = explanation.get("mitre", None)
        if mitre_data:
            st.markdown("### MITRE ATT&CK Enrichment")
            st.write(f"**Tactic:** {mitre_data.get('tactic', 'Unknown')}")
            st.write(f"**Technique:** {mitre_data.get('technique', 'Unknown')}")
            st.caption(mitre_data.get('note', ''))
        else:
            st.write("No MITRE ATT&CK enrichment applicable.")
            
    # ------------------------------------------------
    # WHY THIS WARNING?
    # ------------------------------------------------
    st.header("WHY THIS WARNING?")
    st.write(explanation.get("summary", ""))
    
    top_features = explanation.get("top_changed_features", [])
    if top_features:
        st.markdown("#### Top Forecasted Feature Changes (Ranked by Scaled Magnitude):")
        for f in top_features:
            feat_name = f.get('feature', 'Unknown')
            val_delta = f.get('signed_change', 0.0)
            direction = "↑" if val_delta > 0 else "↓"
            st.write(f"{f.get('rank', '?')}. **{feat_name}** {direction} (Delta: {safe_float(val_delta):.4f})")
    
    st.caption("Note: Changes indicate variables with strongest deviation contributing to forecast explanation. They do not imply strict causality.")
    
    # ------------------------------------------------
    # NETWORK STATE COMPARISON
    # ------------------------------------------------
    st.header("NETWORK STATE: Current vs Forecast")
    
    all_features = explanation.get("all_feature_changes", [])
    
    if all_features:
        df_features = pd.DataFrame(all_features)
        
        # Format columns for display
        def format_rel(x):
            if isinstance(x, str):
                return x
            return f"{safe_float(x):.4f}"
            
        df_display = pd.DataFrame({
            "Feature": df_features["feature"],
            "Current $S_t$": df_features["current_value"].apply(lambda x: f"{safe_float(x):.4f}"),
            "Forecast $\hat{S}_{t+1}$": df_features["forecasted_value"].apply(lambda x: f"{safe_float(x):.4f}"),
            "Absolute Delta": df_features["absolute_change"].apply(lambda x: f"{safe_float(x):.4f}"),
            "Relative Change": df_features["relative_change"].apply(format_rel)
        })
        
        st.dataframe(df_display, use_container_width=True)
        
        # Visualization
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df_features['feature'],
            y=df_features['signed_scaled_change'],
            name='Scaled Forecasted Delta',
            marker_color='indianred'
        ))
        fig.update_layout(
            title="Forecasted Feature Deviation (Scaled Space)",
            xaxis_title="Network Feature",
            yaxis_title="Standardized Deviation",
            barmode='group',
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------
    # HISTORICAL NETWORK TREND & ATTACK TIMELINE
    # ------------------------------------------------
    st.header("HISTORICAL ATTACK TIMELINE")
    st.write("Visualizing risk probabilities over the selected historical sequences.")
    
    # Pre-calculate risk probabilities up to the current sequence
    recent_risks = []
    recent_times = []
    actual_attacks = []
    
    max_history = min(seq_idx + 1, 50) # show up to 50 recent windows
    start_history = seq_idx - max_history + 1
    
    for i in range(start_history, seq_idx + 1):
        hist_seq = inputs[i]
        hist_meta = metadata[i]
        hist_t_end = hist_meta['input_end']
        
        # Check actual label
        hist_t_row = idx_df[idx_df['window_end'] == hist_t_end]
        is_attk = False
        hist_w_id = -1
        if len(hist_t_row) > 0:
            is_attk = bool(hist_t_row.iloc[0]['contains_attack'])
            hist_w_id = hist_t_row.index[0]
            
        # Get risk
        hist_state_unscaled = hist_seq[-1, 64:].numpy()
        hist_state_scaled = torch.tensor(engine.scaler.transform(hist_state_unscaled.reshape(1, -1)), dtype=torch.float32)
        
        hist_warn = engine.evaluate_sequence(
            sequence_tensor=hist_seq,
            current_state_tensor=hist_state_scaled,
            is_currently_attack=is_attk,
            current_window_id=hist_w_id
        )
        
        recent_risks.append(hist_warn.get('risk_probability', 0.0) * 100)
        recent_times.append(str(hist_t_end))
        actual_attacks.append(is_attk)
        
    timeline_df = pd.DataFrame({
        "Time": recent_times,
        "Forecasted Risk %": recent_risks,
        "Actual Attack at t": actual_attacks
    })
    
    fig_timeline = go.Figure()
    fig_timeline.add_trace(go.Scatter(
        x=timeline_df['Time'], 
        y=timeline_df['Forecasted Risk %'],
        mode='lines+markers',
        name='Forecasted Risk',
        line=dict(color='orange')
    ))
    
    # Highlight actual attacks
    attack_times = timeline_df[timeline_df['Actual Attack at t'] == True]
    if not attack_times.empty:
        fig_timeline.add_trace(go.Scatter(
            x=attack_times['Time'],
            y=attack_times['Forecasted Risk %'],
            mode='markers',
            marker=dict(color='red', size=10, symbol='x'),
            name='Actual Attack at t'
        ))
        
    fig_timeline.update_layout(
        title="Recent Minute-Level Forecast Trend",
        xaxis_title="Minute Window End",
        yaxis_title="Forecasted Risk Probability (%)",
        yaxis_range=[0, 100]
    )
    st.plotly_chart(fig_timeline, use_container_width=True)
    st.caption("Minute-level representation of sequential forecasts. Actual attack markers represent ground-truth labels at time t. The forecast provides a 1-minute horizon.")

if __name__ == "__main__":
    run_dashboard()
