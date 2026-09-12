# Research Log: PS Number 153

## Turn 1-3
- Initial research, finding SIH 2026 PS 153: "AI based Network Attack Forecasting from Network Traffic Data".

## Turn 4-6
- Official PS Details retrieved (SIH26153).
- Deep dive initiated on components.

## Deep Dive: SIH26153 Architectural Design
- **Objective:** Design the MVP components (Data, Model, Simulation, XAI) to meet SIH requirements.

---

## Part 1: Data Engineering & Feature Extraction
- **Goal:** Ingest PCAP/CSV and generate structured feature vectors.
- **Terms:**
  - **NetFlow/IPFIX:** Industry standards for flow-level telemetry (summarized traffic metadata).
  - **PCAP:** Packet Capture; the "source of truth" containing raw bytes and timing.
  - **IAT (Inter-arrival Time):** Time elapsed between successive packets; critical for detecting reconnaissance patterns.
- **Implementation Options:**
  1. *Scapy (Python):* Industrial standard for raw packet manipulation.
  2. *nfdump/nfcapd:* Optimized C-based flow collectors (faster for large PCAPs).
- **Unique Factor:** Propose "Multi-modal feature fusion" — creating a hybrid vector that concatenates macro-flow behavior (volume/duration) with micro-packet behavior (timing entropy/TTL jitter).

## Turn 7 - Deep Dive Part 1: Data Engineering & Feature Engineering
- **Terms:** 
  - **NetFlow/IPFIX:** Summarized metadata about a network connection (start/end time, IPs, ports, bytes).
  - **PCAP (Packet Capture):** The actual recorded traffic (raw packet headers and payload).
- **Implementation Options:**
  1. **Raw PCAP (Scapy/PyShark):** High precision, captures everything, but massive processing overhead.
  2. **Flow-level processing (Argus/CICFlowMeter):** High speed, lower granularity, standard for IDS.
  3. **Hybrid (Proposed):** Use flow-level data for base-state tracking, trigger packet-level extraction on anomalies.
- **Comparison:** Raw PCAP is impractical for real-time forecasting. Flow-level lacks the packet-timing insights needed for advanced attacks. Hybrid balances performance with necessary insight.
- **Proposed Approach:** Hybrid approach with selective feature extraction.
- **Unique Factor (Differentiator):** **'Adaptive Depth Feature Extraction'**. Dynamically toggle the resolution of feature extraction based on the entropy/suspicion score of the current flow, minimizing computational latency.

## Turn 8 - Deep Dive: Conceptual Understanding of SIH26153
- **Goal:** Conceptual clarity on 'Why this PS exists' and 'What the examiners want'.
- **Core Concept:** Transition from *reactive classification* (IDS) to *proactive forecasting* (World Models).
- **Key Insight:** The PS requires modeling the network as a dynamic system where attack stages are causal processes, not isolated events.
- **Examiner Intent:** To filter out standard 'classifier' projects and identify teams capable of advanced AI/cybersecurity reasoning and interpretable, operational tooling.

## Turn 9 - Deep Dive Part 1: Conceptual Foundation of Data Engineering
- **Core Concept:** Data engineering here is not just cleaning data; it is *defining the state space* of the network.
- **Examiner Expectation:** Showing 'Data Physics' knowledge—understanding why specific metrics (e.g., TTL, IAT, TCP Flags) describe the *process* of an attack.
- **Key Distinction:** Feature engineering for World Models must describe *system dynamics*, not just *malicious markers*.

## Turn 10 - Implementation Deep Dive: Feature Engineering Strategies
- **Goal:** Compare implementation strategies for Data Engineering (PS153).
- **Approaches Identified:**
  1. **Flow-Centric (Standard):** Using CICFlowMeter/Argus. Best for batch processing, lacks granular packet timing.
  2. **Packet-Centric (Granular):** Using Scapy/PyShark. High depth, prohibitive latency for real-time.
  3. **Hybrid/Adaptive (Proposed):** Flow-records for state, selective PCAP parsing for anomalies. Best fit for PS153 'World Model' requirement.
- **Hackathon Expectation:** Examiners want an 'Operational Tool'—not a lab script. Must demonstrate balance between performance, depth, and interpretability.

## Turn 12 - Decision: Committed to Hybrid GNN Approach
- **Strategy:** Hybrid GNN-GRU 'GCN-lite' World Model.
- **Architecture:** [Static Graph Topology (Predefined Nodes/Edges) + Edge Feature Enrichment (Part 1 Data)] -> GCN Layer (Spatial) -> GRU Layer (Temporal) -> Prediction Engine.
- **Rationale:** Combines topological modeling (GNN) with sequence forecasting (GRU), maximizing innovation potential while keeping hackathon-viable by using a static graph.
- **Next Steps:** Proceed to Part 3: Forward Simulation and MITRE ATT&CK Mapping.

## Turn 15 - Deep Dive Part 4: Explainability (XAI) & Interpretability
- **Goal:** Understand and plan for the 'Interpretability' mandate.
- **Core Concept:** Translating the World Model's hidden dynamics ($S_{t+1}$) into actionable defensive insights.
- **Why It's Mandatory:** PS requirement: 'Black-box outputs are explicitly disqualified.'
- **Implementation Approaches:**
  1. **Attention Mapping (GNN/GRU):** Visualizing which nodes/edges/time-steps 'fired'.
  2. **Feature Importance (SHAP):** Calculating which specific traffic metrics (e.g., IAT, TCP Flags) drove the prediction.
  3. **Node Saliency Maps (GNN-Specific):** Highlighting specific hosts/links contributing most to the attack prediction.
- **Hackathon Expectation:** Need a visual dashboard that explains *why* the alert occurred, not just *that* an alert occurred.

## Turn 16 - Deep Dive: Enhanced Explainability Strategy (The 'Unique Factor')
- **Goal:** Elevate XAI beyond SHAP/Counterfactuals to a full 'Defensive Insight Engine'.
- **Base Strategy:** Retain 'What-If' Counterfactual Toggler (Interaction).
- **Enhancement 1: Adversarial Trajectory Visualizer (GNN/Topological):** Map predicted MITRE phases onto the actual network graph. Visualizing the path of lateral movement.
- **Enhancement 2: Contrastive 'Why-Not' Explainer:** Explicitly visualize the delta between our World Model's trajectory and the Logistic Regression baseline's prediction. Showing *why* the World Model caught it and the baseline didn't.
- **Enhancement 3: Local LLM Threat Narrative:** Use a small local LLM (e.g., Llama-3-8B-Q4) to summarize the technical trajectory into a human-readable 'Adversary Storyboard'.

## Turn 17 - Architecture & UI/XAI Finalization
- **Core Architecture (Decoupled):** [Data Pipeline] -> [World Model (GCN+GRU)] -> [Semantic Mapper (Sidecar)] -> [Defensive Insight Engine (XAI)].
- **UI Strategy (Streamlit):**
  - **Main Dashboard:** Live Network Graph (Interactive/Animated using `streamlit-agraph` or `plotly`) showing attack trajectory.
  - **Control Panel:** 'What-If' Counterfactual Toggler for expert analysis.
  - **Baseline Monitor:** Side-by-side comparison (Our World Model vs. Logistic Regression baseline).
  - **Narrative Panel:** Local-LLM generated threat intelligence report.
- **Visualization Strategy:**
  - **Graph:** Static/Animated graph using `plotly` or `networkx` within Streamlit.
  - **Alerting:** Probability timelines showing infiltration trajectory.
- **Final Confirmation:** The architecture is fully defined, feasible, and high-impact. Ready for implementation planning.

## Turn 18 - Architectural Documentation Finalized
- **Deliverable:** Created `D:\SADMaNS\ARCHITECTURE.md` detailing the Hybrid GNN-GRU World Model, Sidecar Mapper, and Defensive Insight Engine.
- **Status:** Architecture fully defined and hackathon-ready. Ready for Phase A: Coding the Data Pipeline.

## Turn 19 - Full Engineering Documentation Created
- **Deliverable 1:** `PRD.md` — Defines the product vision, user personas, and success KPIs.
- **Deliverable 2:** `ARCHITECTURE.md` (Expanded) — Detailed mathematical and structural breakdown of the GCN-GRU Hybrid model.
- **Status:** Ready for Phase A implementation. The 'How' and 'What' are now officially finalized and documented at a professional grade.

## Turn 20 - Research Synthesis: Finding the 'Unique Factor'
- **Research Summary:**
  - Confirmed 'GCN + GRU' is SOTA for spatio-temporal forecasting.
  - Found high-value coupling: **GNN-Enhanced Reinforcement Learning** for active defense policies.
  - Concept found: **Causal Inference in Network Traffic**—crucial for World Models to distinguish causality from correlation.
- **Proposal for 'Unique Factor':** **'Proactive Defensive Simulator (PDS)'**
  - Instead of only forecasting, use the trained GNN-GRU World Model as a 'Simulated Environment' to train a lightweight RL agent.
  - The RL agent learns the 'best defensive policy' (e.g., 'isolate Host X' vs 'throttle IP Y') by playing against the World Model’s attack-trajectory predictions.
- **Comparison:**
  - *Detection Only (Most teams):* Just alerts.
  - *Detection + Forecast (Our Core):* Alerts + Forecasts trajectory.
  - *Detection + Forecast + Action (Our New Unique Factor):* Alerts + Forecasts trajectory + **Proposes/Simulates optimal defense.**
