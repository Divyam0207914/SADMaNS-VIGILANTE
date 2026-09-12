# SADMaNS / VIGILANTE — MVP Implementation Specification
## Port-Graph GCN + GRU Network World Model

**Version:** MVP-1  
**Dataset:** `02-15-2018.csv` from CSE-CIC-IDS2018  
**Development Environment:** VS Code  
**Frontend/Demo:** Streamlit  
**Graph Representation for MVP:** Destination-Port Interaction Graph  
**Temporal Window:** 1 minute  
**Core Model:** GCN + GRU  
**Explainability:** PyTorch Geometric `torch_geometric.explain` + GNNExplainer  
**MITRE Integration:** `mitreattack-python` / ATT&CK STIX data  
**Primary Goal:** Predict the next network state and use the predicted future state to estimate attack risk and infer the likely attack stage.

---

# 1. Project Objective

SADMaNS is a predictive network-defence prototype based on the idea of a **network world model**.

The system must NOT be implemented as a simple:

```text
Traffic → Classifier → Attack / Benign
```

Instead, the MVP should follow:

```text
Network Traffic
      ↓
1-Minute Network State
      ↓
Port Interaction Graph
      ↓
GCN — spatial/service relationship representation
      ↓
GRU — temporal state evolution
      ↓
Predicted Next Network State
      ↓
 ┌───────────────┬────────────────┐
 ↓               ↓                ↓
Threat Risk   Attack Type     Attack Stage
                                  ↓
                           MITRE ATT&CK
                                  ↓
                           Explanation
                                  ↓
                              Streamlit
```

The central research question is:

> Given only the network state observed up to time `t`, can the system predict the likely next state and identify whether the network is moving toward malicious activity?

The project should prioritize a scientifically meaningful **state-transition prediction** rather than merely maximizing classification accuracy.

---

# 2. Source Research Direction

The project research documents define the core concept as predictive cyber defence:

```text
Observe
  ↓
Build Network State
  ↓
Learn State Dynamics
  ↓
Simulate Future States
  ↓
Forecast Attack Trajectory
  ↓
Predict Next Stage / Target
  ↓
Explain Prediction
  ↓
Support Defender Action
```

The conceptual world-model formulation is:

\[
P(S_{t+1} \mid S_t)
\]

with multi-step forecasting:

\[
S_t \rightarrow S_{t+1} \rightarrow S_{t+2} \rightarrow \cdots \rightarrow S_{t+k}
\]

For the first MVP, implement **one-step prediction first**:

\[
S_t \rightarrow \hat{S}_{t+1}
\]

After the one-step model works correctly, extend it to K-step rollout.

---

# 3. Current Dataset

The first implementation uses:

```text
02-15-2018.csv
```

from the CSE-CIC-IDS2018 flow dataset.

The provided CSV has exactly 80 columns:

```text
1  : Dst Port
2  : Protocol
3  : Timestamp
4  : Flow Duration
5  : Tot Fwd Pkts
6  : Tot Bwd Pkts
7  : TotLen Fwd Pkts
8  : TotLen Bwd Pkts
9  : Fwd Pkt Len Max
10 : Fwd Pkt Len Min
11 : Fwd Pkt Len Mean
12 : Fwd Pkt Len Std
13 : Bwd Pkt Len Max
14 : Bwd Pkt Len Min
15 : Bwd Pkt Len Mean
16 : Bwd Pkt Len Std
17 : Flow Byts/s
18 : Flow Pkts/s
19 : Flow IAT Mean
20 : Flow IAT Std
21 : Flow IAT Max
22 : Flow IAT Min
23 : Fwd IAT Tot
24 : Fwd IAT Mean
25 : Fwd IAT Std
26 : Fwd IAT Max
27 : Fwd IAT Min
28 : Bwd IAT Tot
29 : Bwd IAT Mean
30 : Bwd IAT Std
31 : Bwd IAT Max
32 : Bwd IAT Min
33 : Fwd PSH Flags
34 : Bwd PSH Flags
35 : Fwd URG Flags
36 : Bwd URG Flags
37 : Fwd Header Len
38 : Bwd Header Len
39 : Fwd Pkts/s
40 : Bwd Pkts/s
41 : Pkt Len Min
42 : Pkt Len Max
43 : Pkt Len Mean
44 : Pkt Len Std
45 : Pkt Len Var
46 : FIN Flag Cnt
47 : SYN Flag Cnt
48 : RST Flag Cnt
49 : PSH Flag Cnt
50 : ACK Flag Cnt
51 : URG Flag Cnt
52 : CWE Flag Count
53 : ECE Flag Cnt
54 : Down/Up Ratio
55 : Pkt Size Avg
56 : Fwd Seg Size Avg
57 : Bwd Seg Size Avg
58 : Fwd Byts/b Avg
59 : Fwd Pkts/b Avg
60 : Fwd Blk Rate Avg
61 : Bwd Byts/b Avg
62 : Bwd Pkts/b Avg
63 : Bwd Blk Rate Avg
64 : Subflow Fwd Pkts
65 : Subflow Fwd Byts
66 : Subflow Bwd Pkts
67 : Subflow Bwd Byts
68 : Init Fwd Win Byts
69 : Init Bwd Win Byts
70 : Fwd Act Data Pkts
71 : Fwd Seg Size Min
72 : Active Mean
73 : Active Std
74 : Active Max
75 : Active Min
76 : Idle Mean
77 : Idle Std
78 : Idle Max
79 : Idle Min
80 : Label
```

---

# 4. Critical Dataset Limitation

The current CSV does **not** contain:

```text
Src IP
Dst IP
Src Port
```

Therefore, the MVP cannot honestly construct a host-to-host communication graph such as:

```text
Host A → Host B
Host B → Host C
```

Do NOT fabricate source IPs, destination IPs, host IDs, or source ports.

Instead, the MVP must explicitly use:

> **Destination-Port Interaction Graph**

This is a prototype graph representation based on the information actually available in the CSV.

The application should display a small disclaimer such as:

```text
MVP Graph Mode:
Destination-Port Interaction Graph

Host-level graph requires source/destination IP telemetry
or raw PCAP-derived data.
```

The architecture should be written so that the graph builder can later be replaced by a host-level graph builder without rewriting the GCN/GRU/world-model components.

---

# 5. Port Graph Definition

## 5.1 Nodes

Each unique `Dst Port` appearing in a 1-minute temporal window is represented as a graph node.

Example:

```text
21
22
80
443
445
3389
```

Each node represents a destination service/port observed during that time window.

Do NOT call these nodes "hosts".

---

## 5.2 Node Features

For every destination port in each 1-minute window, aggregate traffic statistics.

Recommended initial node features:

```text
flow_count
total_fwd_packets
total_bwd_packets
total_fwd_bytes
total_bwd_bytes
mean_flow_duration
mean_flow_packets_per_sec
mean_flow_bytes_per_sec
mean_packet_size
mean_flow_iat
syn_flag_count
ack_flag_count
rst_flag_count
psh_flag_count
fin_flag_count
fwd_packets_per_sec
bwd_packets_per_sec
```

The feature list should be configurable.

Do not hard-code assumptions that every feature is useful.

---

# 6. Port Graph Edge Construction

Because source/destination host relationships are unavailable, create an **interaction/co-occurrence graph**.

For every 1-minute window:

1. Identify all destination ports appearing in that window.
2. Create one node for every unique destination port.
3. Connect two port nodes when they co-occur in the same temporal window.
4. Weight the edge according to co-occurrence frequency or a normalized co-occurrence score.

Example:

```text
Window:

22, 80, 443, 445

Graph:

22 ─── 80
│ \     │
│  \    │
│   \   │
443 ─── 445
```

The exact edge-weight strategy should be implemented cleanly and documented.

Possible MVP strategy:

```text
edge_weight(i,j) = number of flows/windows in which i and j co-occur
```

Normalize weights before feeding them into the GCN if required.

Important:

> This graph represents service/port interaction patterns, not physical network topology.

---

# 7. Temporal Windowing

Use a fixed temporal window of:

```text
1 minute
```

The raw data must first be converted to timestamps.

Process:

```text
CSV
 ↓
Parse Timestamp
 ↓
Remove invalid timestamps
 ↓
Sort chronologically
 ↓
Floor/group timestamps into 1-minute windows
```

Example:

```text
10:00:01
10:00:17
10:00:42
10:00:59
       ↓
Window 10:00:00–10:00:59

10:01:02
10:01:31
10:01:55
       ↓
Window 10:01:00–10:01:59
```

Each window becomes:

```text
S_t
```

---

# 8. Network State Definition

For MVP:

\[
S_t = [G_t, X_t]
\]

where:

- `G_t` = destination-port graph for the 1-minute window
- `X_t` = aggregated global traffic state

The global state should include features such as:

```text
total_flow_count
total_packets
total_bytes
mean_flow_duration
mean_packet_size
mean_flow_packets_per_sec
mean_flow_bytes_per_sec
SYN_count
ACK_count
RST_count
unique_destination_ports
port_entropy
mean_IAT
```

The exact global state vector should be configurable.

---

# 9. Why We Need Both Graph and Global State

The GCN handles the graph representation:

```text
Which services/ports appear together?
How does the service interaction structure look?
```

The GRU handles temporal evolution:

```text
How does this state change from minute to minute?
```

The global state captures information that may not be fully represented by graph topology.

Therefore:

```text
Port Graph
    ↓
GCN
    ↓
Graph embedding
       \
        \
         + → State representation
        /
Global state
```

---

# 10. GCN Component

Use PyTorch Geometric.

Recommended initial architecture:

```text
Input node features
        ↓
GCNConv
        ↓
ReLU
        ↓
GCNConv
        ↓
ReLU
        ↓
Global Mean Pooling
        ↓
Graph embedding
```

Conceptually:

\[
H^{(l+1)}
=
\sigma
(
\hat D^{-1/2}
\hat A
\hat D^{-1/2}
H^{(l)}
W^{(l)}
)
\]

The GCN should produce a fixed-size representation for each 1-minute graph.

Example:

```text
G_10:00 → z_10:00
G_10:01 → z_10:01
G_10:02 → z_10:02
```

---

# 11. GRU Component

The GRU receives a sequence of graph/state representations.

For an input history of 5 minutes:

```text
S(t-4)
S(t-3)
S(t-2)
S(t-1)
S(t)
```

the model predicts:

```text
S(t+1)
```

Initial sequence length should be configurable, with:

```text
SEQ_LEN = 5
```

as the default.

Do not assume 5 is optimal; make it easy to change later.

---

# 12. World Model

The core MVP model is:

```text
G(t-4) → GCN → z(t-4)
G(t-3) → GCN → z(t-3)
G(t-2) → GCN → z(t-2)
G(t-1) → GCN → z(t-1)
G(t)   → GCN → z(t)

[z(t-4), ..., z(t)]
                ↓
               GRU
                ↓
         predicted latent state
                ↓
        next-state decoder
                ↓
              Ŝ(t+1)
```

The model should predict a future state rather than directly predicting only a class.

---

# 13. Next-State Prediction

The first prediction target is:

```text
S(t+1)
```

The decoder should predict the chosen global state vector.

For continuous state variables:

```text
MSELoss
```

or another suitable regression loss can be used.

Example:

```text
Actual next state:

flow_count       = 140
packet_rate      = 620
SYN_rate         = 0.35
unique_ports     = 12

Predicted:

flow_count       = 151
packet_rate      = 641
SYN_rate         = 0.38
unique_ports     = 13
```

The state-prediction error can then become a component of anomaly/threat scoring.

---

# 14. Threat-Risk Head

Add a prediction head from the GRU hidden state:

```text
GRU hidden state
      ↓
Linear
      ↓
Attack probability
```

Output:

```text
P(attack in next state)
```

This is a downstream forecasting head, not the definition of the world model.

For binary attack target:

```text
0 = benign
1 = attack
```

The target must be generated from the future 1-minute window, not from future information that would be unavailable at prediction time.

Example:

```text
Input:
10:00–10:04

Target:
10:05

If 10:05 contains attack traffic:
future_attack = 1
```

---

# 15. Attack-Type Head

The prototype should optionally predict the attack type represented by the future window.

Use the actual `Label` values found in the dataset.

Do NOT hard-code attack classes before inspecting the CSV.

The preprocessing pipeline must:

```text
Label values
   ↓
normalize labels
   ↓
inspect unique values
   ↓
create class mapping
```

Example only:

```text
BENIGN
FTP-BruteForce
SSH-Bruteforce
...
```

The exact classes must come from the actual file.

Use:

```text
CrossEntropyLoss
```

for multi-class attack-type prediction.

---

# 16. Attack Stage Inference

The project wants to expose a meaningful attack-stage prediction.

Recommended dashboard taxonomy:

```text
NORMAL
RECONNAISSANCE
INITIAL ACCESS
DISCOVERY
LATERAL MOVEMENT
COMMAND & CONTROL
IMPACT
```

However:

> The CICIDS2018 flow CSV does not directly provide perfect MITRE ATT&CK stage labels.

Therefore the application must use wording such as:

```text
Inferred Attack Stage
```

rather than claiming that the dataset directly labels the stage.

---

# 17. MITRE ATT&CK Integration

Use:

```text
mitreattack-python
```

for ATT&CK data/technique lookup.

Important architectural rule:

> `mitreattack-python` is the ATT&CK knowledge/lookup layer. It is not itself the ML stage predictor.

Correct pipeline:

```text
Observed traffic
      ↓
GCN + GRU
      ↓
Predicted future state
      ↓
Attack type / behaviour prediction
      ↓
Semantic mapping
      ↓
MITRE ATT&CK lookup
      ↓
Technique / tactic information
```

The system should not do:

```text
Port 22 → automatically call it a MITRE technique
```

Port numbers alone are not sufficient evidence.

The mapper should combine:

```text
predicted attack behaviour
+
traffic features
+
attack label information during training/evaluation
+
ATT&CK knowledge
```

and produce an **inferred ATT&CK tactic/technique** with confidence.

---

# 18. MITRE Module Design

Create:

```text
mitre/
├── attack_mapper.py
└── technique_lookup.py
```

Responsibilities:

### `attack_mapper.py`

Maps model-level behaviour predictions to candidate ATT&CK techniques/tactics.

### `technique_lookup.py`

Queries/loads ATT&CK STIX data using the selected `mitreattack-python` APIs.

Return structured information:

```python
{
    "technique_id": "...",
    "technique_name": "...",
    "tactic": "...",
    "description": "...",
    "confidence": 0.78
}
```

Do not invent technique IDs.

---

# 19. Explainability

Use:

```text
torch_geometric.explain
GNNExplainer
```

The explanation should answer:

> Which graph components and node features contributed to the prediction?

The output should include:

```text
Important nodes
Important edges
Important node features
```

For example:

```text
Important Ports

445   ██████████
22    ███████
3389  █████

Important Relationships

22 ↔ 445
445 ↔ 3389

Important Features

SYN Flag Count
Flow Pkts/s
Flow Duration
Pkt Len Mean
```

Do not invent importance values.

All displayed values must come from the explainer output.

---

# 20. Explainability Limitation

GNNExplainer primarily explains the graph/GNN component.

It does not automatically provide a complete explanation of the GRU's temporal reasoning.

Therefore the MVP should show:

```text
Graph Explanation
+
Temporal Context
```

For temporal context, display the evolution of important state variables:

```text
Time       SYN Rate    Port Count    Packet Rate
-------------------------------------------------
10:01       0.12          4            120
10:02       0.18          6            170
10:03       0.29          9            250
10:04       0.41         13            390
```

This gives the user a temporal explanation without falsely claiming that GNNExplainer explains the GRU itself.

---

# 21. Streamlit Dashboard

Build a clean prototype rather than a huge production UI.

Suggested layout:

```text
====================================================
                  SADMaNS
       Predictive Network Defence
====================================================

CURRENT WINDOW
[ 10:04:00 – 10:04:59 ]

----------------------------------------------------
CURRENT / FORECAST RISK

Current Risk       Next-State Risk
    42%                 71%

Forecast:
+1 min    +2 min    +3 min
 55%       71%       79%
----------------------------------------------------

             DESTINATION-PORT GRAPH

             [ 22 ]
             /    \
          [80]    [443]
             \    /
             [445]

----------------------------------------------------

PREDICTED NEXT STATE

Attack Risk: 71%
Attack Type: <model prediction>
Inferred Stage: Lateral Movement
Confidence: 78%

----------------------------------------------------

MITRE ATT&CK

Tactic: <lookup result>
Technique: <lookup result>
Technique ID: <lookup result>

----------------------------------------------------

WHY?

Important nodes:
...

Important edges:
...

Important features:
...

----------------------------------------------------

TEMPORAL SIGNALS

SYN rate       ↑
Port diversity ↑
Packet rate    ↑
IAT            ↑

====================================================
```

---

# 22. Graph Visualization

Use one of:

```text
NetworkX
Plotly
streamlit-agraph
```

Recommended MVP:

```text
NetworkX → Plotly → Streamlit
```

The graph should support:

- node labels
- edge visualization
- node importance
- edge importance
- current-window filtering
- basic interaction

For the first version, do not spend excessive time on animation.

A time-window selector/slider is sufficient.

---

# 23. Time Replay

The Streamlit application should allow the user to select a window:

```text
Window:
[ 37 / 120 ]

10:36:00 – 10:36:59
```

When the window changes:

```text
Graph changes
Risk changes
Predicted next state changes
Explanation changes
MITRE information changes
```

This creates the perception of a network evolving through time.

---

# 24. Multi-Step Forecasting — Phase 2

After one-step forecasting works:

```text
S_t
 ↓
Ŝ_t+1
 ↓
Ŝ_t+2
 ↓
Ŝ_t+3
 ↓
Ŝ_t+4
```

The dashboard can show:

```text
Next 1 min   48%
Next 2 min   61%
Next 3 min   74%
Next 4 min   81%
```

Do not build K-step rollout before validating one-step prediction.

---

# 25. Training Sample Construction

With:

```text
SEQ_LEN = 5
WINDOW = 1 minute
```

create samples:

```text
Input                         Target

S1 S2 S3 S4 S5      →         S6
S2 S3 S4 S5 S6      →         S7
S3 S4 S5 S6 S7      →         S8
```

For every sample:

```text
past states
    ↓
model
    ↓
future state
```

No future features may leak into the input.

---

# 26. Data Leakage Rules

This is mandatory.

If prediction time is:

```text
10:05
```

allowed:

```text
10:00–10:05
```

not allowed:

```text
10:05–10:10
```

Do not randomly mix highly correlated future windows into train/test.

Prefer chronological splitting.

Example:

```text
EARLY TIMELINE
     ↓
TRAIN

MIDDLE TIMELINE
     ↓
VALIDATION

LATE TIMELINE
     ↓
TEST
```

The test set must represent future information relative to training.

---

# 27. Important Evaluation Metrics

Do not make accuracy the headline metric.

Measure:

```text
Precision
Recall
F1
PR-AUC
False Positive Rate
False alarms/hour
Next-state prediction error
Attack-stage accuracy
Attack-type F1
Lead time
```

The most important forecasting metric is:

> **How early did the system forecast malicious progression?**

Example:

```text
Attack stage becomes active:
10:15:00

Model becomes high-confidence:
10:11:20

Lead time:
3 minutes 40 seconds
```

Do not claim a lead-time result until it has been measured experimentally.

---

# 28. Baseline

The project documentation requires comparison with a Logistic Regression baseline.

Baseline:

```text
Current/temporal aggregated features
        ↓
Logistic Regression
        ↓
Attack probability
```

Then compare:

```text
                    Logistic Regression     GCN + GRU
Precision
Recall
F1
PR-AUC
FPR
Lead Time
```

The baseline should be simple and reproducible.

Do not fabricate performance numbers.

---

# 29. Project Directory

Create this structure:

```text
SADMaNS/
│
├── data/
│   └── 02-15-2018.csv
│
├── preprocessing/
│   ├── __init__.py
│   ├── loader.py
│   ├── cleaner.py
│   ├── windowing.py
│   ├── state_builder.py
│   └── graph_builder.py
│
├── models/
│   ├── __init__.py
│   ├── gcn_encoder.py
│   ├── gru_dynamics.py
│   ├── prediction_heads.py
│   └── world_model.py
│
├── mitre/
│   ├── __init__.py
│   ├── attack_mapper.py
│   └── technique_lookup.py
│
├── explainability/
│   ├── __init__.py
│   └── gnn_explainer.py
│
├── visualization/
│   ├── __init__.py
│   ├── graph_plot.py
│   └── charts.py
│
├── training/
│   ├── train_world_model.py
│   ├── train_baseline.py
│   └── evaluate.py
│
├── checkpoints/
│
├── app/
│   └── streamlit_app.py
│
├── config.py
├── requirements.txt
└── README.md
```

---

# 30. Configuration

Create a central configuration file.

Example:

```python
DATA_PATH = "data/02-15-2018.csv"

WINDOW_SIZE = "1min"

SEQ_LEN = 5

GRAPH_MODE = "DESTINATION_PORT"

GCN_HIDDEN_DIM = 64

GRU_HIDDEN_DIM = 128

LEARNING_RATE = 1e-3

BATCH_SIZE = 16

EPOCHS = 20

RANDOM_SEED = 42
```

Do not scatter these values throughout the code.

---

# 31. Implementation Order

Follow this exact order.

## Phase 1 — Environment

Install and verify:

```text
Python
PyTorch
PyTorch Geometric
Pandas
NumPy
Scikit-learn
NetworkX
Plotly
Streamlit
mitreattack-python
```

Verify imports before proceeding.

---

## Phase 2 — Dataset Inspection

Build:

```text
01_dataset_inspection.py
```

Print:

```text
shape
columns
timestamp range
unique labels
label distribution
missing values
infinite values
unique destination ports
```

Also generate basic Matplotlib visualizations if needed.

Do not load unnecessary copies of the entire dataset into memory.

---

## Phase 3 — Cleaning

Implement:

```text
cleaner.py
```

Handle:

- invalid timestamps
- missing labels
- numeric conversion
- infinity values
- invalid destination ports
- NaN values
- duplicate handling if appropriate

Document every cleaning decision.

---

## Phase 4 — Temporal Windows

Implement:

```text
windowing.py
```

Convert rows into:

```text
1-minute windows
```

Return an ordered list/data structure of states.

---

## Phase 5 — State Construction

Implement:

```text
state_builder.py
```

For every window calculate:

```text
global state
node-level aggregated features
attack target
attack type target
```

Do not leak future information.

---

## Phase 6 — Graph Construction

Implement:

```text
graph_builder.py
```

Input:

```text
one-minute window
```

Output:

```text
torch_geometric.data.Data
```

with:

```text
x
edge_index
edge_attr
```

where:

- `x` = port-node features
- `edge_index` = port co-occurrence relationships
- `edge_attr` = optional edge weights

---

## Phase 7 — Visual Graph Validation

Before training GCN:

> Make the graph visible.

Display several 1-minute windows.

Check:

- nodes correspond to actual `Dst Port` values
- edges are sensible according to the defined co-occurrence rule
- node features contain no NaN/inf
- graph sizes are manageable

Do not continue to model training until this works.

---

## Phase 8 — GCN

Implement:

```text
gcn_encoder.py
```

Test:

```text
graph → embedding
```

Print:

```text
input shape
edge shape
embedding shape
```

---

## Phase 9 — GRU

Implement:

```text
gru_dynamics.py
```

Test:

```text
sequence of embeddings → hidden state
```

---

## Phase 10 — World Model

Implement:

```text
world_model.py
```

Combine:

```text
GCN
+
GRU
+
next-state decoder
+
attack-risk head
+
attack-type head
+
stage head if training labels are available
```

The next-state decoder is the central component.

---

## Phase 11 — Training

Train chronologically.

Save:

```text
checkpoints/world_model.pt
```

Also save:

```text
scaler
label encoder
feature configuration
model configuration
```

so inference is reproducible.

---

## Phase 12 — MITRE Mapper

Implement:

```text
mitre/attack_mapper.py
mitre/technique_lookup.py
```

Do not make MITRE lookup the prediction mechanism.

Use it as semantic enrichment after the model produces behaviour/attack predictions.

---

## Phase 13 — GNNExplainer

Implement:

```text
explainability/gnn_explainer.py
```

Test the explainer on a selected prediction.

Return:

```text
node importance
edge importance
feature importance
```

---

## Phase 14 — Streamlit

Implement:

```text
app/streamlit_app.py
```

The app should load the saved model rather than retrain every time.

Provide:

```text
dataset/window selector
network graph
current state
predicted next state
attack probability
attack type
inferred stage
MITRE information
explanation
temporal signals
```

---

## Phase 15 — Evaluation

Run:

```text
Logistic Regression
vs
GCN + GRU World Model
```

Generate the final evaluation table.

---

# 32. MVP Definition of Done

The first prototype is considered complete when all of the following work:

### Data

- [ ] `02-15-2018.csv` loads
- [ ] timestamps parse correctly
- [ ] data sorted chronologically
- [ ] labels inspected
- [ ] 1-minute windows generated

### Graph

- [ ] destination-port nodes created
- [ ] node features created
- [ ] port co-occurrence edges created
- [ ] graph visualized

### AI

- [ ] GCN produces graph embedding
- [ ] GRU processes temporal sequence
- [ ] next-state prediction works
- [ ] attack-risk prediction works
- [ ] attack-type prediction works if class targets are valid

### MITRE

- [ ] ATT&CK data is accessible
- [ ] predicted behaviour is mapped to candidate ATT&CK semantics
- [ ] technique/tactic shown in dashboard
- [ ] inferred nature is clearly communicated

### Explainability

- [ ] GNNExplainer runs
- [ ] important nodes shown
- [ ] important edges shown
- [ ] important features shown

### Dashboard

- [ ] Streamlit launches
- [ ] graph visible
- [ ] time-window selection works
- [ ] future prediction visible
- [ ] attack stage visible
- [ ] MITRE information visible
- [ ] explanation visible

---

# 33. What NOT to Implement Yet

Do not add these before the MVP works:

```text
LLM
RAG
Reinforcement Learning
Counterfactual defence
Real-time packet capture
Full PCAP pipeline
Temporal Graph Network
Transformer
Multi-dataset training
CTU-13
Automatic defensive actions
```

These are future extensions.

The first objective is a **working, explainable GCN + GRU world-model prototype**.

---

# 34. Future Upgrade Path

Once the MVP works:

```text
MVP

Destination-Port Graph
        ↓
GCN + GRU
        ↓
Next State
```

upgrade to:

```text
PCAP / richer telemetry
        ↓
Source IP + Destination IP
        ↓
Host Communication Graph
        ↓
GCN / Temporal GNN
        ↓
GRU / advanced temporal model
        ↓
K-step trajectory
        ↓
Next target
        ↓
Temporal XAI
        ↓
Counterfactual simulation
```

The host-level graph should eventually look like:

```text
Internet
   ↓
Host A
   ↓
Host B
   ↓
Host C
   ↓
Database
```

with edge features such as:

```text
port
protocol
bytes
packets
duration
TCP flags
IAT
frequency
direction
```

---

# 35. Final Product Narrative

The MVP should be presented as:

> **A predictive network world model that learns how destination-service traffic states evolve over time and forecasts the next network state using graph and temporal deep learning.**

The model does not merely ask:

> "Is this traffic malicious?"

It asks:

> "Given what the network has been doing during the previous minutes, what is the network likely to look like next?"

Then:

```text
Predict
   ↓
Assess Risk
   ↓
Infer Attack Stage
   ↓
Enrich with MITRE ATT&CK
   ↓
Explain Why
```

---

# 36. ANTIGRAVITY CODING AGENT PROMPT

Copy the following prompt into Antigravity.

---

## MASTER IMPLEMENTATION PROMPT

You are the lead ML engineer implementing the SADMaNS/VIGILANTE MVP.

Read this entire specification before writing code.

The project is an **AI-based predictive network defence prototype**. The goal is NOT to create a normal binary intrusion classifier. The core idea is a small network world model that learns temporal state transitions:

\[
P(S_{t+1}|S_t)
\]

The MVP must use:

- Python
- PyTorch
- PyTorch Geometric
- GCN
- GRU
- Pandas
- NumPy
- Scikit-learn
- NetworkX
- Plotly
- Streamlit
- `mitreattack-python`
- `torch_geometric.explain`
- GNNExplainer

Dataset:

```text
data/02-15-2018.csv
```

The dataset has 80 columns and contains:

```text
Dst Port
Protocol
Timestamp
Flow Duration
Tot Fwd Pkts
Tot Bwd Pkts
TotLen Fwd Pkts
TotLen Bwd Pkts
Fwd Pkt Len Max
Fwd Pkt Len Min
Fwd Pkt Len Mean
Fwd Pkt Len Std
Bwd Pkt Len Max
Bwd Pkt Len Min
Bwd Pkt Len Mean
Bwd Pkt Len Std
Flow Byts/s
Flow Pkts/s
Flow IAT Mean
Flow IAT Std
Flow IAT Max
Flow IAT Min
Fwd IAT Tot
Fwd IAT Mean
Fwd IAT Std
Fwd IAT Max
Fwd IAT Min
Bwd IAT Tot
Bwd IAT Mean
Bwd IAT Std
Bwd IAT Max
Bwd IAT Min
Fwd PSH Flags
Bwd PSH Flags
Fwd URG Flags
Bwd URG Flags
Fwd Header Len
Bwd Header Len
Fwd Pkts/s
Bwd Pkts/s
Pkt Len Min
Pkt Len Max
Pkt Len Mean
Pkt Len Std
Pkt Len Var
FIN Flag Cnt
SYN Flag Cnt
RST Flag Cnt
PSH Flag Cnt
ACK Flag Cnt
URG Flag Cnt
CWE Flag Count
ECE Flag Cnt
Down/Up Ratio
Pkt Size Avg
Fwd Seg Size Avg
Bwd Seg Size Avg
Fwd Byts/b Avg
Fwd Pkts/b Avg
Fwd Blk Rate Avg
Bwd Byts/b Avg
Bwd Pkts/b Avg
Bwd Blk Rate Avg
Subflow Fwd Pkts
Subflow Fwd Byts
Subflow Bwd Pkts
Subflow Bwd Byts
Init Fwd Win Byts
Init Bwd Win Byts
Fwd Act Data Pkts
Fwd Seg Size Min
Active Mean
Active Std
Active Max
Active Min
Idle Mean
Idle Std
Idle Max
Idle Min
Label
```

CRITICAL LIMITATION:

The CSV does not contain source IP, destination IP, or source port.

Therefore DO NOT fabricate host IDs.

For this MVP use:

```text
GRAPH_MODE = "DESTINATION_PORT"
```

Nodes are unique `Dst Port` values observed during each 1-minute window.

Call this a:

**Destination-Port Interaction Graph**

Do not call it a host communication graph.

Edges should be based on destination-port co-occurrence within the same 1-minute window.

Node features should be aggregated traffic features such as:

```text
flow_count
total_fwd_packets
total_bwd_packets
total_fwd_bytes
total_bwd_bytes
mean_flow_duration
mean_flow_packets_per_sec
mean_flow_bytes_per_sec
mean_packet_size
mean_flow_iat
syn_flag_count
ack_flag_count
rst_flag_count
psh_flag_count
fin_flag_count
fwd_packets_per_sec
bwd_packets_per_sec
```

Also construct a global state vector containing configurable aggregate traffic statistics.

Temporal window:

```text
WINDOW_SIZE = 1 minute
```

Default sequence length:

```text
SEQ_LEN = 5
```

So:

```text
S(t-4), S(t-3), S(t-2), S(t-1), S(t)
```

predict:

```text
S(t+1)
```

The architecture must be:

```text
1-minute Port Graph
        ↓
GCN
        ↓
Graph Embedding
        ↓
combine with global state
        ↓
GRU
        ↓
Predicted Next State
        ↓
 ┌───────────────┬────────────────┬──────────────┐
 ↓               ↓                ↓
Attack Risk   Attack Type    Inferred Stage
                                  ↓
                           MITRE ATT&CK
```

The world-model objective is next-state prediction.

Do NOT replace the model with a direct:

```text
features → attack classifier
```

The attack prediction is a downstream head.

Implement:

1. GCN encoder
2. GRU temporal dynamics model
3. next-state decoder
4. attack-risk head
5. attack-type head
6. optional stage head only if a defensible stage target is available

For the first implementation, one-step prediction is mandatory.

K-step rollout is a later extension.

Use chronological train/validation/test splits.

Do NOT randomly mix highly correlated temporal windows.

Do NOT leak future information.

If prediction time is `t`, only information at or before `t` may be used as input.

The attack target must be generated from the future target window, not from future data accidentally included in the input.

Attack labels must be inspected directly from the CSV. Do not invent the number of attack classes.

For MITRE ATT&CK:

Use `mitreattack-python` as the ATT&CK knowledge/lookup layer.

Do NOT claim that `mitreattack-python` itself predicts attack stages.

The correct flow is:

```text
model predicts behaviour/attack type
        ↓
semantic mapper
        ↓
ATT&CK lookup
        ↓
technique/tactic information
```

Do not hard-code arbitrary technique IDs.

Use wording:

```text
Inferred Attack Stage
```

because the dataset does not directly provide perfect MITRE stage labels.

Recommended dashboard stages:

```text
NORMAL
RECONNAISSANCE
INITIAL ACCESS
DISCOVERY
LATERAL MOVEMENT
COMMAND & CONTROL
IMPACT
```

GNN explainability:

Use:

```python
torch_geometric.explain
```

and:

```text
GNNExplainer
```

The explanation must expose:

- important nodes
- important edges
- important node features

Do not fabricate importance values.

Also show temporal signal evolution separately because GNNExplainer does not automatically explain the GRU's temporal reasoning.

Use:

```text
NetworkX → Plotly → Streamlit
```

for the graph visualization.

The Streamlit dashboard must contain:

```text
Current time window
Current risk
Predicted next-state risk
Predicted attack type
Inferred attack stage
MITRE ATT&CK information
Destination-port graph
Important nodes
Important edges
Important features
Temporal signal trends
```

Use a time-window slider or selector so the graph and predictions update for different windows.

Project structure:

```text
SADMaNS/
│
├── data/
│   └── 02-15-2018.csv
├── preprocessing/
│   ├── loader.py
│   ├── cleaner.py
│   ├── windowing.py
│   ├── state_builder.py
│   └── graph_builder.py
├── models/
│   ├── gcn_encoder.py
│   ├── gru_dynamics.py
│   ├── prediction_heads.py
│   └── world_model.py
├── mitre/
│   ├── attack_mapper.py
│   └── technique_lookup.py
├── explainability/
│   └── gnn_explainer.py
├── visualization/
│   ├── graph_plot.py
│   └── charts.py
├── training/
│   ├── train_world_model.py
│   ├── train_baseline.py
│   └── evaluate.py
├── checkpoints/
├── app/
│   └── streamlit_app.py
├── config.py
├── requirements.txt
└── README.md
```

Implementation order is mandatory:

### Step 1
Verify Python environment and dependencies.

### Step 2
Inspect the actual CSV.

Print:

```text
shape
columns
timestamp range
unique labels
label distribution
missing values
infinite values
unique destination ports
```

Do not assume values before inspecting them.

### Step 3
Implement cleaning.

### Step 4
Implement 1-minute temporal windowing.

### Step 5
Implement state construction.

### Step 6
Implement destination-port graph construction.

### Step 7
VISUALIZE AND VALIDATE THE GRAPH BEFORE TRAINING.

### Step 8
Implement and test GCN.

### Step 9
Implement and test GRU.

### Step 10
Combine them into the world model.

### Step 11
Train one-step next-state prediction.

### Step 12
Add attack-risk and attack-type heads.

### Step 13
Add MITRE ATT&CK semantic enrichment.

### Step 14
Add GNNExplainer.

### Step 15
Build Streamlit.

### Step 16
Implement Logistic Regression baseline.

### Step 17
Evaluate both systems.

Do not jump directly to Streamlit before validating the underlying graph/model.

Do not train the model every time Streamlit loads.

Save the trained model and preprocessing objects.

Save:

```text
model checkpoint
scaler
label encoder
feature configuration
model configuration
```

The application should load those artifacts.

The code must be modular, readable, typed where practical, and documented.

Every major transformation should have a small test or validation output.

Do not fabricate results.

Do not fabricate attack labels.

Do not fabricate MITRE mappings.

Do not fabricate explainability values.

Do not claim the model predicts host-level attacker movement because the MVP only has destination-port information.

The first successful demo should prove:

```text
CSV
 ↓
1-minute state
 ↓
destination-port graph
 ↓
GCN
 ↓
GRU
 ↓
predicted next state
 ↓
attack risk/type
 ↓
inferred stage
 ↓
MITRE enrichment
 ↓
GNN explanation
 ↓
Streamlit dashboard
```

After implementing each phase, report:

1. what was implemented
2. files created/changed
3. how to run it
4. what output should appear
5. any assumptions or limitations
6. what should be tested next

Do not add unnecessary technologies.

The priority is:

```text
CORRECT DATA
    ↓
CORRECT TEMPORAL WINDOWS
    ↓
CORRECT GRAPH
    ↓
WORKING GCN
    ↓
WORKING GRU
    ↓
VALID NEXT-STATE FORECAST
    ↓
EXPLAINABILITY
    ↓
MITRE ENRICHMENT
    ↓
STREAMLIT DEMO
```

The final prototype should communicate one central idea:

> **Don't just detect what is happening now. Learn how the network state evolves and forecast what is likely to happen next.**
