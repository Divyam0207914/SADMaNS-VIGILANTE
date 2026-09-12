# Expanded Architecture Document: Cyber-Forecasting World Model

## 1. Introduction
This document details the technical implementation of SIH26153. The solution leverages a **Hybrid GNN-GRU World Model** to simulate and predict network state transitions ($P(S_{t+1} | S_t)$).

## 2. Theoretical Framework: World Models in Cyber
Traditional AI models are classifiers ($f(X) \rightarrow Y$). Our solution is a **World Model** ($f(S_t, a) \rightarrow S_{t+1}$), where $S_t$ is the current network state. The model learns the "physics" of the network environment—how traffic flows naturally evolve and how malicious perturbations propagate through the topology.

---

## 3. Detailed Data Architecture

### 3.1. State Definition ($S_t$)
The network state $S_t$ at time $t$ is represented as a **Graph Feature Matrix** $X \in \mathbb{R}^{V \times F}$:
*   **$V$ (Nodes):** IPs/Hosts in the network.
*   **$F$ (Features):** Concatenated flow/packet metrics (IAT, Flag entropy, TTL Variance, Payload distribution).

### 3.2. Adaptive Data Ingestion Pipeline
We utilize a **Tiered Sampling Strategy**:
*   **Layer 1 (Light):** Continuous background flow summaries.
*   **Layer 2 (Heavy):** When the World Model’s *Hidden State* diverges significantly from a benign latent space, a trigger activates `scapy`-based deep packet inspection for the specific flow.

---

## 4. Model Architecture & Training

### 4.1. The Spatial Module (GCN)
We use a **Graph Convolutional Network (GCN)** layer:
$$H^{(l+1)} = \sigma(\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} H^{(l)} W^{(l)})$$
*   **Purpose:** To capture topological relationships—e.g., if Host A starts unusual probes to Host B, the GCN captures that Host B’s state is now structurally "linked" to a potential compromise.

### 4.2. The Temporal Module (GRU)
The GCN's output is fed into a **Gated Recurrent Unit (GRU)**:
*   **Hidden State ($h_t$):** Stores the "memory" of the attack sequence.
*   **Output Layer:** A Linear layer that reconstructs the *next* likely feature vector $S_{t+1}$.

### 4.3. The Semantic Mapper (Decoupled Sidecar)
A 2-layer MLP $f_{sidecar}(S_{t+1}) \rightarrow Y_{MITRE}$.
*   **Why decoupled?** This allows the World Model to be trained on large unlabeled traffic datasets (Dynamics learning) while the Sidecar is fine-tuned on smaller labeled attack datasets (Semantics learning).

---

## 5. Explainability (XAI) Pipeline
We implement **Captum's Integrated Gradients** to calculate the contribution of each element in the feature matrix $X$ to the final forecast.
*   **Attribution Map:** Maps the high-importance features back to the GNN nodes.
*   **Counterfactual Analysis:** Perturbs specific features in the input state to see if the predicted MITRE phase changes, validating the model’s "causal" understanding.

## 6. Performance Benchmarking
A **Logistic Regression baseline** is trained on the same data. Success is defined by the World Model’s ability to predict a high-risk state $N$ steps before the Logistic Regression baseline labels a flow as "malicious."

---

## 7. Deployment & Scalability
*   **Environment:** PyTorch/Python running in a contained virtual environment.
*   **Optimization:** Model quantization to ensure real-time performance on a standard laptop.
*   **Evaluation:** Using the `CIC-IDS-2018` dataset as the ground truth.
