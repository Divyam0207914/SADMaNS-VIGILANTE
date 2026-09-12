# PHASE 9 EXPLAINABILITY, MITRE & SOC DASHBOARD REPORT
SADMaNS / VIGILANTE — AI-based Network Attack Forecasting from Network Traffic Data

## 1. Objective
Phase 9 constructs an Explainability Engine, MITRE ATT&CK Enrichment layer, and a Streamlit-based SOC Dashboard. This phase transforms the raw mathematical output of the World Model (Phase 7) and Early Warning Engine (Phase 8) into interpretable cybersecurity intelligence. The dashboard answers "What is happening?", "What is forecasted?", and "Why did the model generate this warning?" without making unsupported causal claims.

## 2. Phase 5–8 Dependencies
Phase 9 relies heavily on the MVP established in previous phases. It leverages the global network states (13 dimensions), the untrained GCN graph structures (Phase 5), chronological forecasting GRU (Phase 6), the world model's multi-head outputs (Phase 7), and the warning logic definitions (Phase 8). Phase 9 does not alter or retrain these dependencies; it only interprets their outputs.

## 3. Explainability Architecture
The `ExplanationEngine` (`models/explanation_engine.py`) takes the current state $S_t$ (unscaled) and the forecasted state $\hat{S}_{t+1}$ (scaled). It uses the frozen Phase 7 `StandardScaler` to align comparisons. It generates absolute, signed, scaled, and relative feature deviations. It ensures edge cases (zeros, NaNs, Infs) are gracefully caught and substituted with finite representations to guarantee JSON serialization stability. 

## 4. Feature-Change Methodology & Scaled Ranking
Features are ranked by their **Absolute Scaled Deviation**. By operating in the standardized space, variables with massively varying scales (e.g., bytes vs probabilities) are fairly compared for deviation severity. The top three features displaying the largest scaled deviations are surfaced as the primary explanation. This explicitly represents "the features showing the strongest deviation contributing to the forecast," preventing any false causal claims ("this feature caused the attack").

## 5. Learned vs Heuristic vs Semantic Components
The dashboard and explanation engines explicitly distinguish output origins:
- **LEARNED**: Next-state forecast $\hat{S}_{t+1}$, Risk Probability, and Attack Type prediction.
- **HEURISTIC**: Stage inference (e.g., `IMPACT`, `NORMAL`). This relies on a static logic mapping, not learned sequence logic.
- **SEMANTIC**: MITRE ATT&CK enrichment (e.g., `T1499 Endpoint Denial of Service`). This is a semantic lookup layered onto heuristic classification.
- **UNTRAINED**: Phase 5 GCN spatial representations.

## 6. SOC Dashboard Architecture & Offline Demo Methodology
The dashboard (`app.py`) is a deterministic offline demonstration built with Streamlit. It loads the `best_world_model.pt` and evaluates single selected sequences natively to simulate real-time operations. It includes:
- **Current Network Status**: Status, Risk, and Warning Level.
- **Attack Assessment**: Probabilities and types alongside MITRE integration.
- **Why this Warning?**: Top 3 feature changes natively displayed.
- **Network State Comparison**: A tabular data view with safe delta formatting.
- **Scaled Deviation Plotly Visualization**: Bar chart visualizing standard deviations natively.
The dashboard prominently displays "Offline dataset demonstration — Not live network traffic".

## 7. Security/Leakage Validation
Leakage safety is strictly preserved. In real-time simulation, the application natively operates only using information from $t-4$ to $t$. The actual target label of $t+1$ is heavily guarded and strictly removed from the inference path, ensuring true forecasting validity.

## 8. Software Validation Tests
Software behaviors were explicitly validated using synthetic edge cases (`run_phase9.py`):
- ✅ Zero-value inputs processed safely (no divide-by-zero crashes).
- ✅ NaN/Inf inputs caught securely by NumPy safeguards.
- ✅ Warning state mapping successfully parsed for ONSET, CONTINUING, and RECOVERY.
- ✅ Feature ranking descending validation succeeded exactly.
- ✅ Leakage preservation succeeded strictly.

## 9. Actual Measured Timings (Apple Silicon MPS)
- **Model Load Time**: ~0.69 seconds
- **Single Sequence Inference Time**: ~44 ms
- **Explanation Generation Time**: ~0.25 ms

## 10. Limitations
- **No Causal Explanations**: Feature deviations are correlations inherent to the forecasted state, not explicit causality.
- **Missing Onsets**: The Phase 8 metric caveat remains; zero attack onset sequences exist within the unseen test subset, rendering exact evaluation of "Early Warning" transitions technically unfeasible within this limited split.
- **Static Semantic Enrichment**: The MITRE and Stage outputs remain structurally static heuristic maps that do not dynamically infer novel kill-chains.

## 11. Conclusion
Phase 9 completes the primary software construction of the SADMaNS / VIGILANTE MVP. The Explainability Engine and SOC Dashboard operate swiftly and deterministically. They natively preserve constraints and strictly evaluate temporal vectors chronologically without mathematical leakage, effectively structuring complex topological/chronological data into readable security abstractions.

## 12. Recommendation for Phase 10
**READY FOR PHASE 10 (Final Integration + SIH Validation/Demo)**
