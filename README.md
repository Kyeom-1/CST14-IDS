# Explainable Multiclass Intrusion Detection System

This project is an interactive Streamlit app for simulating network traffic and explaining IDS predictions in detail.

It compares two approaches side by side:
- **Baseline model**: binary classifier (`NORMAL` vs `ATTACK`)
- **Proposed model**: weighted multiclass ensemble that predicts specific attack families

The app is designed to show not just *what* was predicted, but *why* using SHAP feature attributions.

## What the App Does

- Generates realistic traffic packets (normal and multiple attack patterns)
- Runs both baseline and proposed models on the same packet
- Displays agreement/disagreement between model verdicts
- Breaks down each ensemble member's vote
- Explains feature influence with SHAP for:
  - the predicted class
  - the `normal` class
- Shows full probability distribution across all multiclass labels

## Current Architecture

The main app entrypoint is `app.py`.

### Baseline branch
- Loads `baseline_model.pkl`
- Binary decision with thresholding
- Reports training metrics (accuracy, precision, recall, F1, AUC)

### Proposed branch
- Loads `multiclass_ids_model.pkl`
- Uses a weighted ensemble:
  - XGBoost: 40%
  - LightGBM: 25%
  - Random Forest: 20%
  - Extra Trees: 15%
- Produces:
  - multiclass prediction label
  - confidence
  - attack/normal verdict

### Explainability
- Uses SHAP `TreeExplainer` on XGBoost outputs
- Handles multiclass SHAP direction correctly by interpreting contributions relative to the selected class score
- Presents score comparison between predicted class and `normal` class

## Simulations Included

The sidebar lets you generate and analyze templates such as:
- DDoS SYN Flood
- ICMP Amplification (Smurf-style)
- HTTP Back DoS
- Teardrop fragmentation attack
- Port sweep / service probe
- Brute-force login attack
- Remote-to-Local exploit behavior
- Privilege escalation / root exploit
- Random mixed traffic

## Dataset Basis

The feature design and attack classes are based on NSL-KDD style intrusion detection data (connection metadata, protocol/service flags, error rates, host/service traffic counters, authentication and privilege indicators).

## Project Files

- `app.py` - Streamlit application (UI, simulation, prediction, SHAP explanation)
- `baseline_model.pkl` - serialized baseline binary IDS model bundle
- `multiclass_ids_model.pkl` - serialized multiclass ensemble bundle

## Quick Start

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install streamlit pandas numpy scikit-learn xgboost lightgbm shap joblib
```

3. Run the app:

```bash
streamlit run app.py
```

4. Open the local Streamlit URL shown in your terminal.

## Typical Workflow

1. Choose a simulation type in the sidebar.
2. Click **Generate & Analyze Packet**.
3. Review baseline vs proposed verdicts.
4. Inspect ensemble member votes.
5. Read SHAP sections to understand feature-level reasoning.
6. Check full class probability rankings.

## Notes

- Keep `baseline_model.pkl` and `multiclass_ids_model.pkl` in the same directory as `app.py`.
- The app relies on exact feature preprocessing contained in the saved model bundles.
- If model files were generated in a different environment/library version, compatibility issues can occur during loading.
