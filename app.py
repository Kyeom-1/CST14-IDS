import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import random

# =========================================================
# APP CONFIG
# =========================================================
st.set_page_config(page_title="Explainable IDS", layout="wide")

st.title("🔐 Explainable IDS (SHAP-Corrected Version)")
st.markdown("Manual + Live IDS with safe SHAP explanations (no dtype errors).")

# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_model():
    return joblib.load("ids_low_fn_final.pkl")

saved = load_model()
model = saved["model"]
preprocessor = saved["preprocessor"]
threshold = saved["threshold"]

# SHAP explainer (cached)
@st.cache_resource
def get_explainer():
    return shap.TreeExplainer(model)

explainer = get_explainer()

# =========================================================
# FEATURES
# =========================================================
features = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes",
    "land","wrong_fragment","urgent","hot","num_failed_logins","logged_in",
    "num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
    "dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate"
]

# =========================================================
# SAFE PREPROCESS (CRITICAL FIX)
# =========================================================
def preprocess(packet_dict):
    full = {f: 0 for f in features}
    full.update(packet_dict)

    df = pd.DataFrame([full])

    # numeric matrix for model + SHAP
    X = preprocessor.transform(df)

    return X, df

# =========================================================
# PREDICTION
# =========================================================
def predict(X):
    prob = model.predict_proba(X)[0][1]
    pred = int(prob >= threshold)
    return pred, prob

# =========================================================
# SHAP SAFE EXPLANATION (FIXED)
# =========================================================
def explain_shap(X):

    shap_values = explainer.shap_values(X)

    # handle binary classification safely
    if isinstance(shap_values, list):
        vals = shap_values[1][0]
    else:
        vals = shap_values[0]

    contrib = list(zip(features, vals))
    contrib = sorted(contrib, key=lambda x: abs(x[1]), reverse=True)

    attack = []
    normal = []

    for f, v in contrib[:10]:
        if v > 0:
            attack.append(f"{f}: {v:.4f} → pushes ATTACK")
        else:
            normal.append(f"{f}: {v:.4f} → pushes NORMAL")

    return attack, normal

# =========================================================
# SESSION STATE
# =========================================================
if "packets" not in st.session_state:
    st.session_state.packets = []

# =========================================================
# MODE SELECT
# =========================================================
mode = st.sidebar.radio(
    "Mode",
    ["Manual Inspection", "Live IDS Simulation"]
)

st.sidebar.info("SHAP-safe IDS (no categorical crashes)")

# =========================================================
# 🧪 MANUAL MODE
# =========================================================
if mode == "Manual Inspection":

    st.header("🧪 Manual Packet Analyzer")

    packet = {}

    col1, col2, col3 = st.columns(3)

    for i, f in enumerate(features):
        target = [col1, col2, col3][i % 3]

        with target:

            if f == "protocol_type":
                packet[f] = st.selectbox(f, ["tcp","udp","icmp"])

            elif f == "service":
                packet[f] = st.selectbox(f, ["http","ftp","smtp","private"])

            elif f == "flag":
                packet[f] = st.selectbox(f, ["SF","S0","REJ"])

            else:
                packet[f] = st.number_input(f, value=0.0)

    if st.button("Analyze Packet"):

        X, df = preprocess(packet)
        pred, prob = predict(X)

        attack, normal = explain_shap(X)

        st.subheader("📦 Result")

        if pred == 1:
            st.error(f"🚨 ATTACK DETECTED ({prob:.3f})")
        else:
            st.success(f"✅ NORMAL TRAFFIC ({prob:.3f})")

        col1, col2 = st.columns(2)

        with col1:
            st.write("### Raw Packet")
            st.json(packet)

        with col2:
            st.write("### SHAP Attack Drivers")
            for a in attack:
                st.write("🚨", a)

            st.write("### SHAP Normal Drivers")
            for n in normal:
                st.write("🟢", n)

# =========================================================
# 📡 LIVE MODE (SAFE + CLICKABLE)
# =========================================================
else:

    st.header("📡 Live IDS Simulation")

    if st.button("Generate Packet"):

        packet = {
            "duration": random.randint(0, 5),
            "protocol_type": random.choice(["tcp","udp","icmp"]),
            "service": random.choice(["http","ftp","smtp","private"]),
            "flag": random.choice(["SF","S0","REJ"]),
            "src_bytes": random.randint(0, 50000),
            "dst_bytes": random.randint(0, 50000),
            "land": 0,
            "wrong_fragment": 0,
            "urgent": 0,
            "hot": 0,
            "num_failed_logins": 0,
            "logged_in": random.choice([0,1]),
            "count": random.randint(1, 150),
            "srv_count": random.randint(1, 100),
            "serror_rate": random.random(),
            "srv_serror_rate": random.random(),
            "rerror_rate": random.random(),
            "srv_rerror_rate": random.random(),
            "same_srv_rate": random.random(),
            "diff_srv_rate": random.random(),
            "srv_diff_host_rate": random.random(),
            "dst_host_count": random.randint(0, 255),
            "dst_host_srv_count": random.randint(0, 255),
            "dst_host_same_srv_rate": random.random(),
            "dst_host_diff_srv_rate": random.random(),
            "dst_host_same_src_port_rate": random.random(),
            "dst_host_srv_diff_host_rate": random.random(),
            "dst_host_serror_rate": random.random(),
            "dst_host_srv_serror_rate": random.random(),
            "dst_host_rerror_rate": random.random(),
            "dst_host_srv_rerror_rate": random.random()
        }

        X, df = preprocess(packet)
        pred, prob = predict(X)

        st.session_state.packets.append({
            "packet": packet,
            "X": X,
            "pred": pred,
            "prob": prob
        })

    # =====================================================
    # PACKET TABLE
    # =====================================================
    if len(st.session_state.packets) > 0:

        table = [
            {
                "ID": i,
                "Status": "ATTACK" if p["pred"] == 1 else "NORMAL",
                "Probability": round(p["prob"], 4)
            }
            for i, p in enumerate(st.session_state.packets)
        ]

        df = pd.DataFrame(table)

        selected = st.selectbox("Select Packet", df["ID"])

        st.dataframe(df, use_container_width=True)

        p = st.session_state.packets[selected]

        st.subheader(f"📦 Packet {selected} Forensic View")

        col1, col2 = st.columns(2)

        with col1:
            st.write("### Raw Packet")
            st.json(p["packet"])

        with col2:
            if p["pred"] == 1:
                st.error(f"🚨 ATTACK ({p['prob']:.3f})")
            else:
                st.success(f"✅ NORMAL ({p['prob']:.3f})")

        attack, normal = explain_shap(p["X"])

        st.write("### 🧠 SHAP Explanation")

        col3, col4 = st.columns(2)

        with col3:
            st.write("🚨 Attack Drivers")
            for a in attack:
                st.write("•", a)

        with col4:
            st.write("🟢 Normal Drivers")
            for n in normal:
                st.write("•", n)