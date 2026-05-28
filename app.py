# =============================================================================
# EXPLAINABLE MULTICLASS IDS — FIXED SHAP + BASELINE COMPARISON
# =============================================================================
# ROOT CAUSE OF "SHAP says attack but model says normal" BUG:
#
# SHAP for multiclass XGBoost returns shape (n_samples, n_features, n_classes).
# sv[0, :, pred_idx] = how each feature contributed TO the PREDICTED class.
# Positive SHAP = pushed TOWARD the predicted class.
# Negative SHAP = pushed AWAY from the predicted class.
#
# The old code always labelled positive = "toward ATTACK" — which is
# completely wrong when the prediction is "normal". It was reading the
# right numbers but attaching the wrong meaning to them.
#
# FIX: label direction relative to the PREDICTED class, not hardcoded
# to "attack". Also show a dual view: what pushed toward the predicted
# class AND what pushed toward/away from "normal" class specifically.
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import random
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(page_title="Explainable IDS", layout="wide", page_icon="🔐")
st.title("🔐 Explainable Multiclass Intrusion Detection System")
st.markdown(
    "Simulate real-world network attacks, inspect packet features, "
    "view ensemble voting, and understand **why** the model made its decision."
)

# =============================================================================
# LOAD MODELS
# =============================================================================

@st.cache_resource
def load_proposed():
    return joblib.load("multiclass_ids_model.pkl")

@st.cache_resource
def load_baseline():
    return joblib.load("baseline_model.pkl")

saved            = load_proposed()
xgb_model        = saved["xgb_model"]
rf_model         = saved["rf_model"]
et_model         = saved["et_model"]
lgbm_model       = saved["lgbm_model"]
preprocessor     = saved["preprocessor"]
class_names      = saved["class_names"]
NORMAL_IDX       = class_names.index("normal")   # 11

baseline_saved    = load_baseline()
baseline_model    = baseline_saved["model"]
baseline_features = baseline_saved["feature_names"]
baseline_encoders = baseline_saved["label_encoders"]
baseline_threshold= baseline_saved["threshold"]
baseline_metrics  = baseline_saved["metrics"]

@st.cache_resource
def get_explainer():
    return shap.TreeExplainer(xgb_model)

explainer = get_explainer()
EV = explainer.expected_value   # list of 23 floats, one per class

# =============================================================================
# FEATURES
# =============================================================================
RAW_FEATURES = [
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

# Human-readable descriptions for every raw feature
FEATURE_INFO = {
    "duration":                    "Length of the connection in seconds. Very short or zero often signals automated/attack traffic.",
    "protocol_type":               "Network protocol: TCP (connection-oriented), UDP (connectionless), or ICMP (ping/diagnostics).",
    "service":                     "Target service: HTTP (web), FTP (file transfer), SSH (secure shell), telnet, etc.",
    "flag":                        "TCP handshake state. SF=normal complete. S0=SYN only (never completed). REJ=rejected. S0 is a hallmark of SYN floods.",
    "src_bytes":                   "Bytes sent from source to destination. Zero src_bytes with high count = pure flooding.",
    "dst_bytes":                   "Bytes returned from destination to source. Zero dst_bytes means server never responded — classic DoS sign.",
    "land":                        "1 if source and destination IP/port are identical — spoofed loopback attack indicator.",
    "wrong_fragment":              "Count of incorrect IP fragments. Used in evasion and fragmentation attacks.",
    "urgent":                      "Count of urgent (OOB) packets. Rarely used legitimately; common in older exploits.",
    "hot":                         "Count of 'hot' indicators: access to sensitive dirs, executing commands, etc.",
    "num_failed_logins":           "Failed login attempts in this session. High values = brute force.",
    "logged_in":                   "1 if user successfully authenticated. Attacks often proceed without authentication.",
    "num_compromised":             "Conditions indicating system compromise (e.g. illegal file access, error conditions).",
    "root_shell":                  "1 if a root shell was obtained. Extremely high severity — indicates full system compromise.",
    "su_attempted":                "1 if 'su root' was attempted. Privilege escalation indicator.",
    "num_root":                    "Root-level operations performed during the session.",
    "num_file_creations":          "Files created during the session. Unexpected file creation = possible malware drop.",
    "num_shells":                  "Number of shell prompts spawned. Multiple shells = suspicious.",
    "num_access_files":            "Operations on access control files (passwd, shadow, etc.).",
    "num_outbound_cmds":           "Outbound FTP commands. Non-zero in FTP sessions only.",
    "is_host_login":               "1 if login is to a host account.",
    "is_guest_login":              "1 if login is as guest. Guest access is a risk factor.",
    "count":                       "Connections to the same host in the last 2 seconds. Very high = flooding.",
    "srv_count":                   "Connections to the same service in the last 2 seconds. High = service-targeted flood.",
    "serror_rate":                 "Fraction of connections with SYN errors. Near 1.0 = SYN flood in progress.",
    "srv_serror_rate":             "SYN error rate per service. Confirms service-specific SYN flooding.",
    "rerror_rate":                 "Fraction of connections with REJ errors. High = port scanning or blocked flood.",
    "srv_rerror_rate":             "REJ error rate per service.",
    "same_srv_rate":               "Fraction of connections to the same service. Near 1.0 = focused attack; near 0 = scanning.",
    "diff_srv_rate":               "Fraction of connections to different services. High = port scan or multi-service probe.",
    "srv_diff_host_rate":          "Fraction of connections to different hosts for the same service.",
    "dst_host_count":              "Connections to the destination host (longer window). High = sustained targeting.",
    "dst_host_srv_count":          "Connections to this service at the destination host.",
    "dst_host_same_srv_rate":      "Fraction of same-service connections at destination. Near 0 = service scanning.",
    "dst_host_diff_srv_rate":      "Fraction of different-service connections at destination. High = host-wide scan.",
    "dst_host_same_src_port_rate": "Fraction using the same source port. High = source-port-fixed scanning.",
    "dst_host_srv_diff_host_rate": "Fraction from different hosts to same service. High = distributed attack.",
    "dst_host_serror_rate":        "SYN error rate at destination host level. High = host is being SYN flooded.",
    "dst_host_srv_serror_rate":    "SYN error rate per service at destination. Confirms service-specific flood.",
    "dst_host_rerror_rate":        "REJ error rate at destination host level.",
    "dst_host_srv_rerror_rate":    "REJ error rate per service at destination.",
}

# =============================================================================
# ATTACK TEMPLATES
# =============================================================================
ATTACK_TEMPLATES = {
    "Random Traffic": {
        "_description": "PLACEHOLDER — overwritten at generation time.",
        "_random": True,
    },
    "DDoS — SYN Flood": {
        "_description": (
            "**What it is:** TCP SYN Flood denial-of-service (dataset label: *neptune*).\n\n"
            "**How it works:** Attacker sends thousands of SYN packets but never completes "
            "the TCP handshake. The victim's connection table fills up, blocking legitimate users.\n\n"
            "**Key indicators:** `flag=S0` (SYN never acknowledged), `serror_rate=1.0`, "
            "`count=500` (flooding), `src_bytes=0` and `dst_bytes=0` (no real data exchange)."
        ),
        "duration":0,"protocol_type":"tcp","service":"http","flag":"S0",
        "src_bytes":0,"dst_bytes":0,"logged_in":0,
        "count":500,"srv_count":500,"serror_rate":1.0,"srv_serror_rate":1.0,
        "same_srv_rate":1.0,"diff_srv_rate":0.0,
        "dst_host_count":255,"dst_host_srv_count":255,
        "dst_host_serror_rate":1.0,"dst_host_srv_serror_rate":1.0,
    },
    "DDoS — ICMP Amplification": {
        "_description": (
            "**What it is:** ICMP amplification attack (dataset label: *smurf*).\n\n"
            "**How it works:** Spoofed ICMP Echo Requests are broadcast to a network with the "
            "victim's IP as source. All devices reply to the victim simultaneously.\n\n"
            "**Key indicators:** `protocol_type=icmp`, massive `count` (511), `dst_bytes=0` "
            "(victim overwhelmed, can't respond)."
        ),
        "duration":0,"protocol_type":"icmp","service":"ecr_i","flag":"SF",
        "src_bytes":1032,"dst_bytes":0,"logged_in":0,
        "count":511,"srv_count":511,"serror_rate":0.0,
        "same_srv_rate":1.0,"diff_srv_rate":0.0,
        "dst_host_count":255,"dst_host_srv_count":255,
    },
    "DoS — HTTP Back Attack": {
        "_description": (
            "**What it is:** Apache 'back' DoS attack — overwhelms a web server with "
            "malformed HTTP requests (dataset label: *back*).\n\n"
            "**How it works:** The attacker sends massive HTTP requests with backslash-heavy "
            "URLs to Apache servers. Each request triggers expensive processing, consuming CPU "
            "until the server becomes unresponsive.\n\n"
            "**Key indicators:** `protocol_type=tcp`, `service=http`, very large `src_bytes` (54,000+), "
            "high `count` (300–400 connections), `logged_in=1`, `flag=SF` (full connections — "
            "unlike SYN flood, this completes the handshake deliberately).\n\n"
            "> ⚠️ **Why not UDP?** NSL-KDD has no 'UDP flood' class. "
            "The *back* attack is TCP/HTTP-based. A plain `udp + rerror_rate=1.0` packet "
            "matches no attack the model was trained on — so it correctly calls it normal."
        ),
        "duration":0,"protocol_type":"tcp","service":"http","flag":"SF",
        "src_bytes":54540,"dst_bytes":8314,"logged_in":1,
        "count":400,"srv_count":400,"serror_rate":0.0,"srv_serror_rate":0.0,
        "rerror_rate":0.0,"srv_rerror_rate":0.0,
        "same_srv_rate":1.0,"diff_srv_rate":0.0,
        "dst_host_count":255,"dst_host_srv_count":255,
        "dst_host_same_srv_rate":1.0,"dst_host_diff_srv_rate":0.0,
    },
    "DoS — Teardrop (UDP Fragmentation)": {
        "_description": (
            "**What it is:** Teardrop attack — a UDP fragmentation exploit that crashes "
            "older operating systems (dataset label: *teardrop*).\n\n"
            "**How it works:** The attacker sends overlapping, malformed UDP fragments. "
            "When the victim's IP stack tries to reassemble them, it crashes or hangs "
            "due to the invalid fragment offsets. Classic against Windows 95/NT and early Linux.\n\n"
            "**Key indicators:** `protocol_type=udp`, `service=private`, "
            "`wrong_fragment=3` (the defining signature — malformed fragments), "
            "`dst_bytes=0` (target unable to respond), `src_bytes=28` (tiny fragment payload).\n\n"
            "> ℹ️ **This is the real UDP-based DoS in NSL-KDD.** "
            "The model identifies teardrop via `wrong_fragment`, not just high count or rerror_rate."
        ),
        "duration":0,"protocol_type":"udp","service":"private","flag":"SF",
        "src_bytes":28,"dst_bytes":0,"logged_in":0,
        "wrong_fragment":3,
        "count":100,"srv_count":100,"serror_rate":0.0,"srv_serror_rate":0.0,
        "rerror_rate":0.0,"srv_rerror_rate":0.0,
        "same_srv_rate":1.0,"diff_srv_rate":0.0,
        "dst_host_count":255,"dst_host_srv_count":255,
        "dst_host_same_srv_rate":1.0,"dst_host_diff_srv_rate":0.0,
    },
    "Port Scan (Horizontal)": {
        "_description": (
            "**What it is:** Horizontal port sweep across many hosts (dataset label: *portsweep*).\n\n"
            "**How it works:** Attacker probes the same port on many IP addresses to find live/vulnerable hosts. "
            "Common first reconnaissance step before targeted exploitation.\n\n"
            "**Key indicators:** `dst_host_count=255` (many hosts), `diff_srv_rate=0.95` (touching many services), "
            "`flag=REJ` (connections rejected), near-zero bytes."
        ),
        "duration":0,"protocol_type":"tcp","service":"ftp","flag":"REJ",
        "src_bytes":0,"dst_bytes":0,"logged_in":0,
        "count":1,"srv_count":1,"serror_rate":0.0,
        "same_srv_rate":0.05,"diff_srv_rate":0.95,
        "dst_host_count":255,"dst_host_srv_count":10,
        "dst_host_same_srv_rate":0.04,"dst_host_diff_srv_rate":0.96,
    },
    "Port Scan (Vertical / Service Probe)": {
        "_description": (
            "**What it is:** Vertical port scan — many ports on one host (dataset label: *satan*).\n\n"
            "**How it works:** Attacker scans many ports on a single target to enumerate running services. "
            "Tools like Nmap are commonly used. Maps the attack surface before exploitation.\n\n"
            "**Key indicators:** `diff_srv_rate=0.9` (many services), short duration, "
            "low bytes per connection, `flag=REJ`."
        ),
        "duration":0,"protocol_type":"tcp","service":"private","flag":"REJ",
        "src_bytes":0,"dst_bytes":0,"logged_in":0,
        "count":10,"srv_count":3,"serror_rate":0.3,"srv_serror_rate":0.3,
        "same_srv_rate":0.1,"diff_srv_rate":0.9,
        "dst_host_count":30,"dst_host_srv_count":5,
    },
    "Brute Force Login (SSH/FTP)": {
        "_description": (
            "**What it is:** Credential brute-force against SSH, FTP, or Telnet (dataset label: *guess_passwd*).\n\n"
            "**How it works:** Automated tools try thousands of username/password combinations rapidly. "
            "Each failure is logged. Without rate-limiting the service can be cracked.\n\n"
            "**Key indicators:** `num_failed_logins=10`, `logged_in=0` (all failing), "
            "`srv_count=100` (hammering same service), short duration per connection."
        ),
        "duration":1,"protocol_type":"tcp","service":"ftp","flag":"SF",
        "src_bytes":250,"dst_bytes":150,"logged_in":0,
        "num_failed_logins":10,"count":100,"srv_count":100,
        "serror_rate":0.0,"rerror_rate":0.1,
        "same_srv_rate":1.0,"diff_srv_rate":0.0,
        "dst_host_count":100,"dst_host_srv_count":100,
    },
    "Remote-to-Local Exploit (R2L)": {
        "_description": (
            "**What it is:** Remote-to-Local attack gaining unauthorized local access (dataset label: *ftp_write*).\n\n"
            "**How it works:** Attacker exploits misconfigured or vulnerable services (e.g. anonymous FTP write) "
            "to upload malicious files like cron jobs or web shells onto the target machine.\n\n"
            "**Key indicators:** `num_file_creations=3` (files written to server), "
            "`logged_in=0` (unauthorized), actual data in `dst_bytes` (upload occurred)."
        ),
        "duration":5,"protocol_type":"tcp","service":"ftp","flag":"SF",
        "src_bytes":512,"dst_bytes":1024,"logged_in":0,
        "num_file_creations":3,"num_failed_logins":1,
        "count":2,"srv_count":2,
        "serror_rate":0.0,"same_srv_rate":1.0,"diff_srv_rate":0.0,
        "dst_host_count":5,"dst_host_srv_count":5,
    },
    "Privilege Escalation (Root Exploit)": {
        "_description": (
            "**What it is:** Local privilege escalation to root, often via buffer overflow (dataset label: *buffer_overflow*).\n\n"
            "**How it works:** An attacker already inside the system exploits a vulnerable program — overflows a memory buffer, "
            "injects shellcode, and escalates from low-privilege user to root. Full system control achieved.\n\n"
            "**Key indicators:** `root_shell=1` (root shell spawned — critical), `su_attempted=1`, "
            "`num_root=5`, `num_shells=2` (multiple shell invocations)."
        ),
        "duration":3,"protocol_type":"tcp","service":"telnet","flag":"SF",
        "src_bytes":1200,"dst_bytes":900,"logged_in":1,
        "root_shell":1,"su_attempted":1,"num_root":5,
        "num_compromised":3,"num_shells":2,"num_file_creations":1,
        "count":1,"srv_count":1,
        "serror_rate":0.0,"same_srv_rate":1.0,
        "dst_host_count":2,"dst_host_srv_count":2,
    },
    "Web Attack — HTTP Exploit": {
        "_description": (
            "**What it is:** Web application attack targeting vulnerable CGI scripts (dataset label: *phf*).\n\n"
            "**How it works:** Attacker crafts malicious HTTP requests exploiting poor input sanitization. "
            "Classic example: `/cgi-bin/phf` script allowed arbitrary command execution on early web servers.\n\n"
            "**Key indicators:** `hot=2` (sensitive system areas accessed), `num_access_files=2` "
            "(reading restricted files), `num_compromised=1`, `logged_in=0`."
        ),
        "duration":1,"protocol_type":"tcp","service":"http","flag":"SF",
        "src_bytes":338,"dst_bytes":0,"logged_in":0,
        "hot":2,"num_compromised":1,"num_access_files":2,
        "count":1,"srv_count":1,
        "serror_rate":0.0,"same_srv_rate":1.0,
        "dst_host_count":2,"dst_host_srv_count":2,
    },
}

# =============================================================================
# PREPROCESS
# =============================================================================
def preprocess_proposed(packet):
    full = {f: 0 for f in RAW_FEATURES}
    full.update({k: v for k, v in packet.items() if not k.startswith("_")})
    df = pd.DataFrame([full])
    X  = preprocessor.transform(df)
    if hasattr(X, "toarray"): X = X.toarray()
    names = preprocessor.get_feature_names_out()
    return pd.DataFrame(X, columns=names)

def preprocess_baseline(packet):
    full = {f: 0 for f in baseline_features}
    full.update({k: v for k, v in packet.items()
                 if not k.startswith("_") and k in baseline_features})
    for col, le in baseline_encoders.items():
        val = str(full.get(col, ""))
        full[col] = int(le.transform([val])[0]) if val in le.classes_ else 0
    return pd.DataFrame([full])[baseline_features]

# =============================================================================
# PREDICT
# =============================================================================
def predict_proposed(X):
    p_xgb  = xgb_model.predict_proba(X)[0]
    p_rf   = rf_model.predict_proba(X)[0]
    p_et   = et_model.predict_proba(X)[0]
    p_lgbm = lgbm_model.predict_proba(X)[0]
    ens    = 0.40*p_xgb + 0.25*p_lgbm + 0.20*p_rf + 0.15*p_et
    idx    = int(np.argmax(ens))
    members = {
        "XGBoost (40%)":       p_xgb,
        "LightGBM (25%)":      p_lgbm,
        "Random Forest (20%)": p_rf,
        "Extra Trees (15%)":   p_et,
    }
    return {"prediction": class_names[idx], "confidence": float(ens[idx]),
            "verdict": "ATTACK" if idx != NORMAL_IDX else "NORMAL",
            "ensemble_probs": ens, "pred_idx": idx, "members": members}

def predict_baseline(X):
    proba = baseline_model.predict_proba(X)[0]
    atk   = float(proba[1]); nrm = float(proba[0])
    v     = "ATTACK" if atk >= baseline_threshold else "NORMAL"
    return {"verdict": v, "confidence": atk if v=="ATTACK" else nrm,
            "attack_prob": atk, "normal_prob": nrm}

# =============================================================================
# SHAP — FIXED
#
# For multiclass XGBoost, shap_values shape = (1, n_features, n_classes)
# sv[0, :, c] = contribution of each feature toward class c's raw score.
# Positive = pushed raw score of class c UP (more likely to be class c).
# Negative = pushed raw score of class c DOWN (less likely to be class c).
#
# The direction label must say "toward <predicted class>" not "toward ATTACK".
# =============================================================================
def compute_shap(X, pred_idx):
    sv  = np.array(explainer.shap_values(X))    # (1, 120, 23)
    ev  = np.array(EV)                           # (23,)

    vals_pred   = sv[0, :, pred_idx]             # toward predicted class
    vals_normal = sv[0, :, NORMAL_IDX]           # toward normal class

    ev_pred   = float(ev[pred_idx])
    ev_normal = float(ev[NORMAL_IDX])

    score_pred   = ev_pred   + float(vals_pred.sum())
    score_normal = ev_normal + float(vals_normal.sum())

    feat_names = list(X.columns)

    def top10(vals):
        pairs = sorted(zip(feat_names, vals.tolist()),
                       key=lambda x: abs(x[1]), reverse=True)[:10]
        return [(n.split("__")[-1], float(v)) for n, v in pairs]

    return {
        "pred_class":    class_names[pred_idx],
        "vals_pred":     top10(vals_pred),
        "vals_normal":   top10(vals_normal),
        "ev_pred":       ev_pred,
        "ev_normal":     ev_normal,
        "score_pred":    score_pred,
        "score_normal":  score_normal,
        "gap":           score_pred - score_normal,
    }

# =============================================================================
# RANDOM NORMAL PACKET GENERATOR
# Produces realistic variety: different services, protocols, byte sizes,
# connection counts — all within normal ranges so the model classifies it clean.
# =============================================================================

# =============================================================================
# RANDOM TRAFFIC GENERATOR
# Each call randomly picks a traffic type — ~40% normal, ~60% attack spread
# across DoS, probe, brute-force, R2L, and privilege escalation categories.
# Returns (packet_dict, description_string) so the UI can update accordingly.
# =============================================================================

NORMAL_SERVICES_TCP  = ["http", "ftp", "ssh", "smtp", "pop_3", "imap4",
                         "domain_u", "telnet", "finger", "nntp", "courier",
                         "netbios_ssn", "exec", "login", "shell", "sql_net"]
NORMAL_SERVICES_UDP  = ["domain_u", "ntp_u", "tftp_u", "other"]
NORMAL_SERVICES_ICMP = ["eco_i", "ecr_i"]

def _make_normal():
    proto = random.choices(["tcp","udp","icmp"], weights=[0.75,0.15,0.10])[0]
    if proto == "tcp":
        service = random.choice(NORMAL_SERVICES_TCP)
        flag    = random.choices(["SF","S1","S2"], weights=[0.92,0.05,0.03])[0]
        logged  = random.choices([1, 0], weights=[0.85, 0.15])[0]
        src_b   = random.randint(50, 8000)
        dst_b   = random.randint(100, 20000)
    elif proto == "udp":
        service = random.choice(NORMAL_SERVICES_UDP)
        flag = "SF"; logged = 0
        src_b = random.randint(28, 512); dst_b = random.randint(0, 512)
    else:
        service = random.choice(NORMAL_SERVICES_ICMP)
        flag = "SF"; logged = 0
        src_b = random.choice([8,28,40,56,64]); dst_b = random.choice([8,28,40,56,64])

    serror = round(random.uniform(0.0, 0.04), 4)
    rerror = round(random.uniform(0.0, 0.04), 4)
    same   = round(random.uniform(0.7, 1.0),  4)
    count  = random.randint(1, 20)
    dhc    = random.randint(1, 30)

    pkt = {
        "duration": random.randint(0,60), "protocol_type": proto,
        "service": service, "flag": flag,
        "src_bytes": src_b, "dst_bytes": dst_b,
        "land":0,"wrong_fragment":0,"urgent":0,
        "hot": random.choices([0,1],[0.95,0.05])[0],
        "num_failed_logins":0, "logged_in": logged,
        "num_compromised":0,"root_shell":0,"su_attempted":0,"num_root":0,
        "num_file_creations": random.choices([0,1],[0.92,0.08])[0],
        "num_shells":0,"num_access_files":0,"num_outbound_cmds":0,
        "is_host_login":0,"is_guest_login":0,
        "count": count, "srv_count": random.randint(1, count),
        "serror_rate":serror,"srv_serror_rate":serror,
        "rerror_rate":rerror,"srv_rerror_rate":rerror,
        "same_srv_rate":same,"diff_srv_rate":round(1-same,4),
        "srv_diff_host_rate":round(random.uniform(0,0.15),4),
        "dst_host_count": dhc, "dst_host_srv_count": random.randint(1,dhc),
        "dst_host_same_srv_rate":same,"dst_host_diff_srv_rate":round(1-same,4),
        "dst_host_same_src_port_rate":round(random.uniform(0,0.5),4),
        "dst_host_srv_diff_host_rate":round(random.uniform(0,0.1),4),
        "dst_host_serror_rate":serror,"dst_host_srv_serror_rate":serror,
        "dst_host_rerror_rate":rerror,"dst_host_srv_rerror_rate":rerror,
    }
    desc = (
        "**What it is:** Legitimate network traffic.\n\n"
        f"**Randomly generated:** `{proto.upper()}` · service `{service}` · flag `{flag}`\n\n"
        "**Why it's normal:** Low error rates, clean handshake, balanced byte flow, "
        "no suspicious login failures or privilege operations."
    )
    return pkt, desc

def _make_syn_flood():
    serror = round(random.uniform(0.95, 1.0), 4)
    count  = random.randint(450, 511)
    pkt = {
        "duration":0,"protocol_type":"tcp",
        "service": random.choice(["http","ftp","smtp","telnet"]),
        "flag":"S0","src_bytes":0,"dst_bytes":0,"logged_in":0,
        "land":0,"wrong_fragment":0,"urgent":0,"hot":0,
        "num_failed_logins":0,"num_compromised":0,"root_shell":0,
        "su_attempted":0,"num_root":0,"num_file_creations":0,
        "num_shells":0,"num_access_files":0,"num_outbound_cmds":0,
        "is_host_login":0,"is_guest_login":0,
        "count": count,"srv_count": count,
        "serror_rate":serror,"srv_serror_rate":serror,
        "rerror_rate":0.0,"srv_rerror_rate":0.0,
        "same_srv_rate":1.0,"diff_srv_rate":0.0,"srv_diff_host_rate":0.0,
        "dst_host_count":255,"dst_host_srv_count":255,
        "dst_host_same_srv_rate":1.0,"dst_host_diff_srv_rate":0.0,
        "dst_host_same_src_port_rate":round(random.uniform(0,0.1),4),
        "dst_host_srv_diff_host_rate":0.0,
        "dst_host_serror_rate":serror,"dst_host_srv_serror_rate":serror,
        "dst_host_rerror_rate":0.0,"dst_host_srv_rerror_rate":0.0,
    }
    desc = (
        "**What it is:** TCP SYN Flood DoS attack (dataset label: *neptune*).\n\n"
        "**How it works:** Attacker floods target with half-open TCP connections — "
        "SYN sent but handshake never completed — exhausting the server's connection table.\n\n"
        f"**Generated values:** `flag=S0`, `serror_rate={serror}`, `count={count}`, "
        "`src_bytes=0`, `dst_bytes=0` (no real data exchange)."
    )
    return pkt, desc

def _make_icmp_flood():
    count = random.randint(480, 511)
    src_b = random.choice([1032, 1024, 512, 64])
    pkt = {
        "duration":0,"protocol_type":"icmp","service":"ecr_i","flag":"SF",
        "src_bytes":src_b,"dst_bytes":0,"logged_in":0,
        "land":0,"wrong_fragment":0,"urgent":0,"hot":0,
        "num_failed_logins":0,"num_compromised":0,"root_shell":0,
        "su_attempted":0,"num_root":0,"num_file_creations":0,
        "num_shells":0,"num_access_files":0,"num_outbound_cmds":0,
        "is_host_login":0,"is_guest_login":0,
        "count":count,"srv_count":count,
        "serror_rate":0.0,"srv_serror_rate":0.0,
        "rerror_rate":0.0,"srv_rerror_rate":0.0,
        "same_srv_rate":1.0,"diff_srv_rate":0.0,"srv_diff_host_rate":0.0,
        "dst_host_count":255,"dst_host_srv_count":255,
        "dst_host_same_srv_rate":1.0,"dst_host_diff_srv_rate":0.0,
        "dst_host_same_src_port_rate":round(random.uniform(0,0.05),4),
        "dst_host_srv_diff_host_rate":0.0,
        "dst_host_serror_rate":0.0,"dst_host_srv_serror_rate":0.0,
        "dst_host_rerror_rate":0.0,"dst_host_srv_rerror_rate":0.0,
    }
    desc = (
        "**What it is:** ICMP Amplification / Smurf DDoS attack (dataset label: *smurf*).\n\n"
        "**How it works:** Spoofed ICMP Echo Requests are broadcast to the network "
        "with the victim's IP as source; every device replies, overwhelming the victim.\n\n"
        f"**Generated values:** `protocol=icmp`, `count={count}`, `src_bytes={src_b}`, "
        "`dst_bytes=0` (victim unable to respond)."
    )
    return pkt, desc

def _make_portsweep():
    rerror = round(random.uniform(0.0, 0.3), 4)
    dcount = random.randint(200, 255)
    pkt = {
        "duration":0,"protocol_type":"tcp",
        "service": random.choice(["ftp","http","ssh","smtp"]),
        "flag": random.choice(["REJ","S0"]),
        "src_bytes":0,"dst_bytes":0,"logged_in":0,
        "land":0,"wrong_fragment":0,"urgent":0,"hot":0,
        "num_failed_logins":0,"num_compromised":0,"root_shell":0,
        "su_attempted":0,"num_root":0,"num_file_creations":0,
        "num_shells":0,"num_access_files":0,"num_outbound_cmds":0,
        "is_host_login":0,"is_guest_login":0,
        "count":1,"srv_count":1,
        "serror_rate":0.0,"srv_serror_rate":0.0,
        "rerror_rate":rerror,"srv_rerror_rate":rerror,
        "same_srv_rate":round(random.uniform(0.02,0.08),4),
        "diff_srv_rate":round(random.uniform(0.88,0.98),4),
        "srv_diff_host_rate":round(random.uniform(0.0,0.1),4),
        "dst_host_count": dcount,"dst_host_srv_count": random.randint(5,20),
        "dst_host_same_srv_rate":round(random.uniform(0.02,0.08),4),
        "dst_host_diff_srv_rate":round(random.uniform(0.88,0.98),4),
        "dst_host_same_src_port_rate":round(random.uniform(0,0.05),4),
        "dst_host_srv_diff_host_rate":round(random.uniform(0,0.1),4),
        "dst_host_serror_rate":0.0,"dst_host_srv_serror_rate":0.0,
        "dst_host_rerror_rate":rerror,"dst_host_srv_rerror_rate":rerror,
    }
    desc = (
        "**What it is:** Horizontal port sweep / reconnaissance (dataset label: *portsweep*).\n\n"
        "**How it works:** Attacker probes the same port across many IP addresses "
        "looking for live/vulnerable hosts — the first step before targeted exploitation.\n\n"
        f"**Generated values:** `dst_host_count={dcount}`, `diff_srv_rate≈0.9+`, "
        "`count=1` per host (sampling behavior), connections rejected or incomplete."
    )
    return pkt, desc

def _make_satan():
    serror = round(random.uniform(0.2, 0.5), 4)
    pkt = {
        "duration":0,"protocol_type":"tcp",
        "service": random.choice(["private","ftp","http","telnet"]),
        "flag": random.choice(["REJ","S0"]),
        "src_bytes":random.randint(0,30),"dst_bytes":0,"logged_in":0,
        "land":0,"wrong_fragment":0,"urgent":0,"hot":0,
        "num_failed_logins":0,"num_compromised":0,"root_shell":0,
        "su_attempted":0,"num_root":0,"num_file_creations":0,
        "num_shells":0,"num_access_files":0,"num_outbound_cmds":0,
        "is_host_login":0,"is_guest_login":0,
        "count": random.randint(5,15),"srv_count": random.randint(2,6),
        "serror_rate":serror,"srv_serror_rate":serror,
        "rerror_rate":round(random.uniform(0.1,0.4),4),
        "srv_rerror_rate":round(random.uniform(0.1,0.4),4),
        "same_srv_rate":round(random.uniform(0.05,0.2),4),
        "diff_srv_rate":round(random.uniform(0.75,0.95),4),
        "srv_diff_host_rate":round(random.uniform(0,0.2),4),
        "dst_host_count": random.randint(20,60),
        "dst_host_srv_count": random.randint(3,10),
        "dst_host_same_srv_rate":round(random.uniform(0.05,0.2),4),
        "dst_host_diff_srv_rate":round(random.uniform(0.75,0.95),4),
        "dst_host_same_src_port_rate":round(random.uniform(0,0.1),4),
        "dst_host_srv_diff_host_rate":round(random.uniform(0,0.2),4),
        "dst_host_serror_rate":serror,"dst_host_srv_serror_rate":serror,
        "dst_host_rerror_rate":round(random.uniform(0.1,0.4),4),
        "dst_host_srv_rerror_rate":round(random.uniform(0.1,0.4),4),
    }
    desc = (
        "**What it is:** Vertical service probe (dataset label: *satan*).\n\n"
        "**How it works:** Attacker scans many ports on one target to map which services "
        "are running. Used to identify attack vectors before exploitation.\n\n"
        f"**Generated values:** `diff_srv_rate≈{pkt['diff_srv_rate']}` (many services probed), "
        "short duration, minimal bytes, connections mostly rejected."
    )
    return pkt, desc

def _make_brute_force():
    fails  = random.randint(6, 15)
    count  = random.randint(80, 120)
    pkt = {
        "duration": random.randint(0,3),
        "protocol_type":"tcp",
        "service": random.choice(["ftp","telnet","ssh","imap4","pop_3"]),
        "flag":"SF","src_bytes":random.randint(200,400),
        "dst_bytes":random.randint(100,250),"logged_in":0,
        "land":0,"wrong_fragment":0,"urgent":0,"hot":0,
        "num_failed_logins": fails,"num_compromised":0,
        "root_shell":0,"su_attempted":0,"num_root":0,
        "num_file_creations":0,"num_shells":0,"num_access_files":0,
        "num_outbound_cmds":0,"is_host_login":0,"is_guest_login":0,
        "count": count,"srv_count": count,
        "serror_rate":0.0,"srv_serror_rate":0.0,
        "rerror_rate":round(random.uniform(0.05,0.15),4),
        "srv_rerror_rate":round(random.uniform(0.05,0.15),4),
        "same_srv_rate":1.0,"diff_srv_rate":0.0,"srv_diff_host_rate":0.0,
        "dst_host_count": count,"dst_host_srv_count": count,
        "dst_host_same_srv_rate":1.0,"dst_host_diff_srv_rate":0.0,
        "dst_host_same_src_port_rate":round(random.uniform(0.8,1.0),4),
        "dst_host_srv_diff_host_rate":0.0,
        "dst_host_serror_rate":0.0,"dst_host_srv_serror_rate":0.0,
        "dst_host_rerror_rate":round(random.uniform(0.05,0.15),4),
        "dst_host_srv_rerror_rate":round(random.uniform(0.05,0.15),4),
    }
    desc = (
        "**What it is:** Credential brute-force attack (dataset label: *guess_passwd*).\n\n"
        "**How it works:** Automated tools rapidly try username/password combinations "
        "against a login service until one succeeds.\n\n"
        f"**Generated values:** `num_failed_logins={fails}`, `logged_in=0`, "
        f"`count={count}` (hammering same service), short connection duration."
    )
    return pkt, desc

def _make_root_exploit():
    num_root = random.randint(3, 8)
    pkt = {
        "duration": random.randint(1,5),
        "protocol_type":"tcp",
        "service": random.choice(["telnet","ftp","ssh"]),
        "flag":"SF","src_bytes":random.randint(800,2000),
        "dst_bytes":random.randint(500,1500),"logged_in":1,
        "land":0,"wrong_fragment":0,"urgent":0,"hot":random.randint(1,4),
        "num_failed_logins":0,"num_compromised":random.randint(1,5),
        "root_shell":1,"su_attempted":1,"num_root": num_root,
        "num_file_creations":random.randint(0,3),
        "num_shells":random.randint(1,3),"num_access_files":random.randint(0,2),
        "num_outbound_cmds":0,"is_host_login":0,"is_guest_login":0,
        "count":random.randint(1,3),"srv_count":random.randint(1,3),
        "serror_rate":0.0,"srv_serror_rate":0.0,
        "rerror_rate":0.0,"srv_rerror_rate":0.0,
        "same_srv_rate":1.0,"diff_srv_rate":0.0,"srv_diff_host_rate":0.0,
        "dst_host_count":random.randint(1,5),"dst_host_srv_count":random.randint(1,5),
        "dst_host_same_srv_rate":1.0,"dst_host_diff_srv_rate":0.0,
        "dst_host_same_src_port_rate":round(random.uniform(0,0.3),4),
        "dst_host_srv_diff_host_rate":0.0,
        "dst_host_serror_rate":0.0,"dst_host_srv_serror_rate":0.0,
        "dst_host_rerror_rate":0.0,"dst_host_srv_rerror_rate":0.0,
    }
    desc = (
        "**What it is:** Privilege escalation / root exploit (dataset label: *buffer_overflow*).\n\n"
        "**How it works:** Attacker exploits a vulnerable program to overflow a memory buffer, "
        "inject shellcode, and escalate to root — gaining full system control.\n\n"
        f"**Generated values:** `root_shell=1` ⚠️, `su_attempted=1`, "
        f"`num_root={num_root}`, `num_shells={pkt['num_shells']}` (multiple shells spawned)."
    )
    return pkt, desc

def _make_r2l():
    files = random.randint(1, 4)
    pkt = {
        "duration": random.randint(2,8),
        "protocol_type":"tcp","service":"ftp","flag":"SF",
        "src_bytes":random.randint(300,800),
        "dst_bytes":random.randint(800,2000),"logged_in":0,
        "land":0,"wrong_fragment":0,"urgent":0,"hot":0,
        "num_failed_logins":random.randint(0,2),
        "num_compromised":0,"root_shell":0,"su_attempted":0,"num_root":0,
        "num_file_creations": files,"num_shells":0,"num_access_files":0,
        "num_outbound_cmds":0,"is_host_login":0,"is_guest_login":0,
        "count":random.randint(1,4),"srv_count":random.randint(1,4),
        "serror_rate":0.0,"srv_serror_rate":0.0,
        "rerror_rate":0.0,"srv_rerror_rate":0.0,
        "same_srv_rate":1.0,"diff_srv_rate":0.0,"srv_diff_host_rate":0.0,
        "dst_host_count":random.randint(2,8),"dst_host_srv_count":random.randint(2,8),
        "dst_host_same_srv_rate":1.0,"dst_host_diff_srv_rate":0.0,
        "dst_host_same_src_port_rate":round(random.uniform(0,0.4),4),
        "dst_host_srv_diff_host_rate":0.0,
        "dst_host_serror_rate":0.0,"dst_host_srv_serror_rate":0.0,
        "dst_host_rerror_rate":0.0,"dst_host_srv_rerror_rate":0.0,
    }
    desc = (
        "**What it is:** Remote-to-Local file write exploit (dataset label: *ftp_write*).\n\n"
        "**How it works:** Attacker uses anonymous or misconfigured FTP access to upload "
        "malicious files (cron jobs, web shells) onto the victim machine.\n\n"
        f"**Generated values:** `num_file_creations={files}` (files written), "
        "`logged_in=0` (unauthorized access), `dst_bytes` shows data successfully uploaded."
    )
    return pkt, desc

# Traffic type pool — weights control how often each appears
_TRAFFIC_POOL = [
    (_make_normal,      0.38),   # ~38% normal
    (_make_syn_flood,   0.14),   # ~14% SYN flood
    (_make_icmp_flood,  0.12),   # ~12% ICMP flood
    (_make_portsweep,   0.10),   # ~10% port sweep
    (_make_satan,       0.09),   # ~9%  service probe
    (_make_brute_force, 0.09),   # ~9%  brute force
    (_make_root_exploit,0.04),   # ~4%  root exploit
    (_make_r2l,         0.04),   # ~4%  R2L upload
]

def make_random_traffic():
    """Pick a random traffic type and generate a packet + description."""
    fns, weights = zip(*_TRAFFIC_POOL)
    fn = random.choices(fns, weights=weights)[0]
    return fn()


# =============================================================================
# SIDEBAR
# =============================================================================
st.sidebar.header("🧪 Attack Simulator")
attack_type = st.sidebar.selectbox("Simulation Type", list(ATTACK_TEMPLATES.keys()))
st.sidebar.markdown("---")
if st.sidebar.button("⚡ Generate & Analyze Packet", use_container_width=True):
    tmpl = ATTACK_TEMPLATES[attack_type]
    if tmpl.get("_random"):
        # Fully random — could be normal OR any attack type
        packet, generated_desc = make_random_traffic()
        desc_to_use = generated_desc
    else:
        # Fixed attack template with small jitter
        packet = {}
        for k, v in tmpl.items():
            if k.startswith("_"): continue
            if isinstance(v, int):     packet[k] = max(0, v + random.randint(-2, 2))
            elif isinstance(v, float): packet[k] = round(max(0.0, min(1.0, v + random.uniform(-0.02, 0.02))), 4)
            else:                      packet[k] = v
        desc_to_use = tmpl.get("_description", "")
    st.session_state["packet"]      = packet
    st.session_state["description"] = desc_to_use
    st.session_state["sim_name"]    = attack_type

# =============================================================================
# MAIN
# =============================================================================
if "packet" not in st.session_state:
    st.info("👈 Select a simulation from the sidebar and click **Generate & Analyze Packet**.")
    st.markdown("### Available Simulations")
    cols = st.columns(2)
    for i, (name, tmpl) in enumerate(ATTACK_TEMPLATES.items()):
        with cols[i % 2]:
            st.markdown(f"**{name}**")
            st.caption(tmpl["_description"].split("\n\n")[0].replace("**",""))
            st.markdown("---")
    st.stop()

packet      = st.session_state["packet"]
description = st.session_state["description"]
sim_name    = st.session_state["sim_name"]

display_pkt = {k: v for k, v in packet.items() if not k.startswith("_")}
with st.expander("📦 Raw Packet Features", expanded=False):
    st.json(display_pkt)

X_prop = preprocess_proposed(packet)
X_base = preprocess_baseline(packet)
rp     = predict_proposed(X_prop)
rb     = predict_baseline(X_base)
shap_d = compute_shap(X_prop, rp["pred_idx"])

is_attack = rp["verdict"] == "ATTACK"
pred_class = rp["prediction"]

# ── HEAD-TO-HEAD VERDICT ───────────────────────────────────────────────────
st.header("⚔️ Model Comparison — Head-to-Head Verdict")
cb, _, cp = st.columns([5, 1, 5])

with cb:
    st.markdown("### 🔷 Baseline Model")
    st.caption("Single binary XGBoost — attack or normal only")
    (st.error if rb["verdict"]=="ATTACK" else st.success)(
        f"{'🔴 ATTACK DETECTED' if rb['verdict']=='ATTACK' else '✅ NORMAL TRAFFIC'}"
    )
    st.metric("Confidence",  f"{rb['confidence']:.2%}")
    st.metric("Attack Prob", f"{rb['attack_prob']:.4f}")
    st.metric("Normal Prob", f"{rb['normal_prob']:.4f}")
    st.caption(f"Threshold: {baseline_threshold}")

with _:
    st.markdown("<div style='text-align:center;font-size:2rem;margin-top:60px'>VS</div>",
                unsafe_allow_html=True)

with cp:
    st.markdown("### 🔶 Proposed Model")
    st.caption("Weighted ensemble — identifies specific attack class")
    (st.error if is_attack else st.success)(
        f"{'🔴 ATTACK: ' + pred_class.upper() if is_attack else '✅ NORMAL TRAFFIC'}"
    )
    st.metric("Confidence",    f"{rp['confidence']:.2%}")
    st.metric("Attack Class",  pred_class.upper())
    st.metric("Total Classes", f"{len(class_names)} classes")

st.markdown("---")
if rb["verdict"] == rp["verdict"]:
    st.success(f"✅ **Both models AGREE — {rp['verdict']}**")
else:
    st.warning(
        f"⚠️ **Models DISAGREE** — Baseline: **{rb['verdict']}** · Proposed: **{rp['verdict']}**  \n"
        "Review the SHAP section below to understand which features drove each decision."
    )

# ── BASELINE METRICS ───────────────────────────────────────────────────────
st.header("📋 Baseline Training Metrics")
m = baseline_metrics
c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Accuracy",  f"{m.get('accuracy',0):.4f}")
c2.metric("Precision", f"{m.get('precision',0):.4f}")
c3.metric("Recall",    f"{m.get('recall',0):.4f}")
c4.metric("F1 Score",  f"{m.get('f1',0):.4f}")
c5.metric("AUC",       f"{m.get('auc',0):.4f}")

# ── ENSEMBLE VOTES ─────────────────────────────────────────────────────────
st.header("🗳️ Proposed Model — Individual Member Votes")
rows = []
for mname, probs in rp["members"].items():
    midx    = int(np.argmax(probs))
    mcls    = class_names[midx]
    mverdict= "🔴 ATTACK" if midx != NORMAL_IDX else "✅ NORMAL"
    rows.append({"Model": mname, "Verdict": mverdict,
                 "Predicted Class": mcls.upper(), "Confidence": f"{float(np.max(probs)):.2%}"})
vote_df = pd.DataFrame(rows)
st.dataframe(vote_df, use_container_width=True, hide_index=True)
atk_v = sum(1 for r in rows if "ATTACK" in r["Verdict"])
st.markdown(f"**Tally:** 🔴 {atk_v}× ATTACK · ✅ {4-atk_v}× NORMAL → Ensemble: **{rp['verdict']}**")

# ── ATTACK EXPLANATION ─────────────────────────────────────────────────────
st.header("🧠 What Is This Attack?")
st.markdown(description)

# ── SHAP — FIXED ───────────────────────────────────────────────────────────
st.header("📊 SHAP Feature Explanation")

# ── WHY THE PREDICTION IS WHAT IT IS ──────────────────────────────────────
st.markdown("---")
st.subheader("🔍 Why did the model say this?")

score_diff = shap_d["gap"]
if is_attack:
    st.markdown(
        f"The model predicted **{pred_class.upper()}** (an attack class) because its raw score "
        f"(**{shap_d['score_pred']:.2f}**) was **{abs(score_diff):.2f} points higher** than the "
        f"normal class score (**{shap_d['score_normal']:.2f}**). The larger the gap, the more "
        f"confident the model is that this is an attack."
    )
else:
    st.markdown(
        f"The model predicted **NORMAL** because the normal class raw score "
        f"(**{shap_d['score_normal']:.2f}**) was **{abs(score_diff):.2f} points higher** than the "
        f"{pred_class} attack class score (**{shap_d['score_pred']:.2f}**). "
        f"Even when individual features look suspicious, **the model compares raw scores across all "
        f"23 classes** — if normal wins that comparison, it outputs normal."
    )
    st.info(
        "💡 **This is why SHAP can show features 'pushing toward attack' but the model still says normal.**\n\n"
        "SHAP measures each feature's contribution to the *predicted class's* raw score — not to "
        "a binary attack/normal outcome. A feature can push the neptune score up by +2.0, but if "
        "the normal score is still higher overall, the model picks normal. "
        "The two tables below show both sides of that comparison."
    )

st.markdown("---")

# ── IMPORTANT CORRECTION NOTE ─────────────────────────────────────────────
st.markdown(
    "> **How to read these tables:** Positive SHAP = feature pushed the score for that class **UP**. "
    "Negative SHAP = feature pushed it **DOWN**. "
    "A feature can push the attack score up AND the normal score up at the same time — "
    "what matters is which class ends up with the higher total score."
)

col_a, col_b = st.columns(2)

# LEFT: SHAP for predicted class
with col_a:
    icon = "🔴" if is_attack else "✅"
    st.subheader(f"{icon} Features → `{pred_class.upper()}` score")
    st.caption(
        f"Base score for {pred_class}: **{shap_d['ev_pred']:.3f}** · "
        f"After features: **{shap_d['score_pred']:.3f}**"
    )
    for feat, val in shap_d["vals_pred"]:
        info = FEATURE_INFO.get(feat, "")
        if val > 0:
            label = f"⬆️ +{val:.4f} — pushed **{pred_class}** score UP"
            st.error(f"**`{feat}`** &nbsp; {label}\n\n_{info}_")
        else:
            label = f"⬇️ {val:.4f} — pushed **{pred_class}** score DOWN"
            st.success(f"**`{feat}`** &nbsp; {label}\n\n_{info}_")

# RIGHT: SHAP for normal class
with col_b:
    st.subheader("✅ Features → `NORMAL` score")
    st.caption(
        f"Base score for normal: **{shap_d['ev_normal']:.3f}** · "
        f"After features: **{shap_d['score_normal']:.3f}**"
    )
    for feat, val in shap_d["vals_normal"]:
        info = FEATURE_INFO.get(feat, "")
        if val > 0:
            label = f"⬆️ +{val:.4f} — pushed **normal** score UP"
            st.success(f"**`{feat}`** &nbsp; {label}\n\n_{info}_")
        else:
            label = f"⬇️ {val:.4f} — pushed **normal** score DOWN"
            st.error(f"**`{feat}`** &nbsp; {label}\n\n_{info}_")

# ── SCORE SUMMARY BAR ─────────────────────────────────────────────────────
st.markdown("---")
st.subheader("⚖️ Final Score Comparison")
st.markdown(
    f"| Class | Base Score | Feature Shift | **Final Score** | Winner? |\n"
    f"|---|---|---|---|---|\n"
    f"| `{pred_class}` | {shap_d['ev_pred']:.3f} | "
    f"{shap_d['score_pred']-shap_d['ev_pred']:+.3f} | "
    f"**{shap_d['score_pred']:.3f}** | "
    f"{'✅ YES' if is_attack or pred_class=='normal' else '❌'} |\n"
    f"| `normal` | {shap_d['ev_normal']:.3f} | "
    f"{shap_d['score_normal']-shap_d['ev_normal']:+.3f} | "
    f"**{shap_d['score_normal']:.3f}** | "
    f"{'✅ YES' if not is_attack else '❌'} |"
)
st.caption(
    "The model picks the class with the highest final raw score across all 23 classes. "
    "SHAP explains how features shifted each class's score from its base value."
)

# ── FULL PROBABILITY TABLE ─────────────────────────────────────────────────
st.header("📈 Full Class Probability Breakdown")
probs_df = pd.DataFrame({
    "Attack Class": [c.upper() for c in class_names],
    "Probability":  [f"{p:.4f}" for p in rp["ensemble_probs"]],
    "_sort":        rp["ensemble_probs"],
}).sort_values("_sort", ascending=False).drop(columns=["_sort"])
st.dataframe(probs_df, use_container_width=True, hide_index=True)