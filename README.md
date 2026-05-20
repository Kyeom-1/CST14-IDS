# 🔐 Explainable Intrusion Detection System (IDS)

An AI-powered Intrusion Detection System (IDS) that uses Machine Learning and SHAP explainability to classify network traffic as **Normal or Attack**, with full forensic-level reasoning for each prediction.

---

## 🚀 Features

### 🧪 Manual Packet Analysis
- Input custom network traffic features
- Get real-time prediction (Normal / Attack)
- See full SHAP-based explanations per feature

### 📡 Live IDS Simulation
- Generates synthetic network packets
- Streams real-time detection results
- Stores packet history for review
- Click any packet for forensic inspection

### 🧠 Explainable AI (SHAP)
- Feature-level impact analysis
- Shows which features push toward:
  - 🚨 Attack classification
  - ✅ Normal classification
- Makes ML decisions transparent

---

## 🧰 Tech Stack

- Python
- Streamlit
- Scikit-learn
- XGBoost
- SHAP (Explainable AI)
- Pandas / NumPy
- Matplotlib

---

## 📊 Dataset

This project is based on the **NSL-KDD dataset**, a widely used benchmark dataset for Intrusion Detection Systems.

It includes network traffic features such as:
- Protocol type
- Service type
- Connection duration
- Error rates
- Traffic statistics

---

## 🧠 Model Overview

- Model Type: Gradient Boosted Trees (XGBoost / ML pipeline)
- Input: Preprocessed network traffic features
- Output:
  - Probability of attack
  - Binary classification (Normal / Attack)

### 🎯 Optimization Goal
The model is tuned to:
> Reduce False Negatives (important for security systems)

---

## 📦 Project Structure
