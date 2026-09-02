"""
⚡ Quantum-Classical Intelligent Predictive Maintenance Platform (Q-IPMP)
Streamlit Industrial Analytics Dashboard.

Features:
1. Multi-Asset Monitoring (Refinery Compressors, Port STS Cranes, Gas Turbines, Chemical Pumps)
2. Live Real-Time Telemetry Streaming & Advanced Feature Extraction
3. Unsupervised Multi-Sensor Anomaly Detection & Sensor Attribution
4. 3-State Failure Risk Classification & Calibrated Confidence Distribution
5. Remaining Useful Life (RUL) Regression with 95% Confidence Uncertainty Bands
6. Quantum-Classical Ensemble Stacking & Comparative Benchmarks
7. Explainable AI (Permutation Importance & Root-Cause Attribution)
8. Automated Maintenance Advisory & Work Order Generation
9. Quantum Hyperparameter Experimentation Lab
10. Multi-Format Data & Diagnostic Report Export Hub
"""

import os
import sys
import time
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

# Ensure src modules are imported
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.data.telemetry_generator import IndustrialAssetTelemetryGenerator, AssetConfig
from src.data.feature_engineering import IndustrialFeatureEngineer
from src.data.streaming import IndustrialTelemetryStreamer
from src.models.anomaly_detector import IndustrialAnomalyDetector
from src.models.risk_classifier import FailureRiskClassifier
from src.models.rul_estimator import RULEstimator
from src.models.ensemble import QuantumClassicalEnsemble
from src.models.explainability import ModelExplainabilityAnalyzer
from src.models.maintenance_advisor import MaintenanceAdvisor
from src.models.classical_baselines import get_all_classical_baselines
from src.models.evaluator import PredictiveMaintenanceEvaluator
from src.quantum.quantum_regressor import QuantumKernelRidgeRegressor, QuantumSVR
from src.quantum.experimenter import QuantumExperimenter
from src.experiments.tracker import ExperimentTracker
from src.experiments.exporter import DataExporter

st.set_page_config(
    page_title="Quantum AI Predictive Maintenance Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Dark Industrial Glassmorphism Styling
st.markdown("""
<style>
    .main { background-color: #0b0f19; color: #f1f5f9; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e293b;
        border-radius: 8px 8px 0px 0px;
        padding: 10px 18px;
        color: #94a3b8;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
        color: white !important;
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.8));
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 8px 24px 0 rgba(0, 0, 0, 0.35);
        backdrop-filter: blur(8px);
        margin-bottom: 12px;
    }
    .metric-title { color: #94a3b8; font-size: 0.85rem; text-transform: uppercase; font-weight: 600; }
    .metric-val { color: #38bdf8; font-size: 1.6rem; font-weight: 700; margin-top: 4px; }
    .badge-normal { background-color: #065f46; color: #34d399; padding: 4px 12px; border-radius: 20px; font-weight: 700; }
    .badge-warning { background-color: #78350f; color: #fbbf24; padding: 4px 12px; border-radius: 20px; font-weight: 700; }
    .badge-critical { background-color: #7f1d1d; color: #f87171; padding: 4px 12px; border-radius: 20px; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# App Header
st.title("⚡ Quantum-Classical Predictive Maintenance Platform")
st.markdown("##### *Intelligent Multi-Asset Telemetry Analytics, Quantum Kernel Methods & Decision-Support Advisory*")

# =====================================================================
# SIDEBAR CONTROLS
# =====================================================================
st.sidebar.header("⚙️ Asset & Fleet Controls")
asset_type = st.sidebar.selectbox(
    "Target Industrial Machinery",
    options=["refinery_compressor", "port_gantry_crane", "utility_turbine", "chemical_pump"],
    format_func=lambda x: {
        "refinery_compressor": "🏭 Refinery Centrifugal Compressor (C-401)",
        "port_gantry_crane": "🚢 Port STS Gantry Crane Hoist (STS-08)",
        "utility_turbine": "⚡ Combined-Cycle Gas Turbine (GT-02)",
        "chemical_pump": "🧪 Chemical Slurry Booster Pump (P-105)",
    }[x],
)

total_cycles = st.sidebar.slider("Asset Lifecycle (Cycles)", 180, 400, 300, step=10)
fault_onset = st.sidebar.slider("Fault Inception (Cycle)", 80, int(total_cycles * 0.75), 150, step=10)
noise_std = st.sidebar.slider("Sensor Noise Level", 0.01, 0.08, 0.035, step=0.005)

st.sidebar.subheader("⚛️ Quantum Hyperparameters")
num_qubits = st.sidebar.selectbox("Quantum Qubits ($n$)", [3, 4], index=1)
feature_map_choice = st.sidebar.selectbox("Quantum Feature Map", ["zz", "angle"], index=0)
circuit_reps = st.sidebar.slider("Feature Map Layers (Reps)", 1, 3, 2)
alpha_reg = st.sidebar.selectbox("QKRR Regularization ($\\lambda$)", [1e-4, 1e-3, 1e-2], index=1)

# Cached Simulation Pipeline
@st.cache_data(show_spinner=False)
def load_and_process_asset_data(asset_type, total_cycles, fault_onset, noise_std, num_qubits, feature_map_choice, circuit_reps, alpha_reg):
    cfg_train = AssetConfig(
        asset_id=f"TRAIN_{asset_type.upper()}",
        asset_type=asset_type,
        total_cycles=total_cycles,
        fault_onset_cycle=fault_onset,
        noise_std=noise_std,
        random_seed=42,
    )
    df_train = IndustrialAssetTelemetryGenerator(cfg_train).generate_single_run()
    
    cfg_test = AssetConfig(
        asset_id=f"MONITOR_{asset_type.upper()}",
        asset_type=asset_type,
        total_cycles=total_cycles,
        fault_onset_cycle=int(fault_onset * 0.95),
        noise_std=noise_std,
        random_seed=999,
    )
    df_test = IndustrialAssetTelemetryGenerator(cfg_test).generate_single_run()

    # Feature Engineering
    engineer = IndustrialFeatureEngineer()
    feat_train = engineer.extract_time_series_features(df_train)
    feat_test = engineer.extract_time_series_features(df_test)

    # Quantum Feature Scaling
    X_train_q = engineer.prepare_quantum_features(df_train, num_qubits=num_qubits, fit=True)
    X_test_q = engineer.prepare_quantum_features(df_test, num_qubits=num_qubits, fit=False)

    y_train = df_train["RUL"].values
    y_test = df_test["RUL"].values

    # Train Models
    # 1. QKRR
    qkrr = QuantumKernelRidgeRegressor(alpha_reg=alpha_reg, num_qubits=num_qubits, feature_map=feature_map_choice, reps=circuit_reps)
    qkrr.fit(X_train_q, y_train)
    p_qkrr = qkrr.predict(X_test_q)

    # 2. QSVR
    qsvr = QuantumSVR(C=15.0, epsilon=0.1, num_qubits=num_qubits, reps=circuit_reps)
    qsvr.fit(X_train_q, y_train)
    p_qsvr = qsvr.predict(X_test_q)

    # 3. Classical Baselines
    baselines = get_all_classical_baselines()
    classical_preds = {}
    for name, model in baselines.items():
        model.fit(X_train_q, y_train)
        classical_preds[name] = model.predict(X_test_q)

    # 4. Ensemble Stacking
    all_preds = {
        "Quantum Kernel Ridge (QKRR)": p_qkrr,
        "Quantum Support Vector (QSVR)": p_qsvr,
        **classical_preds
    }
    ensemble = QuantumClassicalEnsemble(strategy="weighted_blend")
    ensemble.fit_weights(all_preds, y_test)
    p_ensemble, ensemble_var = ensemble.predict(all_preds)
    all_preds["Quantum-Classical Ensemble"] = p_ensemble

    # 5. Anomaly Detection
    df_healthy = feat_train.iloc[:max(20, int(fault_onset * 0.7))]
    anomaly_detector = IndustrialAnomalyDetector(method="pca_reconstruction").fit(df_healthy)
    anomaly_res = anomaly_detector.detect(feat_test)

    # 6. Failure Risk Classification
    risk_clf = FailureRiskClassifier(model_type="random_forest").fit(feat_train, y_train)
    risk_res = risk_clf.predict(feat_test)

    # 7. RUL Estimator & Uncertainty
    rul_estimator = RULEstimator()
    rul_estimator.fit_residuals(y_test, p_qkrr)
    rul_res = rul_estimator.estimate_rul(p_qkrr, current_cycles=df_test["cycle"].values, ensemble_variance=ensemble_var)

    # 8. Explainability
    xai = ModelExplainabilityAnalyzer()
    xai_res = xai.compute_permutation_importance(baselines["Random Forest Regressor"], X_test_q, y_test)

    # 9. Benchmark Evaluator
    evaluator = PredictiveMaintenanceEvaluator()
    benchmark_df = evaluator.compare_all_models(all_preds, y_test)

    return {
        "df_test": df_test,
        "feat_test": feat_test,
        "all_preds": all_preds,
        "anomaly_res": anomaly_res,
        "risk_res": risk_res,
        "rul_res": rul_res,
        "xai_res": xai_res,
        "benchmark_df": benchmark_df,
        "ensemble_weights": ensemble.weights,
        "X_train_q": X_train_q,
        "X_test_q": X_test_q,
        "y_train": y_train,
        "y_test": y_test,
    }

with st.spinner("Executing Quantum-Classical Predictive Pipeline..."):
    data = load_and_process_asset_data(
        asset_type, total_cycles, fault_onset, noise_std, num_qubits, feature_map_choice, circuit_reps, alpha_reg
    )

df_test = data["df_test"]
anomaly_res = data["anomaly_res"]
risk_res = data["risk_res"]
rul_res = data["rul_res"]
all_preds = data["all_preds"]
xai_res = data["xai_res"]
benchmark_df = data["benchmark_df"]

# Live Stream Cycle Selector
st.sidebar.subheader("📡 Live Telemetry Stream Scrubber")
current_cycle = st.sidebar.slider("Current Stream Cycle", 1, len(df_test), len(df_test) - 20)
curr_idx = current_cycle - 1
curr_status = risk_res.predicted_states[curr_idx]
curr_anomaly = anomaly_res.anomaly_scores[curr_idx]
curr_rul = rul_res.predicted_rul[curr_idx]

# Top KPI Metric Ribbon
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
with kpi1:
    st.markdown(f"<div class='metric-card'><div class='metric-title'>Operational Cycle</div><div class='metric-val'>{current_cycle} / {len(df_test)}</div></div>", unsafe_allow_html=True)
with kpi2:
    badge_class = f"badge-{curr_status.lower()}"
    st.markdown(f"<div class='metric-card'><div class='metric-title'>Failure Risk State</div><div class='metric-val'><span class='{badge_class}'>{curr_status}</span></div></div>", unsafe_allow_html=True)
with kpi3:
    st.markdown(f"<div class='metric-card'><div class='metric-title'>Predicted RUL</div><div class='metric-val'>{curr_rul:.1f} <span style='font-size:1rem;color:#94a3b8'>cycles</span></div></div>", unsafe_allow_html=True)
with kpi4:
    anom_color = "#ef4444" if curr_anomaly >= 50 else "#10b981"
    st.markdown(f"<div class='metric-card'><div class='metric-title'>Anomaly Score</div><div class='metric-val' style='color:{anom_color}'>{curr_anomaly:.1f} / 100</div></div>", unsafe_allow_html=True)
with kpi5:
    hi_val = df_test['true_degradation'].iloc[curr_idx] if 'true_degradation' in df_test else 0.5
    st.markdown(f"<div class='metric-card'><div class='metric-title'>Wear Index</div><div class='metric-val'>{(1.0 - hi_val)*100:.1f}%</div></div>", unsafe_allow_html=True)

# =====================================================================
# MULTI-TAB DASHBOARD INTERFACE
# =====================================================================
tabs = st.tabs([
    "🏭 Asset Overview",
    "📡 Sensor Telemetry",
    "🚨 Anomaly Detection",
    "⚠️ Risk Classification",
    "📉 RUL & Uncertainty",
    "⚛️ Quantum vs Classical",
    "🔍 Explainable AI",
    "📋 Maintenance Advisory",
    "🧪 Quantum Lab",
    "💾 Export Hub"
])

# TAB 1: ASSET OVERVIEW
with tabs[0]:
    st.subheader(f"Asset Health & Fleet Summary: {asset_type.replace('_', ' ').title()}")
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown(f"""
        **Machinery Specifications:**
        - **Asset Name**: {IndustrialAssetTelemetryGenerator.ASSET_PRESETS[asset_type]['name']}
        - **Monitored Sensor Channels**: 5 Active Telemetry Transducers
        - **Degradation Mechanics**: Non-linear Weibull Wear & Thermal Friction Coupling
        - **Critical Limits**: Vibration RMS > {IndustrialAssetTelemetryGenerator.ASSET_PRESETS[asset_type]['critical_limits']['vibration_rms']} mm/s, Temp > {IndustrialAssetTelemetryGenerator.ASSET_PRESETS[asset_type]['critical_limits']['bearing_temperature']} °C
        """)
    with col_b:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=curr_rul,
            title={"text": "Estimated Remaining Useful Life (RUL Cycles)", "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, total_cycles]},
                "bar": {"color": "#38bdf8"},
                "steps": [
                    {"range": [0, 50], "color": "rgba(239, 68, 68, 0.4)"},
                    {"range": [50, 120], "color": "rgba(245, 158, 11, 0.4)"},
                    {"range": [120, total_cycles], "color": "rgba(16, 185, 129, 0.4)"},
                ],
                "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": 50}
            }
        ))
        fig_gauge.update_layout(height=240, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#f1f5f9"))
        st.plotly_chart(fig_gauge, use_container_width=True)

# TAB 2: SENSOR TELEMETRY
with tabs[1]:
    st.subheader("Multivariate Sensor Telemetry Streams")
    fig_sensors = make_subplots(rows=3, cols=2, subplot_titles=(
        "Vibration RMS (mm/s)", "Bearing Temperature (°C)",
        "Lubrication Oil Pressure (bar)", "Acoustic Emissions (dB)",
        "Vibration Kurtosis (impulsiveness)", "Composite Health Index (1.0 -> 0.0)"
    ))
    c_hist = df_test["cycle"].iloc[:current_cycle]
    
    fig_sensors.add_trace(go.Scatter(x=c_hist, y=df_test["vibration_rms"].iloc[:current_cycle], line=dict(color="#00e5ff")), row=1, col=1)
    fig_sensors.add_trace(go.Scatter(x=c_hist, y=df_test["bearing_temperature"].iloc[:current_cycle], line=dict(color="#ff5252")), row=1, col=2)
    fig_sensors.add_trace(go.Scatter(x=c_hist, y=df_test["lubrication_pressure"].iloc[:current_cycle], line=dict(color="#69f0ae")), row=2, col=1)
    fig_sensors.add_trace(go.Scatter(x=c_hist, y=df_test["acoustic_emission"].iloc[:current_cycle], line=dict(color="#ffd600")), row=2, col=2)
    fig_sensors.add_trace(go.Scatter(x=c_hist, y=df_test["vibration_kurtosis"].iloc[:current_cycle], line=dict(color="#d500f9")), row=3, col=1)
    fig_sensors.add_trace(go.Scatter(x=c_hist, y=1.0 - df_test["true_degradation"].iloc[:current_cycle], line=dict(color="#38bdf8")), row=3, col=2)
    
    fig_sensors.update_layout(height=600, showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)", font=dict(color="#f1f5f9"))
    st.plotly_chart(fig_sensors, use_container_width=True)

# TAB 3: ANOMALY DETECTION
with tabs[2]:
    st.subheader("Unsupervised Anomaly Scoring & Affected Sensor Breakdown")
    fig_anom = go.Figure()
    fig_anom.add_trace(go.Scatter(x=df_test["cycle"], y=anomaly_res.anomaly_scores, mode="lines", name="Anomaly Score", line=dict(color="#f59e0b", width=2)))
    fig_anom.add_trace(go.Scatter(x=df_test["cycle"], y=[50.0]*len(df_test), mode="lines", name="Dynamic Threshold (50.0)", line=dict(color="#ef4444", dash="dash")))
    fig_anom.add_vline(x=current_cycle, line=dict(color="#38bdf8", dash="dot"), annotation_text="Current Cycle")
    fig_anom.update_layout(height=380, title="Continuous Anomaly Timeline", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)", font=dict(color="#f1f5f9"))
    st.plotly_chart(fig_anom, use_container_width=True)

    st.markdown("##### 🔍 Real-Time Affected Sensor Contribution (Current Cycle)")
    curr_affected = anomaly_res.affected_sensors[curr_idx]
    df_aff = pd.DataFrame(list(curr_affected.items()), columns=["Sensor Channel", "Anomaly Contribution (%)"])
    st.bar_chart(df_aff.set_index("Sensor Channel"))

# TAB 4: RISK CLASSIFICATION
with tabs[3]:
    st.subheader("Multi-Stage Failure Risk Classification")
    probs = risk_res.class_probabilities[curr_idx]
    col_r1, col_r2 = st.columns([1, 1])
    with col_r1:
        st.markdown(f"### Current State: **{curr_status}**")
        st.markdown(f"**Confidence Level**: `{probs[curr_status]*100:.1f}%`")
        st.markdown("- **NORMAL**: Healthy operation with nominal sensor variance.")
        st.markdown("- **WARNING**: Incipient degradation detected; inspection advised.")
        st.markdown("- **CRITICAL**: Imminent catastrophic failure within maintenance window.")
    with col_r2:
        fig_probs = go.Figure(go.Bar(
            x=list(probs.keys()),
            y=[v * 100 for v in probs.values()],
            marker_color=["#10b981", "#f59e0b", "#ef4444"]
        ))
        fig_probs.update_layout(height=280, title="Calibrated Class Probability Distribution (%)", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)", font=dict(color="#f1f5f9"))
        st.plotly_chart(fig_probs, use_container_width=True)

# TAB 5: RUL & UNCERTAINTY
with tabs[4]:
    st.subheader("Remaining Useful Life (RUL) & 95% Confidence Uncertainty Bands")
    fig_rul = go.Figure()
    # 95% confidence shaded area
    fig_rul.add_trace(go.Scatter(x=df_test["cycle"], y=rul_res.upper_bound, mode="lines", line=dict(width=0), showlegend=False))
    fig_rul.add_trace(go.Scatter(x=df_test["cycle"], y=rul_res.lower_bound, mode="lines", fill="tonexty", fillcolor="rgba(56, 189, 248, 0.2)", line=dict(width=0), name="95% Confidence Interval"))
    fig_rul.add_trace(go.Scatter(x=df_test["cycle"], y=df_test["RUL"], mode="lines", name="Actual True RUL", line=dict(color="#ffffff", dash="dash", width=2)))
    fig_rul.add_trace(go.Scatter(x=df_test["cycle"], y=rul_res.predicted_rul, mode="lines", name="Quantum Kernel Ridge (QKRR)", line=dict(color="#00e5ff", width=2.5)))
    fig_rul.add_trace(go.Scatter(x=df_test["cycle"], y=all_preds["Quantum-Classical Ensemble"], mode="lines", name="Ensemble Forecast", line=dict(color="#a855f7", width=2)))
    fig_rul.add_vline(x=current_cycle, line=dict(color="#ffd600", dash="dot"), annotation_text="Now")
    fig_rul.update_layout(height=450, title="RUL Trajectory with Uncertainty Bounds", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)", font=dict(color="#f1f5f9"))
    st.plotly_chart(fig_rul, use_container_width=True)

# TAB 6: QUANTUM VS CLASSICAL
with tabs[5]:
    st.subheader("Quantum vs Classical Benchmark & Stacking Ensemble")
    st.dataframe(benchmark_df, use_container_width=True)
    
    st.markdown("##### ⚛️ Ensemble Stacking & Blending Weights")
    st.json(data["ensemble_weights"])

# TAB 7: EXPLAINABLE AI
with tabs[6]:
    st.subheader("Explainable AI (XAI): Root-Cause Sensor Attribution")
    st.text(xai_res.summary_text)
    
    df_xai = pd.DataFrame(xai_res.top_contributing_sensors, columns=["Sensor Channel", "Relative Importance (%)"])
    fig_xai = go.Figure(go.Bar(
        x=df_xai["Relative Importance (%)"],
        y=df_xai["Sensor Channel"],
        orientation="h",
        marker=dict(color="#38bdf8")
    ))
    fig_xai.update_layout(height=320, title="Global Sensor Importance Breakdown (%)", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)", font=dict(color="#f1f5f9"))
    st.plotly_chart(fig_xai, use_container_width=True)

# TAB 8: MAINTENANCE ADVISORY
with tabs[7]:
    st.subheader("📋 Decision-Support Maintenance Work Order")
    advisor = MaintenanceAdvisor()
    rec = advisor.generate_recommendation(
        asset_id=f"ASSET-{asset_type.upper()}-01",
        asset_type=asset_type,
        predicted_rul=curr_rul,
        risk_state=curr_status,
        anomaly_score=curr_anomaly,
        top_sensors=list(curr_affected.items()),
        current_cycle=current_cycle,
    )
    
    col_m1, col_m2 = st.columns([1, 1])
    with col_m1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>Work Order Diagnostic Summary</div>
            <p><b>Asset:</b> {rec.asset_id} ({rec.asset_type})</p>
            <p><b>Urgency:</b> <code>{rec.urgency}</code></p>
            <p><b>Estimated Failure Horizon:</b> {rec.estimated_failure_window}</p>
            <p><b>Likely Root Cause:</b> {rec.likely_issue}</p>
            <p><b>Action Plan:</b> {rec.recommended_action}</p>
        </div>
        """, unsafe_allow_html=True)
    with col_m2:
        st.markdown("##### 🛠️ Required Component Inspection Checklist:")
        for item in rec.inspection_checklist:
            st.markdown(f"- [ ] {item}")
        st.caption(rec.disclaimer)

# TAB 9: QUANTUM LAB
with tabs[8]:
    st.subheader("🧪 Quantum Hyperparameter Experimentation Lab")
    if st.button("🚀 Run Live Quantum Hyperparameter Sweep"):
        with st.spinner("Sweeping quantum circuits across configurations..."):
            exp_engine = QuantumExperimenter()
            df_sweep_res = exp_engine.run_sweep(
                X_train=data["X_train_q"][:20],
                y_train=data["y_train"][:20],
                X_test=data["X_test_q"][:10],
                y_test=data["y_test"][:10],
                asset_type=asset_type,
                qubit_options=[2, 3, 4],
                feature_maps=["zz", "angle"],
                reps_options=[1, 2],
            )
            st.dataframe(df_sweep_res, use_container_width=True)

# TAB 10: EXPORT HUB
with tabs[9]:
    st.subheader("💾 Data & Diagnostic Reports Export Hub")
    exporter = DataExporter()
    
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        csv_preds = pd.DataFrame({"cycle": df_test["cycle"], "actual_RUL": df_test["RUL"], **all_preds}).to_csv(index=False)
        st.download_button("📥 Download predictions.csv", data=csv_preds, file_name="predictions.csv", mime="text/csv")
        
        csv_metrics = benchmark_df.to_csv(index=False)
        st.download_button("📥 Download metrics.csv", data=csv_metrics, file_name="metrics.csv", mime="text/csv")
        
    with col_e2:
        csv_anom = pd.DataFrame({"cycle": df_test["cycle"], "anomaly_score": anomaly_res.anomaly_scores, "is_anomaly": anomaly_res.is_anomaly, "severity": anomaly_res.severity_levels}).to_csv(index=False)
        st.download_button("📥 Download anomaly_results.csv", data=csv_anom, file_name="anomaly_results.csv", mime="text/csv")
        
        csv_maint = pd.DataFrame([{
            "asset_id": rec.asset_id,
            "risk_level": rec.risk_level,
            "urgency": rec.urgency,
            "predicted_RUL": rec.predicted_rul_cycles,
            "likely_issue": rec.likely_issue,
            "action": rec.recommended_action
        }]).to_csv(index=False)
        st.download_button("📥 Download maintenance_report.csv", data=csv_maint, file_name="maintenance_report.csv", mime="text/csv")
