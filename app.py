"""
Streamlit Web Dashboard for Quantum AI / ML Predictive Maintenance.

Features:
- Fleet Asset Monitoring (Refineries, Ports, Utilities, Chemical Plants)
- Real-time Multi-Sensor Telemetry (Vibration, Temperature, Pressure, Acoustic)
- Quantum Kernel Gram Matrix & Hilbert-Space State Overlap Visualizer
- Remaining Useful Life (RUL) Trajectory & Early Warning Detection
- Quantum vs Classical Benchmark Comparison & Economic Downtime ROI
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

from src.data.telemetry_generator import (
    IndustrialAssetTelemetryGenerator,
    AssetConfig,
    prepare_quantum_timeseries_dataset,
)
from src.quantum.quantum_kernel import QuantumKernel
from src.quantum.quantum_regressor import QuantumKernelRidgeRegressor, QuantumSVR
from src.models.classical_baselines import get_all_classical_baselines
from src.models.evaluator import PredictiveMaintenanceEvaluator

st.set_page_config(
    page_title="Quantum AI Predictive Maintenance",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for dark glassmorphism styling
st.markdown("""
<style>
    .main {
        background-color: #0b0f19;
        color: #f1f5f9;
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.8));
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(8px);
        margin-bottom: 12px;
    }
    .metric-title {
        color: #94a3b8;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    .metric-val {
        color: #38bdf8;
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 4px;
    }
    .status-healthy { color: #10b981; font-weight: 700; }
    .status-warning { color: #f59e0b; font-weight: 700; }
    .status-critical { color: #ef4444; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Quantum AI / ML Predictive Maintenance Dashboard")
st.markdown("##### *Quantum Kernel Methods & Time-Series Degradation Forecasting for Heavy Industry, Utilities, Ports & Refineries*")

# Sidebar Controls
st.sidebar.header("⚙️ Simulation & Asset Parameters")
asset_type = st.sidebar.selectbox(
    "Select Industrial Asset",
    options=["refinery_compressor", "port_gantry_crane", "utility_turbine", "chemical_pump"],
    format_func=lambda x: {
        "refinery_compressor": "🏭 Refinery Centrifugal Compressor (C-401)",
        "port_gantry_crane": "🚢 Port STS Gantry Crane Hoist (STS-08)",
        "utility_turbine": "⚡ Combined-Cycle Gas Turbine (GT-02)",
        "chemical_pump": "🧪 Chemical Slurry Booster Pump (P-105)",
    }[x],
)

total_cycles = st.sidebar.slider("Asset Operational Lifecycle (Cycles)", 200, 450, 320, step=10)
fault_onset = st.sidebar.slider("Incipient Fault Onset (Cycle)", 100, int(total_cycles * 0.8), 160, step=10)
noise_std = st.sidebar.slider("Sensor Noise (Std Dev)", 0.01, 0.08, 0.035, step=0.005)

st.sidebar.subheader("⚛️ Quantum Circuit Architecture")
num_qubits = st.sidebar.selectbox("Quantum Qubits ($n$)", [3, 4, 5], index=1)
circuit_reps = st.sidebar.slider("ZZ-Feature Map Layers (Reps)", 1, 3, 2)
alpha_reg = st.sidebar.selectbox("QKRR Ridge Regularization ($\\lambda$)", [1e-4, 1e-3, 1e-2, 1e-1], index=1)

# Generate Live Asset Telemetry
@st.cache_data(show_spinner=False)
def get_telemetry_data(asset_type, total_cycles, fault_onset, noise_std):
    cfg = AssetConfig(
        asset_id=f"MONITOR_{asset_type.upper()}",
        asset_type=asset_type,
        total_cycles=total_cycles,
        fault_onset_cycle=fault_onset,
        degradation_rate=0.038,
        noise_std=noise_std,
        random_seed=42,
    )
    gen = IndustrialAssetTelemetryGenerator(cfg)
    return gen.generate_single_run()

df = get_telemetry_data(asset_type, total_cycles, fault_onset, noise_std)

# Main Dashboard Tabs
tab_monitor, tab_quantum, tab_benchmark, tab_theory = st.tabs([
    "📡 Live Telemetry & Fleet State",
    "⚛️ Quantum Kernel & Hilbert Space",
    "🏆 Model Benchmarks & ROI",
    "📚 Quantum Time-Series Theory",
])

# Tab 1: Live Telemetry
with tab_monitor:
    current_cycle = st.slider("Select Current Inspection Cycle", 1, total_cycles, total_cycles // 2, step=1)
    current_row = df.iloc[current_cycle - 1]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Health State</div>
            <div class="metric-val status-{current_row['health_state'].lower()}">{current_row['health_state']}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Degradation Index</div>
            <div class="metric-val">{current_row['true_degradation_index']*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">True Remaining Life (RUL)</div>
            <div class="metric-val">{int(current_row['RUL'])} cycles</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Vibration RMS</div>
            <div class="metric-val">{current_row['vibration_rms']:.2f} mm/s</div>
        </div>
        """, unsafe_allow_html=True)

    # Multi-sensor telemetry charts
    st.subheader("Multivariate Sensor Telemetry Over Machine Lifecycle")
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=["Vibration RMS (mm/s)", "Bearing Temperature (°C)", "Lubrication Oil Pressure (bar)", "Acoustic Emissions (dB)"],
    )

    history_df = df.iloc[:current_cycle]
    future_df = df.iloc[current_cycle - 1:]

    # Sensor 1: Vibration
    fig.add_trace(go.Scatter(x=history_df["cycle"], y=history_df["vibration_rms"], mode="lines", name="Vibration (History)", line=dict(color="#00e5ff", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=future_df["cycle"], y=future_df["vibration_rms"], mode="lines", name="Vibration (Projected)", line=dict(color="#00e5ff", width=1.5, dash="dot")), row=1, col=1)
    
    # Sensor 2: Temperature
    fig.add_trace(go.Scatter(x=history_df["cycle"], y=history_df["bearing_temperature"], mode="lines", name="Temperature (History)", line=dict(color="#ff5252", width=2)), row=1, col=2)
    fig.add_trace(go.Scatter(x=future_df["cycle"], y=future_df["bearing_temperature"], mode="lines", name="Temperature (Projected)", line=dict(color="#ff5252", width=1.5, dash="dot")), row=1, col=2)

    # Sensor 3: Pressure
    fig.add_trace(go.Scatter(x=history_df["cycle"], y=history_df["lubrication_pressure"], mode="lines", name="Pressure (History)", line=dict(color="#69f0ae", width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=future_df["cycle"], y=future_df["lubrication_pressure"], mode="lines", name="Pressure (Projected)", line=dict(color="#69f0ae", width=1.5, dash="dot")), row=2, col=1)

    # Sensor 4: Acoustic
    fig.add_trace(go.Scatter(x=history_df["cycle"], y=history_df["acoustic_emission"], mode="lines", name="Acoustic (History)", line=dict(color="#ffd600", width=2)), row=2, col=2)
    fig.add_trace(go.Scatter(x=future_df["cycle"], y=future_df["acoustic_emission"], mode="lines", name="Acoustic (Projected)", line=dict(color="#ffd600", width=1.5, dash="dot")), row=2, col=2)

    # Vertical current position lines
    for r in [1, 2]:
        for c in [1, 2]:
            fig.add_vline(x=current_cycle, line=dict(color="white", dash="dash", width=1.5), row=r, col=c)
            fig.add_vline(x=fault_onset, line=dict(color="red", dash="dot", width=1), row=r, col=c)

    fig.update_layout(height=520, template="plotly_dark", showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

# Tab 2: Quantum Kernel & Hilbert Space
with tab_quantum:
    st.subheader("Quantum State Overlap & Feature Map Architecture")
    st.markdown("""
    The **$ZZ$-Entangling Quantum Feature Map** encodes continuous multi-sensor windows into the phases of entangled qubits:
    $$\\mathcal{U}_{\\Phi}(\\mathbf{x}) = \\exp\\left(i \\sum_i x_i Z_i + \\sum_{j < k} 2(\\pi - x_j)(\\pi - x_k) Z_j Z_k \\right) H^{\\otimes n}$$
    """)

    col_q1, col_q2 = st.columns([1, 1])

    with col_q1:
        st.write("##### Quantum Circuit Topology")
        st.code(f"""
Qubit 0: ──H──RZ(2x₀)───────●───────RZ(2(π-x₀)(π-x₁))───────●──────
                             │                               │
Qubit 1: ──H──RZ(2x₁)───────X───────────────────────────────X──●───
                                                               │
Qubit 2: ──H──RZ(2x₂)───────●───────RZ(2(π-x₂)(π-x₃))───────●──X───
                             │                               │
Qubit 3: ──H──RZ(2x₃)───────X───────────────────────────────X──────
(Layers = {circuit_reps}, Wires = {num_qubits})
        """, language="text")

    with col_q2:
        st.write("##### Quantum Kernel Overlap Fidelity Matrix $K_{ij} = |\\langle \\phi(x_i) | \\phi(x_j) \\rangle|^2$")
        X_q, y_rul_q, _, _ = prepare_quantum_timeseries_dataset(df, window_size=5, stride=4, num_qubits=num_qubits)
        
        # Subsample for swift visualization
        sub_X = X_q[:40]
        qk = QuantumKernel(num_qubits=num_qubits, feature_map="zz", reps=circuit_reps)
        K_mat = qk.compute_matrix(sub_X)

        fig_k, ax = plt.subplots(figsize=(5, 4), facecolor="#0b0f19")
        im = ax.imshow(K_mat, cmap="magma", origin="lower")
        ax.set_facecolor("#0b0f19")
        ax.tick_params(colors="white")
        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        ax.set_title("Quantum Hilbert Space Kernel", color="white", fontsize=11)
        st.pyplot(fig_k)

# Tab 3: Benchmarks & ROI
with tab_benchmark:
    st.subheader("Quantum vs Classical Predictive Maintenance Benchmarks")
    
    if st.button("🚀 Run Full Fleet Quantum vs Classical Benchmark", key="run_btn"):
        with st.spinner("Computing Quantum Kernels and Training Models..."):
            X_train, y_train_r, _, _ = prepare_quantum_timeseries_dataset(df, window_size=5, stride=3, num_qubits=num_qubits)
            
            # Unseen test run
            cfg_test = AssetConfig(
                asset_id="TEST_RUN",
                asset_type=asset_type,
                total_cycles=300,
                fault_onset_cycle=150,
                noise_std=noise_std,
                random_seed=777,
            )
            df_test_bench = IndustrialAssetTelemetryGenerator(cfg_test).generate_single_run()
            X_test, y_test_r, _, _ = prepare_quantum_timeseries_dataset(df_test_bench, window_size=5, stride=3, num_qubits=num_qubits)

            # Subsample
            X_tr = X_train[:80]
            y_tr = y_train_r[:80]
            X_te = X_test[:60]
            y_te = y_test_r[:60]

            qk = QuantumKernel(num_qubits=num_qubits, feature_map="zz", reps=circuit_reps)
            K_tr = qk.compute_matrix(X_tr)
            K_te = qk.compute_matrix(X_te, X_tr)

            # Train QKRR
            qkrr = QuantumKernelRidgeRegressor(alpha_reg=alpha_reg, num_qubits=num_qubits, reps=circuit_reps)
            qkrr.fit(X_tr, y_tr, K_train=K_tr)
            y_pred_qkrr = qkrr.predict(X_te, K_test=K_te)

            # Train Classical
            classical = get_all_classical_baselines()
            preds = {"Quantum Kernel Ridge (QKRR)": (y_te, y_pred_qkrr)}
            for c_name, c_m in classical.items():
                c_m.fit(X_tr, y_tr)
                preds[c_name] = (y_te, c_m.predict(X_te))

            evaluator = PredictiveMaintenanceEvaluator(hourly_downtime_cost=45000.0)
            df_bench = evaluator.benchmark_fleet(preds)

            st.dataframe(df_bench.style.format({
                "rmse": "{:.2f}",
                "mae": "{:.2f}",
                "r2_score": "{:.3f}",
                "earliness_cycles": "{:.1f}",
                "false_alarm_rate": "{:.2%}",
                "estimated_cost_savings_usd": "${:,.2f}",
            }), use_container_width=True)

            # Forecast trajectory plot
            st.subheader("Remaining Useful Life (RUL) Forecasting Trajectory")
            fig_rul = go.Figure()
            fig_rul.add_trace(go.Scatter(y=y_te, mode="lines", name="Actual True RUL", line=dict(color="white", width=3, dash="dash")))
            fig_rul.add_trace(go.Scatter(y=y_pred_qkrr, mode="lines", name="Quantum Kernel Ridge (QKRR)", line=dict(color="#00e5ff", width=2.5)))
            fig_rul.add_trace(go.Scatter(y=preds["Classical SVR (RBF Kernel)"][1], mode="lines", name="Classical SVR (RBF)", line=dict(color="#ff5252", width=1.5)))
            fig_rul.add_trace(go.Scatter(y=preds["Random Forest Regressor"][1], mode="lines", name="Random Forest", line=dict(color="#ffd600", width=1.5)))

            fig_rul.add_hline(y=50, line=dict(color="red", dash="dot"), annotation_text="Critical Limit (50 cyc)")
            fig_rul.update_layout(template="plotly_dark", height=450, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_rul, use_container_width=True)
    else:
        st.info("Click the button above to run the live training and benchmarking pipeline.")

# Tab 4: Theory
with tab_theory:
    st.markdown("""
    ### 🔬 Quantum AI / ML for Industrial Time-Series Degradation
    
    #### 1. Why Quantum Kernel Methods?
    Industrial sensor telemetry operates in complex, multi-modal regimes where degradation often begins as subtle, cross-channel covariance shifts long before absolute amplitude thresholds (e.g. vibration alerts) are breached.
    
    Classical Gaussian RBF kernels calculate distances in Euclidean space:
    $$k_{\\text{RBF}}(\\mathbf{x}, \\mathbf{x}') = \\exp(-\\gamma ||\\mathbf{x} - \\mathbf{x}'||^2)$$
    
    In contrast, the **Quantum Kernel** implicitly maps temporal windows into a high-dimensional quantum Hilbert state space $\\mathcal{H}$:
    $$K_Q(\\mathbf{x}, \\mathbf{x}') = |\\langle 0^{\\otimes n} | \\mathcal{U}_{\\Phi}^\\dagger(\\mathbf{x}') \\mathcal{U}_{\\Phi}(\\mathbf{x}) | 0^{\\otimes n} \\rangle|^2$$
    
    #### 2. Cross-Sensor Entanglement
    By using entangling two-qubit gates ($ZZ$ interactions), the quantum circuit introduces non-linear feature interactions between disparate physical domains (e.g. mechanical vibration $\\times$ thermal conductivity $\\times$ hydraulic pressure drops), allowing earlier fault detection and drastically reducing false alarm overhead.
    
    #### 3. Industrial Applications
    - **Refineries & Petrochemicals**: Centrifugal compressors, cracked gas turbines, high-pressure hydrogen pumps.
    - **Ports & Maritime Logistics**: Ship-to-shore (STS) container crane gearboxes, propulsion shaft bearings.
    - **Power & Utilities**: Combined-cycle gas turbines, boiler feed pumps, electrical grid step-up transformers.
    """)
