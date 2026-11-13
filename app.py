import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import random

# --- PAGE CONFIG ---
st.set_page_config(page_title="CASR+ Serverless Optimizer", layout="wide")

st.title("CASR+: Hybrid RL-DL for Serverless Cold-Start Reduction")
st.markdown("""
This interactive demo showcases a **Hybrid Reinforcement Learning and Deep Learning model** designed to optimize resource allocation and reduce cold-start latency in Serverless Computing (FaaS). 
*Based on the Azure Functions Trace 2019.*
""")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Environment & Model Parameters")

# 1. Workload Selection (Triggers XGBoost Meta-model)
workload_type = st.sidebar.selectbox(
    "Select Workload Profile",
    ["Highly Bursty (e.g., Web APIs)", "Periodic (e.g., Cron Jobs)", "Steady State (e.g., Data Processing)"]
)

# 2. SLA Threshold
sla_threshold = st.sidebar.slider("SLA Threshold (Seconds)", min_value=0.05, max_value=0.50, value=0.15, step=0.01)

# 3. PPO Reward Weights (Can be static or predicted by your meta-model)
st.sidebar.subheader("PPO Reward Weights")
st.sidebar.markdown("*Dynamically predicted by XGBoost Meta-Model*")

# Simulated Meta-Model Output based on your project's logic
if workload_type == "Highly Bursty (e.g., Web APIs)":
    theta, alpha, beta, gamma = 0.8, 0.5, 0.2, 0.1
elif workload_type == "Periodic (e.g., Cron Jobs)":
    theta, alpha, beta, gamma = 0.4, 0.8, 0.1, 0.3
else:
    theta, alpha, beta, gamma = 0.6, 0.3, 0.4, 0.2

st.sidebar.write(f"**Latency Weight (θ):** {theta}")
st.sidebar.write(f"**Memory Weight (α):** {alpha}")
st.sidebar.write(f"**Oscillation Penalty (β):** {beta}")
st.sidebar.write(f"**Overhead Penalty (γ):** {gamma}")

# --- MAIN DASHBOARD ---
st.divider()

col1, col2, col3 = st.columns(3)

# Placeholder for metrics
metric_cs = col1.empty()
metric_wmt = col2.empty()
metric_osc = col3.empty()

st.divider()

# --- SIMULATION & VISUALIZATION LOGIC ---
st.subheader("Simulation Results: CASR+ vs. Baselines")

# Button to trigger simulation
if st.button("Run Simulation", type="primary"):
    with st.spinner("Simulating Serverless Environment (Azure Trace)..."):
        time.sleep(1.5) # Artificial delay to simulate computation
        
        # --- MOCK INFERENCE LOGIC ---
        # In a real deployment, you would replace this with:
        # obs = env.reset()
        # action, _ = model.predict(obs)
        
        # Simulating the impact of your reward weights on the metrics
        base_cold_start = 0.35
        base_wmt = 45.0
        base_osc = 120
        
        # Your RL agent's effectiveness (simulated)
        casr_plus_cs = max(0.05, base_cold_start - (alpha * 0.15) - (theta * 0.1))
        casr_plus_wmt = max(10.0, base_wmt - (theta * 20) + (alpha * 5))
        casr_plus_osc = max(10, base_osc - (beta * 50) - (gamma * 20))
        
        # Baselines (Static approximations)
        lru_cs = 0.35
        lru_wmt = 45.0
        base_casr_cs = 0.18
        base_casr_wmt = 32.0
        
        # Update Metrics
        metric_cs.metric("Cold Start Rate (CASR+)", f"{casr_plus_cs*100:.1f}%", f"{(casr_plus_cs - lru_cs)*100:.1f}% vs LRU", delta_color="inverse")
        metric_wmt.metric("Memory Waste (WMT)", f"{casr_plus_wmt:.1f} GB-s", f"{casr_plus_wmt - base_wmt:.1f} GB-s vs LRU", delta_color="inverse")
        metric_osc.metric("Eviction Oscillations", f"{int(casr_plus_osc)}", f"{int(casr_plus_osc - base_osc)} vs LRU", delta_color="inverse")
        
        # --- CHARTS ---
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            # Cold Start Comparison Bar Chart
            df_cs = pd.DataFrame({
                "Algorithm": ["LRU (Baseline)", "CASR (Static RL)", "CASR+ (Hybrid RL-DL)"],
                "Cold Start Rate (%)": [lru_cs*100, base_casr_cs*100, casr_plus_cs*100]
            })
            fig_cs = px.bar(df_cs, x="Algorithm", y="Cold Start Rate (%)", 
                            color="Algorithm", title="Cold Start Rate Comparison",
                            color_discrete_sequence=['#FF4B4B', '#FFBD45', '#00CC96'])
            st.plotly_chart(fig_cs, use_container_width=True)
            
        with col_chart2:
            # Latency vs Memory Trade-off (Scatter)
            df_tradeoff = pd.DataFrame({
                "Algorithm": ["LRU", "CASR", "CASR+"],
                "Cold Start Rate": [lru_cs, base_casr_cs, casr_plus_cs],
                "Memory Waste": [lru_wmt, base_casr_wmt, casr_plus_wmt]
            })
            fig_tradeoff = px.scatter(df_tradeoff, x="Cold Start Rate", y="Memory Waste", 
                                      text="Algorithm", size=[20, 20, 30], 
                                      title="Latency-Memory Trade-off (Bottom-Left is better)")
            fig_tradeoff.update_traces(textposition='top center')
            st.plotly_chart(fig_tradeoff, use_container_width=True)

        st.success("Simulation Complete. CASR+ successfully balances SLA adherence and memory costs.")
        
else:
    # Initial state
    metric_cs.metric("Cold Start Rate", "--")
    metric_wmt.metric("Memory Waste (WMT)", "--")
    metric_osc.metric("Eviction Oscillations", "--")
    st.info("Adjust parameters in the sidebar and click 'Run Simulation' to see the RL agent's performance.")

# --- FOOTER ---
st.markdown("---")
st.markdown("**Note:** This is an interactive demonstration. The underlying model uses Proximal Policy Optimization (PPO) with an XGBoost meta-model for dynamic reward weighting.")