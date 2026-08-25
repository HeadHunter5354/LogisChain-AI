import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# Add src to system path
sys.path.append(os.path.abspath("."))
from src.simulation.engine import LogisChainSimulationEngine

st.set_page_config(page_title="LogisChain Lab | ZeTheta", layout="wide", page_icon="🚢")

st.title("🚢 LogisChain Lab: Dual-Domain Trade Finance & Logistics Simulator")
st.markdown("---")

# Session state initialization
if "engine" not in st.session_state:
    st.session_state.engine = LogisChainSimulationEngine()

engine = st.session_state.engine

# Sidebar: Game Controls & Scenario Triggers
st.sidebar.header("🎮 Simulation Controls")
game_mode = st.sidebar.selectbox("Select Game Mode", ["Mode 1: Trade Finance Portfolio Management", "Mode 2: Supply Chain Finance (SCF) Dynamic Pricing"])
scenario_name = st.sidebar.selectbox("Inject Disruption Shock Scenario", list(engine.SCENARIOS.keys()))
engine.trigger_scenario(scenario_name)

st.sidebar.info(f"**Active Shock Context:**\n{engine.SCENARIOS[scenario_name]['desc']}")

# Main KPI Row
col1, col2, col3, col4 = st.columns(4)
col1.metric("Simulated Turn (Week)", f"{engine.turn} / {engine.max_turns}")
col2.metric("Portfolio Facility Pool", f"${engine.portfolio_capital_mil:.1f} M")
col3.metric("Active Scenario", engine.active_scenario)

# Display Current Score if history exists
last_score = engine.history[-1]["total_score"] if engine.history else 0
col4.metric("Current Score (1000 pts)", f"{last_score} pts")

st.markdown("### 📋 Credit & Trade Facility Ledger (Underwriting Review)")

# Underwriting Interface
ledger_df = engine.ledger.copy()
selected_approvals = {}

st.write("Review borrower applications. Use cross-domain indicators (SC-PD, Port Dwell, Expected Loss) to accept or reject exposures:")

# Display sample of 10 applications for interactive turns
sample_ledger = ledger_df.head(10)
cols = st.columns(len(sample_ledger))

for idx, row in sample_ledger.iterrows():
    with st.container():
        c1, c2, c3, c4, c5 = st.columns([2, 2, 2, 2, 2])
        c1.write(f"**{row['node_id']}**")
        c2.write(f"Facility: ${row['facility_limit_mil']}M")
        c3.write(f"SC-PD: **{row['sc_adjusted_pd']*100:.2f}%**")
        c4.write(f"Spread: {row['pricing_spread_bps']} bps")
        approved = c5.checkbox("Approve", value=(row['sc_adjusted_pd'] < 0.08), key=f"app_{row['node_id']}")
        selected_approvals[row['node_id']] = approved

# Fill remaining defaults
for idx, row in ledger_df.iloc[10:].iterrows():
    selected_approvals[row['node_id']] = (row['sc_adjusted_pd'] < 0.08)

st.markdown("---")

if st.button("🚀 Execute Turn & Advance Week", type="primary"):
    summary = engine.run_turn(selected_approvals)
    st.success(f"Turn {summary['turn']} completed under scenario: {summary['scenario']}")

# Performance Analytics & Comparison with AI Opponent
if engine.history:
    st.markdown("### 📊 Performance Benchmark: Player vs AI Opponent")
    hist_df = pd.DataFrame(engine.history)
    
    col_a, col_b = st.columns(2)
    with col_a:
        fig_pnl = go.Figure()
        fig_pnl.add_trace(go.Scatter(x=hist_df["turn"], y=hist_df["player_pnl_mil"], mode="lines+markers", name="Player P&L ($M)", line=dict(color="royalblue", width=3)))
        fig_pnl.add_trace(go.Scatter(x=hist_df["turn"], y=hist_df["ai_pnl_mil"], mode="lines+markers", name="AI Opponent P&L ($M)", line=dict(color="firebrick", dash="dash")))
        fig_pnl.update_layout(title="Cumulative P&L Comparison ($M)", xaxis_title="Turn (Week)", yaxis_title="P&L ($M)")
        st.plotly_chart(fig_pnl, use_container_width=True)
        
    with col_b:
        fig_def = go.Figure()
        fig_def.add_trace(go.Bar(x=hist_df["turn"], y=hist_df["player_defaults"], name="Player Defaults", marker_color="royalblue"))
        fig_def.add_trace(go.Bar(x=hist_df["turn"], y=hist_df["ai_defaults"], name="AI Defaults", marker_color="indianred"))
        fig_def.update_layout(title="Default Count Comparison", barmode="group", xaxis_title="Turn (Week)", yaxis_title="Defaults")
        st.plotly_chart(fig_def, use_container_width=True)