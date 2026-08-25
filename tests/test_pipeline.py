import pytest
import os
import pandas as pd
import numpy as np
from src.simulation.engine import LogisChainSimulationEngine

def test_feature_matrix_integrity():
    assert os.path.exists("data/features/feature_matrix_50.csv"), "Feature catalog missing"
    df = pd.read_csv("data/features/feature_matrix_50.csv")
    assert df.shape[1] >= 50, "Feature catalog must have at least 50 features"
    assert "sc_adjusted_pd" in df.columns, "SC-PD column missing"
    assert "ccc" in df.columns, "CCC column missing"

def test_simulation_engine_initialization():
    engine = LogisChainSimulationEngine()
    assert engine.turn == 1
    assert engine.portfolio_capital_mil == 500.0
    assert len(engine.SCENARIOS) >= 5

def test_scenario_trigger_and_turn_progression():
    engine = LogisChainSimulationEngine()
    engine.trigger_scenario("Suez Canal Blockage")
    assert engine.active_scenario == "Suez Canal Blockage"
    
    # Simulate turn decisions
    decisions = {row["node_id"]: True for _, row in engine.ledger.head(5).iterrows()}
    summary = engine.run_turn(decisions)
    
    assert summary["turn"] == 1
    assert engine.turn == 2
    assert "player_pnl_mil" in summary
    assert "total_score" in summary
    assert 0 <= summary["total_score"] <= 1000

def test_trade_finance_ledger():
    assert os.path.exists("data/processed/trade_finance_risk_ledger.csv")
    df = pd.read_csv("data/processed/trade_finance_risk_ledger.csv")
    assert not df.isnull().values.any(), "Ledger contains NaN values"
    assert (df["sc_adjusted_pd"] >= 0).all() and (df["sc_adjusted_pd"] <= 1).all()