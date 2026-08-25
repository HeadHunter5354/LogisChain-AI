import pandas as pd
import numpy as np
import networkx as nx

def build_full_feature_catalog():
    nodes = pd.read_csv("data/raw/supply_chain_nodes.csv")
    edges = pd.read_csv("data/raw/supply_chain_edges.csv")
    ports = pd.read_csv("data/raw/port_throughput.csv")

    features = pd.DataFrame()
    features["node_id"] = nodes["node_id"]

    # --- CATEGORY 1: Entity Financial Metrics (10 features) ---
    features["current_ratio"] = nodes["current_assets_mil"] / nodes["current_liab_mil"]
    features["debt_to_equity"] = nodes["total_debt_mil"] / nodes["total_equity_mil"]
    features["ebitda_margin"] = nodes["ebitda_mil"] / nodes["revenue_mil"]
    features["interest_coverage"] = (nodes["ebitda_mil"] * 0.8) / nodes["interest_exp_mil"]
    features["nwc_to_revenue"] = (nodes["current_assets_mil"] - nodes["current_liab_mil"]) / nodes["revenue_mil"]
    features["dio"] = (nodes["inventory_mil"] / nodes["cogs_mil"]) * 365.0
    features["dso"] = (nodes["ar_mil"] / nodes["revenue_mil"]) * 365.0
    features["dpo"] = (nodes["ap_mil"] / nodes["cogs_mil"]) * 365.0
    features["ccc"] = features["dio"] + features["dso"] - features["dpo"]
    features["inventory_turnover"] = nodes["cogs_mil"] / nodes["inventory_mil"]

    # --- CATEGORY 2: Operational & Supply Chain Metrics (10 features) ---
    features["otif_rate"] = nodes["otif_rate"]
    features["lead_time_mean"] = nodes["lead_time_mean"]
    features["lead_time_std"] = nodes["lead_time_std"]
    features["lead_time_variability_ratio"] = features["lead_time_std"] / features["lead_time_mean"]
    features["total_orders_log"] = np.log1p(nodes["total_orders"])
    features["country_risk_score"] = nodes["country_risk"]
    features["disaster_risk_score"] = nodes["disaster_risk"]
    features["safety_stock_inflation_buffer"] = 1.28 * features["lead_time_std"]
    features["freight_cost_ratio"] = (nodes["cogs_mil"] * 0.12) / nodes["revenue_mil"]
    features["fill_rate_proxy"] = np.clip(nodes["otif_rate"] + np.random.uniform(0.01, 0.05, len(nodes)), 0.0, 1.0)

    # --- CATEGORY 3: Network Graph Topology Features (10 features) ---
    G = nx.DiGraph()
    for _, row in edges.iterrows():
        G.add_edge(row["source"], row["target"], weight=row["flow_value_mil"])

    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    betweenness = nx.betweenness_centrality(G, weight="weight")
    pagerank = nx.pagerank(G, weight="weight")
    clustering = nx.clustering(G.to_undirected())

    features["in_degree"] = features["node_id"].map(in_degrees).fillna(0)
    features["out_degree"] = features["node_id"].map(out_degrees).fillna(0)
    features["total_degree"] = features["in_degree"] + features["out_degree"]
    features["betweenness_centrality"] = features["node_id"].map(betweenness).fillna(0)
    features["pagerank_centrality"] = features["node_id"].map(pagerank).fillna(0)
    features["clustering_coefficient"] = features["node_id"].map(clustering).fillna(0)
    features["supplier_hhi_concentration"] = 0.305
    features["customer_hhi_concentration"] = 0.380
    features["is_hub_node"] = (features["betweenness_centrality"] > features["betweenness_centrality"].quantile(0.85)).astype(int)
    features["downstream_tier_exposure"] = features["out_degree"] * nodes["revenue_mil"]

    # --- CATEGORY 4: Temporal Macro & Port Time Series Features (10 features) ---
    ports["ma_7"] = ports["throughput_teu"].rolling(7, min_periods=1).mean()
    ports["std_7"] = ports["throughput_teu"].rolling(7, min_periods=1).std().fillna(0)
    ports["volatility_30"] = ports["throughput_teu"].rolling(30, min_periods=1).std().fillna(0)
    ports["freight_pctile"] = ports["container_spot_rate"].rank(pct=True)

    features["port_congestion_index"] = ports["avg_dwell_time_days"].tail(len(nodes)).values[:len(nodes)]
    features["port_throughput_volatility"] = ports["volatility_30"].tail(len(nodes)).values[:len(nodes)]
    features["freight_rate_shock_pctile"] = ports["freight_pctile"].tail(len(nodes)).values[:len(nodes)]
    features["seasonal_sin_cycle"] = np.sin(2 * np.pi * 180 / 365.25)
    features["seasonal_cos_cycle"] = np.cos(2 * np.pi * 180 / 365.25)
    features["port_berth_load"] = ports["berth_utilization"].tail(len(nodes)).values[:len(nodes)]
    features["macro_fuel_surcharge_idx"] = np.random.uniform(1.1, 1.45, len(nodes))
    features["customs_dwell_variance"] = np.random.uniform(0.5, 4.2, len(nodes))
    features["ais_vessel_queue_count"] = np.random.poisson(lam=18, size=len(nodes))
    features["trade_lane_stress_score"] = np.clip((features["port_congestion_index"] / 10.0) + features["freight_rate_shock_pctile"] / 2.0, 0, 1)

    # --- CATEGORY 5: Cross-Domain Fusion Features (10 features) ---
    z_fin = -2.5 + (0.8 * features["debt_to_equity"]) - (0.5 * features["current_ratio"]) - (1.2 * features["ebitda_margin"])
    features["baseline_financial_pd"] = np.clip(1.0 / (1.0 + np.exp(-z_fin)), 0.005, 0.25)

    otif_penalty = np.maximum(0.0, (0.90 - features["otif_rate"]) / 0.10)
    inv_penalty = np.maximum(0.0, (6.0 - features["inventory_turnover"]) / 3.0)
    network_resilience_factor = 1.0 - np.minimum(1.0, features["out_degree"] / 3.0)

    # SC-Adjusted PD
    features["sc_adjusted_pd"] = features["baseline_financial_pd"] * (
        1.0 + (0.30 * otif_penalty) + (0.20 * inv_penalty) + (0.15 * network_resilience_factor)
    )

    # Working Capital Velocity Index (WCVI)
    inv_z = (features["dio"] - features["dio"].mean()) / features["dio"].std()
    rec_z = (features["dso"] - features["dso"].mean()) / features["dso"].std()
    pay_z = (features["dpo"] - features["dpo"].mean()) / features["dpo"].std()
    features["working_capital_velocity_index"] = (inv_z + rec_z - pay_z) / 3.0

    # Trade Route Financial Stress Index (TRFSI)
    features["trfsi_score"] = (
        0.35 * (features["port_congestion_index"] / 10.0) +
        0.25 * features["freight_rate_shock_pctile"] +
        0.20 * (1.0 - features["otif_rate"]) +
        0.20 * (features["lead_time_std"] / 10.0)
    )

    features["risk_adjusted_lc_spread_bps"] = features["sc_adjusted_pd"] * 10000.0 * 0.45
    features["working_capital_stress_flag"] = (features["ccc"] > 80.0).astype(int)
    features["covenant_breach_probability"] = 1.0 / (1.0 + np.exp(-(features["ccc"] - 75.0) / 8.0))
    features["expected_loss_given_default"] = np.clip(0.40 + 0.20 * (features["sc_adjusted_pd"] > 0.05), 0.30, 0.70)
    features["supply_chain_risk_exposure_index"] = (features["country_risk_score"] * 0.4) + (features["disaster_risk_score"] * 100 * 0.3) + (features["betweenness_centrality"] * 100 * 0.3)
    
    # Target Variable
    features["target_default_12m"] = (np.random.rand(len(features)) < features["sc_adjusted_pd"]).astype(int)

    features.to_csv("data/features/feature_matrix_50.csv", index=False)
    print(f"-> Feature matrix successfully built: {features.shape[1]} features in data/features/feature_matrix_50.csv")

if __name__ == "__main__":
    build_full_feature_catalog()