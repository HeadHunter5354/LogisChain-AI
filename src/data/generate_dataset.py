import os
import numpy as np
import pandas as pd

np.random.seed(42)
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# 1. 365 Days of Port Throughput Time Series
dates = pd.date_range(start="2025-01-01", periods=365, freq="D")
t = np.arange(365)
seasonal = 10000 * np.sin(2 * np.pi * t / 365.25) + 3000 * np.cos(2 * np.pi * t / 7)
trend = 50 * t
noise = np.random.normal(0, 4000, 365)
base_throughput = 120000 + seasonal + trend + noise

port_df = pd.DataFrame({
    "date": dates,
    "port_id": "PORT_SHANGHAI",
    "throughput_teu": np.clip(base_throughput, 50000, 200000).astype(int),
    "berth_utilization": np.clip(0.75 + 0.15 * np.sin(2 * np.pi * t / 30) + np.random.normal(0, 0.05, 365), 0.4, 0.98),
    "avg_dwell_time_days": np.clip(3.5 + 2.0 * (base_throughput > 135000) + np.random.exponential(1.0, 365), 1.5, 14.0),
    "container_spot_rate": np.clip(2200 + 15 * t + 800 * np.sin(2 * np.pi * t / 180) + np.random.normal(0, 150, 365), 1200, 8500)
})
port_df.to_csv("data/raw/port_throughput.csv", index=False)

# 2. Supply Chain Network Graph Entities (250 Nodes)
n_suppliers, n_manufacturers = 150, 60
nodes = []

for i in range(n_suppliers):
    nodes.append({
        "node_id": f"SUP_{i:03d}",
        "tier": "Supplier",
        "revenue_mil": np.random.uniform(20, 200),
        "cogs_mil": np.random.uniform(15, 160),
        "inventory_mil": np.random.uniform(2, 30),
        "ar_mil": np.random.uniform(3, 40),
        "ap_mil": np.random.uniform(2, 25),
        "current_assets_mil": np.random.uniform(10, 60),
        "current_liab_mil": np.random.uniform(8, 45),
        "total_debt_mil": np.random.uniform(5, 50),
        "total_equity_mil": np.random.uniform(10, 80),
        "ebitda_mil": np.random.uniform(2, 35),
        "interest_exp_mil": np.random.uniform(0.5, 6),
        "total_orders": np.random.randint(500, 3000),
        "on_time_orders": 0,
        "lead_time_mean": np.random.uniform(8, 30),
        "lead_time_std": np.random.uniform(1.2, 7.5),
        "country_risk": np.random.uniform(20, 85),
        "disaster_risk": np.random.uniform(0.05, 0.65)
    })

node_df = pd.DataFrame(nodes)
node_df["otif_rate"] = np.clip(np.random.beta(a=9, b=1.5, size=len(node_df)), 0.50, 0.99)
node_df["on_time_orders"] = (node_df["total_orders"] * node_df["otif_rate"]).astype(int)

# 3. Supply Network Edges
edges = []
for s_idx in range(n_suppliers):
    m_targets = np.random.choice(range(n_suppliers, n_suppliers + n_manufacturers), size=np.random.randint(1, 4), replace=False)
    for m in m_targets:
        edges.append({
            "source": f"SUP_{s_idx:03d}",
            "target": f"MFG_{m - n_suppliers:03d}",
            "flow_value_mil": np.random.uniform(1.0, 25.0),
            "edge_type": "material_flow",
            "transit_mode": np.random.choice(["ocean", "truck", "rail"], p=[0.6, 0.3, 0.1])
        })

edge_df = pd.DataFrame(edges)
node_df.to_csv("data/raw/supply_chain_nodes.csv", index=False)
edge_df.to_csv("data/raw/supply_chain_edges.csv", index=False)
print("-> Successfully generated raw data in data/raw/")