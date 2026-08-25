import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss

def run_trade_finance_pricing_engine():
    print("Running Trade Finance & SC-Adjusted Credit Scorer...")
    df = pd.read_csv("data/features/feature_matrix_50.csv")

    # 1. Baseline Financial PD vs SC-Enhanced PD
    baseline_pd = df["baseline_financial_pd"].values
    sc_adjusted_pd = df["sc_adjusted_pd"].values
    y_true = df["target_default_12m"].values

    # Evaluation comparison: Financial-only vs SC-Enhanced
    auc_base = roc_auc_score(y_true, baseline_pd)
    auc_enhanced = roc_auc_score(y_true, sc_adjusted_pd)
    gini_enhanced = 2 * auc_enhanced - 1
    ece = np.abs(sc_adjusted_pd.mean() - y_true.mean())

    print("=" * 55)
    print("CROSS-DOMAIN CREDIT RISK BENCHMARK (SC-PD)")
    print(f"Financial-Only Baseline AUC: {auc_base:.4f}")
    print(f"SC-Enhanced SC-PD AUC:       {auc_enhanced:.4f} (Target: >0.84)")
    print(f"Model Gini Index:            {gini_enhanced:.4f} (Target: >0.65)")
    print(f"Expected Calibration Error:  {ece:.4f} (Target: <0.03)")
    print("=" * 55)

    # 2. Portfolio Exposure at Default (EAD) & Loss Given Default (LGD)
    ead_facility_mil = np.random.uniform(2.0, 30.0, len(df))
    lgd_rate = df["expected_loss_given_default"].values
    expected_loss_mil = sc_adjusted_pd * lgd_rate * ead_facility_mil

    # 3. Dynamic Trade Finance Pricing Engine
    # Base spread 100 bps + SC-Risk Premium + Port Congestion penalty
    base_spread_bps = 100.0
    sc_risk_spread_bps = sc_adjusted_pd * 2500.0
    port_congestion_bps = (df["port_congestion_index"] / 5.0) * 80.0
    total_financing_spread_bps = base_spread_bps + sc_risk_spread_bps + port_congestion_bps

    # Letter of Credit (LC) Approval Decision Rule (Max risk tolerance: PD < 8%)
    lc_approved = (sc_adjusted_pd < 0.08).astype(int)

    trade_finance_df = pd.DataFrame({
        "node_id": df["node_id"],
        "facility_limit_mil": np.round(ead_facility_mil, 2),
        "baseline_pd": np.round(baseline_pd, 4),
        "sc_adjusted_pd": np.round(sc_adjusted_pd, 4),
        "lgd_rate": np.round(lgd_rate, 2),
        "expected_loss_mil": np.round(expected_loss_mil, 3),
        "pricing_spread_bps": np.round(total_financing_spread_bps, 1),
        "lc_approval_status": lc_approved
    })

    os.makedirs("data/processed", exist_ok=True)
    trade_finance_df.to_csv("data/processed/trade_finance_risk_ledger.csv", index=False)
    print("\nTrade Finance Risk Ledger saved to data/processed/trade_finance_risk_ledger.csv")

if __name__ == "__main__":
    run_trade_finance_pricing_engine()