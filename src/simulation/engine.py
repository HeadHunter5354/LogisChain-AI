import numpy as np
import pandas as pd

class LogisChainSimulationEngine:
    """Three-Layer Gamified Simulation Engine for Trade Finance & SCF"""
    
    SCENARIOS = {
        "None": {"desc": "Normal market conditions", "otif_drop": 0.0, "dwell_add": 0.0, "freight_mult": 1.0},
        "Suez Canal Blockage": {"desc": "Chokepoint blocked: +14d transit, European congestion cascade", "otif_drop": 0.25, "dwell_add": 6.5, "freight_mult": 2.2},
        "Major Port Congestion": {"desc": "Severe berth backlog at major transshipment hub", "otif_drop": 0.15, "dwell_add": 5.0, "freight_mult": 1.4},
        "Carrier Bankruptcy": {"desc": "Major shipping line insolvancy: cargo stranded globally", "otif_drop": 0.30, "dwell_add": 8.0, "freight_mult": 1.8},
        "Demand Bullwhip Shock": {"desc": "Rapid 40% demand swing causing severe inventory mismatch", "otif_drop": 0.10, "dwell_add": 2.0, "freight_mult": 1.2},
        "Extreme Monsoon Flood": {"desc": "Regional manufacturing hub disrupted by natural disaster", "otif_drop": 0.35, "dwell_add": 4.0, "freight_mult": 1.3}
    }

    def __init__(self, data_path="data/processed/trade_finance_risk_ledger.csv"):
        self.ledger = pd.read_csv(data_path)
        self.turn = 1
        self.max_turns = 12  # 1 Quarter (12 Weeks)
        self.portfolio_capital_mil = 500.0
        self.active_scenario = "None"
        
        # Player and AI Opponent State
        self.player_approvals = {}
        self.ai_approvals = {}
        self.history = []

    def trigger_scenario(self, scenario_name):
        if scenario_name in self.SCENARIOS:
            self.active_scenario = scenario_name

    def run_turn(self, player_decisions):
        """
        player_decisions: dict of {node_id: bool (approved or rejected)}
        """
        sc_params = self.SCENARIOS[self.active_scenario]
        
        # 1. Update Physical Layer
        shocked_pd = np.clip(self.ledger["sc_adjusted_pd"] + (sc_params["otif_drop"] * 0.4), 0.01, 0.40)
        
        # 2. AI Opponent Strategy (Approves if shocked PD < 0.085)
        ai_decisions = {row["node_id"]: (shocked_pd[i] < 0.085) for i, row in self.ledger.iterrows()}
        
        # 3. Calculate Financial Outcomes
        player_pnl, player_defaults, player_financed = 0.0, 0, 0.0
        ai_pnl, ai_defaults, ai_financed = 0.0, 0, 0.0

        for i, row in self.ledger.iterrows():
            nid = row["node_id"]
            exposure = row["facility_limit_mil"]
            spread = row["pricing_spread_bps"] / 10000.0
            actual_default = np.random.rand() < shocked_pd[i]

            # Player Outcome
            if player_decisions.get(nid, False):
                player_financed += exposure
                if actual_default:
                    player_pnl -= exposure * row["lgd_rate"]
                    player_defaults += 1
                else:
                    player_pnl += exposure * spread

            # AI Outcome
            if ai_decisions.get(nid, False):
                ai_financed += exposure
                if actual_default:
                    ai_pnl -= exposure * row["lgd_rate"]
                    ai_defaults += 1
                else:
                    ai_pnl += exposure * spread

        # 4. 1000-Point Scoring Framework Calculation
        score_financial = np.clip(int(150 + (player_pnl * 20)), 0, 300)
        score_risk_mgmt = np.clip(int(250 - (player_defaults * 15)), 0, 250)
        score_sc_intel = 175 if self.active_scenario != "None" else 120
        score_speed = 85
        score_learning = np.clip(int(self.turn * 12), 0, 150)
        total_score = score_financial + score_risk_mgmt + score_sc_intel + score_speed + score_learning

        turn_summary = {
            "turn": self.turn,
            "scenario": self.active_scenario,
            "player_pnl_mil": player_pnl,
            "player_defaults": player_defaults,
            "player_financed_mil": player_financed,
            "ai_pnl_mil": ai_pnl,
            "ai_defaults": ai_defaults,
            "ai_financed_mil": ai_financed,
            "total_score": total_score
        }
        self.history.append(turn_summary)
        self.turn += 1
        return turn_summary