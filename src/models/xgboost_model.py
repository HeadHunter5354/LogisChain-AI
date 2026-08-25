import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss, classification_report
import shap

def train_xgboost_pipeline():
    # 1. Load Feature Matrix from Day 1
    data_path = "data/features/feature_matrix_50.csv"
    df = pd.read_csv(data_path)
    
    # 2. Separate Features and Target
    drop_cols = ["node_id", "target_default_12m"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    X = df[feature_cols]
    y = df["target_default_12m"]
    
    # Stratified Train-Test Split to avoid data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # 3. Handle Imbalance & Train Model
    scale_pos = (len(y_train) - sum(y_train)) / max(1, sum(y_train))
    
    model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.03,
        scale_pos_weight=scale_pos,
        eval_metric="logloss",
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # 4. Evaluation
    preds_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds_proba)
    brier = brier_score_loss(y_test, preds_proba)
    gini = 2 * auc - 1
    
    print("=" * 50)
    print("XGBOOST DEFAULT PREDICTION BENCHMARK")
    print(f"AUC-ROC Score: {auc:.4f} (Target: >0.80)")
    print(f"Gini Index:    {gini:.4f} (Target: >0.60)")
    print(f"Brier Score:   {brier:.4f} (Calibration check)")
    print("=" * 50)
    
    # 5. SHAP Explainability Analysis
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # Compute Top 5 Global Features
    mean_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_shap)[::-1][:5]
    
    print("\nTop 5 Predictive Features by SHAP Importance:")
    for idx in top_indices:
        print(f" - {feature_cols[idx]}: {mean_shap[idx]:.4f}")
        
    return model, feature_cols

if __name__ == "__main__":
    train_xgboost_pipeline()