import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import train_test_split

np.random.seed(42)
torch.manual_seed(42)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=20):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class ShipmentRiskTransformer(nn.Module):
    def __init__(self, input_dim=8, d_model=32, nhead=4, num_layers=2, dim_feedforward=64, dropout=0.1):
        super(ShipmentRiskTransformer, self).__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Multi-output prediction heads
        self.delay_head = nn.Linear(d_model, 1)        # Probability of >5 day delay
        self.damage_head = nn.Linear(d_model, 1)       # Probability of cargo damage
        self.discrepancy_head = nn.Linear(d_model, 1)  # Document discrepancy for LC

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        h = self.input_projection(x)
        h = self.pos_encoder(h)
        encoded = self.transformer_encoder(h)
        
        # Aggregate sequence representation (mean pooling across shipment events)
        pooled = torch.mean(encoded, dim=1)
        
        delay_prob = torch.sigmoid(self.delay_head(pooled))
        damage_prob = torch.sigmoid(self.damage_head(pooled))
        discrepancy_prob = torch.sigmoid(self.discrepancy_head(pooled))
        
        return delay_prob, damage_prob, discrepancy_prob

def generate_synthetic_shipment_data(num_shipments=1000, seq_len=6):
    # 6 event milestones: Booked, Loaded, Departed, In-Transit, Arrived, Customs
    # 8 features per milestone (dwell, vessel speed, temp shock, port index, route risk, etc.)
    X = np.random.randn(num_shipments, seq_len, 8).astype(np.float32)
    
    # Inject synthetic delay signals: elevated dwell time at step 3 & 4 creates delay
    delay_signal = X[:, 2, 0] * 0.8 + X[:, 3, 1] * 1.2 + np.random.normal(0, 0.5, num_shipments)
    y_delay = (delay_signal > 0.5).astype(np.float32)
    
    return X, y_delay

def train_transformer_pipeline():
    print("Generating and processing shipment event streams...")
    X, y_delay = generate_synthetic_shipment_data(num_shipments=1500, seq_len=6)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_delay, test_size=0.25, random_state=42)
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    
    model = ShipmentRiskTransformer(input_dim=8, d_model=32, nhead=4, num_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    criterion = nn.BCELoss()
    
    model.train()
    for epoch in range(1, 81):
        optimizer.zero_grad()
        pred_delay, _, _ = model(X_train_t)
        loss = criterion(pred_delay, y_train_t)
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        test_preds, _, _ = model(X_test_t)
        probs = test_preds.numpy().flatten()
        
    auc = roc_auc_score(y_test, probs)
    brier = brier_score_loss(y_test, probs)
    
    print("=" * 50)
    print("TRANSFORMER SHIPMENT RISK MODEL RESULTS")
    print(f"Delay AUC-ROC:  {auc:.4f} (Target: >0.80)")
    print(f"Brier Score:    {brier:.4f} (Target: <0.18)")
    print("=" * 50)
    
    torch.save(model.state_dict(), "data/processed/transformer_shipment_model.pt")
    print("Model weights saved to data/processed/transformer_shipment_model.pt")

if __name__ == "__main__":
    train_transformer_pipeline()