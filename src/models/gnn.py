import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split

class GraphAttentionLayer(nn.Module):
    """Custom Attention Message Passing Layer for Supply Chain Graphs"""
    def __init__(self, in_features, out_features, dropout=0.2, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        # Linear transformation
        Wh = torch.mm(h, self.W)
        N = Wh.size()[0]

        # Attention mechanism
        a_input = torch.cat([Wh.repeat(1, N).view(N * N, -1), Wh.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        h_prime = torch.matmul(attention, Wh)
        return F.elu(h_prime)

class SupplyChainGNN(nn.Module):
    """3-Layer HetGAT Network Risk Embedding Model"""
    def __init__(self, nfeat, nhid, nclass, dropout=0.2):
        super(SupplyChainGNN, self).__init__()
        self.gat1 = GraphAttentionLayer(nfeat, nhid, dropout=dropout)
        self.gat2 = GraphAttentionLayer(nhid, nhid, dropout=dropout)
        self.classifier = nn.Linear(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gat1(x, adj)
        x = F.dropout(x, self.dropout, training=self.training)
        embeddings = self.gat2(x, adj)
        out = self.classifier(embeddings)
        return out, embeddings

def train_gnn_pipeline():
    print("Loading graph data for GNN training...")
    nodes_df = pd.read_csv("data/raw/supply_chain_nodes.csv")
    edges_df = pd.read_csv("data/raw/supply_chain_edges.csv")
    features_df = pd.read_csv("data/features/feature_matrix_50.csv")

    # Map node IDs to indices
    node_list = nodes_df["node_id"].tolist()
    node2idx = {n: i for i, n in enumerate(node_list)}
    N = len(node_list)

    # 1. Build Adjacency Matrix
    adj = np.eye(N, dtype=np.float32)  # Self-loops
    for _, row in edges_df.iterrows():
        if row["source"] in node2idx and row["target"] in node2idx:
            u, v = node2idx[row["source"]], node2idx[row["target"]]
            adj[u, v] = 1.0
            adj[v, u] = 1.0  # Bi-directional flow propagation

    adj_tensor = torch.tensor(adj, dtype=torch.float32)

    # 2. Extract Feature Tensor (Numerical columns)
    feature_cols = [c for c in features_df.columns if c not in ["node_id", "target_default_12m"]]
    X = features_df[feature_cols].fillna(0).values
    # Standardize
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-7)
    x_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(features_df["target_default_12m"].values, dtype=torch.long)

    # Train / Test Mask split
    train_idx, test_idx = train_test_split(np.arange(N), test_size=0.30, random_state=42, stratify=y_tensor.numpy())
    
    # 3. Model Setup
    model = SupplyChainGNN(nfeat=X.shape[1], nhid=64, nclass=2, dropout=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    # 4. Training Loop
    model.train()
    for epoch in range(1, 101):
        optimizer.zero_grad()
        logits, _ = model(x_tensor, adj_tensor)
        loss = criterion(logits[train_idx], y_tensor[train_idx])
        loss.backward()
        optimizer.step()

    # 5. Evaluation
    model.eval()
    with torch.no_grad():
        logits, embeddings = model(x_tensor, adj_tensor)
        probs = F.softmax(logits, dim=1)[:, 1].numpy()
        preds = np.argmax(logits.numpy(), axis=1)

    test_auc = roc_auc_score(y_tensor[test_idx].numpy(), probs[test_idx])
    test_acc = accuracy_score(y_tensor[test_idx].numpy(), preds[test_idx])

    print("=" * 50)
    print("SUPPLY CHAIN GNN EVALUATION RESULTS")
    print(f"Graph Nodes:      {N} facilities/suppliers")
    print(f"Graph Edges:      {len(edges_df)} commercial links")
    print(f"Link / Risk AUC:  {test_auc:.4f} (Target: >0.75)")
    print(f"Node Acc:         {test_acc:.4f} (Target: >0.70)")
    print("Embedding Shape: ", embeddings.shape)
    print("=" * 50)

    # Save embeddings for stacking ensemble
    np.save("data/processed/gnn_embeddings.npy", embeddings.numpy())
    print("GNN Node Risk Embeddings saved to data/processed/gnn_embeddings.npy")

if __name__ == "__main__":
    train_gnn_pipeline()