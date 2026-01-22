import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
import numpy as np
import plotly.express as px

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GraphSAGE(torch.nn.Module):
    """
    Enhanced GraphSAGE model with dropout and batch normalization.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super(GraphSAGE, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        # First layer
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        # Output layer
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        # Batch normalization layers
        self.batch_norms = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_channels))

    def forward(self, x, edge_index):
        # Apply convolutions
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Final layer (no activation)
        x = self.convs[-1](x, edge_index)

        return x

    def encode(self, x, edge_index):
        """Generate embeddings (same as forward but clearer naming)"""
        return self.forward(x, edge_index)

def train_with_link_prediction(data: Data, epochs=200, embedding_dim=32,
                                hidden_dim=64, learning_rate=0.01,
                                weight_decay=5e-4):
    """
    Trains GraphSAGE using link prediction as unsupervised objective.
    """
    print("\n" + "=" * 70)
    print(f"TRAINING GRAPHSAGE (Link Prediction Loss)")
    print("=" * 70)
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Hidden dimension: {hidden_dim}")
    print(f"Epochs: {epochs}")
    print(f"Device: {device}")

    # Move data to device
    data = data.to(device)

    # Initialize model
    model = GraphSAGE(
        in_channels=data.x.shape[1],
        hidden_channels=hidden_dim,
        out_channels=embedding_dim,
        num_layers=2,
        dropout=0.5
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # Training loop
    model.train()
    loss_history = []

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        # Generate embeddings
        embeddings = model(data.x, data.edge_index)

        # Positive edges (actual edges in the graph)
        pos_edge_index = data.edge_index

        # Generate negative samples (non-existent edges)
        num_pos_edges = pos_edge_index.size(1)

        neg_edge_index = negative_sampling(
            edge_index=pos_edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=min(num_pos_edges, data.num_nodes * 2)
        )

        # Skip batch if no negative edges were generated
        if neg_edge_index.numel() == 0:
            continue

        # Calculate positive scores (should be high)
        pos_src = embeddings[pos_edge_index[0]]
        pos_dst = embeddings[pos_edge_index[1]]
        pos_scores = (pos_src * pos_dst).sum(dim=1)

        # Calculate negative scores (should be low)
        neg_src = embeddings[neg_edge_index[0]]
        neg_dst = embeddings[neg_edge_index[1]]
        neg_scores = (neg_src * neg_dst).sum(dim=1)

        # Binary cross-entropy loss with epsilon for stability
        pos_loss = -torch.log(torch.sigmoid(pos_scores) + 1e-15).mean()

        if neg_scores.numel() > 0:
            neg_loss = -torch.log(1 - torch.sigmoid(neg_scores) + 1e-15).mean()
        else:
            neg_loss = torch.tensor(0.0, device=device)

        loss = pos_loss + neg_loss

        # Backpropagation
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        # Print progress
        if epoch % 20 == 0 or epoch == 1:
            print(f'Epoch {epoch:03d} | Loss: {loss:.4f} | Pos: {pos_loss:.4f} | Neg: {neg_loss:.4f}')

    # Extract final embeddings
    model.eval()
    with torch.no_grad():
        final_embeddings = model(data.x, data.edge_index).cpu().numpy()

    print(f"\n✅ Training complete!")
    print(f"Final embeddings shape: {final_embeddings.shape}")
    print("=" * 70)

    return final_embeddings, model, loss_history

def train_with_contrastive_loss(data: Data, epochs=200, embedding_dim=32,
                                 hidden_dim=64, learning_rate=0.01):
    """
    Alternative training using contrastive learning.
    """
    print("\n" + "=" * 70)
    print(f"TRAINING GRAPHSAGE (Contrastive Learning)")
    print("=" * 70)

    data = data.to(device)

    model = GraphSAGE(
        in_channels=data.x.shape[1],
        hidden_channels=hidden_dim,
        out_channels=embedding_dim,
        num_layers=2,
        dropout=0.3
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)

    loss_history = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        # Get embeddings
        embeddings = model(data.x, data.edge_index)

        # Create corrupted features (shuffle node features)
        # Ensure random permutation is on the same device
        idx = torch.randperm(data.num_nodes, device=device)
        corrupted_x = data.x[idx]
        corrupted_embeddings = model(corrupted_x, data.edge_index)

        # Discriminator: positive samples should have high scores
        pos_scores = torch.sigmoid((embeddings * embeddings).sum(dim=1))

        # Negative samples should have low scores
        neg_scores = torch.sigmoid((embeddings * corrupted_embeddings).sum(dim=1))

        # Binary cross-entropy loss
        pos_loss = -torch.log(pos_scores + 1e-15).mean()
        neg_loss = -torch.log(1 - neg_scores + 1e-15).mean()
        loss = pos_loss + neg_loss

        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if epoch % 20 == 0 or epoch == 1:
            print(f'Epoch {epoch:03d} | Loss: {loss:.4f}')

    model.eval()
    with torch.no_grad():
        final_embeddings = model(data.x, data.edge_index).cpu().numpy()

    print(f"\n✅ Training complete! Embeddings shape: {final_embeddings.shape}")
    print("=" * 70)

    return final_embeddings, model, loss_history
