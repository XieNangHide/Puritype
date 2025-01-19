import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class CausalGNNRecommender(nn.Module):
    def __init__(self, num_users, num_items, hidden_channels, num_layers, dropout, causal_adj):
        super(CausalGNNRecommender, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.hidden_channels = hidden_channels
        
        # Convert causal adjacency matrix to tensor and ensure it's float
        self.causal_adj = torch.tensor(causal_adj, dtype=torch.float32)
        
        # User and item embeddings
        self.user_embedding = nn.Embedding(num_users, hidden_channels)
        self.item_embedding = nn.Embedding(num_items, hidden_channels)
        
        # Causal attention layer
        self.causal_attention = nn.Linear(hidden_channels, hidden_channels)
        
        # GNN layers
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            
        self.dropout = dropout
        
    def apply_causal_attention(self, x):
        """Apply causal attention using causal adjacency matrix"""
        # Ensure causal_adj is on the same device as x
        self.causal_adj = self.causal_adj.to(x.device)
        
        # Apply attention
        attention_weights = torch.matmul(
            self.causal_attention(x),
            self.causal_adj
        )
        attention_weights = F.softmax(attention_weights, dim=1)
        return torch.matmul(attention_weights, x)
    
    def forward(self, edge_index):
        # Get initial embeddings
        x = torch.cat([
            self.user_embedding.weight,
            self.item_embedding.weight
        ], dim=0)
        
        # Apply causal attention
        x = self.apply_causal_attention(x)
        
        # Apply GNN layers
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        # Split back into user and item embeddings
        user_embeddings = x[:self.num_users]
        item_embeddings = x[self.num_users:]
        
        return user_embeddings, item_embeddings
    
    def predict(self, user_indices, item_indices, edge_index):
        """Predict interaction scores for user-item pairs"""
        user_embeddings, item_embeddings = self.forward(edge_index)
        
        # Get specific user and item embeddings
        user_emb = user_embeddings[user_indices]
        item_emb = item_embeddings[item_indices]
        
        # Calculate prediction scores
        scores = torch.sum(user_emb * item_emb, dim=1)
        return torch.sigmoid(scores) 