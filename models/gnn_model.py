import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.data import Data

class GNNRecommender(nn.Module):
    def __init__(self, num_users, num_items, hidden_channels, num_layers, dropout):
        super(GNNRecommender, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.hidden_channels = hidden_channels
        
        # User and item embeddings
        self.user_embedding = nn.Embedding(num_users, hidden_channels)
        self.item_embedding = nn.Embedding(num_items, hidden_channels)
        
        # GNN layers
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            
        self.dropout = dropout
        
    def forward(self, edge_index):
        # Get initial embeddings
        x = torch.cat([
            self.user_embedding.weight,
            self.item_embedding.weight
        ], dim=0)
        
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