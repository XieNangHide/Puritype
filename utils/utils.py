import torch
import numpy as np
import random

def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def create_edge_index(interaction_matrix):
    """Create edge index for PyTorch Geometric from interaction matrix"""
    edges = np.where(interaction_matrix > 0)
    edge_index = torch.tensor([edges[0], edges[1]], dtype=torch.long)
    return edge_index

def split_data(data, train_ratio=0.8, val_ratio=0.1):
    """Split data into train, validation and test sets"""
    n = len(data)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)
    
    indices = np.random.permutation(n)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    return train_indices, val_indices, test_indices 