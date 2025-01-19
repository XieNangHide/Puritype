import numpy as np
from causallearn.search.ScoreBased.GES import ges
from causallearn.utils.GraphUtils import GraphUtils

class CausalDiscovery:
    def __init__(self, data):
        self.data = data
        self.causal_graph = None
        
    def prepare_features(self):
        """Prepare features for causal discovery"""
        features = self.data[['behavior_code', 'hour', 'day', 'category_idx']].values
        return features
    
    def discover_causal_graph(self):
        """Discover causal relationships using GES algorithm"""
        print("Discovering causal relationships...")
        
        # Prepare feature matrix
        X = self.prepare_features()
        
        # Run GES algorithm
        record = ges(X)
        self.causal_graph = record['G']
        
        # Convert to adjacency matrix
        adj_matrix = GraphUtils.to_adj_matrix(self.causal_graph)
        
        return adj_matrix
    
    def get_causal_effects(self):
        """Extract causal effects from the discovered graph"""
        if self.causal_graph is None:
            raise ValueError("Must run discover_causal_graph first!")
            
        # Get direct causal effects
        effects = {}
        adj_matrix = GraphUtils.to_adj_matrix(self.causal_graph)
        
        feature_names = ['behavior', 'hour', 'day', 'category']
        for i in range(len(feature_names)):
            for j in range(len(feature_names)):
                if adj_matrix[i][j] != 0:
                    effects[(feature_names[i], feature_names[j])] = adj_matrix[i][j]
                    
        return effects 