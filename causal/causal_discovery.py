import numpy as np
import networkx as nx
from causallearn.search.ScoreBased.GES import ges
from typing import Dict, List, Tuple, Optional

class CausalDiscovery:
    def __init__(self, data):
        """Initialize CausalDiscovery with data
        
        Args:
            data: pandas DataFrame containing behavior data
        """
        self.data = data
        self.causal_graph = None
        self.feature_names = ['behavior', 'hour', 'day', 'category']
        
    def prepare_features(self) -> np.ndarray:
        """Prepare features for causal discovery
        
        Returns:
            np.ndarray: Feature matrix for causal discovery
        """
        # Standardize features before causal discovery
        features = self.data[['behavior_code', 'hour', 'day', 'category_idx']].values
        features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)
        return features
    
    def convert_to_adjacency_matrix(self, graph: Dict) -> np.ndarray:
        """Convert graph dictionary to adjacency matrix
        
        Args:
            graph: Dictionary containing graph structure from GES
            
        Returns:
            np.ndarray: Adjacency matrix
        """
        n_features = len(self.feature_names)
        adj_matrix = np.zeros((n_features, n_features))
        
        # Extract edges from GES result
        if hasattr(graph, 'G'):
            # Get the graph structure
            G = graph.G
            # Convert to adjacency matrix
            for i in range(n_features):
                for j in range(n_features):
                    if G[i, j] != 0:
                        adj_matrix[i][j] = 1
        
        return adj_matrix
    
    def discover_causal_graph(self) -> np.ndarray:
        """Discover causal relationships using GES algorithm
        
        Returns:
            np.ndarray: Adjacency matrix representing causal graph
        """
        print("Discovering causal relationships...")
        
        try:
            # Prepare feature matrix
            X = self.prepare_features()
            
            # Run GES algorithm with parameters
            record = ges(X, 
                        score_func='local_score_BIC',
                        maxP=4,  # Maximum number of parents
                        parameters=None)
            self.causal_graph = record
            
            # Convert to adjacency matrix
            adj_matrix = self.convert_to_adjacency_matrix(record)
            
            print("Causal discovery completed!")
            return adj_matrix
            
        except Exception as e:
            print(f"Error in causal discovery: {str(e)}")
            # Return empty adjacency matrix in case of error
            return np.zeros((len(self.feature_names), len(self.feature_names)))
    
    def get_causal_effects(self) -> Dict[Tuple[str, str], float]:
        """Extract causal effects from the discovered graph
        
        Returns:
            Dict[Tuple[str, str], float]: Dictionary mapping feature pairs to their causal effect strengths
        """
        if self.causal_graph is None:
            raise ValueError("Must run discover_causal_graph first!")
            
        # Get direct causal effects
        effects = {}
        adj_matrix = self.convert_to_adjacency_matrix(self.causal_graph)
        
        for i in range(len(self.feature_names)):
            for j in range(len(self.feature_names)):
                if adj_matrix[i][j] != 0:
                    effects[(self.feature_names[i], self.feature_names[j])] = adj_matrix[i][j]
                    
        return effects 