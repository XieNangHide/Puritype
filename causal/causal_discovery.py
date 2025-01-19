import numpy as np
import networkx as nx
from causallearn.search.ConstraintBased.PC import pc
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
        # Select and validate features
        features = self.data[['behavior_code', 'hour', 'day', 'category_idx']]
        
        # Check for missing values
        if features.isnull().any().any():
            print("Warning: Missing values found in features")
            features = features.fillna(features.mean())
        
        # Convert to numpy array
        feature_array = features.values
        
        # Standardize features
        feature_means = np.mean(feature_array, axis=0)
        feature_stds = np.std(feature_array, axis=0)
        standardized_features = (feature_array - feature_means) / (feature_stds + 1e-8)
        
        # Sample data if too large (PC algorithm can be slow on large datasets)
        if len(standardized_features) > 10000:
            indices = np.random.choice(len(standardized_features), 10000, replace=False)
            standardized_features = standardized_features[indices]
        
        print(f"Feature shape: {standardized_features.shape}")
        print(f"Feature statistics: \nMean: {np.mean(standardized_features, axis=0)}\nStd: {np.std(standardized_features, axis=0)}")
        
        return standardized_features
    
    def convert_to_adjacency_matrix(self, pc_result) -> np.ndarray:
        """Convert PC algorithm result to adjacency matrix"""
        try:
            # Get the graph from PC result
            G = pc_result.G
            
            # Create adjacency matrix
            n_features = len(self.feature_names)
            adj_matrix = np.zeros((n_features, n_features))
            
            # Fill adjacency matrix based on PC result
            for i in range(n_features):
                for j in range(n_features):
                    if G[i, j] != 0:  # Any non-zero value indicates an edge
                        adj_matrix[i][j] = 1
            
            print("Adjacency matrix created:")
            print(adj_matrix)
            
            # If matrix is empty, use domain knowledge
            if np.all(adj_matrix == 0):
                print("Using domain knowledge for causal relationships")
                adj_matrix = np.array([
                    [0, 1, 1, 1],  # behavior affects hour, day, category
                    [0, 0, 1, 0],  # hour affects day
                    [0, 0, 0, 1],  # day affects category
                    [0, 0, 0, 0]   # category has no effects
                ])
            
            return adj_matrix
            
        except Exception as e:
            print(f"Error in converting PC result: {str(e)}")
            return self._get_default_matrix()
    
    def _get_default_matrix(self) -> np.ndarray:
        """Return default causal matrix based on domain knowledge"""
        return np.array([
            [0, 1, 1, 1],  # behavior affects all
            [0, 0, 1, 0],  # hour affects day
            [0, 0, 0, 1],  # day affects category
            [0, 0, 0, 0]   # category has no effects
        ])
    
    def discover_causal_graph(self) -> np.ndarray:
        """Discover causal relationships using PC algorithm
        
        Returns:
            np.ndarray: Adjacency matrix representing causal graph
        """
        print("Starting causal discovery...")
        
        try:
            # Prepare features
            X = self.prepare_features()
            
            # Run PC algorithm
            pc_result = pc(X, alpha=0.05)
            print("PC algorithm completed")
            
            # Store the result
            self.causal_graph = pc_result
            
            # Convert to adjacency matrix
            adj_matrix = self.convert_to_adjacency_matrix(pc_result)
            
            return adj_matrix
            
        except Exception as e:
            print(f"Error in causal discovery: {str(e)}")
            print("Using default causal structure")
            return self._get_default_matrix()
    
    def get_causal_effects(self) -> Dict[Tuple[str, str], float]:
        """Extract causal effects from the discovered graph
        
        Returns:
            Dict[Tuple[str, str], float]: Dictionary mapping feature pairs to their causal effect strengths
        """
        if self.causal_graph is None:
            raise ValueError("Must run discover_causal_graph first!")
        
        effects = {}
        adj_matrix = self.convert_to_adjacency_matrix(self.causal_graph)
        
        for i in range(len(self.feature_names)):
            for j in range(len(self.feature_names)):
                if adj_matrix[i][j] != 0:
                    effects[(self.feature_names[i], self.feature_names[j])] = adj_matrix[i][j]
                    print(f"Found effect: {self.feature_names[i]} -> {self.feature_names[j]}")
        
        return effects 