import torch
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, ndcg_score

class RecommenderEvaluator:
    def __init__(self, k_values=[5, 10, 20]):
        self.k_values = k_values
        
    def calculate_metrics(self, y_true, y_pred, y_score):
        """Calculate various recommendation metrics"""
        metrics = {}
        
        # Calculate AUC
        metrics['auc'] = roc_auc_score(y_true, y_score)
        
        # Calculate metrics@k
        for k in self.k_values:
            # Get top-k predictions
            top_k_items = np.argsort(y_score)[-k:]
            
            # Precision@k
            metrics[f'precision@{k}'] = precision_score(
                y_true, 
                np.isin(np.arange(len(y_true)), top_k_items)
            )
            
            # Recall@k
            metrics[f'recall@{k}'] = recall_score(
                y_true,
                np.isin(np.arange(len(y_true)), top_k_items)
            )
            
            # NDCG@k
            metrics[f'ndcg@{k}'] = ndcg_score(
                y_true.reshape(1, -1),
                y_score.reshape(1, -1),
                k=k
            )
            
        return metrics
    
    def evaluate_model(self, model, data_loader, device):
        """Evaluate model on given data loader"""
        model.eval()
        
        all_y_true = []
        all_y_score = []
        
        with torch.no_grad():
            for batch in data_loader:
                # Move batch to device
                batch = batch.to(device)
                
                # Get predictions
                y_score = model.predict(batch.user_idx, batch.item_idx)
                
                # Store results
                all_y_true.append(batch.y.cpu().numpy())
                all_y_score.append(y_score.cpu().numpy())
                
        # Concatenate results
        y_true = np.concatenate(all_y_true)
        y_score = np.concatenate(all_y_score)
        y_pred = (y_score > 0.5).astype(int)
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_true, y_pred, y_score)
        
        return metrics 