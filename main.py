from config import Config
from data.data_preprocessing import DataPreprocessor
from causal.causal_discovery import CausalDiscovery
from causal.causal_effects import CausalEffectEstimator
from models.gnn_model import GNNRecommender
from models.causal_gnn_model import CausalGNNRecommender
from evaluation.metrics import RecommenderEvaluator
from utils.utils import set_seed, split_data, create_edge_index
import torch
from torch_geometric.data import DataLoader

def train_model(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        pred = model.predict(batch.user_idx, batch.item_idx)
        loss = criterion(pred, batch.y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(train_loader)

def main():
    # Validate paths
    Config.validate_paths()
    
    # Set random seed and device
    set_seed(Config.SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize data preprocessor
    try:
        preprocessor = DataPreprocessor(Config.DATA_PATH)
        data, interaction_matrix = preprocessor.preprocess()
    except Exception as e:
        print(f"Error during data preprocessing: {str(e)}")
        return
    
    # Split data
    train_indices, val_indices, test_indices = split_data(data)
    
    # Perform causal discovery
    causal_discoverer = CausalDiscovery(data)
    causal_graph = causal_discoverer.discover_causal_graph()
    
    # Estimate causal effects
    effect_estimator = CausalEffectEstimator(data, causal_graph)
    treatment_effects = effect_estimator.estimate_treatment_effects()
    propensity_scores = effect_estimator.estimate_propensity_scores()
    
    # Create edge index for GNN
    edge_index = create_edge_index(interaction_matrix)
    
    # Initialize models
    num_users = len(preprocessor.user_mapping)
    num_items = len(preprocessor.item_mapping)
    
    base_model = GNNRecommender(
        num_users=num_users,
        num_items=num_items,
        hidden_channels=Config.HIDDEN_CHANNELS,
        num_layers=Config.NUM_LAYERS,
        dropout=Config.DROPOUT
    ).to(device)
    
    causal_model = CausalGNNRecommender(
        num_users=num_users,
        num_items=num_items,
        hidden_channels=Config.HIDDEN_CHANNELS,
        num_layers=Config.NUM_LAYERS,
        dropout=Config.DROPOUT,
        causal_adj=causal_graph
    ).to(device)
    
    # Initialize evaluator
    evaluator = RecommenderEvaluator(k_values=Config.TOP_K)
    
    # Train and evaluate models
    criterion = torch.nn.BCELoss()
    base_optimizer = torch.optim.Adam(base_model.parameters(), lr=Config.LEARNING_RATE)
    causal_optimizer = torch.optim.Adam(causal_model.parameters(), lr=Config.LEARNING_RATE)
    
    # Create data loaders
    train_loader = DataLoader(data[train_indices], batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(data[val_indices], batch_size=Config.BATCH_SIZE)
    test_loader = DataLoader(data[test_indices], batch_size=Config.BATCH_SIZE)
    
    # Training loop
    for epoch in range(Config.NUM_EPOCHS):
        # Train base model
        base_loss = train_model(base_model, train_loader, base_optimizer, criterion, device)
        
        # Train causal model
        causal_loss = train_model(causal_model, train_loader, causal_optimizer, criterion, device)
        
        # Evaluate models
        if epoch % 5 == 0:
            base_metrics = evaluator.evaluate_model(base_model, val_loader, device)
            causal_metrics = evaluator.evaluate_model(causal_model, val_loader, device)
            
            print(f"Epoch {epoch}:")
            print(f"Base Model - Loss: {base_loss:.4f}, AUC: {base_metrics['auc']:.4f}")
            print(f"Causal Model - Loss: {causal_loss:.4f}, AUC: {causal_metrics['auc']:.4f}")
    
    # Final evaluation on test set
    base_metrics = evaluator.evaluate_model(base_model, test_loader, device)
    causal_metrics = evaluator.evaluate_model(causal_model, test_loader, device)
    
    print("\nFinal Test Results:")
    print("Base Model Metrics:", base_metrics)
    print("Causal Model Metrics:", causal_metrics)
    
    # Save models and results
    torch.save({
        'base_model_state': base_model.state_dict(),
        'causal_model_state': causal_model.state_dict(),
        'base_metrics': base_metrics,
        'causal_metrics': causal_metrics,
        'treatment_effects': treatment_effects,
        'propensity_scores': propensity_scores
    }, 'model_results.pt')

if __name__ == "__main__":
    main() 