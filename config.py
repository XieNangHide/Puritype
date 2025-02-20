import os

class Config:
    # Data paths
    DATA_PATH = os.path.join("/kaggle/input/taobao1m", "Taobao1M.csv")
    
    # Model parameters
    HIDDEN_CHANNELS = 64
    NUM_LAYERS = 2
    DROPOUT = 0.2
    LEARNING_RATE = 0.001
    
    # Training parameters
    BATCH_SIZE = 1024
    NUM_EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 5
    
    # Random seed for reproducibility
    SEED = 42
    
    # Evaluation metrics
    TOP_K = [5, 10, 20]
    
    @staticmethod
    def validate_paths():
        """Validate that all required paths exist"""
        if not os.path.exists(Config.DATA_PATH):
            raise FileNotFoundError(f"Data file not found at {Config.DATA_PATH}") 