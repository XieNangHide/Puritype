try:
    import pandas as pd
    import numpy as np
    from datetime import datetime
except ImportError as e:
    raise ImportError(f"Required package not found. Please install requirements.txt: {str(e)}")

class DataPreprocessor:
    def __init__(self, file_path):
        """Initialize the DataPreprocessor with the path to the data file"""
        if not isinstance(file_path, str):
            raise TypeError("file_path must be a string")
            
        self.file_path = file_path
        self.data = None
        self.user_mapping = {}
        self.item_mapping = {}
        self.category_mapping = {}
        
    def load_data(self):
        """Load and preprocess the UserBehavior dataset"""
        print("Loading data...")
        try:
            # Read CSV with header row
            self.data = pd.read_csv(
                self.file_path,
                header=0,  # Read first row as header
                dtype={
                    'userid': np.int64,
                    'itemid': np.int64,
                    'categoryid': np.int64,
                    'type': str,
                    'timestamp': np.int64
                },
                low_memory=False
            )
            
            # Rename columns to match our expected format
            self.data = self.data.rename(columns={
                'userid': 'user_id',
                'itemid': 'item_id',
                'categoryid': 'category_id',
                'type': 'behavior_type'
            })
            
            return self.data
            
        except Exception as e:
            raise RuntimeError(f"Error loading data: {str(e)}")
    
    def create_mappings(self):
        """Create mappings for user, item and category IDs"""
        print("Creating ID mappings...")
        self.user_mapping = {id_: idx for idx, id_ in enumerate(self.data['user_id'].unique())}
        self.item_mapping = {id_: idx for idx, id_ in enumerate(self.data['item_id'].unique())}
        self.category_mapping = {id_: idx for idx, id_ in enumerate(self.data['category_id'].unique())}
        
        # Apply mappings
        self.data['user_idx'] = self.data['user_id'].map(self.user_mapping)
        self.data['item_idx'] = self.data['item_id'].map(self.item_mapping)
        self.data['category_idx'] = self.data['category_id'].map(self.category_mapping)
        
    def encode_behavior(self):
        """Encode behavior types"""
        print("Encoding behavior types...")
        behavior_mapping = {'pv': 0, 'cart': 1, 'fav': 2, 'buy': 3}
        self.data['behavior_code'] = self.data['behavior_type'].map(behavior_mapping)
        
    def process_timestamps(self):
        """Process timestamps into datetime and add time-based features"""
        print("Processing timestamps...")
        # Convert timestamp column to numeric, handling any string values
        self.data['timestamp'] = pd.to_numeric(self.data['timestamp'], errors='coerce')
        
        # Convert to datetime
        self.data['datetime'] = pd.to_datetime(self.data['timestamp'], unit='s')
        self.data['hour'] = self.data['datetime'].dt.hour
        self.data['day'] = self.data['datetime'].dt.day
        
    def create_interaction_matrix(self):
        """Create user-item interaction matrix"""
        print("Creating interaction matrix...")
        n_users = len(self.user_mapping)
        n_items = len(self.item_mapping)
        
        # Create sparse interaction matrix
        interaction_matrix = np.zeros((n_users, n_items))
        for _, row in self.data.iterrows():
            interaction_matrix[row['user_idx'], row['item_idx']] = row['behavior_code'] + 1
            
        return interaction_matrix
    
    def preprocess(self):
        """Run all preprocessing steps"""
        try:
            self.load_data()
            self.create_mappings()
            self.encode_behavior()
            self.process_timestamps()
            interaction_matrix = self.create_interaction_matrix()
            
            print("Preprocessing completed!")
            return self.data, interaction_matrix
        except Exception as e:
            print(f"Error during preprocessing: {str(e)}")
            raise 