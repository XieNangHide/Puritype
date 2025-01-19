import numpy as np
from dowhy import CausalModel
import pandas as pd
from sklearn.preprocessing import StandardScaler

class CausalEffectEstimator:
    def __init__(self, data, causal_graph):
        self.data = data
        self.causal_graph = causal_graph
        self.scaler = StandardScaler()
        
    def estimate_treatment_effects(self):
        """Estimate treatment effects for different behaviors"""
        print("Estimating treatment effects...")
        
        # Prepare data
        features = ['behavior_code', 'hour', 'day', 'category_idx']
        X = self.data[features].copy()
        X = pd.DataFrame(self.scaler.fit_transform(X), columns=features)
        
        effects = {}
        treatments = ['behavior_code']
        outcomes = ['category_idx']  # We want to estimate effect on category selection
        common_causes = ['hour', 'day']
        
        # Create causal model
        model = CausalModel(
            data=X,
            treatment=treatments,
            outcome=outcomes,
            common_causes=common_causes,
            graph=self.causal_graph
        )
        
        # Identify causal effect
        identified_estimand = model.identify_effect()
        
        # Estimate causal effect
        estimate = model.estimate_effect(
            identified_estimand,
            method_name="backdoor.linear_regression"
        )
        
        effects['behavior_on_category'] = estimate.value
        
        return effects
    
    def estimate_propensity_scores(self):
        """Estimate propensity scores for different behaviors"""
        from sklearn.linear_model import LogisticRegression
        
        print("Estimating propensity scores...")
        
        # Prepare features
        features = ['hour', 'day', 'category_idx']
        X = self.data[features].copy()
        X = self.scaler.fit_transform(X)
        
        # For each behavior type
        propensity_scores = {}
        for behavior in range(4):  # 4 behavior types
            y = (self.data['behavior_code'] == behavior).astype(int)
            
            # Train logistic regression
            model = LogisticRegression(max_iter=1000)
            model.fit(X, y)
            
            # Calculate propensity scores
            scores = model.predict_proba(X)[:, 1]
            propensity_scores[behavior] = scores
            
        return propensity_scores 