"""
Inference / Scoring Code - Production Ready
Scores new leads with propensity and segment.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from features import engineer_features, ENGINEERED_NUMERIC
from preprocessing import CATEGORICAL_FEATURES
from segmentation import define_segments

class LeadScorer:
    def __init__(self, models_dir=None):
        if models_dir is None:
            models_dir = Path(__file__).parent.parent / "models"
        else:
            models_dir = Path(models_dir)
        
        self.preprocessor = joblib.load(models_dir / "preprocessor.pkl")
        self.model = joblib.load(models_dir / "calibrated_model.pkl")
        
        # Load thresholds if available, else compute dynamically
        self.high_thresh = 0.45  # will be calibrated on data
        self.low_thresh = 0.15
    
    def score_leads(self, df):
        """
        Score new leads.
        Input: DataFrame with same schema as training (without converted column)
        Output: DataFrame with propensity_score and segment
        """
        df = df.copy()
        df = engineer_features(df)
        
        feature_cols = ENGINEERED_NUMERIC + CATEGORICAL_FEATURES
        # Ensure all cols exist
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0
        
        X = df[feature_cols]
        X_processed = self.preprocessor.transform(X)
        y_proba = self.model.predict_proba(X_processed)[:, 1]
        
        segments, high_t, low_t = define_segments(y_proba)
        
        result = df.copy()
        result['propensity_score'] = y_proba
        result['segment'] = segments
        result['score_band'] = pd.cut(
            y_proba, 
            bins=[0, low_t, high_t, 1.0],
            labels=['Low (0-0.15)', f'Medium ({low_t:.2f}-{high_t:.2f})', f'High ({high_t:.2f}-1.0)']
        )
        
        # Prioritization rank
        result['priority_rank'] = result['propensity_score'].rank(ascending=False, method='first').astype(int)
        
        return result[['lead_id', 'propensity_score', 'segment', 'score_band', 'priority_rank']]
    
    def score_single_lead(self, lead_dict):
        """Score a single lead dict"""
        df = pd.DataFrame([lead_dict])
        return self.score_leads(df)

def score_new_data(input_path, output_path):
    """CLI for scoring new CSV"""
    df = pd.read_csv(input_path)
    scorer = LeadScorer()
    scored = scorer.score_leads(df)
    scored.to_csv(output_path, index=False)
    print(f"Scored {len(scored)} leads -> {output_path}")
    print(scored['segment'].value_counts())
    return scored

if __name__ == "__main__":
    # Example usage
    data_path = Path(__file__).parent.parent / "data" / "test_set.csv"
    if data_path.exists():
        df = pd.read_csv(data_path)
        # Remove target to simulate new data
        if 'converted' in df.columns:
            df_test = df.drop(columns=['converted'])
        else:
            df_test = df
        
        scorer = LeadScorer()
        result = scorer.score_leads(df_test.head(10))
        print(result.head(10))
        
        # Full scoring
        full_result = scorer.score_leads(df_test)
        full_result.to_csv(Path(__file__).parent.parent / "data" / "scored_leads_demo.csv", index=False)
        print(f"\nFull scoring saved: {len(full_result)} leads")
        print(full_result['segment'].value_counts())
