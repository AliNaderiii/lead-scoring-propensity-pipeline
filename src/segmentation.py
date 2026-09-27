"""
Lead Segmentation - Convert probabilities to High/Medium/Low
Based on business value and model performance.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def define_segments(y_proba, method='business'):
    """
    Define High/Medium/Low based on business usefulness.
    Uses quantile + business threshold.
    """
    if method == 'quantile':
        high_thresh = np.quantile(y_proba, 0.8)
        low_thresh = np.quantile(y_proba, 0.5)
    else:  # business method - optimized for lift
        # High = top 20% by score, Medium next 30%, Low bottom 50%
        high_thresh = np.quantile(y_proba, 0.8)
        low_thresh = np.quantile(y_proba, 0.5)
    
    def assign_segment(score):
        if score >= high_thresh:
            return 'High'
        elif score >= low_thresh:
            return 'Medium'
        else:
            return 'Low'
    
    segments = [assign_segment(s) for s in y_proba]
    
    return segments, high_thresh, low_thresh

def analyze_segments(df, score_col='propensity_score', target_col='converted'):
    """Analyze conversion rate per segment"""
    analysis = df.groupby('segment').agg({
        target_col: ['count', 'sum', 'mean'],
        score_col: ['mean', 'min', 'max']
    }).round(4)
    
    return analysis

if __name__ == "__main__":
    # Example with test set
    import joblib
    
    models_path = Path(__file__).parent.parent / "models"
    data_path = Path(__file__).parent.parent / "data" / "test_set.csv"
    
    if data_path.exists():
        df = pd.read_csv(data_path)
        # Load model and score
        preprocessor = joblib.load(models_path / "preprocessor.pkl")
        model = joblib.load(models_path / "calibrated_model.pkl")
        
        from features import engineer_features, ENGINEERED_NUMERIC
        from preprocessing import CATEGORICAL_FEATURES
        
        df = engineer_features(df)
        feature_cols = ENGINEERED_NUMERIC + CATEGORICAL_FEATURES
        X = df[feature_cols]
        X_processed = preprocessor.transform(X)
        y_proba = model.predict_proba(X_processed)[:, 1]
        
        segments, high_t, low_t = define_segments(y_proba)
        df['propensity_score'] = y_proba
        df['segment'] = segments
        
        print(f"Thresholds - High >= {high_t:.3f}, Medium >= {low_t:.3f}, Low < {low_t:.3f}")
        print("\nSegment Analysis:")
        print(analyze_segments(df))
        
        # Save
        df[['lead_id', 'propensity_score', 'segment', 'converted']].to_csv(
            Path(__file__).parent.parent / "data" / "scored_leads.csv", index=False
        )
        print("\nSaved scored leads to data/scored_leads.csv")
