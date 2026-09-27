"""
SHAP Explainability - Real SHAP plots
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
import shap

def generate_shap_plots():
    models_path = Path(__file__).parent.parent / "models"
    reports_path = Path(__file__).parent.parent / "reports"
    reports_path.mkdir(exist_ok=True)
    
    X_val = np.load(models_path / "X_val_processed.npy")
    y_val = np.load(models_path / "y_val.npy")
    preprocessor = joblib.load(models_path / "preprocessor.pkl")
    best_model = joblib.load(models_path / "calibrated_model.pkl")
    
    # For SHAP, use underlying model if calibrated
    try:
        explainer_model = best_model.calibrated_classifiers_[0].base_estimator
    except:
        explainer_model = best_model
    
    # Sample for speed (SHAP is expensive)
    sample_idx = np.random.choice(len(X_val), min(500, len(X_val)), replace=False)
    X_sample = X_val[sample_idx]
    
    print("Creating SHAP explainer...")
    # Use TreeExplainer for tree models, otherwise Kernel
    try:
        explainer = shap.TreeExplainer(explainer_model)
        shap_values = explainer.shap_values(X_sample)
        # For binary classification, shap_values might be list
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    except Exception as e:
        print(f"TreeExplainer failed {e}, using KernelExplainer")
        # Fallback - use smaller sample
        background = shap.sample(X_val, 50)
        explainer = shap.KernelExplainer(explainer_model.predict_proba, background)
        shap_values = explainer.shap_values(X_sample, nsamples=100)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    
    # Try to get feature names
    try:
        from features import ENGINEERED_NUMERIC
        from preprocessing import CATEGORICAL_FEATURES
        cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES)
        feature_names = ENGINEERED_NUMERIC + list(cat_names)
        # Truncate to match shape
        feature_names = feature_names[:X_sample.shape[1]]
    except:
        feature_names = [f'feature_{i}' for i in range(X_sample.shape[1])]
    
    print(f"SHAP values shape: {shap_values.shape}")
    
    # 1. Summary Plot (Beeswarm)
    plt.figure()
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(reports_path / "shap_summary_real.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Bar Plot (Mean |SHAP|)
    plt.figure()
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, plot_type='bar', show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(reports_path / "shap_bar_real.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Feature Importance from model (if available)
    try:
        if hasattr(explainer_model, 'feature_importances_'):
            importances = explainer_model.feature_importances_
            indices = np.argsort(importances)[-15:]
            plt.figure(figsize=(8,6))
            plt.barh(range(len(indices)), importances[indices])
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
            plt.xlabel('Feature Importance')
            plt.title('Top 15 Feature Importance - Real Model')
            plt.tight_layout()
            plt.savefig(reports_path / "feature_importance_real.png", dpi=300, bbox_inches='tight')
            plt.close()
    except Exception as e:
        print(f"Feature importance plot failed: {e}")
    
    print("SHAP plots saved.")
    
    # Save SHAP values
    np.save(reports_path / "shap_values.npy", shap_values)
    
    return shap_values

if __name__ == "__main__":
    generate_shap_plots()
