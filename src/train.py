"""
Model Training Pipeline
Baseline + XGBoost + LightGBM with proper handling of imbalance and leakage.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
import xgboost as xgb
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV

from preprocessing import get_preprocessor, split_data, check_data_leakage, NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET
from features import engineer_features, ENGINEERED_NUMERIC

def load_and_prepare_data():
    data_path = Path(__file__).parent.parent / "data" / "leads_raw.csv"
    df = pd.read_csv(data_path)
    
    print(f"Loaded {len(df)} rows")
    check_data_leakage(df)
    
    # Feature engineering
    df = engineer_features(df)
    
    # Split
    train_df, val_df, test_df = split_data(df)
    
    return train_df, val_df, test_df

def train_models():
    train_df, val_df, test_df = load_and_prepare_data()
    
    # Prepare features
    feature_cols = ENGINEERED_NUMERIC + CATEGORICAL_FEATURES
    X_train = train_df[feature_cols]
    y_train = train_df[TARGET]
    X_val = val_df[feature_cols]
    y_val = val_df[TARGET]
    X_test = test_df[feature_cols]
    y_test = test_df[TARGET]
    
    preprocessor = get_preprocessor()
    # Override numeric features with engineered list
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, ENGINEERED_NUMERIC),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ])
    
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)
    
    print(f"Processed shape: {X_train_processed.shape}")
    
    # Calculate scale_pos_weight for imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Scale pos weight: {scale_pos_weight:.2f}")
    
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42, C=0.8),
        'RandomForest': RandomForestClassifier(n_estimators=400, max_depth=12, class_weight='balanced_subsample', random_state=42, n_jobs=-1, min_samples_leaf=5),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=600, max_depth=8, learning_rate=0.03, subsample=0.9, colsample_bytree=0.9,
            scale_pos_weight=scale_pos_weight, random_state=42, n_jobs=-1, eval_metric='logloss', reg_alpha=0.1, reg_lambda=1.0
        ),
        'LightGBM': lgb.LGBMClassifier(
            n_estimators=800, num_leaves=63, max_depth=10, learning_rate=0.03, subsample=0.9, colsample_bytree=0.9,
            class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1, reg_alpha=0.1, reg_lambda=1.0, min_child_samples=10
        )
    }
    
    results = []
    trained_models = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_processed, y_train)
        
        # Val predictions
        y_val_pred = model.predict(X_val_processed)
        y_val_proba = model.predict_proba(X_val_processed)[:, 1]
        
        roc = roc_auc_score(y_val, y_val_proba)
        pr_auc = average_precision_score(y_val, y_val_proba)
        f1 = f1_score(y_val, y_val_pred)
        prec = precision_score(y_val, y_val_pred)
        rec = recall_score(y_val, y_val_pred)
        
        print(f"{name} - ROC-AUC: {roc:.4f}, PR-AUC: {pr_auc:.4f}, F1: {f1:.4f}")
        
        results.append({
            'model': name,
            'roc_auc': roc,
            'pr_auc': pr_auc,
            'f1': f1,
            'precision': prec,
            'recall': rec
        })
        trained_models[name] = model
    
    results_df = pd.DataFrame(results).sort_values('roc_auc', ascending=False)
    print("\n=== Model Comparison ===")
    print(results_df)
    
    # Save results
    reports_path = Path(__file__).parent.parent / "reports"
    reports_path.mkdir(exist_ok=True)
    results_df.to_csv(reports_path / "model_comparison.csv", index=False)
    
    # Best model = LightGBM (usually)
    best_model_name = results_df.iloc[0]['model']
    best_model = trained_models[best_model_name]
    print(f"\nBest model: {best_model_name}")
    
    # Calibrate best model for true propensity scores
    print("Calibrating best model...")
    calibrated = CalibratedClassifierCV(best_model, method='isotonic', cv=3)
    calibrated.fit(X_train_processed, y_train)
    
    # Save models
    models_path = Path(__file__).parent.parent / "models"
    models_path.mkdir(exist_ok=True)
    joblib.dump(preprocessor, models_path / "preprocessor.pkl")
    joblib.dump(best_model, models_path / f"best_model_{best_model_name}.pkl")
    joblib.dump(calibrated, models_path / "calibrated_model.pkl")
    joblib.dump(trained_models, models_path / "all_models.pkl")
    
    # Save test data for evaluation
    test_df.to_csv(Path(__file__).parent.parent / "data" / "test_set.csv", index=False)
    
    # Also save processed test for quick eval
    np.save(models_path / "X_test_processed.npy", X_test_processed)
    np.save(models_path / "y_test.npy", y_test.values)
    np.save(models_path / "X_val_processed.npy", X_val_processed)
    np.save(models_path / "y_val.npy", y_val.values)
    
    return results_df, best_model_name, calibrated, preprocessor

if __name__ == "__main__":
    train_models()
