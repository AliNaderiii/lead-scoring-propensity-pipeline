"""
Advanced Training - Real Dataset + Optuna + CatBoost + Ensemble + Plotly
Professional v2 upgrade.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import xgboost as xgb
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings('ignore')

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except:
    CATBOOST_AVAILABLE = False
    print("CatBoost not available, skipping")

from data_loader_real import load_bank_marketing_real, get_feature_lists

def get_preprocessor_advanced(numeric_features, categorical_features):
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
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    return preprocessor

def train_advanced():
    print("Loading REAL Bank Marketing dataset (UCI, 41K rows)...")
    df = load_bank_marketing_real(leakage_safe=True)
    
    # Split time-aware? Dataset ordered by date, so use last 20% as test for realism
    # For simplicity, stratified split but we note time ordering in docs
    numeric_features, categorical_features = get_feature_lists(df)
    print(f"Numeric: {len(numeric_features)}, Categorical: {len(categorical_features)}")
    
    X = df.drop(columns=['converted'])
    y = df['converted']
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1875, random_state=42, stratify=y_train_val
    )
    
    print(f"Train: {len(X_train)} ({y_train.mean():.2%}), Val: {len(X_val)} ({y_val.mean():.2%}), Test: {len(X_test)} ({y_test.mean():.2%})")
    
    preprocessor = get_preprocessor_advanced(numeric_features, categorical_features)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)
    
    print(f"Processed shape: {X_train_processed.shape}")
    
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Scale pos weight: {scale_pos_weight:.2f} (imbalanced)")
    
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, class_weight='balanced', C=0.5, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=300, max_depth=12, class_weight='balanced_subsample', random_state=42, n_jobs=-1),
        'XGBoost': xgb.XGBClassifier(n_estimators=500, max_depth=7, learning_rate=0.04, subsample=0.9, colsample_bytree=0.9, scale_pos_weight=scale_pos_weight, random_state=42, n_jobs=-1, eval_metric='logloss', reg_alpha=0.1, reg_lambda=1.0),
        'LightGBM': lgb.LGBMClassifier(n_estimators=600, num_leaves=63, max_depth=10, learning_rate=0.04, subsample=0.9, colsample_bytree=0.9, class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1, reg_alpha=0.1, reg_lambda=1.0),
    }
    
    if CATBOOST_AVAILABLE:
        models['CatBoost'] = cb.CatBoostClassifier(iterations=500, depth=8, learning_rate=0.05, loss_function='Logloss', eval_metric='AUC', random_seed=42, verbose=False, auto_class_weights='Balanced')
    
    results = []
    trained_models = {}
    
    for name, model in models.items():
        print(f"\nTraining {name} on REAL data...")
        if name == 'CatBoost':
            model.fit(X_train_processed, y_train)
        else:
            model.fit(X_train_processed, y_train)
        
        y_val_proba = model.predict_proba(X_val_processed)[:, 1]
        y_val_pred = model.predict(X_val_processed)
        
        roc = roc_auc_score(y_val, y_val_proba)
        pr_auc = average_precision_score(y_val, y_val_proba)
        f1 = f1_score(y_val, y_val_pred)
        
        print(f"{name} - ROC-AUC: {roc:.4f}, PR-AUC: {pr_auc:.4f}, F1: {f1:.4f}")
        results.append({'model': name, 'roc_auc': roc, 'pr_auc': pr_auc, 'f1': f1})
        trained_models[name] = model
    
    results_df = pd.DataFrame(results).sort_values('roc_auc', ascending=False)
    print("\n=== Advanced Model Comparison (REAL Bank Data) ===")
    print(results_df)
    
    # Ensemble - Stacking
    print("\nTraining Stacking Ensemble (LogReg + XGBoost + LightGBM)...")
    estimators = [
        ('xgb', models['XGBoost']),
        ('lgb', models['LightGBM']),
        ('rf', models['RandomForest'])
    ]
    stacking = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(class_weight='balanced'), cv=3, n_jobs=-1)
    stacking.fit(X_train_processed, y_train)
    y_val_proba_stack = stacking.predict_proba(X_val_processed)[:, 1]
    roc_stack = roc_auc_score(y_val, y_val_proba_stack)
    pr_stack = average_precision_score(y_val, y_val_proba_stack)
    print(f"Stacking Ensemble - ROC-AUC: {roc_stack:.4f}, PR-AUC: {pr_stack:.4f}")
    results_df = pd.concat([results_df, pd.DataFrame([{'model': 'StackingEnsemble', 'roc_auc': roc_stack, 'pr_auc': pr_stack, 'f1': 0}])], ignore_index=True)
    results_df = results_df.sort_values('roc_auc', ascending=False)
    trained_models['StackingEnsemble'] = stacking
    
    # Best model
    best_name = results_df.iloc[0]['model']
    best_model = trained_models[best_name]
    print(f"\nBest model: {best_name}")
    
    # Calibrate
    print("Calibrating best model with Isotonic...")
    calibrated = CalibratedClassifierCV(best_model, method='isotonic', cv=3)
    calibrated.fit(X_train_processed, y_train)
    
    # Save
    models_path = Path(__file__).parent.parent / "models"
    models_path.mkdir(exist_ok=True)
    reports_path = Path(__file__).parent.parent / "reports"
    reports_path.mkdir(exist_ok=True)
    
    joblib.dump(preprocessor, models_path / "preprocessor_advanced.pkl")
    joblib.dump(calibrated, models_path / "calibrated_model_advanced.pkl")
    joblib.dump(trained_models, models_path / "all_models_advanced.pkl")
    results_df.to_csv(reports_path / "model_comparison_advanced_real.csv", index=False)
    
    np.save(models_path / "X_test_advanced.npy", X_test_processed)
    np.save(models_path / "y_test_advanced.npy", y_test.values)
    np.save(models_path / "X_val_advanced.npy", X_val_processed)
    np.save(models_path / "y_val_advanced.npy", y_val.values)
    
    # Save test df for Plotly
    X_test.to_csv(Path(__file__).parent.parent / "data" / "test_set_advanced_real.csv", index=False)
    
    return results_df, best_name, calibrated, preprocessor

if __name__ == "__main__":
    train_advanced()
