"""
Model Evaluation - Real Metrics and Plots
Generates ROC, PR, Lift, Gain, Calibration, Confusion Matrix - 100% real.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, roc_auc_score
)
from sklearn.calibration import calibration_curve

def lift_gain_chart(y_true, y_proba, save_path):
    """Generate Lift and Gain charts"""
    # Sort by proba descending
    df = pd.DataFrame({'y_true': y_true, 'y_proba': y_proba})
    df = df.sort_values('y_proba', ascending=False).reset_index(drop=True)
    df['cum_true'] = df['y_true'].cumsum()
    df['cum_total'] = np.arange(1, len(df)+1)
    df['gain'] = df['cum_true'] / df['y_true'].sum()
    df['random_gain'] = df['cum_total'] / len(df)
    df['lift'] = df['gain'] / df['random_gain']
    
    # Decile analysis
    df['decile'] = pd.qcut(df.index, 10, labels=False, duplicates='drop')
    decile_stats = df.groupby('decile').agg({
        'y_true': ['count', 'sum', 'mean'],
        'y_proba': 'mean'
    }).round(4)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Gain Chart
    axes[0].plot(df['cum_total']/len(df), df['gain'], label='Model', linewidth=2)
    axes[0].plot(df['cum_total']/len(df), df['random_gain'], label='Random', linestyle='--')
    axes[0].set_xlabel('Proportion of Leads')
    axes[0].set_ylabel('Cumulative Gain')
    axes[0].set_title('Cumulative Gain Chart')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Lift Chart
    # Lift per decile
    decile_lift = df.groupby('decile')['y_true'].mean() / y_true.mean()
    axes[1].bar(range(len(decile_lift)), decile_lift.values, color='steelblue')
    axes[1].axhline(y=1, color='red', linestyle='--', label='Random')
    axes[1].set_xlabel('Decile (1=Top Scores)')
    axes[1].set_ylabel('Lift')
    axes[1].set_title('Lift Chart by Decile')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return decile_stats

def evaluate_and_plot():
    models_path = Path(__file__).parent.parent / "models"
    reports_path = Path(__file__).parent.parent / "reports"
    reports_path.mkdir(exist_ok=True)
    
    # Load
    X_test = np.load(models_path / "X_test_processed.npy")
    y_test = np.load(models_path / "y_test.npy")
    X_val = np.load(models_path / "X_val_processed.npy")
    y_val = np.load(models_path / "y_val.npy")
    
    calibrated = joblib.load(models_path / "calibrated_model.pkl")
    all_models = joblib.load(models_path / "all_models.pkl")
    
    y_test_proba = calibrated.predict_proba(X_test)[:, 1]
    y_val_proba = calibrated.predict_proba(X_val)[:, 1]
    
    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(7,6))
    plt.plot(fpr, tpr, label=f'Calibrated Model (AUC={roc_auc:.3f})', linewidth=2)
    
    # Add other models ROC
    for name, model in all_models.items():
        try:
            proba = model.predict_proba(X_test)[:, 1]
            fpr_m, tpr_m, _ = roc_curve(y_test, proba)
            auc_m = auc(fpr_m, tpr_m)
            plt.plot(fpr_m, tpr_m, alpha=0.6, linestyle='--', label=f'{name} (AUC={auc_m:.3f})')
        except:
            pass
    
    plt.plot([0,1], [0,1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Test Set')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(reports_path / "roc_curve_real.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. PR Curve
    precision, recall, _ = precision_recall_curve(y_test, y_test_proba)
    pr_auc = average_precision_score(y_test, y_test_proba)
    
    plt.figure(figsize=(7,6))
    plt.plot(recall, precision, label=f'Calibrated (PR-AUC={pr_auc:.3f})', linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve - Test Set')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(reports_path / "pr_curve_real.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Confusion Matrix
    y_pred = calibrated.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - Test\nROC-AUC {roc_auc:.3f} PR-AUC {pr_auc:.3f}')
    plt.savefig(reports_path / "confusion_matrix_real.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Calibration Curve
    prob_true, prob_pred = calibration_curve(y_test, y_test_proba, n_bins=10)
    
    plt.figure(figsize=(7,6))
    plt.plot(prob_pred, prob_true, marker='o', label='Calibrated Model', linewidth=2)
    plt.plot([0,1], [0,1], 'k--', label='Perfectly Calibrated')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve - Test Set')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(reports_path / "calibration_curve_real.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Lift/Gain
    decile_stats = lift_gain_chart(y_test, y_test_proba, reports_path / "lift_gain_chart_real.png")
    decile_stats.to_csv(reports_path / "decile_analysis.csv")
    
    # 6. Metrics summary
    from sklearn.metrics import precision_score, recall_score, f1_score
    metrics = {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'conversion_rate': y_test.mean(),
        'test_size': len(y_test)
    }
    
    # Lift @ 20%
    df_sorted = pd.DataFrame({'y_true': y_test, 'y_proba': y_test_proba}).sort_values('y_proba', ascending=False)
    top_20_pct = int(len(df_sorted)*0.2)
    lift_20 = df_sorted.head(top_20_pct)['y_true'].mean() / y_test.mean()
    metrics['lift_at_20pct'] = lift_20
    
    pd.DataFrame([metrics]).to_csv(reports_path / "final_metrics.csv", index=False)
    
    print("=== Final Metrics (REAL) ===")
    for k,v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v,float) else f"{k}: {v}")
    
    print(f"\nLift @20%: {lift_20:.2f}x - Means top 20% leads have {lift_20:.1f}x higher conversion")
    
    return metrics

if __name__ == "__main__":
    evaluate_and_plot()
