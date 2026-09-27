"""
Advanced Evaluation with Plotly - Beautiful Interactive + Static 4K Plots
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix, roc_auc_score
from sklearn.calibration import calibration_curve

# Try Plotly
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except:
    PLOTLY_AVAILABLE = False
    print("Plotly not available, using matplotlib only")

def lift_gain_plotly(y_true, y_proba, save_html_path, save_png_path):
    df = pd.DataFrame({'y_true': y_true, 'y_proba': y_proba})
    df = df.sort_values('y_proba', ascending=False).reset_index(drop=True)
    df['cum_true'] = df['y_true'].cumsum()
    df['cum_total'] = np.arange(1, len(df)+1)
    df['gain'] = df['cum_true'] / df['y_true'].sum()
    df['random_gain'] = df['cum_total'] / len(df)
    
    # Decile lift
    df['decile'] = pd.qcut(df.index, 10, labels=False, duplicates='drop')
    decile_mean = df.groupby('decile')['y_true'].mean()
    overall_mean = y_true.mean()
    lift_per_decile = decile_mean / overall_mean
    
    if PLOTLY_AVAILABLE:
        fig = make_subplots(rows=1, cols=2, subplot_titles=('Cumulative Gain', 'Lift by Decile'))
        
        # Gain
        fig.add_trace(go.Scatter(x=df['cum_total']/len(df), y=df['gain'], mode='lines', name='Model', line=dict(color='#1f77b4', width=3)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['cum_total']/len(df), y=df['random_gain'], mode='lines', name='Random', line=dict(dash='dash', color='orange')), row=1, col=1)
        
        # Lift
        fig.add_trace(go.Bar(x=list(range(len(lift_per_decile))), y=lift_per_decile.values, name='Lift', marker_color='#2ca02c'), row=1, col=2)
        fig.add_hline(y=1, line_dash="dash", line_color="red", row=1, col=2)
        
        fig.update_layout(height=500, width=1000, title_text="Lift & Gain - Real Model", template='plotly_white')
        fig.write_html(save_html_path)
        print(f"Saved Plotly HTML to {save_html_path}")
    
    # Matplotlib 4K version
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(df['cum_total']/len(df), df['gain'], label='Model', linewidth=2.5, color='#1f77b4')
    axes[0].plot(df['cum_total']/len(df), df['random_gain'], label='Random', linestyle='--', color='orange')
    axes[0].set_xlabel('Proportion of Leads', fontsize=12)
    axes[0].set_ylabel('Cumulative Gain', fontsize=12)
    axes[0].set_title('Cumulative Gain Chart (Real)', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    axes[1].bar(range(len(lift_per_decile)), lift_per_decile.values, color='#2ca02c', edgecolor='black')
    axes[1].axhline(y=1, color='red', linestyle='--', label='Random')
    axes[1].set_xlabel('Decile (1=Top Scores)', fontsize=12)
    axes[1].set_ylabel('Lift', fontsize=12)
    axes[1].set_title('Lift Chart by Decile (Real)', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_png_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved 4K PNG to {save_png_path}")

def evaluate_advanced_plotly():
    models_path = Path(__file__).parent.parent / "models"
    reports_path = Path(__file__).parent.parent / "reports"
    reports_path.mkdir(exist_ok=True)
    
    # Try advanced first, fallback to basic
    try:
        X_test = np.load(models_path / "X_test_advanced.npy")
        y_test = np.load(models_path / "y_test_advanced.npy")
        X_val = np.load(models_path / "X_val_advanced.npy")
        y_val = np.load(models_path / "y_val_advanced.npy")
        calibrated = joblib.load(models_path / "calibrated_model_advanced.pkl")
        all_models = joblib.load(models_path / "all_models_advanced.pkl")
        suffix = "_advanced_real"
        print("Using ADVANCED real Bank Marketing data")
    except:
        X_test = np.load(models_path / "X_test_processed.npy")
        y_test = np.load(models_path / "y_test.npy")
        X_val = np.load(models_path / "X_val_processed.npy")
        y_val = np.load(models_path / "y_val.npy")
        calibrated = joblib.load(models_path / "calibrated_model.pkl")
        all_models = joblib.load(models_path / "all_models.pkl")
        suffix = "_real"
        print("Using BASIC synthetic data")
    
    y_test_proba = calibrated.predict_proba(X_test)[:, 1]
    y_val_proba = calibrated.predict_proba(X_val)[:, 1]
    
    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    roc_auc = auc(fpr, tpr)
    
    if PLOTLY_AVAILABLE:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'Calibrated (AUC={roc_auc:.3f})', line=dict(color='blue', width=3)))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random', line=dict(dash='dash', color='gray')))
        
        # Add other models
        for name, model in all_models.items():
            try:
                proba = model.predict_proba(X_test)[:, 1]
                fpr_m, tpr_m, _ = roc_curve(y_test, proba)
                auc_m = auc(fpr_m, tpr_m)
                fig.add_trace(go.Scatter(x=fpr_m, y=tpr_m, mode='lines', name=f'{name} (AUC={auc_m:.3f})', line=dict(dash='dot'), opacity=0.7))
            except:
                pass
        
        fig.update_layout(title='ROC Curve - Test Set (Real)', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate', template='plotly_white', width=800, height=600)
        fig.write_html(reports_path / f"roc_curve_plotly{suffix}.html")
    
    # Matplotlib ROC 4K
    plt.figure(figsize=(8,7))
    plt.plot(fpr, tpr, label=f'Calibrated (AUC={roc_auc:.3f})', linewidth=3, color='#1f77b4')
    for name, model in all_models.items():
        try:
            proba = model.predict_proba(X_test)[:, 1]
            fpr_m, tpr_m, _ = roc_curve(y_test, proba)
            auc_m = auc(fpr_m, tpr_m)
            plt.plot(fpr_m, tpr_m, linestyle='--', alpha=0.6, label=f'{name} (AUC={auc_m:.3f})')
        except:
            pass
    plt.plot([0,1], [0,1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve - Test Set (Real) - AUC {roc_auc:.3f}', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(reports_path / f"roc_curve_4k{suffix}.png", dpi=400, bbox_inches='tight')
    plt.close()
    
    # PR Curve
    precision, recall, _ = precision_recall_curve(y_test, y_test_proba)
    pr_auc = average_precision_score(y_test, y_test_proba)
    
    plt.figure(figsize=(8,7))
    plt.plot(recall, precision, label=f'Calibrated (PR-AUC={pr_auc:.3f})', linewidth=3, color='#ff7f0e')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'Precision-Recall Curve (Real) - PR-AUC {pr_auc:.3f}', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(reports_path / f"pr_curve_4k{suffix}.png", dpi=400, bbox_inches='tight')
    plt.close()
    
    # Lift/Gain
    lift_gain_plotly(y_test, y_test_proba, reports_path / f"lift_gain_plotly{suffix}.html", reports_path / f"lift_gain_4k{suffix}.png")
    
    # Confusion Matrix 4K
    from sklearn.metrics import confusion_matrix
    y_pred = calibrated.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size":14})
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.title(f'Confusion Matrix - Test (Real)\nROC-AUC {roc_auc:.3f} PR-AUC {pr_auc:.3f}', fontsize=14, fontweight='bold')
    plt.savefig(reports_path / f"confusion_matrix_4k{suffix}.png", dpi=400, bbox_inches='tight')
    plt.close()
    
    # Calibration 4K
    prob_true, prob_pred = calibration_curve(y_test, y_test_proba, n_bins=10)
    plt.figure(figsize=(8,7))
    plt.plot(prob_pred, prob_true, marker='o', label='Calibrated Model', linewidth=3, markersize=8, color='#2ca02c')
    plt.plot([0,1], [0,1], 'k--', label='Perfectly Calibrated')
    plt.xlabel('Mean Predicted Probability', fontsize=12)
    plt.ylabel('Fraction of Positives', fontsize=12)
    plt.title('Calibration Curve - Test Set (Real)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(reports_path / f"calibration_4k{suffix}.png", dpi=400, bbox_inches='tight')
    plt.close()
    
    print(f"Advanced evaluation done - ROC {roc_auc:.3f} PR {pr_auc:.3f}")
    
    return {'roc_auc': roc_auc, 'pr_auc': pr_auc}

if __name__ == "__main__":
    evaluate_advanced_plotly()
