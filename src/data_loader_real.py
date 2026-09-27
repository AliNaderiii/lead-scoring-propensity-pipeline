"""
Real Dataset Loader - UCI Bank Marketing
Real-world inspiration for advanced lead scoring.
Handles leakage-safe version (excludes duration).
"""

import pandas as pd
import numpy as np
from pathlib import Path

def load_bank_marketing_real(leakage_safe=True):
    """
    Load real UCI Bank Marketing dataset.
    leakage_safe=True excludes 'duration' which is known to cause leakage
    (duration=0 => y=no, but duration only known after call).
    This is best practice per UCI docs.
    """
    data_path = Path(__file__).parent.parent / "data" / "bank-additional-full-real.csv"
    
    # File uses ; separator and quoted strings
    df = pd.read_csv(data_path, sep=';', quotechar='"')
    
    # Clean column names
    df.columns = df.columns.str.replace('.', '_').str.replace('"', '').str.strip()
    
    # Target: y -> converted (yes=1, no=0)
    df['converted'] = (df['y'] == 'yes').astype(int)
    df = df.drop(columns=['y'])
    
    print(f"Loaded real Bank Marketing: {len(df)} rows, {df['converted'].mean():.2%} conversion")
    print(f"Columns: {df.columns.tolist()}")
    
    # Leakage-safe: Remove duration (UCI says it highly affects target but only known after call)
    if leakage_safe:
        if 'duration' in df.columns:
            print("Leakage-safe mode: Removing 'duration' (post-call info, causes leakage)")
            df = df.drop(columns=['duration'])
    
    # Handle pdays 999 = not previously contacted
    df['was_previously_contacted'] = (df['pdays'] != 999).astype(int)
    df['pdays'] = df['pdays'].replace(999, -1)
    
    # Feature engineering for real dataset
    df = engineer_bank_features(df)
    
    return df

def engineer_bank_features(df):
    df = df.copy()
    
    # Age groups
    df['age_group'] = pd.cut(df['age'], bins=[0, 30, 40, 50, 60, 100], labels=['<30', '30-40', '40-50', '50-60', '60+'])
    
    # Economic indicators combined
    df['economic_stress'] = df['emp_var_rate'] + df['cons_conf_idx']
    df['price_vs_conf'] = df['cons_price_idx'] / (df['cons_conf_idx'] + 100)
    
    # Campaign intensity
    df['campaign_intensity'] = df['campaign'] / (df['previous'] + 1)
    df['is_first_contact'] = (df['previous'] == 0).astype(int)
    df['high_campaign'] = (df['campaign'] > 3).astype(int)
    
    # Contact recency
    df['recent_contact'] = (df['pdays'] != -1) & (df['pdays'] < 30)
    df['recent_contact'] = df['recent_contact'].astype(int)
    
    # Poutcome success is strong signal
    df['was_previous_success'] = (df['poutcome'] == 'success').astype(int)
    df['was_previous_failure'] = (df['poutcome'] == 'failure').astype(int)
    
    # Job + education quality
    high_jobs = ['management', 'admin.', 'professional.course']
    df['is_high_job'] = df['job'].isin(high_jobs).astype(int)
    
    # Housing + loan risk
    df['has_both_loans'] = ((df['housing'] == 'yes') & (df['loan'] == 'yes')).astype(int)
    df['no_loans'] = ((df['housing'] == 'no') & (df['loan'] == 'no')).astype(int)
    
    return df

def get_feature_lists(df):
    numeric = df.select_dtypes(include=[np.number]).columns.drop('converted').tolist()
    categorical = df.select_dtypes(include=['object', 'category']).columns.tolist()
    return numeric, categorical

if __name__ == "__main__":
    df = load_bank_marketing_real(leakage_safe=True)
    print(df.head())
    print(f"\nConversion: {df['converted'].value_counts()}")
    numeric, cat = get_feature_lists(df)
    print(f"Numeric: {numeric}")
    print(f"Categorical: {cat}")
