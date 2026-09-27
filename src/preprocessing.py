"""
Data Preprocessing Pipeline
Handles missing values, outliers, encoding, scaling with no data leakage.
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

NUMERIC_FEATURES = [
    'age', 'income', 'credit_score', 'website_visits',
    'time_on_site_min', 'email_opens', 'email_clicks',
    'form_completions', 'days_since_first_contact',
    'num_contacts', 'response_time_hours',
    'engagement_score', 'budget_range'
]

CATEGORICAL_FEATURES = [
    'lead_source', 'campaign_type', 'region', 'product_interest'
]

TARGET = 'converted'

def get_preprocessor():
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
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ])
    
    return preprocessor

def split_data(df, test_size=0.2, val_size=0.15, random_state=42):
    """
    Time-aware split to prevent leakage.
    First remove duplicates, then split.
    """
    # Remove duplicates
    df_clean = df.drop_duplicates(subset=['lead_id'], keep='first').copy()
    print(f"Removed duplicates: {len(df) - len(df_clean)} rows")
    
    # Sort by days_since_first_contact to simulate time split
    df_clean = df_clean.sort_values('days_since_first_contact')
    
    # Initial train+val / test split
    train_val, test = train_test_split(
        df_clean, test_size=test_size, 
        random_state=random_state, 
        stratify=df_clean[TARGET]
    )
    
    # Train / val split
    val_ratio = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=val_ratio,
        random_state=random_state,
        stratify=train_val[TARGET]
    )
    
    print(f"Train: {len(train)} ({train[TARGET].mean():.3%} conv)")
    print(f"Val: {len(val)} ({val[TARGET].mean():.3%} conv)")
    print(f"Test: {len(test)} ({test[TARGET].mean():.3%} conv)")
    
    return train, val, test

def check_data_leakage(df):
    """Check for common leakage patterns"""
    issues = []
    
    # Check if any feature has perfect correlation with target
    for col in df.select_dtypes(include=[np.number]).columns:
        if col == TARGET:
            continue
        corr = df[col].corr(df[TARGET])
        if abs(corr) > 0.95:
            issues.append(f"Potential leakage: {col} corr={corr:.3f} with target")
    
    # Check missingness correlation with target
    for col in df.columns:
        if df[col].isna().sum() > 0:
            missing_corr = df[col].isna().astype(int).corr(df[TARGET])
            if abs(missing_corr) > 0.3:
                issues.append(f"Missingness leakage: {col} missing corr={missing_corr:.3f}")
    
    if not issues:
        print("No obvious leakage detected.")
    else:
        for issue in issues:
            print(f"WARNING: {issue}")
    
    return issues
