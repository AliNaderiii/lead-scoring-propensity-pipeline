"""
Feature Engineering for Lead Scoring
Creates meaningful predictors without leakage.
"""

import pandas as pd
import numpy as np

def engineer_features(df):
    df = df.copy()
    
    # Interaction features
    df['email_ctr'] = df['email_clicks'] / (df['email_opens'] + 1)
    df['visits_per_contact'] = df['website_visits'] / df['num_contacts']
    df['time_per_visit'] = df['time_on_site_min'] / (df['website_visits'] + 1)
    df['budget_to_income_ratio'] = df['budget_range'] / (df['income'] + 1)
    df['high_engagement'] = (df['engagement_score'] > df['engagement_score'].quantile(0.8)).astype(int)
    df['quick_response'] = (df['response_time_hours'] < 2).astype(int)
    df['multiple_forms'] = (df['form_completions'] >= 2).astype(int)
    df['is_referral'] = (df['lead_source'] == 'referral').astype(int)
    df['is_retargeting'] = (df['campaign_type'] == 'retargeting').astype(int)
    df['is_high_credit'] = (df['credit_score'] > 750).astype(int)
    df['recent_lead'] = (df['days_since_first_contact'] <= 3).astype(int)
    
    # Risk / Quality scores
    df['contact_intensity'] = df['num_contacts'] * df['email_opens']
    df['value_score'] = df['income'] * df['credit_score'] / 100000
    
    return df

def get_feature_importance_names(preprocessor, numeric_features, categorical_features):
    """Get feature names after preprocessing"""
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
    cat_feature_names = cat_encoder.get_feature_names_out(categorical_features)
    all_names = list(numeric_features) + list(cat_feature_names)
    return all_names

# Updated numeric features after engineering
ENGINEERED_NUMERIC = [
    'age', 'income', 'credit_score', 'website_visits',
    'time_on_site_min', 'email_opens', 'email_clicks',
    'form_completions', 'days_since_first_contact',
    'num_contacts', 'response_time_hours',
    'engagement_score', 'budget_range',
    'email_ctr', 'visits_per_contact', 'time_per_visit',
    'budget_to_income_ratio', 'high_engagement',
    'quick_response', 'multiple_forms', 'is_referral',
    'is_retargeting', 'is_high_credit', 'recent_lead',
    'contact_intensity', 'value_score'
]
