"""
Synthetic Lead Scoring Data Generator
Simulates high-volume B2C lead data with realistic conversion patterns.
No client data used - 100% synthetic for portfolio demonstration.
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

def generate_lead_data(n_samples=12000, conversion_rate=0.038):
    """
    Generate realistic lead scoring dataset.
    Features mimic CRM / marketing / automotive lead data.
    """
    # Demographics
    age = np.random.normal(38, 12, n_samples).clip(18, 75).astype(int)
    income = np.random.lognormal(10.8, 0.6, n_samples).clip(20000, 250000)
    credit_score = np.random.normal(680, 90, n_samples).clip(300, 850).astype(int)
    
    # Behavioral
    website_visits = np.random.poisson(4, n_samples) + np.random.exponential(2, n_samples).astype(int)
    time_on_site = np.random.exponential(8, n_samples).clip(0.5, 120)  # minutes
    email_opens = np.random.binomial(10, 0.35, n_samples)
    email_clicks = np.random.binomial(email_opens, 0.4)
    form_completions = np.random.binomial(3, 0.5, n_samples)
    
    # Lead Source
    sources = ['organic_search', 'paid_search', 'referral', 'social_media', 'direct', 'partner', 'event']
    lead_source = np.random.choice(sources, n_samples, p=[0.25, 0.20, 0.15, 0.15, 0.10, 0.08, 0.07])
    
    # Marketing
    campaign_types = ['discount', 'premium', 'awareness', 'retargeting']
    campaign = np.random.choice(campaign_types, n_samples, p=[0.35, 0.25, 0.25, 0.15])
    
    days_since_first_contact = np.random.exponential(12, n_samples).clip(0, 90).astype(int)
    num_contacts = np.random.poisson(2.5, n_samples) + 1
    response_time_hours = np.random.exponential(8, n_samples).clip(0.1, 72)
    
    # Engagement Score (engineered logic)
    engagement_score = (
        website_visits * 0.3 +
        time_on_site * 0.15 +
        email_clicks * 2.0 +
        form_completions * 3.0 +
        np.where(lead_source == 'referral', 5, 0) +
        np.where(campaign == 'retargeting', 3, 0) -
        days_since_first_contact * 0.05 -
        response_time_hours * 0.02
    )
    
    # Geographic / Product
    regions = ['north', 'south', 'east', 'west', 'central']
    region = np.random.choice(regions, n_samples)
    
    product_interest = np.random.choice(['sedan', 'suv', 'truck', 'hybrid'], n_samples, p=[0.35, 0.30, 0.20, 0.15])
    budget_range = income * np.random.uniform(0.15, 0.45, n_samples)
    
    # Target: ~9-11% conversion, ROC ~0.85
    base_logit = -5.6
    logit = (
        base_logit +
        (engagement_score - engagement_score.mean()) * 0.20 +
        (credit_score - 680) * 0.007 +
        np.where(lead_source == 'referral', 1.5, 0) +
        np.where(lead_source == 'paid_search', 0.55, 0) +
        np.where(lead_source == 'social_media', -0.75, 0) +
        np.where(lead_source == 'direct', -0.3, 0) +
        np.where(campaign == 'retargeting', 1.0, 0) +
        np.where(campaign == 'discount', 0.65, 0) +
        (income / 100000) * 0.30 -
        days_since_first_contact * 0.04 -
        response_time_hours * 0.022 +
        form_completions * 0.75 +
        email_clicks * 0.5 +
        website_visits * 0.07
    )
    
    prob = 1 / (1 + np.exp(-logit))
    converted = np.random.binomial(1, prob)
    
    # Adjust to exact conversion rate if needed
    # (prob already calibrated to ~3.8%)
    
    df = pd.DataFrame({
        'lead_id': [f'LEAD_{100000+i}' for i in range(n_samples)],
        'age': age,
        'income': income.round(2),
        'credit_score': credit_score,
        'website_visits': website_visits,
        'time_on_site_min': time_on_site.round(2),
        'email_opens': email_opens,
        'email_clicks': email_clicks,
        'form_completions': form_completions,
        'lead_source': lead_source,
        'campaign_type': campaign,
        'days_since_first_contact': days_since_first_contact,
        'num_contacts': num_contacts,
        'response_time_hours': response_time_hours.round(2),
        'engagement_score': engagement_score.round(2),
        'region': region,
        'product_interest': product_interest,
        'budget_range': budget_range.round(2),
        'converted': converted
    })
    
    # Introduce realistic missing values (5-8%)
    for col in ['income', 'credit_score', 'response_time_hours', 'budget_range']:
        mask = np.random.random(n_samples) < 0.06
        df.loc[mask, col] = np.nan
    
    # Add some duplicates (0.5%)
    dup_idx = np.random.choice(n_samples, int(n_samples * 0.005), replace=False)
    df = pd.concat([df, df.iloc[dup_idx]], ignore_index=True)
    
    return df

if __name__ == "__main__":
    output_path = Path(__file__).parent.parent / "data" / "leads_raw.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("Generating synthetic lead data...")
    df = generate_lead_data(n_samples=12000, conversion_rate=0.038)
    df.to_csv(output_path, index=False)
    
    print(f"Saved {len(df)} rows to {output_path}")
    print(f"Conversion rate: {df['converted'].mean():.3%}")
    print(df['converted'].value_counts())
    print("\nColumns:", df.columns.tolist())
