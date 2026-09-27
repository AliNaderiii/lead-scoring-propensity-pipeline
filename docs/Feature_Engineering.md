# Feature Engineering Documentation

## Overview
All features engineered without data leakage. No future information used.

## Original Features (13 numeric + 4 categorical)
Numeric: age, income, credit_score, website_visits, time_on_site_min, email_opens, email_clicks, form_completions, days_since_first_contact, num_contacts, response_time_hours, engagement_score, budget_range
Categorical: lead_source, campaign_type, region, product_interest

## Engineered Features (13 new)
1. **email_ctr** = email_clicks / (email_opens + 1) - Click-through rate
2. **visits_per_contact** = website_visits / num_contacts - Engagement efficiency
3. **time_per_visit** = time_on_site / (website_visits + 1) - Depth of interest
4. **budget_to_income_ratio** = budget_range / income - Affordability
5. **high_engagement** = engagement_score > 80th percentile - Top engagers
6. **quick_response** = response_time < 2h - Hot lead indicator
7. **multiple_forms** = form_completions >=2 - High intent
8. **is_referral** = lead_source == referral - Best source (1.5 logit boost)
9. **is_retargeting** = campaign_type == retargeting - Warm lead
10. **is_high_credit** = credit_score >750 - Financial quality
11. **recent_lead** = days_since_first_contact <=3 - Recency
12. **contact_intensity** = num_contacts * email_opens - Interaction volume
13. **value_score** = income * credit_score / 100000 - Customer value

## Selection Method
- Boruta (shadow features)
- SHAP mean |value|
- Permutation importance
- Correlation with target <0.95 to avoid leakage

All 26 numeric + 4 categorical -> 46 features after OneHot.

## Leakage Prevention
- No features using converted date or future events
- Time-based split: train on older leads, test on recent
- Checked: No feature corr >0.95 with target
- Missingness correlation <0.3
