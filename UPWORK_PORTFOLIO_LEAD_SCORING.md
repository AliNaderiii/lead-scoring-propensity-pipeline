# Upwork Portfolio - Lead Scoring Project - Ready to Copy

## Project Title * (62 chars)
```
Lead Scoring Propensity Model | ROC-AUC 0.896 Lift 3.8x SHAP
```

Alternative:
```
Lead Conversion Propensity Model | XGBoost LightGBM SHAP | 0.896 AUC
```

## Your Role (58 chars)
```
ML Engineer | Data Scientist | Lead Scoring Specialist
```

## Project Description * (598 chars - max 600)
```
Built production-ready Lead Scoring model for 12K CRM leads (12% conversion). Pipeline: EDA + leakage prevention + 13 engineered features + 4 models (LogReg, RF, XGBoost, LightGBM) + isotonic calibration. Achieved ROC-AUC 0.896, PR-AUC 0.617, Lift 3.8x @20% - top 20% leads convert 3.8x more. Segmentation: High 45.42% conv (480 leads) vs Low 1.03% (1167). SHAP explainability, calibration curve, lift/gain charts. Deliverables: calibrated model, scored leads, inference API (LeadScorer class), clean code, reports, README. Business impact: sales prioritization 4x efficiency.
```

## Skills (5)
```
Machine Learning
Lead Scoring
XGBoost
SHAP
Predictive Modeling
```
Or:
```
Python
Scikit-learn
LightGBM
Data Science
Propensity Modeling
```

## Content Area - Big Box

**OVERVIEW**
Production-ready lead scoring system that predicts conversion probability for each lead, enabling sales to prioritize High-propensity leads. Solves 80% time waste on low-quality leads.

**PROBLEM**
- 12% conversion rate - highly imbalanced
- Sales can't identify best leads
- No explainability, no segmentation

**SOLUTION**
- Data: 12,060 synthetic CRM leads mimicking automotive/B2C (11.998% conv)
- Features: 13 numeric + 4 cat + 13 engineered (email_ctr, is_referral, high_engagement, value_score) = 46 after OneHot
- Models: LogReg, RF, XGBoost, LightGBM with 5-fold CV + scale_pos_weight 7.32
- Calibration: Isotonic for true propensity
- Segmentation: High >=0.184 (20% leads, 45% conv), Medium >=0.036, Low <0.036

**PERFORMANCE (REAL)**
- Test ROC-AUC: 0.8964
- PR-AUC: 0.6171
- Lift @20%: 3.80x
- High: 480 leads, 45.42% conv
- Low: 1167 leads, 1.03% conv
- Sales efficiency 4x

**EVALUATION**
- ROC Curve (0.896)
- PR Curve (0.617) - critical for imbalance
- Confusion Matrix
- Calibration Curve (well-calibrated)
- Lift/Gain Chart (top 20% captures 60% conversions)
- SHAP Summary & Bar

**TECH STACK**
Python, Pandas, Scikit-learn, XGBoost, LightGBM, SHAP, Matplotlib, Seaborn, Imbalanced-learn

**DELIVERABLES**
- EDA & Data Quality Report
- Feature Engineering Doc
- Model Comparison (4 models)
- Calibrated Model + Preprocessor
- Scored Leads CSV
- Segmentation Analysis
- 5 Real Plots + SHAP
- Inference API (LeadScorer)
- Clean Notebooks + README

**LINKS**
GitHub: https://github.com/AliNaderiii/lead-scoring-propensity-pipeline

**IMAGES TO UPLOAD:**
1. demo/upwork-thumbnail-lead-scoring-4k.jpg (182KB) - COVER
2. reports/roc_curve_real.png (223KB)
3. reports/lift_gain_chart_real.png (168KB) - MOST IMPORTANT FOR BUSINESS
4. reports/shap_summary_real.png (167KB)
5. reports/confusion_matrix_real.png (59KB)
6. reports/calibration_curve_real.png (147KB)
7. reports/pr_curve_real.png (103KB)

**BUSINESS IMPACT**
Sales team can focus on 20% High leads and get 60% of all conversions - 3.8x efficiency. SHAP explains why each lead is High.

