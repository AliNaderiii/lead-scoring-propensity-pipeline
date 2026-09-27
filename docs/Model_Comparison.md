# Model Comparison - Real Results

## Dataset
- 12060 rows synthetic (mimics CRM/automotive), 12000 after dedup
- Train 7800 (12.01% conv), Val 1800 (12% conv), Test 2400 (12% conv)
- Imbalance: 7.32 scale_pos_weight
- Features: 46 after preprocessing

## Results (Validation Set)

| Model | ROC-AUC | PR-AUC | F1 | Precision | Recall |
|-------|---------|--------|----|-----------|--------|
| LogisticRegression | 0.8740 | 0.5613 | 0.4774 | 0.3435 | 0.7824 |
| RandomForest | 0.8618 | 0.5262 | 0.5000 | 0.4457 | 0.5694 |
| XGBoost | 0.8583 | 0.5237 | 0.4738 | 0.5135 | 0.4398 |
| LightGBM | 0.8526 | 0.5219 | 0.4494 | 0.4815 | 0.4213 |

## Final Test Set (Calibrated LogisticRegression)
- ROC-AUC: 0.8964
- PR-AUC: 0.6171
- Precision: 0.6988
- Recall: 0.4028
- F1: 0.5110
- Lift @20%: 3.80x

## Why LogisticRegression Wins Here
- Linear separability after strong feature engineering
- Better calibration for true propensity scores (isotonic)
- More interpretable for business (SHAP + coefficients)
- Still compared with XGBoost/LightGBM as required

## Calibration
Isotonic calibration used for true probability scores.
Calibration curve shows well-calibrated predictions (close to diagonal).

## Business Impact
Top 20% scored leads (High segment) have 3.8x higher conversion than random.
High segment: 480 leads, 45.42% conversion vs Low: 1.03% conversion.
Sales team can prioritize High first, 4x efficiency gain.
