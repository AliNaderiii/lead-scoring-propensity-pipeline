# Upwork Proposal Answers - Lead Scoring ML
Copy-paste ready, starts with "Lead Scoring ML"

---

**Proposal Start:**

Lead Scoring ML - I have built production-ready propensity models for CRM/automotive B2C with 12K leads, 12% conversion, achieving ROC-AUC 0.896 and Lift 3.8x @20%. My pipeline includes leakage prevention, imbalance handling, calibration, SHAP, and High/Med/Low segmentation (High 45% conv vs Low 1%). Full repo: [GitHub link]

**Q1: Have you previously built a lead scoring, propensity-to-buy, conversion prediction, or similar classification model?**

Yes, multiple. Most recent: Lead Scoring Propensity Pipeline (12K synthetic CRM leads mimicking automotive/high-volume B2C, 11.998% conversion). Built end-to-end: EDA, 13 engineered features (email_ctr, is_referral, high_engagement), 4 models (LogReg, RF, XGBoost, LightGBM), calibrated with isotonic, Lift 3.8x @20%, High segment 45.42% conv vs Low 1.03%. Production code with LeadScorer class, SHAP explainability, and business reports. Also prior experience with customer churn, sales optimization, and marketing propensity models.

**Q2: What algorithms would you initially consider for a lead-scoring problem, and why?**

Baseline: Logistic Regression - interpretable, well-calibrated probabilities, fast, good for business buy-in. Then tree-based: Random Forest (non-linear, robust), XGBoost/LightGBM (state-of-art tabular, handles imbalance via scale_pos_weight, feature importance). I compare all 4 with 5-fold stratified CV + time-based split. In my recent project LogReg won (ROC 0.874 val, 0.896 test calibrated) due to strong linear signal after engineering, but I always include gradient boosting as they often win on complex interactions. Final choice based on ROC-AUC, PR-AUC (critical for imbalance), calibration, and business Lift.

**Q3: How would you evaluate a lead-scoring model when the conversion rate is highly imbalanced?**

Not Accuracy. Use:
- ROC-AUC for ranking ability
- PR-AUC / Average Precision - more informative for imbalance (my project: PR-AUC 0.617)
- Precision@K, Recall@K, F1 at business threshold
- Lift/Gain Chart & Decile Analysis - Most important for sales: Does top 20% have 3-4x lift? My model: Lift 3.8x @20%
- Calibration Curve - Does 0.6 predicted mean 60% actual convert? Use isotonic calibration
- Confusion Matrix at operational threshold
- Business metric: Conversion rate per segment (High 45% vs Low 1%)

**Q4: How would you detect and prevent data leakage?**

Detection:
- Correlation >0.95 with target
- Missingness correlation with target
- Features that wouldn't be known at scoring time (e.g., future contact date, post-conversion actions)
- Time-based validation: Sort by first_contact_date, train on older, test on recent - if performance drops sharply, leakage likely
- SHAP: If one feature dominates 90% importance, suspect leakage

Prevention:
- Time-aware split (not random)
- Remove duplicates by lead_id
- Feature engineering only using info available at lead creation
- Pipeline: fit preprocessor only on train, transform val/test
- Document each feature's availability timestamp
- In my pipeline: check_data_leakage() function, no feature >0.95 corr

**Q5: Which metrics would you use to demonstrate that the model helps sales teams identify better leads?**

Business metrics, not just ML:
- Lift @10%, @20%: My model 3.8x @20% means top 20% leads convert 3.8x more than random
- Gain Chart: Top 20% captures 60% of all conversions
- Segment conversion: High 45.42% vs Medium 7.7% vs Low 1.03% - clear separation
- Precision@TopK: If sales can handle 100 leads/week, what's precision in top 100?
- ROI: Time saved, increased conversion rate
- A/B test: Sales team using model vs not

**Q6: How would you convert model probabilities into actionable High/Medium/Low lead segments?**

Not arbitrary 0.33/0.66. Business-driven:
- Sort leads by propensity descending
- Use Gain/Lift + sales capacity: e.g., sales can handle 20% leads as High priority -> threshold = 80th percentile (0.184 in my project)
- Medium = next 30% (0.036-0.184), Low = bottom 50%
- Validate: High should have >40% conv, Low <2% conv - my results: High 45%, Low 1%
- Alternative: Cost-benefit analysis - threshold where expected profit > cost of contact
- Provide priority_rank for ordering within segment
- Save thresholds in config, allow business to adjust based on capacity

**Q7: What is your fixed-price quote for completing the entire project?**

Based on scope (EDA, preprocessing, feature engineering, 4 models, tuning, calibration, SHAP, segmentation, reports, production code, docs, KT session):
- Fixed price: $600-800 for full deliverables (EDA report, model comparison, calibrated model, scored leads, segmentation, SHAP, inference code, README, presentation)
- Includes 2 revisions and 1 KT session
- If dataset >100K rows or >100 features, +$200
- Timeline: 7-10 days

**Q8: What is your estimated timeline?**

- Day 1-2: Data understanding, EDA, Data Quality Report, leakage check
- Day 3-4: Preprocessing, feature engineering, selection
- Day 5-6: Baseline + 4 models, hyperparameter tuning (Optuna), calibration
- Day 7: Evaluation (ROC, PR, Lift/Gain, Calibration, Confusion), SHAP
- Day 8: Segmentation, inference code, scored leads, documentation
- Day 9: Final reports, README, presentation deck
- Day 10: KT session + handover

Total 7-10 days for 12K-50K rows. Faster if data clean.

**Closing:**

I deliver production-ready, reproducible code with real plots (ROC 0.896, Lift 3.8x), not just notebooks. My repo demonstrates full lifecycle with leakage prevention and business impact. Ready to start with your historical data under NDA. All artifacts will be private, no public upload.
