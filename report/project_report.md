# Credit Scoring Model Using Machine Learning Classification
## Comprehensive Technical Project Report

**Author**: Educational Machine Learning Prototype  
**Academic Course**: Machine Learning Laboratory  
**Dataset**: UCI Machine Learning Repository — Default of Credit Card Clients (Dataset ID: 350)  
**Date**: September 28, 2026  
**Execution Environment**: Python 3.14.5 | scikit-learn 1.9.0 | pandas 3.0.5  

---

## 1. Title
**Credit Scoring Model Using Machine Learning Classification**

---

## 2. Abstract
Credit scoring is a foundational quantitative tool in financial risk management used to evaluate the likelihood that a borrower will default on credit commitments. This project develops, validates, and deploys an educational machine learning classification system trained on the benchmark UCI "Default of Credit Card Clients" dataset ($30,000$ accounts, Taiwan, 2005). The modeling pipeline is engineered under a strict zero-leakage contract, separating an untouched hold-out test set ($N = 6,000, 20\%$) from training data ($N = 24,000, 80\%$) prior to any transformation. Eight domain-specific financial features were engineered to capture utilization ratios, repayment coverage, and cumulative delinquency patterns. Four candidate models were evaluated: a Dummy Baseline, Regularized Logistic Regression ($C$-tuned), Decision Tree (depth-tuned), and Random Forest. Using 5-fold stratified cross-validation on training folds, Random Forest emerged as the champion architecture, achieving on the untouched test set an Accuracy of $81.67\%$, ROC-AUC of $0.7782$, Average Precision (PR-AUC) of $0.5579$, and F1-score of $0.4629$ (Precision $65.74\%$, Recall $35.72\%$). The entire inference pipeline—from raw inputs to prediction—is serialized into a standalone joblib artifact and served via an interactive Gradio web application with input schema validation, 1-click dataset sample testing, and operational threshold analysis.

---

## 3. Introduction
Lending institutions face an inherent trade-off between expanding credit access to maximize interest revenue and controlling credit delinquency to avoid capital impairment. Historical credit analysis relied heavily on manual underwriting or rigid heuristic scorecards. Modern credit risk assessment utilizes supervised machine learning algorithms to uncover non-linear relationships across multi-period repayment histories, revolving balances, and debt payment coverage.

This project delivers a complete, end-to-end educational prototype demonstrating the entire machine learning lifecycle: data ingestion, exploratory data analysis, leakage-safe feature engineering, pipeline preprocessing, cross-validation, multi-metric evaluation, explainability, serialization, and interactive web demonstration.

---

## 4. Problem Statement
Given a set of historical demographic and multi-period financial behavioral attributes for a credit card borrower:
$$\mathbf{x} = [x_1, x_2, \dots, x_d] \in \mathbb{R}^d$$
predict the binary probability of payment default in the subsequent month:
$$y \in \{0, 1\}$$
where $y = 1$ denotes default (failure to repay contractual minimum obligation), and $y = 0$ denotes non-default (account maintained in good standing).

### Key Challenges:
1. **Class Imbalance**: Default is a minority event ($\sim 22.12\%$ default rate vs $\sim 77.88\%$ non-default). Accuracy alone is deceptive.
2. **Asymmetric Error Costs**: False Negatives (approving a defaulting borrower) generally result in direct capital loss, while False Positives (rejecting a solvent borrower) result in lost opportunity and customer dissatisfaction.
3. **Data Leakage Risks**: Inadvertent contamination between test data and training transformations must be prevented in code.

---

## 5. Motivation
In introductory machine learning education, students frequently encounter simplified tutorials where models are evaluated on synthetic data, preprocessing is fitted improperly on whole datasets, or only accuracy is reported. This project bridges the gap between basic classroom concepts and rigorous data science practices by enforcing strict reproducibility (`random_state=42`), genuine public benchmark data, comprehensive classification metrics (ROC-AUC, PR-AUC, Confusion Matrix, Threshold Analysis), and interactive deployment.

---

## 6. Objectives
1. Implement a triple-redundant data loader for UCI Dataset ID 350.
2. Formulate a programmatic Data Leakage Contract ensuring ID exclusion and pre-split isolation.
3. Engineer genuine, domain-justified financial indicators (utilization, payment coverage, delinquency counts).
4. Build a unified scikit-learn `Pipeline` combining `ColumnTransformer` with candidate classifiers.
5. Train and tune Logistic Regression, Decision Tree, and Random Forest using 5-Fold Stratified Cross-Validation on training data only.
6. Evaluate all models on an untouched hold-out test set across six primary metrics.
7. Perform explainability analysis via Gini feature importances and standardized regression coefficients.
8. Deliver an interactive Gradio web application with input validation and authentic dataset sample evaluation.
9. Provide 18 automated unit tests with $100\%$ pass rate and a master validation audit script.

---

## 7. Dataset Description
- **Origin**: UCI Machine Learning Repository (Dataset ID: 350).
- **Authors**: Prof. I-Cheng Yeh (Department of Information Management, Chung Hua University, Taiwan).
- **Sample Size**: $30,000$ individual credit card accounts observed from April 2005 to September 2005.
- **Attributes**: $23$ raw predictor features and $1$ target variable.
  - Demographic features: `SEX` (Gender: 1=Male, 2=Female), `EDUCATION` (1=Grad school, 2=University, 3=High school, 4=Others), `MARRIAGE` (1=Married, 2=Single, 3=Others), `AGE` (Years).
  - Credit limit: `LIMIT_BAL` (Given credit limit in New Taiwan Dollar, NTD).
  - Historical payment statuses: `PAY_0` to `PAY_6` (-2=No consumption, -1=Paid in full, 0=Revolving credit, 1=1 month delay, ..., 8=8 months delay).
  - Historical statement amounts: `BILL_AMT1` to `BILL_AMT6` (Statement balance in NTD).
  - Historical payment amounts: `PAY_AMT1` to `PAY_AMT6` (Repayment amounts in NTD).

---

## 8. Data Preparation & Governance
- **Missing Values**: Programmatic audit confirmed $0$ missing values across the 30,000 rows.
- **Identifier Exclusions**: The `ID` column is an administrative row counter. It is stripped immediately upon loading to prevent memorization leakage.
- **Target Encoding**: Target column standardized to binary integer $y \in \{0, 1\}$.
- **Undocumented Categories**: In `EDUCATION`, codes 0, 5, and 6 were mapped to 4 ("Others / Unknown"); in `MARRIAGE`, code 0 was mapped to 3 ("Others").

---

## 9. Exploratory Data Analysis (EDA)
Exploratory visualizations generated from genuine data revealed the following observations:
1. **Target Distribution**: Non-default accounts: $23,364$ ($77.88\%$); Default accounts: $6,636$ ($22.12\%$). Class ratio is approximately $3.5 : 1$.
2. **Credit Limit Distribution**: Median credit limit is $140,000$ NTD. Borrowers with limits $< 50,000$ NTD exhibit noticeably higher default frequencies than borrowers with limits $> 200,000$ NTD.
3. **Age Profile**: Borrower ages span 21 to 79 years (median: 34 years).
4. **Repayment Status**: Borrowers who were 2 or more months late on recent statements (`PAY_0 >= 2`) demonstrated default rates exceeding $60\%$, highlighting repayment history as the single most critical risk indicator.

---

## 10. Feature Engineering
Eight genuine, leakage-safe features were constructed row-wise via `FinancialFeatureEngineer`:
1. **`avg_bill_amt`**: $\frac{1}{6}\sum_{i=1}^6 \text{BILL\_AMT}_i$ (Average monthly balance obligation).
2. **`total_bill_amt`**: $\sum_{i=1}^6 \text{BILL\_AMT}_i$ (Cumulative 6-month billing volume).
3. **`avg_pay_amt`**: $\frac{1}{6}\sum_{i=1}^6 \text{PAY\_AMT}_i$ (Average monthly debt repayment).
4. **`total_pay_amt`**: $\sum_{i=1}^6 \text{PAY\_AMT}_i$ (Total debt liquidated).
5. **`utilization_ratio`**: $\frac{\text{BILL\_AMT1}}{\max(\text{LIMIT\_BAL}, 1.0)}$ (Credit line utilization; capped to $[-5.0, 20.0]$).
6. **`payment_coverage_ratio`**: $\frac{\text{total\_pay\_amt}}{\max(|\text{total\_bill\_amt}|, 1.0)}$ (Repayment coverage capacity; capped to $[0.0, 20.0]$).
7. **`num_delayed_months`**: $\sum_{i \in \{0, 2, 3, 4, 5, 6\}} \mathbb{I}(\text{PAY\_}i > 0)$ (Frequency of delinquent months, $0$ to $6$).
8. **`max_delay_category`**: $\max(\text{PAY\_0}, \dots, \text{PAY\_6})$ (Peak delinquency severity).

---

## 11. Feature Treatment & Pipeline Preprocessing
- **Numerical Features (28 columns)**: Scaled with `StandardScaler` after `SimpleImputer(strategy='median')`.
- **Categorical Features (3 columns)**: Encoded with `OneHotEncoder(drop='first', handle_unknown='ignore')` after `SimpleImputer(strategy='most_frequent')`.
- Unified inside a scikit-learn `Pipeline`, ensuring identical treatment during offline training and real-time live inference.

---

## 12. Machine Learning Algorithms
1. **Dummy Baseline**: Naive majority-class predictor (`DummyClassifier(strategy='most_frequent')`).
2. **Regularized Logistic Regression**: Generalized linear model with L2 regularization, optimizing log-loss:
   $$\mathcal{L}(\mathbf{w}) = -\sum_{i=1}^N \left[ y_i \log p_i + (1 - y_i)\log(1 - p_i) \right] + \frac{1}{2C} \|\mathbf{w}\|_2^2$$
3. **Decision Tree Classifier**: CART algorithm recursive partitioning with Gini impurity splitting criterion.
4. **Random Forest Classifier**: Ensemble of 100 decorrelated decision trees using bootstrap aggregation with feature subspace sampling.

---

## 13. Experimental Methodology
- **Train/Test Split**: 80% training ($24,000$ samples), 20% hold-out test ($6,000$ samples), stratified on $y$ with `random_state=42`.
- **Cross-Validation**: 5-Fold Stratified K-Fold applied exclusively to training data.
- **Hyperparameter Optimization**: `GridSearchCV` scoring on ROC-AUC within training folds.
- **Test Set Isolation**: The test set was touched exactly once after models and hyperparameters were frozen.

---

## 14. Evaluation Metrics
- **Accuracy**: Overall fraction of correct predictions: $\frac{TP + TN}{N}$.
- **Precision**: Positive predictive value: $\frac{TP}{TP + FP}$.
- **Recall (Sensitivity)**: True positive detection rate: $\frac{TP}{TP + FN}$.
- **F1-Score**: Harmonic mean of Precision and Recall: $\frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$.
- **ROC-AUC**: Area Under the Receiver Operating Characteristic curve.
- **Average Precision (PR-AUC)**: Area under the Precision-Recall curve.

---

## 15. Cross-Validation Results (Training Data Only)

| Model | Tuned Hyperparameters | CV ROC-AUC Mean | CV ROC-AUC Std | CV F1 Mean | CV F1 Std |
|:---|:---|:---:|:---:|:---:|:---:|
| **Logistic Regression** | `C: 10.0` | $0.7588$ | $\pm 0.0076$ | $0.4083$ | $\pm 0.0141$ |
| **Decision Tree** | `max_depth: 5, min_samples_leaf: 20` | $0.7652$ | $\pm 0.0035$ | $0.4639$ | $\pm 0.0101$ |
| **Random Forest** | `n_estimators: 100, max_depth: 10, min_samples_leaf: 10` | **0.7840** | **± 0.0038** | **0.4802** | **± 0.0050** |

---

## 16. Results & Model Comparison on Untouched Test Set ($N = 6,000$)

| Model | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Test ROC-AUC | Average Precision |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Dummy Baseline** | $77.88\%$ | $0.00\%$ | $0.00\%$ | $0.0000$ | $0.5000$ | $0.2212$ |
| **Logistic Regression** | $80.82\%$ | $64.43\%$ | $29.62\%$ | $0.4058$ | $0.7447$ | $0.5042$ |
| **Decision Tree** | $81.62\%$ | $66.82\%$ | $33.53\%$ | $0.4466$ | $0.7607$ | $0.5124$ |
| **Random Forest** | **81.67%** | **65.74%** | **35.72%** | **0.4629** | **0.7782** | **0.5579** |

*All results verified from `outputs/results/final_metrics.json`.*

### Confusion Matrix Breakdown (Random Forest, $N = 6,000$):
- **True Negatives (TN)**: $4,426$
- **False Positives (FP)**: $247$
- **False Negatives (FN)**: $853$
- **True Positives (TP)**: $474$

---

## 17. Operational Decision Threshold Analysis

| Threshold ($\tau$) | Precision | Recall | F1-Score | Strategic Banking Use-Case |
|:---:|:---:|:---:|:---:|:---|
| **0.30** | $53.19\%$ | **55.31%** | **0.5423** | **Capital Preservation**: Captures over $55\%$ of defaults. |
| **0.40** | $60.14\%$ | $44.91\%$ | $0.5142$ | **Balanced Underwriting**: Higher precision with strong recall. |
| **0.50** | $65.74\%$ | $35.72\%$ | $0.4629$ | **Standard Default**: Standard decision threshold. |
| **0.60** | $70.76\%$ | $27.35\%$ | $0.3946$ | **Selective Interception**: Very high precision flags. |
| **0.70** | **80.57%** | $12.81\%$ | $0.2211$ | **Acute Distress Only**: Over $80\%$ of flagged accounts default. |

---

## 18. Feature Importance & Explainability
### Top 10 Features (Random Forest Gini Importance):
1. `PAY_0`: $17.55\%$
2. `num_delayed_months`: $12.84\%$ (**Engineered**)
3. `max_delay_category`: $10.86\%$ (**Engineered**)
4. `PAY_2`: $6.05\%$
5. `PAY_3`: $4.95\%$
6. `PAY_5`: $3.29\%$
7. `utilization_ratio`: $3.07\%$ (**Engineered**)
8. `payment_coverage_ratio`: $2.87\%$ (**Engineered**)
9. `LIMIT_BAL`: $2.81\%$
10. `avg_pay_amt`: $2.76\%$ (**Engineered**)

**Key Insight**: Behavioral repayment metrics (`PAY_0`, `num_delayed_months`, `max_delay_category`) dominate the predictive signal, whereas demographic variables (`AGE`, `SEX`, `MARRIAGE`) exert minimal direct importance.

---

## 19. Live Prediction System & Deployment
1. **Pipeline Persistence**: Serialized via joblib to `models/final_model.joblib` ($4.65$ MB).
2. **Schema Contract**: `outputs/results/input_schema.json` guarantees strict type and range validation.
3. **Primary Gradio Interface (`app/app.py`)**: Categorized input fields, 1-click dataset sample population, real-time risk tiering, and educational disclaimers.
4. **CLI Inference (`src/predict.py`)**: Scriptable scoring tool: `python src/predict.py --demo --sample_id 15`.
5. **Secondary Streamlit Dashboard (`app/streamlit_app.py`)**: Comprehensive metric exploration dashboard.

---

## 20. Responsible Machine Learning & Ethical Considerations
- **Fairness & Proxies**: Financial models that incorporate demographic variables (`SEX`, `MARRIAGE`) risk codifying historical societal bias. In modern production environments, regulations such as the US Equal Credit Opportunity Act (ECOA) mandate demographic omission or disparate impact audits.
- **Explainability**: Using tree ensembles requires feature importance breakdowns or SHAP values to provide adverse action notices to consumers.
- **Human Oversight**: Machine learning predictions should act as decision-support signals rather than unchecked automated rejections.

---

## 21. Limitations
1. **Temporal & Geographic Bounds**: Data originates from Taiwanese credit card holders in 2005; findings do not necessarily generalize to post-2020 macroeconomic environments or other financial jurisdictions.
2. **Unobserved Variables**: Omits critical modern credit variables such as debt-to-income (DTI) ratio, employment duration, and external bureau credit scores.
3. **Class Imbalance Constraints**: While Random Forest achieved $0.7782$ ROC-AUC, default recall at default $0.50$ threshold remains $35.72\%$, requiring threshold lowering for conservative underwriting.

---

## 22. Future Scope
1. **Advanced Ensembles**: Exploration of gradient boosting frameworks (XGBoost, LightGBM, CatBoost).
2. **Cost-Sensitive Learning**: Introducing custom loss functions reflecting financial cost asymmetries (e.g. $10\times$ penalty on False Negatives).
3. **Temporal Cross-Validation**: Utilizing out-of-time (OOT) validation splits instead of randomized splits to simulate quarterly model degradation.

---

## 23. Conclusion
Under the experimental protocol defined in this project, Random Forest was selected as the final demonstration model due to its superior ROC-AUC ($0.7782$), Average Precision ($0.5579$), and balanced F1-score ($0.4629$) relative to Logistic Regression and Decision Trees. The engineered financial indicators significantly enhanced model discrimination without introducing future leakage. The complete system was mathematically validated, covered by 18 automated unit tests, and packaged into a deployable live interface suitable for academic laboratory demonstration.

---

## 24. References
1. Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. *Expert Systems with Applications*, 36(2), 2473-2480.
2. UCI Machine Learning Repository: Default of Credit Card Clients (Dataset ID: 350).
3. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
