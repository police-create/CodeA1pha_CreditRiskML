# Credit Scoring Model Using Machine Learning Classification
## 12-Slide Classroom Presentation Content & Speaking Notes

---

### Slide 1: Title & Overview
- **Header**: CREDIT SCORING MODEL USING MACHINE LEARNING CLASSIFICATION
- **Subheader**: Educational Supervised Classification Prototype for Credit-Default Risk
- **Course**: Machine Learning Laboratory Demonstration
- **Candidate Models**: Dummy Baseline, Logistic Regression, Decision Tree, Random Forest
- **Presenter Speaking Notes**:
  > "Good morning, respected examiners. Today I am presenting my Machine Learning project on Credit Scoring and Credit-Default Risk Prediction. In this project, we built, evaluated, and deployed an end-to-end classification system using the official UCI credit default benchmark dataset, strictly enforcing data leakage prevention, reproducibility, and rigorous evaluation."

---

### Slide 2: Problem Statement & Motivation
- **Header**: The Credit Default Challenge
- **Key Points**:
  - **Financial Context**: Lenders must distinguish solvent borrowers from those likely to default on contractual payments.
  - **Class Imbalance**: Default is a minority event ($\sim 22.12\%$ of accounts). A naive model predicting 'no default' achieves $77.88\%$ accuracy while providing zero risk protection.
  - **Asymmetric Losses**: False Negatives (unintercepted defaults) carry severe principal write-offs, while False Positives (lost loans) carry opportunity costs.
- **Presenter Speaking Notes**:
  > "The primary challenge in credit scoring is asymmetric risk under class imbalance. In our dataset, only 22% of accounts default. If we only look at accuracy, a naive model that approves everyone gets 78% accuracy, but causes massive financial losses. We must optimize for metrics like ROC-AUC, Recall, and PR-AUC."

---

### Slide 3: Project Objectives
- **Header**: Project Scope & Engineering Goals
- **Key Points**:
  - Implement a zero-leakage ML pipeline using scikit-learn `Pipeline` and `ColumnTransformer`.
  - Engineer 8 domain-specific financial features (utilization, payment coverage, delinquency frequencies).
  - Perform hyperparameter tuning strictly within 5-Fold Stratified Cross-Validation training partitions.
  - Evaluate candidate architectures on an untouched hold-out test set ($N = 6,000$).
  - Deploy a live interactive Gradio application (`app/app.py`) with input validation and authentic dataset demo testing.
- **Presenter Speaking Notes**:
  > "Our engineering objective was not just to write code, but to build a verified, reproducible machine learning system with automated tests, an untouched hold-out test set, and a working live demo for classroom evaluation."

---

### Slide 4: Dataset Overview (UCI ID: 350)
- **Header**: Benchmark Dataset: Default of Credit Card Clients
- **Key Points**:
  - **Source**: UCI Machine Learning Repository (Dataset ID: 350, Prof. I-Cheng Yeh).
  - **Volume**: $30,000$ historical customer records observed over 6 months (April–Sept 2005).
  - **Attributes**: $23$ raw predictive features + $1$ binary target (`default_payment_next_month`).
  - **Target Distribution**: Non-Default ($0$): $23,364$ accounts ($77.88\%$); Default ($1$): $6,636$ accounts ($22.12\%$).
  - **Data Governance**: Administrative `ID` column is programmatically excluded to prevent memorization leakage.
- **Presenter Speaking Notes**:
  > "We utilized the official UCI benchmark dataset of 30,000 credit card accounts. We verified the absence of missing values, excluded the ID column so the model cannot memorize borrower indices, and verified the target is binary."

---

### Slide 5: Data Preparation & Leakage Prevention Contract
- **Header**: Ensuring Scientific Reproducibility & Zero Leakage
- **Key Points**:
  - **Stratified Split**: 80% Training ($24,000$ samples), 20% Test ($6,000$ samples) with `random_state=42`.
  - **Pre-split Isolation**: Preprocessing and encoders are fitted **strictly on training data**.
  - **Leakage Contracts**: Programmatic assertions (`assert "ID" not in X.columns`, `assert "default_payment_next_month" not in X.columns`).
  - **Preprocessing**: Numerical scaling (`StandardScaler`) + Categorical encoding (`OneHotEncoder`).
- **Presenter Speaking Notes**:
  > "A major methodological mistake in beginner ML projects is data leakage—such as fitting scalers on the whole dataset before splitting. In our project, we implemented programmatic contracts in code: the 20% test set was locked away and never influenced scaling, encoding, or hyperparameter selection."

---

### Slide 6: Exploratory Data Analysis (EDA)
- **Header**: Empirical Insights & Behavioral Associations
- **Key Points**:
  - **Credit Limit Disparity**: Borrowers with credit limits $< 50,000$ NTD exhibit markedly higher default rates than borrowers above $200,000$ NTD.
  - **Payment Delinquency**: Repayment status in recent months (`PAY_0`, `PAY_2`) is the strongest single visual indicator of subsequent default.
  - **Non-Causal Disclaimer**: Observed associations reflect correlation within this historical sample, not direct financial causation.
- **Presenter Speaking Notes**:
  > "EDA clearly highlighted that recent delinquency—specifically being 2 or more months late in September—is the most pronounced risk factor. However, we maintain academic honesty: correlation in historical data does not prove financial causation."

---

### Slide 7: Domain-Specific Feature Engineering
- **Header**: Constructing Genuine Financial Indicators
- **Key Points**:
  - `utilization_ratio`: $\text{avg\_bill\_amt} / \text{LIMIT\_BAL}$ (Revolving debt strain).
  - `payment_coverage_ratio`: $\text{total\_pay\_amt} / |\text{total\_bill\_amt}|$ (Debt service capacity).
  - `num_delayed_months`: Count of months with payment delay status $> 0$ ($0$ to $6$).
  - `max_delay_category`: Peak delinquency score across the 6-month horizon.
  - `total_bill_amt` & `avg_bill_amt`, `total_pay_amt` & `avg_pay_amt`.
  - **Impact**: 5 of the top 10 most predictive features in Random Forest are our engineered features!
- **Presenter Speaking Notes**:
  > "Rather than using raw billing numbers alone, we engineered 8 domain-justified features such as credit utilization, payment coverage, and delinquent month counts. As we will see, our engineered features proved to be among the top predictors in the trained model."

---

### Slide 8: Machine Learning Algorithms
- **Header**: Modeling Strategy & Candidate Architectures
- **Key Points**:
  - **Dummy Baseline**: Majority-class heuristic benchmark.
  - **Logistic Regression**: Scaled linear classifier with L2 regularization penalty ($C=10.0$).
  - **Decision Tree**: Non-linear CART classifier (`max_depth=5`, `min_samples_leaf=20`).
  - **Random Forest**: Bagged ensemble of 100 decorrelated trees (`max_depth=10`, `min_samples_leaf=10`).
  - **Tuning**: Conducted using `GridSearchCV` scoring on ROC-AUC with 5-fold Stratified CV.
- **Presenter Speaking Notes**:
  > "We compared four models representing different inductive biases: from a naive baseline to an interpretable linear model, a single decision tree, and an ensemble random forest. All hyperparameters were tuned exclusively on the training folds."

---

### Slide 9: Evaluation Metrics & Protocol
- **Header**: Beyond Accuracy: Multi-Metric Evaluation
- **Key Points**:
  - **Accuracy**: Overall fraction of correct classifications.
  - **Precision**: How many flagged defaults are true defaults? ($65.74\%$ for Random Forest).
  - **Recall**: What fraction of actual defaults did we intercept? ($35.72\%$ at standard threshold).
  - **F1-Score**: Harmonic mean balancing Precision and Recall ($0.4629$).
  - **ROC-AUC & PR-AUC**: Global threshold-independent discrimination metrics.
- **Presenter Speaking Notes**:
  > "Because of the 78:22 class imbalance, accuracy is insufficient. We evaluate all models on Precision, Recall, F1, ROC-AUC, and Average Precision to obtain an honest assessment of detection power."

---

### Slide 10: Actual Experimental Results & Comparison
- **Header**: Model Performance on Untouched Test Set ($N = 6,000$)
- **Table**:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Average Precision | 5-Fold CV ROC-AUC |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Dummy Baseline** | $77.88\%$ | $0.00\%$ | $0.00\%$ | $0.0000$ | $0.5000$ | $0.2212$ | $0.5000 \pm 0.0000$ |
| **Logistic Regression** | $80.82\%$ | $64.43\%$ | $29.62\%$ | $0.4058$ | $0.7447$ | $0.5042$ | $0.7588 \pm 0.0076$ |
| **Decision Tree** | $81.62\%$ | $66.82\%$ | $33.53\%$ | $0.4466$ | $0.7607$ | $0.5124$ | $0.7652 \pm 0.0035$ |
| **Random Forest** | **81.67%** | **65.74%** | **35.72%** | **0.4629** | **0.7782** | **0.5579** | **0.7840 ± 0.0038** |

- **Presenter Speaking Notes**:
  > "These are the actual experimental numbers recorded in our final metrics JSON. Random Forest achieved the highest ROC-AUC of 0.7782 and Average Precision of 0.5579, outperforming both Logistic Regression and Decision Tree while demonstrating high CV stability with a standard deviation of only 0.0038."

---

### Slide 11: Live Demonstration & Responsible Machine Learning
- **Header**: Interactive Deployment & Ethical Considerations
- **Key Points**:
  - **Live Interfaces**: Primary Gradio application (`app/app.py`), CLI inference (`src/predict.py`), Streamlit dashboard.
  - **End-to-End Pipeline**: Saved in `models/final_model.joblib`; inputs are validated before scoring.
  - **1-Click Authentic Sample**: Evaluates real dataset records and compares with historical ground truth.
  - **Responsible ML**: Acknowledges that demographic features (`SEX`, `MARRIAGE`) cannot be used indiscriminately under fair lending statutes (e.g. ECOA/GDPR).
- **Presenter Speaking Notes**:
  > "We persisted the complete pipeline into final_model.joblib so that live scoring requires zero retraining. In our Gradio app, users can test custom inputs or load authentic dataset profiles. We also emphasize Responsible ML: models in banking require fair lending compliance and human oversight."

---

### Slide 12: Conclusion & Viva Readiness
- **Header**: Summary & Academic Readiness
- **Key Takeaways**:
  - **Champion Architecture**: Random Forest selected based on empirical superiority across ROC-AUC ($0.7782$) and PR-AUC ($0.5579$).
  - **Threshold Flexibility**: In conservative lending, lowering the threshold to $0.30$ increases recall to $55.31\%$.
  - **Testing & Verification**: 18/18 pytest tests passed, master validation passed, zero fabricated numbers.
- **Presenter Speaking Notes**:
  > "In conclusion, this project provides a mathematically validated, reproducible, and fully demonstrable credit scoring prototype. I am now ready to demonstrate the live application and answer your questions. Thank you!"
