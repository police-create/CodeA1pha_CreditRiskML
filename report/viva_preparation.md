# Comprehensive Viva Voce Preparation Guide
## 38 Technical Viva Questions and Clear, Beginner-Friendly Answers

---

### Q1: What is Credit Scoring, and what problem does your project solve?
**Answer**:  
Credit scoring is a statistical method used by lenders to assess the likelihood that a borrower will fail to repay their loan obligations. Our project frames credit scoring as a **binary classification problem**: using a borrower's credit limit, demographic details, billing records, and past repayment behavior, the model predicts whether they will default next month ($1 = \text{Default}, 0 = \text{No Default}$).

---

### Q2: What dataset did you use, and what is its official source?
**Answer**:  
We used the **Default of Credit Card Clients** dataset from the official **UCI Machine Learning Repository** (Dataset ID: 350), collected by Prof. I-Cheng Yeh from a major bank in Taiwan in 2005. It contains $30,000$ accounts with $23$ raw input features and $1$ target variable.

---

### Q3: What is the distribution of the target class in this dataset?
**Answer**:  
The dataset is **imbalanced**:
- Non-Default ($0$): $23,364$ accounts ($77.88\%$)
- Default ($1$): $6,636$ accounts ($22.12\%$)  
This means approximately $3.5$ non-defaulting borrowers exist for every $1$ defaulting borrower.

---

### Q4: Why is Accuracy a misleading metric for this project?
**Answer**:  
Because of class imbalance, a naive dummy model that predicts "No Default" for every single borrower achieves an accuracy of $77.88\%$ while having a Recall of $0\%$. In credit risk, missing a defaulting borrower leads to loan write-offs. Therefore, we prioritize **ROC-AUC**, **Recall**, **Precision**, **F1-Score**, and **Average Precision (PR-AUC)** over accuracy alone.

---

### Q5: What is Data Leakage, and how did you prevent it in your code?
**Answer**:  
Data leakage occurs when information from outside the training dataset (such as the target variable or future test data) is inadvertently used to create the model. We prevented it through a strict **Data Leakage Contract**:
1. The administrative `ID` column was dropped immediately.
2. The 80/20 train/test split was performed **before** any scaling or transformation.
3. Preprocessing (imputation and standard scaling) was fitted **only** on the training set.
4. Hyperparameter tuning using 5-fold cross-validation used **only** the training folds.
5. The test set ($6,000$ samples) was held out untouched until final evaluation.

---

### Q6: Why did you drop the ID column?
**Answer**:  
`ID` is just an arbitrary sequential identifier. If included as a predictive feature, models like Decision Trees can memorize which ID numbers defaulted in the training data, leading to massive overfitting and false high performance that fails completely on fresh borrowers.

---

### Q7: What features did you engineer, and why?
**Answer**:  
We engineered 8 domain-specific financial features:
1. `avg_bill_amt` & `total_bill_amt`: Captures overall monthly balance and cumulative debt.
2. `avg_pay_amt` & `total_pay_amt`: Measures repayment magnitude.
3. `utilization_ratio`: $\text{avg\_bill\_amt} / \text{LIMIT\_BAL}$, measuring how close the borrower is to exhausting their credit limit.
4. `payment_coverage_ratio`: $\text{total\_pay\_amt} / |\text{total\_bill\_amt}|$, measuring debt servicing ability.
5. `num_delayed_months`: Total count of months with late payments ($0$ to $6$).
6. `max_delay_category`: The worst payment delay status recorded across 6 months.

---

### Q8: How do you know your engineered features actually helped?
**Answer**:  
In our feature importance analysis of the champion Random Forest model, **5 of the top 10 most predictive features were our engineered features**. Specifically, `num_delayed_months` ranked #2 ($12.84\%$ importance) and `max_delay_category` ranked #3 ($10.86\%$), proving they added significant predictive signal.

---

### Q9: What candidate models did you evaluate?
**Answer**:  
We evaluated four models:
1. **Dummy Baseline Classifier**: Majority class benchmark.
2. **Logistic Regression**: Linear benchmark with L2 regularization ($C$-tuned).
3. **Decision Tree Classifier**: Single tree tuned for depth and leaf sizes.
4. **Random Forest Classifier**: Ensemble of 100 decorrelated decision trees.

---

### Q10: Which model was selected as the final champion, and what were its exact results?
**Answer**:  
**Random Forest** was selected based on empirical evidence on the untouched test set ($N = 6,000$):
- **Accuracy**: $81.67\%$
- **ROC-AUC**: $0.7782$ (Highest among all models)
- **Average Precision (PR-AUC)**: $0.5579$ (Highest)
- **F1-Score**: $0.4629$ (Precision: $65.74\%$, Recall: $35.72\%$)
- **5-Fold CV ROC-AUC**: $0.7840 \pm 0.0038$ (Highest stability)

---

### Q11: What is a Confusion Matrix, and what are your model's exact counts?
**Answer**:  
A confusion matrix breaks down predictions into 4 outcomes:
- **True Negatives (TN)**: $4,426$ solvent borrowers correctly predicted.
- **False Positives (FP)**: $247$ solvent borrowers mistakenly flagged as defaulting.
- **False Negatives (FN)**: $853$ defaulting borrowers missed by the model.
- **True Positives (TP)**: $474$ defaulting borrowers successfully caught.
- **Total**: $4,426 + 247 + 853 + 474 = 6,000$ test borrowers.

---

### Q12: In credit scoring, which is worse: a False Positive or a False Negative?
**Answer**:  
It depends on the lender's business strategy, but typically **False Negatives are financially more dangerous** because failing to catch a default results in unrecoverable loan principal loss. False Positives mean rejecting a solvent applicant, resulting in lost interest revenue and customer friction. Lenders can adjust the decision threshold depending on their risk tolerance.

---

### Q13: What is Threshold Analysis, and what happens when you change the threshold?
**Answer**:  
By default, classifiers predict default if probability $\ge 0.50$. In threshold analysis, we evaluate performance at different cut-offs:
- At $\tau = 0.30$: Recall increases to **$55.31\%$** (catches more defaults), but Precision drops to $53.19\%$.
- At $\tau = 0.70$: Precision rises to **$80.57\%$** (high confidence when flagged), but Recall drops to $12.81\%$.  
There is no universally best threshold; conservative lenders prefer lower thresholds like $0.30$ to prevent bad loans.

---

### Q14: What is ROC-AUC, and how do you interpret a score of 0.7782?
**Answer**:  
ROC-AUC (Area Under the Receiver Operating Characteristic curve) measures the probability that the model ranks a randomly chosen defaulting borrower higher than a randomly chosen solvent borrower. A random guess score is $0.50$, and a perfect model is $1.00$. Our score of **$0.7782$** indicates strong, reliable ranking discrimination.

---

### Q15: What is Average Precision / PR-AUC, and why is it important here?
**Answer**:  
PR-AUC measures the area under the Precision-Recall curve. Under class imbalance, ROC-AUC can sometimes look optimistic because True Negatives are so numerous. PR-AUC focuses directly on the minority class. Our Random Forest achieved an Average Precision of **$0.5579$**, which is more than **2.5 times higher than the no-skill baseline of $0.2212$**.

---

### Q16: How did you implement Cross-Validation?
**Answer**:  
We used **Stratified 5-Fold Cross-Validation** (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`) on the training data. Stratification ensures each of the 5 folds contains the exact $22.12\%$ default proportion. The mean and standard deviation of ROC-AUC and F1 were recorded to confirm model stability.

---

### Q17: What hyperparameters were tuned for Random Forest?
**Answer**:  
Using `GridSearchCV` scoring on ROC-AUC over 5 folds:
- `max_depth`: tested $6$ and $10$ (Best: $10$)
- `min_samples_leaf`: tested $5$ and $10$ (Best: $10$)
- `n_estimators`: fixed at $100$ trees  
Restricting `min_samples_leaf` to 10 prevented individual trees from memorizing outlier accounts.

---

### Q18: Did your model overfit? How do you prove it?
**Answer**:  
We performed a dedicated Overfitting Audit comparing Train vs Test performance:
- Train ROC-AUC: $0.8537$
- Test ROC-AUC: $0.7782$
- Generalization Gap: $+0.0755$ ($7.5\%$)  
This gap is modest and expected for an ensemble of 100 trees with restricted depth. An overfitted tree without constraints typically exhibits a gap $> 25\%$.

---

### Q19: What is Gini Feature Importance?
**Answer**:  
Gini importance (or Mean Decrease in Impurity) measures the total reduction in node impurity brought by a feature across all 100 decision trees in the forest. A higher percentage indicates that splits on that feature contributed more to separating defaulters from non-defaulters.

---

### Q20: What was the single most important feature according to the model?
**Answer**:  
`PAY_0` (repayment status in September 2005) was the most important feature, accounting for **$17.55\%$** of overall importance. Borrowers who were already 2 months late in September had the highest probability of subsequent default.

---

### Q21: What information do Logistic Regression coefficients give you?
**Answer**:  
Logistic Regression coefficients indicate the direction and relative magnitude of effect for each standardized feature. Positive coefficients (like `PAY_0` and `utilization_ratio`) increase the predicted log-odds of default, while negative coefficients (like `LIMIT_BAL` and `PAY_AMT1`) decrease the log-odds of default.

---

### Q22: What is the purpose of the Pipeline and ColumnTransformer?
**Answer**:  
They package feature engineering, numerical scaling, categorical encoding, and classification into a **single callable object**. This guarantees that whenever fresh raw data is presented to `pipeline.predict()`, the exact same transformations fitted during training are applied automatically, eliminating training-serving skew.

---

### Q23: How did you handle categorical variables?
**Answer**:  
`SEX`, `EDUCATION`, and `MARRIAGE` were encoded using `OneHotEncoder(drop='first', handle_unknown='ignore')`. Setting `handle_unknown='ignore'` ensures that if an unexpected category value appears during live user input, the application does not crash.

---

### Q24: What is model persistence, and how is it implemented?
**Answer**:  
Model persistence is saving the trained model pipeline to disk so it can be reloaded for live inference without retraining. We saved our pipeline as `models/final_model.joblib` ($4.65$ MB). In unit tests, we proved that pre-save and post-reload predictions match with an error of $0.0000$.

---

### Q25: How does your live Gradio web application work?
**Answer**:  
The Gradio app (`app/app.py`):
1. Loads `models/final_model.joblib` and `outputs/results/input_schema.json`.
2. Provides intuitive input fields organized into tabs.
3. Includes a 1-click **Quick Demo Selector** to populate authentic dataset profiles.
4. Validates user inputs (non-negative credit limit, age 18–100, non-negative payments).
5. Runs the pipeline and displays Risk Classification, Default Probability %, Assigned Risk Tier, and Educational Comparison.

---

### Q26: How do you prove your live application does not use hardcoded predictions?
**Answer**:  
The Gradio app imports and calls `final_model.joblib` on the exact `DataFrame` constructed from user inputs. In our unit tests (`test_prediction.py`), changing input values dynamically changes the output probabilities.

---

### Q27: What automated tests did you implement?
**Answer**:  
We implemented **18 automated unit tests** in `tests/` using `pytest`:
- 5 data integrity and leakage tests (`test_data.py`).
- 4 preprocessing and division-by-zero tests (`test_preprocessing.py`).
- 5 prediction, schema validation, and error-handling tests (`test_prediction.py`).
- 4 pipeline persistence and reload equivalence tests (`test_pipeline.py`).  
All 18 tests passed with a $100\%$ pass rate in $14.5$ seconds.

---

### Q28: What is `tests/run_all_checks.py`?
**Answer**:  
It is our master QA validation script. It sequentially audits 10 critical system components (dataset, leakage rules, feature engineering, model artifact, reload equivalence, schema contract, result files, 14 figures, CLI engine, and pytest). It prints `PROJECT VALIDATION PASSED` only when all 10 checks succeed.

---

### Q29: What ethical and Responsible ML considerations apply to this dataset?
**Answer**:  
The dataset contains demographic features (`SEX`, `MARRIAGE`, `EDUCATION`). In modern financial regulations (such as the US Equal Credit Opportunity Act and EU GDPR), using demographic attributes directly for credit denial is legally restricted to prevent algorithmic discrimination. In our educational project, we demonstrate feature encoding and emphasize that real-world deployment requires fair lending audits and human oversight.

---

### Q30: What is the difference between a Decision Tree and a Random Forest?
**Answer**:  
A Decision Tree makes decisions based on a single hierarchical set of rules, which often suffers from high variance and overfitting. A Random Forest builds an ensemble of many decision trees (100 in our project), each trained on a bootstrap sample of data with randomized feature subsets, and averages their predictions. This bagging technique dramatically reduces variance and boosts generalization.

---

### Q31: Why did you use `random_state=42`?
**Answer**:  
`random_state=42` initializes pseudo-random number generators to a fixed seed. This ensures that train/test splitting, cross-validation fold generation, and Random Forest bootstrapping produce identical, reproducible results every time the code is executed.

---

### Q32: What are the main limitations of this project?
**Answer**:  
1. **Historical & Geographic Scope**: Data is from Taiwan in 2005; consumer credit behaviors may differ across countries and economic cycles.
2. **Missing Variables**: Modern credit underwriting incorporates debt-to-income (DTI) ratios, bureau credit scores (e.g. FICO/CIBIL), and employment duration, which are not in this dataset.
3. **Recall Trade-off**: At the default $0.50$ threshold, recall is $35.72\%$, requiring threshold lowering for conservative underwriting.

---

### Q33: How would you improve this project in the future?
**Answer**:  
1. Implement gradient boosting algorithms (XGBoost, LightGBM, CatBoost) for potential incremental gains in Average Precision.
2. Incorporate cost-sensitive learning to place an explicit monetary penalty on False Negatives during model training.
3. Implement out-of-time (OOT) temporal validation splits to simulate quarterly model drift over time.

---

### Q34: What is the difference between a continuous numerical, categorical, and ordinal feature?
**Answer**:  
- **Continuous Numerical**: Numeric measurements with meaningful infinite gradations (e.g. `LIMIT_BAL`, `AGE`, `BILL_AMT1`).
- **Categorical (Nominal)**: Discrete categories without natural ordering (e.g. `SEX`, `MARRIAGE`).
- **Ordinal**: Discrete categories with a clear sequential order (e.g. `PAY_0` where $-1 < 0 < 1 < 2$ represents increasing months of payment delay).

---

### Q35: What happens if an applicant enters an Age of 15 or a negative Credit Limit in the live demo?
**Answer**:  
Our input validator catches this immediately before inference and returns a clear, user-friendly notice: *"Credit Limit must be greater than 0 NTD"* or *"Age must be between 18 and 100 years"*. The application does not crash.

---

### Q36: What is the purpose of `SimpleImputer` if the raw dataset had 0 missing values?
**Answer**:  
Even if the training dataset had zero missing values, real-world user inputs or future batch scoring might contain missing fields. Including `SimpleImputer` in the pipeline makes the system robust against runtime `None` or `NaN` inputs.

---

### Q37: How do you interpret an estimated default probability of 62.88%?
**Answer**:  
An estimated probability of $62.88\%$ means the model assesses a $62.88\%$ statistical likelihood that this borrower profile will fail to meet their minimum payment obligation next month. Because $62.88\% \ge 50\%$, the model classifies them in **Tier 3 / Elevated Delinquency Risk (Default)**.

---

### Q38: Why is this project described as an "Educational Prototype"?
**Answer**:  
Because production credit scoring systems require compliance with banking regulatory frameworks (e.g. Basel II/III capital adequacy standards, FCRA adverse action explanations, and fair lending statutes), extensive stress-testing, and continuous monitoring for concept drift. Describing it honestly as an educational prototype reflects academic integrity.
