# 5-Minute Classroom Laboratory Demonstration Script
## Exact Minute-by-Minute Script with Simple Sentences for Beginners

---

### Timing Breakdown:
- **0:00 – 0:30**: Problem Introduction & Objective
- **0:30 – 1:00**: Show the Genuine UCI Dataset
- **1:00 – 1:40**: Show Exploratory Data Analysis (EDA)
- **1:40 – 2:20**: Show Domain Feature Engineering
- **2:20 – 3:00**: Show Model Training & Leakage Prevention
- **3:00 – 3:40**: Show Actual Metrics & Model Comparison
- **3:40 – 4:40**: Perform Live Prediction Demo (Gradio)
- **4:40 – 5:00**: Limitations, Ethical AI & Conclusion

---

### [0:00 – 0:30] Problem Introduction & Objective
*(Stand confidently, click open the presentation title slide or README)*

**Say Exactly This:**
> "Good morning, respected examiners. Today I am presenting my project: **Credit Scoring Model Using Machine Learning Classification**.
> In retail banking, evaluating whether a credit card holder will default on their payment is critical to prevent financial losses.
> In our project, we framed this as a binary classification problem: predicting whether an account will default next month based on their credit limit, demographic details, billing statements, and past repayment history.
> We built a complete, reproducible machine learning system with zero data leakage, automated testing, and a live web application."

---

### [0:30 – 1:00] Show the Genuine Dataset
*(Show the notebook or `data/README.md` or printout of `X.head()`)*

**Say Exactly This:**
> "Here is our dataset. We used the official benchmark dataset from the **UCI Machine Learning Repository**, Dataset ID 350, titled **Default of Credit Card Clients**.
> It contains **30,000 real customer records** observed over a 6-month period in Taiwan.
> The target variable is `default_payment_next_month`.
> As you can see on the screen, there are **23,364 non-defaulting accounts (78%)** and **6,636 defaulting accounts (22%)**.
> We programmatically dropped the `ID` column right away so the model cannot memorize borrower row indices."

---

### [1:00 – 1:40] Show Exploratory Data Analysis (EDA)
*(Open `outputs/figures/target_distribution.png` and `outputs/figures/credit_limit_distribution.png`)*

**Say Exactly This:**
> "Next, we performed Exploratory Data Analysis.
> Looking at the target distribution, there is a clear class imbalance of about 3.5 to 1. This means accuracy alone will be misleading, because a naive model that predicts 'no default' would achieve 78% accuracy with zero recall.
> In the second plot, we observe that borrowers with lower credit limits—under 50,000 NT dollars—exhibited significantly higher rates of default compared to borrowers with limits above 200,000.
> Most importantly, the correlation analysis showed that the repayment status in the most recent month—feature `PAY_0`—had the strongest correlation with default."

---

### [1:40 – 2:20] Show Feature Engineering
*(Open `src/feature_engineering.py` or `outputs/figures/feature_importance.png`)*

**Say Exactly This:**
> "To give the models stronger financial signals, we engineered 8 domain-specific features without causing data leakage.
> First, we calculated `utilization_ratio`—which is average bill amount divided by credit limit, showing how close someone is to exhausting their line of credit.
> Second, we calculated `payment_coverage_ratio`—the total repayments divided by total bill amounts, measuring debt service capacity.
> Third, we created `num_delayed_months`, counting how many months a borrower was late.
> When we look at feature importance in our final model, **5 of the top 10 most predictive features were our engineered features**, proving that domain engineering significantly boosted predictive power."

---

### [2:20 – 3:00] Show Model Training & Leakage Prevention
*(Show `src/train.py` or the pipeline diagram)*

**Say Exactly This:**
> "For our modeling pipeline, we enforced a strict Data Leakage Contract in code.
> We split the data into an 80% training set of 24,000 samples and a held-out test set of 6,000 samples with `random_state=42`.
> All preprocessing—such as standard scaling and one-hot encoding—was fitted **strictly on the training data** inside a scikit-learn `Pipeline`.
> We compared four models: a Dummy Baseline, Logistic Regression, a Decision Tree, and a Random Forest.
> All hyperparameters were tuned using **5-Fold Stratified Cross-Validation on the training folds only**, and the test set was evaluated exactly once at the end."

---

### [3:00 – 3:40] Show Actual Metrics & Model Comparison
*(Show `outputs/results/model_comparison.csv` and `outputs/figures/roc_curves.png`)*

**Say Exactly This:**
> "Here are our actual experimental results evaluated on the untouched 6,000 test samples.
> The Dummy Baseline achieved 77.88% accuracy, but an F1-score and Recall of zero.
> Logistic Regression achieved an Accuracy of 80.82% and ROC-AUC of 0.7447.
> Decision Tree achieved an Accuracy of 81.62% and ROC-AUC of 0.7607.
> **Random Forest was our champion model**, achieving:
> - **Accuracy**: 81.67%
> - **ROC-AUC**: 0.7782
> - **Average Precision (PR-AUC)**: 0.5579
> - **F1-Score**: 0.4629 with 65.74% Precision.
> In our confusion matrix, Random Forest correctly identified 4,426 solvent borrowers and intercepted 474 defaults."

---

### [3:40 – 4:40] Perform Live Prediction Demo
*(Open the Gradio app in the browser: `http://127.0.0.1:7860`)*

**Say Exactly This:**
> "Now, let me demonstrate our live application.
> The interface loads the complete serialized pipeline from `models/final_model.joblib`. No models are retrained on the fly.
> First, let me select an authentic record from our dataset using the **Quick Demo Selector**: **'Sample 1: Low-Risk Prime Borrower'**.
> I click **'Load Selected Demo Sample'**. As you see, it automatically populates a credit limit of 260,000 NT dollars, zero repayment delays, and consistent payments.
> When I click **'Run Credit Risk Assessment'**, the model immediately outputs:
> **'NO DEFAULT'** with an estimated default probability of **10.31%**, placing them in **Tier 1: Prime / Low Risk**.
> And the educational comparison confirms: the actual historical outcome in the dataset was indeed **No Default**.
> Next, let's load **'Sample 2: High-Risk Delinquent Borrower'**.
> This borrower has a small limit of 30,000 NT dollars and is 2 months late on recent payments.
> Clicking Assess outputs: **'DEFAULT RISK DETECTED'** with a probability of **62.88%**, placing them in **Tier 3: Elevated Risk**, which matches the true historical default label."

---

### [4:40 – 5:00] Limitations, Responsible ML & Conclusion
*(Conclude with confidence)*

**Say Exactly This:**
> "Finally, we addressed Responsible Machine Learning and limitations.
> Our dataset includes demographic variables like sex and marital status. In production banking, regulations like the Equal Credit Opportunity Act strictly prohibit demographic discrimination. Furthermore, this dataset reflects Taiwanese credit cards from 2005.
> All 18 automated unit tests in our project passed with 100% success, and the master system validation check passed completely.
> Thank you, and I am now ready to answer any questions!"
