# Dataset Information: Default of Credit Card Clients

## Source & Provenance
- **Repository**: UCI Machine Learning Repository
- **Dataset Title**: Default of Credit Card Clients
- **Dataset ID**: 350
- **Official UCI Link**: https://archive.ics.uci.edu/dataset/350/default%2Bof%2Bcredit%2Bcard%2Bclients
- **Original Source**: Department of Information Management, Chung Hua University, Taiwan and Department of Civil Engineering, Tamkang University, Taiwan (Prof. I-Cheng Yeh).
- **Date Donated**: 2016-01-26

## Dataset Overview
- **Number of Instances**: 30,000
- **Number of Raw Attributes**: 24 (23 predictive features + 1 target variable)
- **Target Variable**: `default_payment_next_month` (Binary: 1 = Default, 0 = Non-Default)
- **Time Period**: April 2005 to September 2005

## Acquisition Strategy
1. **Primary**: Dynamic retrieval via `ucimlrepo.fetch_ucirepo(id=350)`.
2. **Fallback 1**: Direct HTTP download of the dataset XLS/CSV from the official UCI repository URL.
3. **Fallback 2**: Local offline cache located at `data/default_of_credit_card_clients.csv`.

## Data Governance & Leakage Prevention
- The `ID` column is an administrative identifier and is **strictly excluded** from all feature engineering, modeling, and evaluation pipelines.
- Target column `default_payment_next_month` is isolated immediately from the predictor feature matrix `X`.
