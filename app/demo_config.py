"""Configuration, input definitions, and demo samples for the live applications.
"""

from typing import Dict, Any, List

EDUCATION_MAP = {
    "Graduate School (1)": 1,
    "University (2)": 2,
    "High School (3)": 3,
    "Others (4)": 4,
}

MARRIAGE_MAP = {
    "Married (1)": 1,
    "Single (2)": 2,
    "Others (3)": 3,
}

SEX_MAP = {
    "Male (1)": 1,
    "Female (2)": 2,
}

PAY_STATUS_MAP = {
    "-2: No consumption / Account inactive": -2,
    "-1: Paid in full": -1,
    "0: Revolving credit / Minimum paid": 0,
    "1: Payment delayed by 1 month": 1,
    "2: Payment delayed by 2 months": 2,
    "3: Payment delayed by 3 months": 3,
    "4: Payment delayed by 4 months": 4,
    "5: Payment delayed by 5 months": 5,
    "6: Payment delayed by 6 months": 6,
    "7: Payment delayed by 7 months": 7,
    "8: Payment delayed by 8+ months": 8,
}

# Pre-packaged authentic dataset profiles for 1-click classroom demo
PRESET_DEMO_SAMPLES = {
    "Sample 1: Low-Risk Prime Borrower (No Default)": {
        "description": "High credit limit, zero delay, consistent full repayments.",
        "actual_target": 0,
        "values": {
            "LIMIT_BAL": 260000,
            "SEX": 2,
            "EDUCATION": 1,
            "MARRIAGE": 2,
            "AGE": 32,
            "PAY_0": -1,
            "PAY_2": -1,
            "PAY_3": -1,
            "PAY_4": -1,
            "PAY_5": -1,
            "PAY_6": -1,
            "BILL_AMT1": 2500,
            "BILL_AMT2": 2600,
            "BILL_AMT3": 2400,
            "BILL_AMT4": 3000,
            "BILL_AMT5": 2800,
            "BILL_AMT6": 2100,
            "PAY_AMT1": 2600,
            "PAY_AMT2": 2400,
            "PAY_AMT3": 3000,
            "PAY_AMT4": 2800,
            "PAY_AMT5": 2100,
            "PAY_AMT6": 2500,
        },
    },
    "Sample 2: High-Risk Delinquent Borrower (Default)": {
        "description": "Low credit limit, multi-month delinquency (PAY_0=2), low debt repayment coverage.",
        "actual_target": 1,
        "values": {
            "LIMIT_BAL": 30000,
            "SEX": 1,
            "EDUCATION": 2,
            "MARRIAGE": 1,
            "AGE": 38,
            "PAY_0": 2,
            "PAY_2": 2,
            "PAY_3": 0,
            "PAY_4": 0,
            "PAY_5": 0,
            "PAY_6": 0,
            "BILL_AMT1": 29400,
            "BILL_AMT2": 29800,
            "BILL_AMT3": 28500,
            "BILL_AMT4": 28000,
            "BILL_AMT5": 27500,
            "BILL_AMT6": 27000,
            "PAY_AMT1": 1500,
            "PAY_AMT2": 0,
            "PAY_AMT3": 1200,
            "PAY_AMT4": 1000,
            "PAY_AMT5": 1000,
            "PAY_AMT6": 1000,
        },
    },
    "Sample 3: Moderate-Risk Revolver (Borderline)": {
        "description": "Revolving balance user with 1 month delay.",
        "actual_target": 0,
        "values": {
            "LIMIT_BAL": 100000,
            "SEX": 2,
            "EDUCATION": 2,
            "MARRIAGE": 2,
            "AGE": 27,
            "PAY_0": 1,
            "PAY_2": 0,
            "PAY_3": 0,
            "PAY_4": 0,
            "PAY_5": 0,
            "PAY_6": 0,
            "BILL_AMT1": 45000,
            "BILL_AMT2": 42000,
            "BILL_AMT3": 40000,
            "BILL_AMT4": 38000,
            "BILL_AMT5": 35000,
            "BILL_AMT6": 32000,
            "PAY_AMT1": 3000,
            "PAY_AMT2": 3000,
            "PAY_AMT3": 2500,
            "PAY_AMT4": 2000,
            "PAY_AMT5": 2000,
            "PAY_AMT6": 2000,
        },
    },
}
