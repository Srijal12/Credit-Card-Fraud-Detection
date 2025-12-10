# Credit-Card-Fraud-Detection
Credit card fraud detection using Python, Scikit-Learn, SMOTE, and XGBoost to minimize false negatives.
Overview

This project focuses on detecting fraudulent credit card transactions using machine learning techniques. Fraud detection is a critical problem because fraudulent activities are rare and highly costly. The goal of the project is to build a model that identifies fraudulent transactions with high recall while minimizing false negatives.

Dataset

A credit card transaction dataset (such as the Kaggle dataset) is used. It contains anonymized numerical feature values along with a target variable indicating whether a transaction is fraudulent (1) or legitimate (0).

Tools and Libraries
Python
Pandas, NumPy
Scikit-learn
XGBoost
Imbalanced-learn (SMOTE)
Matplotlib, Seaborn

Key Steps

Loaded and explored the credit card dataset.
Handled severe class imbalance using SMOTE oversampling.
Performed exploratory data analysis to understand patterns and correlations.
Trained and evaluated multiple models, including:
Logistic Regression
Random Forest
XGBoost
Evaluated model performance using ROC-AUC, confusion matrix, precision, recall, and F1-score.
Selected the best-performing model based on recall and AUC scores.

Outcome
The project produced a fraud detection model capable of identifying most fraudulent transactions with high recall. Handling class imbalance significantly improved model performance. The final model can be integrated into real-world banking or financial applications to reduce financial losses caused by fraudulent activities.

Files in Repository
fraud_detection.py (main script)
creditcard.csv (dataset, if included)
README.md (project documentation)

creditcard.csv (dataset, if included)

README.md (project documentation
