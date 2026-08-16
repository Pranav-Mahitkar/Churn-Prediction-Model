# Customer Churn Prediction

A machine learning project that predicts telecom customer churn by benchmarking 10 classification algorithms and tuning the top performers for maximum ROC-AUC.

## Overview

Customer churn — when a subscriber stops using a company's service — is one of the most expensive problems in subscription businesses. This project builds an end-to-end pipeline that cleans a telecom customer dataset, explores the drivers of churn, trains and compares 10 classifiers, and tunes the best-performing models with cross-validated hyperparameter search.

## Dataset

- **Source:** Telco Customer Churn dataset (`data.csv`)
- **Size:** 7,043 customer records, 21 features
- **Target:** `Churn` (Yes / No)
- **Features:** Demographics (gender, senior citizen, partner, dependents), account info (tenure, contract type, payment method, paperless billing), and subscribed services (phone, internet, online security, tech support, streaming TV/movies, etc.)

## Data Cleaning & Preprocessing

- Dropped the non-predictive `customerID` column
- Converted `TotalCharges` from string to numeric, coercing blank entries
- Removed 11 records with zero tenure (new customers with missing `TotalCharges`)
- Mapped `SeniorCitizen` from binary (0/1) to categorical (No/Yes) for consistency with other fields
- Split categorical features into three groups for encoding:
  - **One-hot encoding** for multi-category nominal features (`PaymentMethod`, `Contract`, `InternetService`)
  - **Label encoding** for binary/ordinal categorical features
  - **Standard scaling** for continuous features (`tenure`, `MonthlyCharges`, `TotalCharges`)
- Checked multicollinearity via Variance Inflation Factor (VIF)

## Exploratory Data Analysis

Key patterns identified before modeling:
- Customers on **month-to-month contracts** churn at a much higher rate than those on annual/two-year contracts
- **Higher monthly charges** correlate with higher churn likelihood
- **Newer customers** (low tenure) are significantly more likely to churn than long-tenured ones

## Models Benchmarked

10 classifiers were evaluated using 10-fold cross-validation, scored on ROC-AUC:

| Rank | Algorithm | ROC AUC Mean | Accuracy Mean |
|------|-----------|-------------:|---------------:|
| 1 | **Voting Classifier** (soft, GBC+LR+AdaBoost) | 84.93% | 80.23% |
| 2 | Gradient Boosting Classifier | 84.72% | 79.72% |
| 3 | **AdaBoost Classifier** | 84.55% | 80.09% |
| 4 | Logistic Regression | 84.39% | 74.38% |
| 5 | SVC (linear) | 82.99% | 79.11% |
| 6 | Random Forest | 82.75% | 78.67% |
| 7 | Gaussian Naive Bayes | 82.32% | 75.38% |
| 8 | SVC (RBF kernel) | 79.65% | 79.26% |
| 9 | K-Nearest Neighbors | 77.14% | 75.90% |
| 10 | Decision Tree | 66.67% | 73.73% |

## Hyperparameter Tuning

The top boosting models were tuned further using `GridSearchCV` and `RandomizedSearchCV` (10-fold CV, ROC-AUC scoring):

- **AdaBoost** — grid search over `n_estimators` and `learning_rate` → best ROC-AUC: **84.71%**
- **Gradient Boosting** — grid/randomized search over `loss`, `n_estimators`, `max_depth` → best ROC-AUC: **84.82%**

## Feature Importance

Across both Gradient Boosting and AdaBoost, the strongest churn predictors were:
- **Contract type** (month-to-month vs. longer terms)
- **Tenure**
- **Monthly Charges** / **Total Charges**
- **Online Security** and **Tech Support** subscription status

## Evaluation Metrics

Each model was assessed on accuracy, precision, recall, F1-score, and F2-score (weighting recall higher, since catching potential churners matters more than false alarms), along with confusion matrices and ROC curves for visual comparison.

## Tech Stack

- **Language:** Python
- **Data handling:** pandas, NumPy
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Modeling:** scikit-learn, XGBoost, CatBoost
- **Statistics:** statsmodels (VIF analysis)

## Key Takeaways

- Ensemble/boosting methods (AdaBoost, Gradient Boosting, and a soft-voting ensemble of the two plus Logistic Regression) consistently outperformed simpler models like Decision Trees and KNN on this dataset.
- Contract type and tenure are the dominant signals for churn — actionable insights a business could use to target retention offers at month-to-month, low-tenure customers.
- The soft Voting Classifier combining Gradient Boosting, Logistic Regression, and AdaBoost achieved the best overall ROC-AUC (84.93%), showing that model averaging captured complementary strengths of its base learners.
