# Panic Disorder Prediction using Machine Learning

This project applies machine learning techniques to predict the likelihood of panic disorder in individuals using a dataset containing demographic, behavioral, and clinical features. The workflow includes data cleaning, feature engineering, statistical testing, and classification using multiple ML algorithms. The goal is to identify key predictors and build interpretable, high-performing models for early risk detection.

---

## 📌 Objective

To develop and evaluate classification models that can predict panic disorder diagnosis using structured survey data. The analysis also explores feature relevance and interprets model behavior through visualizations and performance metrics.

---

## Dataset Overview

- **Source**: Dataset obtained from Kaggle - Link - https://www.kaggle.com/datasets/muhammadshahidazeem/panic-disorder-detection-dataset
- **Target Variable**: Panic Disorder Diagnosis (binary)
- **Features include**:
  - Demographics (age, gender, location)
  - Psychosocial data (stressors, coping mechanisms, social support)
  - Medical & psychiatric history
  - Lifestyle and behavioral factors

---

## Data Preprocessing

- Handled missing values and Dropped duplicates
- Performed Chi-Square test for assessing significance of categorical features
- Encoded categorical variables using Label Encoding and One-Hot Encoding
- Scaled features with `StandardScaler`

---

## Exploratory Data Analysis

- Class distribution visualization
- Correlation heatmap for numeric features
- Boxplots to observe distribution across target variable

---

## ⚙️ Machine Learning Models

| Model                | Highlights                      |
|---------------------|----------------------------------|
| Logistic Regression | Baseline model, interpretable   |
| Random Forest       | Ensemble learning, robust       |
| Decision Tree       | Interpretable, fast             |
| XGBoost             | Gradient boosting, high accuracy|

All models were evaluated using:
- Accuracy
- Classification Report (Precision, Recall, F1-Score)
- Confusion Matrix
- ROC-AUC Curve

---

## Key Results

- **Best performance** observed with XGBoost and Random Forest (AUC > 0.90)
- Logistic regression coefficients helped interpret predictor impact
- Class imbalance was handled using SMOTE
- Extracted feature importance from logistic regression and tree-based models

---

## Tools & Libraries

- Python, Pandas, NumPy, Scikit-learn
- Matplotlib, Seaborn, XGBoost
- SMOTE (imbalanced-learn)
- Statistical tests (Chi-Square)

---

## Limitations

- The dataset is synthetic and may not fully capture real-world complexity
- Clinical diagnosis often depends on temporal/contextual factors not included here. 

