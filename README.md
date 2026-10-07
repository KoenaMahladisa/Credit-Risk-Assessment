# Credit-Risk-Assessment
Credit Risk Analysis
1. Project Overview
   
1.1 Introduction

Creditworthiness assessment is an important data-driven process used to evaluate the likelihood that an applicant will meet their financial obligations. This project analyzes the CreditWorthiness dataset and uses the accompanying Python notebook to explore applicant characteristics and develop a machine-learning approach for supporting creditworthiness assessment.  

The dataset contains 1,000 records and 21 variables. The project performs data preparation, exploratory analysis, and predictive modeling to identify patterns in the available credit information. The notebook applies a machine-learning workflow to assess and predict creditworthiness from the available applicant information.   


1.2 Problem Statement
Credit decisions often involve large volumes of applicant information with different numerical and categorical characteristics. The challenge addressed in this project is to transform the available credit data into meaningful insights and to develop a predictive approach that can support the assessment of creditworthiness.   

The primary problem addressed by the data is the quantification of financial risk during the loan approval process. Specifically, the institution needs to determine whether a potential borrower is "creditworthy" (Good) or "risky" (Bad) based on a variety of personal and financial attributes. The manual assessment of these risks is often inconsistent or slow. Therefore, the challenge is to use historical data—including factors like checking account balances, loan duration, credit history, purpose of the loan, and demographic info (age, job type)—to build a model that can accurately predict the credit score of new applicants.   

1.3 Purpose of the Project
The purpose of this project is to:

- Analyze and understand the structure and characteristics of the creditworthiness dataset.   

- Identify data quality issues and prepare the data for analysis and modeling.   

- Explore relationships and patterns among variables that may be associated with creditworthiness.   

- Apply the preprocessing and feature preparation steps implemented in the notebook.   

- Develop and evaluate machine-learning model(s) for predicting or assessing creditworthiness.   

- Use the findings to provide practical insights and recommendations for improving data-driven credit assessment.  

2. Technical Stack & Architecture
   
2.1 Model Training (train_model.py)

- Dataset Loading & Preprocessing: Reads the Excel dataset, capitalizes column names, and applies one-hot encoding using pandas (pd.get_dummies with drop_first=True).   

- Feature Preservation: Saves the exact feature columns (model_columns.joblib) to ensure consistent alignment of user inputs during inference.   

- Scaling & Splitting: Splits the data into training and testing sets (80-20 split) and scales features using StandardScaler.   

- Machine Learning Model: Trains a RandomForestClassifier and exports the trained model (credit_model.joblib) and scaler (scaler.joblib) using joblib.   

2.2 Web Application (app.py)
Built with Streamlit as an interactive dashboard (AI Creditworthiness Assessment System).   

Collects comprehensive applicant details across a multi-column form (checking/savings balances, loan duration, credit history, purpose, age, employment duration, housing type, etc.).   

Aligns user inputs with model schema, handles missing dummy columns gracefully, scales features, and generates predictions alongside model confidence and risk metrics (Low/Medium/High).   
