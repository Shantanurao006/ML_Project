Scalable Analysis and Predictive Modelling on Large-Scale E-commerce Data
Submitted By
Name: Shantanu Singh Rao
Roll Number: M24DE2026
Abstract

This project explores scalable analysis and predictive modelling using a large-scale e-commerce transaction dataset. The dataset contains over 19 million records, reflecting real-world user interactions such as product views, cart additions, and purchases.

To handle the scale and complexity of the data, both traditional (Pandas) and Big Data (PySpark) processing techniques were employed. Multiple machine learning models were developed and evaluated to predict whether a user is likely to make a purchase.

A key highlight of this work is the identification and removal of data leakage, which significantly affected model performance. Finally, a Django-based web dashboard was built to present insights, compare models, and allow interactive predictions, making the solution practical and deployment-ready.

1. Introduction

In today’s digital economy, e-commerce platforms generate enormous volumes of data from user interactions. Every click, product view, or purchase contributes to a growing dataset that can be leveraged for business insights and predictive analytics.

However, analysing such large datasets using conventional tools can be inefficient and time-consuming. This is where Big Data technologies, such as PySpark, become essential.

This project brings together machine learning and Big Data techniques to:

Process large-scale transactional data efficiently
Understand customer behaviour
Predict purchase likelihood
Present insights through an interactive dashboard

The overall aim is to simulate a real-world data pipeline, from raw data processing to deployment.

2. Objectives

The primary objectives of this project are:

To perform scalable analysis on large-scale e-commerce data
To compare traditional data processing (Pandas) with distributed processing (PySpark)
To develop and evaluate multiple machine learning models
To identify and eliminate data leakage for reliable predictions
To design a user-friendly dashboard for visualisation and prediction
3. Dataset Description

Dataset Source: Kaggle — E-commerce Behaviour Data from Multi Category Store

Total Rows: 19,610,775

The dataset represents user activity on an e-commerce platform and includes the following key features:

event_time – Timestamp of user interaction
event_type – Type of action (view, cart, purchase)
brand – Product brand
category_code – Product category
price – Price of the product
user_id – Unique identifier of the user

A new binary target variable, purchase, was derived from event_type, where:

1 indicates a purchase
0 indicates no purchase

Screenshot:
Insert: screenshots/dashboard.png

4. Methodology

The project follows a structured and systematic workflow:

Data Collection → Data Cleaning → Feature Engineering → Model Training → Evaluation → Deployment

Steps involved:
Data Loading
The dataset was loaded using Pandas for initial analysis and PySpark for large-scale processing.
Data Cleaning
Missing values in categorical columns were handled appropriately.
Feature Engineering
New time-based features were extracted from the timestamp.
Model Development
Multiple machine learning models were trained and evaluated.
Performance Comparison
Models were compared using standard evaluation metrics.
Deployment
The final model was integrated into a Django dashboard for real-time interaction.
5. Data Preprocessing and Feature Engineering
5.1 Data Cleaning
Missing values in brand and category_code were addressed
A manageable subset (500,000 rows) was used for model training
5.2 Target Variable Creation

A new column purchase was created from event_type:

Purchase → 1
Non-purchase → 0
5.3 Feature Extraction

From the event_time column, the following features were derived:

Month
Day
Hour
Weekday

These features help capture temporal patterns in user behaviour.

5.4 Encoding

Categorical features such as brand and category_code were converted into numerical values for model compatibility.

6. Exploratory Data Analysis (EDA)

EDA was conducted to gain insights into the dataset.

Key Observations:
Brands such as Samsung and Apple appeared most frequently
Electronics, particularly smartphones, dominated the dataset
The dataset was highly imbalanced, with very few purchase events compared to views

This imbalance played a crucial role in model performance and evaluation.

Screenshot:
Insert: screenshots/charts.png

7. Machine Learning Approaches

Four different approaches were implemented:

Approach 1 — Logistic Regression
Accuracy: 96.69%
Recall: 0%

Although the accuracy was high, the model failed to identify any purchase events due to class imbalance.

Approach 2 — Random Forest (With Data Leakage)
Accuracy: 100%

This result was misleading because the model used event_type, which directly indicates the target variable. This is a classic example of data leakage.

Approach 3 — Random Forest (Without Data Leakage)
Accuracy: 74.72%
Recall: 52.87%

After removing leakage, the model produced more realistic and reliable results.
This approach was selected as the final model.

Approach 4 — XGBoost
Accuracy: 69.01%
Recall: 59.12%

XGBoost achieved a higher recall, meaning it captured more actual purchases, but its overall balance was slightly lower than Random Forest.

8. Model Comparison
Model	Accuracy	Recall	F1 Score
Logistic Regression	0.9669	0.0000	0.0000
Random Forest (Leakage)	1.0000	1.0000	1.0000
Random Forest (No Leakage)	0.7472	0.5287	0.0755
XGBoost	0.6901	0.5912	0.0693

Screenshot:
Insert: screenshots/dashboard.png

9. Pandas vs PySpark Comparison
Framework	Rows Processed	Time
Pandas	500,000	1.46 sec
PySpark	19,610,775	90.66 sec
Insight:
Pandas is efficient for smaller datasets
PySpark is better suited for large-scale distributed processing

Screenshot:
Insert: screenshots/pyspark_table.png

10. Django Dashboard

A Django-based dashboard was developed to present results in an interactive and user-friendly manner.

Features:
Display of best-performing model
Comparison of all machine learning approaches
Visual performance charts
Pandas vs PySpark comparison
Navigation to prediction page

Screenshot:
Insert: screenshots/dashboard.png

11. Prediction System

The system allows users to input:

Brand
Category
Price
Hour

Based on these inputs, the trained model predicts whether a purchase is likely.

Example Output:

Purchase Not Likely

Screenshots:
Insert:

screenshots/predict_page.png
screenshots/prediction_result.png
12. Technologies Used
Technology	Purpose
Python	Core programming
Pandas	Data analysis
NumPy	Numerical operations
Matplotlib	Visualisation
Scikit-learn	Machine learning
XGBoost	Advanced modelling
PySpark	Big Data processing
Django	Web application
Joblib	Model persistence
13. Conclusion

This project demonstrates a complete pipeline for handling large-scale data, from preprocessing to deployment. It highlights the importance of careful feature selection, particularly in avoiding data leakage, which can otherwise lead to misleading results.

The comparison between Pandas and PySpark further reinforces the need for scalable tools in real-world scenarios.

Among all approaches, Random Forest without data leakage was selected as the final model due to its balanced and realistic performance.

14. Future Work

The project can be extended in several ways:

Integration with real-time streaming data
Deployment on cloud platforms such as AWS or Azure
Use of deep learning techniques
Improved handling of class imbalance
Development of recommendation systems
