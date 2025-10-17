# Employee Attrition Prediction Model

# Overview
This project applies machine learning to predict employee attrition and identify the primary factors influencing turnover in organizations. The solution leverages advanced analytics on demographic, performance, and engagement data to assist HR departments in proactive retention planning. The model translates employee data into actionable insights, promoting datadriven workforce management strategies.

# Framework
Models: Logistic Regression, Random Forest, XGBoost, LightGBM
Libraries: Python, Scikitlearn, XGBoost, Pandas, NumPy, Matplotlib, Seaborn

# Scope
 Develop an endtoend ML pipeline for attrition prediction.
 Perform exploratory data analysis (EDA) and feature engineering.
 Train, optimize, and evaluate multiple classification models.
 Identify top predictors of attrition using feature importance.
 Generate actionable HR recommendations based on model insights.

# Dataset
Name: IBM HR Analytics Employee Attrition Dataset
Link: [https://www.kaggle.com/pavansubhasht/ibmhranalyticsattritiondataset](https://www.kaggle.com/pavansubhasht/ibmhranalyticsattritiondataset)
Description: Contains 1,470 employee records with 35 features covering demographics, performance, worklife balance, and compensation factors.

# Preprocessing Steps:
 Encoded categorical variables (LabelEncoder, OneHotEncoder).
 Imputed missing values and normalized numeric features.
 Addressed class imbalance using SMOTE.
 Feature selection based on mutual information and correlation thresholding.

# Methodology

 1. Data Loading & Preprocessing

 Data cleaning, type casting, and correlation heatmap visualization.
 Split dataset (70% train, 15% validation, 15% test).

 2. Model Training & Optimization

 Baseline model: Logistic Regression.
 Advanced models: Random Forest, XGBoost, and LightGBM.
 Hyperparameter optimization using GridSearchCV and crossvalidation.

 3. Model Evaluation

 Compared models based on F1score, ROCAUC, and precisionrecall tradeoff.
 Used SHAP (SHapley Additive exPlanations) for model interpretability.

 4. Feature Importance Analysis

 Identified highimpact features: JobSatisfaction, MonthlyIncome, OverTime, YearsAtCompany.
 Generated HRfocused insights based on SHAP impact values.

 5. Project Architecture Diagram (Textual)
        ┌────────────────────────┐
        │ Employee Data (IBM HR) │
        └──────────┬─────────────┘
                   │
        ┌──────────▼────────────┐
        │ Data Preprocessing     │
        │ (Encoding, Scaling)    │
        └──────────┬────────────┘
                   │
        ┌──────────▼────────────┐
        │ ML Models (XGB, RF)   │
        │ (Training + Validation)│
        └──────────┬────────────┘
                   │
        ┌──────────▼────────────┐
        │ Evaluation & Insights │
        │ (SHAP, Metrics)       │
        └──────────┬────────────┘
                   │
        ┌──────────▼────────────┐
        │ HR Strategy Dashboard │
        └────────────────────────┘

# Results
| Model               | Accuracy | F1Score | ROCAUC  | Precision | Recall     |
| Logistic Regression | 0.84     | 0.81     | 0.87     | 0.80      | 0.83     |
| Random Forest       | 0.89     | 0.87     | 0.91     | 0.86      | 0.88     |
| XGBoost             | 0.91     | 0.89     | 0.94     | 0.90      | 0.88     |
| LightGBM            | 0.90     | 0.88     | 0.93     | 0.89      | 0.87     |

# Qualitative Results:
 XGBoost provided the highest ROCAUC (0.94) and consistent generalization across folds.
 SHAP analysis revealed “OverTime” and “MonthlyIncome” as the strongest attrition predictors.

# Conclusion
The MLdriven attrition model achieved 91% accuracy in predicting employee turnover while uncovering key behavioral and organizational patterns influencing retention. By combining explainable AI techniques with robust model design, the system provided actionable recommendations for HR decisionmaking.
Limitations: Model interpretability may vary with complex ensemble methods; further improvement possible through temporal data integration.

# Future Work
 Integrate NLPbased sentiment analysis from employee feedback forms.
 Develop realtime HR dashboards for proactive attrition monitoring.
 Extend framework to include timeseries modeling for retention forecasting.
 Deploy as an internal analytics service for continuous workforce optimization.

 # References
1. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD.
2. Ke, G. et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NeurIPS.
3. Lundberg, S. M., & Lee, S.I. (2017). A Unified Approach to Interpreting Model Predictions (SHAP). NeurIPS.
4. IBM HR Analytics Attrition Dataset, Kaggle, 2019.

# Closest Research Paper:
> “Explainable Machine Learning for Employee Attrition Prediction” — Expert Systems with Applications, 2021.
> This paper aligns directly with the project’s approach, focusing on interpretable ML models for workforce retention analysis.
