#### **Telecom Customer Churn Prediction**
#### Author: **Fausto Pucheta Fortin**


#### <u>**Table of Contents**</u>
---
-[Business Context](#Business-Context)
- [Project Overview](#Project-Overview)
- [Project Structure](#Project-Structure)
- [Key Highlights](#Key-Highlights)
- [Model Characteristics](#Model-Characteristics)
- [Run Locally](#Run-Locally)


#### **Business Context**
---
Customer retention is critical for sustained business growth. Understanding and predicting customer churn can significantly impact strategic decision-making and resource allocation. This project aims to enhance our ability to foresee and mitigate customer churn by leveraging advanced analytics and machine learning.


#### **Project Overview**
---
- This project addresses Telecom Customer Churn Prediction. The analysis encompasses extensive exploratory data analysis (EDA), including univariate and multivariate analyses.
- Key findings from EDA inform subsequent feature engineering in the Feature_Engineer notebook. The process involves binning, creating new binary features, interaction features, and scaling.
-  Following this, the Modeling_&_Evaluation notebook applies Decision Tree, Random Forest, and XGBoost algorithms. The XGBoost model emerges as the best performer, achieving high precision and ROC AUC.
-  The top three influential features identified are 'high_support_calls,' 'low_spender,' and 'high_payment_delay.' 
- Finally, the creation of a Flask application is included to make live predictions using the model trained previously.

The Flask application app.py allows users to interact with the trained model for live customer churn prediction.
![Untitled_Project_V1](https://github.com/fausstoo/Telecom-Customer-Churn-Prediction/assets/59534169/9d0ba389-91ae-4612-965e-576227772f69)


#### **Project Structure** 
---
fausstoo/telecom-customer-churn-prediction\
│\
├── artifacts\
│   ├── flask\
│   │   ├── flask_preprocessor.pkl\
│   │\
│   ├── Models\
│   │   ├── .gitattributes\
│   │   ├── RandomForestClassifier.pkl\
│   │\
│   ├── preprocessor.pkl\
│   ├── xgboost_classifier.pkl\
│\
├── data\
│   ├── external\
│   │   ├── .gitkeep\
│   │   ├── customer_churn_dataset.pkl\
│   │\
│   ├── processed\
│   │   ├── test_df\
│   │   │   ├── X_test.pkl\
│   │   │   ├── y_test.pkl\
│   │   │\
│   │   ├── train_df\
│   │   │   ├── X_train.pkl\
│   │   │   ├── y_train.pkl\
│   │   │\
│   │   ├── validation_df\
│   │   │   ├── X_val.pkl\
│   │   │   ├── y_val.pkl\
│   │   │\
│   │   ├── .gitkeep\
│   │   ├── features_df.pkl\
│   │   ├── imputed_df.pkl\
│   │   ├── modeling_df.pkl\
│   ├── raw\
│      ├── .gitkeep\
│      ├── archive.zip\
│      ├── customer_churn_dataset-test.pkl\
│      ├── customer_churn_dataset-train.pkl\
│      ├── customer_churn_dataset.pkl\
│\
├── logs\
│\
├── notebooks\
│   ├── .gitkeep\
│   ├── code_test.py\
│   ├── EDA_01.ipynb\
│   ├── EDA_02.ipynb\
│   ├── Feature_Engineer.ipynb\
│   ├── Modeling_&_Evaluation.ipynb\
│\
├── reports\
│   ├── figures\
│   ├── EDA_01\
│   ├── EDA_02\
│   ├── Feature_Engineer\
│   ├── Modeling_&_Evaluation\
│\
├── src\
│   ├── components\
│   │   ├── __init__.py\
│   │   ├── data_ingestion.py\
│   │   ├── data_transformation.py\
│   │   ├── model_trainer.py\
│   │\
│   ├── functions\
│   │   ├── pycache\
│   │   ├── feature_engineer.py\
│   │   ├── modeling.py\
│   │   ├── null_imputation.py\
│   │   ├── outliers_removal.py\
│   │   ├── plot_functions.py\
│   │   ├── tabular_report_functions.py\
│\
│   ├── pipeline\
│   │   ├── pycache\
│   │   ├── __init__.py\
│   │   ├── predict_pipeline.py\
│   │   ├── train_pipeline.py\
│\
│   ├── visualizations\
│   │   ├── .gitkeep\
│   │   ├── __init__.py\
│   │   ├── visualize_EDA_1.py\
│   │   ├── visualize_EDA_2.py\
│   │   ├── visualize_Feature_Engineer.py\
│   ├── __init__.py\
│   ├── utils.py\
│   ├── exception.py\
│   └── login.py\
│\
├── templates\
│   ├── home.html\
│   └── index.html\
├── .gitattributes\
├── .gitignore\
├── app.py\
├── setup.py\
├── requirements.txt\
└── README.md\

#### **Key Highlights**
---
- Exploratory data analysis to uncover patterns and features influencing customer churn. *EDA_01.ipynb & EDA_02.ipynb notebooks*
- Feature engineering and selection to create predictive indicators. *Feature_Engineer.ipynb*
- Model training and selection based on Precision Score using Randomized Search CV, Cross-Validation, and Hyperparameter tuning. *model_trainer.py*
- Feature importance analysis for further iterations. *Modeling_&_Evaluation.ipynb*


#### **Model Characteristics**
---
- The model for this project is an XGBoostClassifier, and it was chosen by iterating and experimenting with *RandomizedSearchCV*, *Cross-validation*, and *Hyperparameter tuning*.
- It delivers high precision and a strong ROC AUC score. (**0.928** for Precision-Recall Curve and **0.920** for ROC Curve)

![roc_auc_curve](https://github.com/fausstoo/Telecom-Customer-Churn-Prediction/assets/59534169/1643e6ec-db6a-433b-ad8a-9420208590bc)

![precision_recall_curve](https://github.com/fausstoo/Telecom-Customer-Churn-Prediction/assets/59534169/b9f7ea54-aa8a-4a3c-a8d6-814ed63e3e0a)
  
- The top features influencing churn are 'high_support_calls', 'low_spender', and 'high_payment_delay'.
![feature_importance](https://github.com/fausstoo/Telecom-Customer-Churn-Prediction/assets/59534169/37ca26b7-c27a-451d-a93f-3afe05a2c7b1)


#### **Run Locally**
---
You can interact with the trained algorithm Flask app by following these steps: 

1. Initialize Git 
   ```bash
   git init
   ```
   
2. Clone the Repository: 
   ```bash
   git clone https://github.com/fausstoo/Telecom-Customer-Churn-Prediction.git
   ```
   
3. Create environment
```bash
conda create -n <env_name> python=<python_version> -y
```

4. Activate environment
```bash
conda activate <env_name>
```

5. Install Dependencies: 
   ```bash
   pip install -r requirements.txt
   ```
   
6. Start the App: 
   ```bash
   python app.py
   ```
   
7. Access the App: 
   ```bash
   http://127.0.0.1:5000/predictdata
   ```
   
8. Input Customer Data: 
On the home page, you'll find a form where you can input customer data. Fill out the form with relevant customer information, such as age, support calls, payment delays, and spending.

9. Get Predictions: 
After submitting the data, the application will utilize the trained machine learning model to predict customer churn. The prediction result will be displayed on the screen.
