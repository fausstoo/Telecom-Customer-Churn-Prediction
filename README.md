#### **Telecom Customer Churn Prediction**
#### Author: **Fausto Pucheta Fortin**

**Project Overview**
This project aims to predict customer churn in a telecom company, providing insights and solutions to improve customer retention. The dataset used can be found at: https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset. 

Key highlights include: \
- Data exploration to uncover patterns and features influencing customer churn.
- Feature engineering to create predictive indicators.
- Machine learning model development and evaluation.
- Feature importance analysis.

**Model characteristics** 
- The model for this project is an XGBoostClassifier, and it was chosen by iterating and experimenting with *RandomizedSearchCV*, *Cross-validation*, and *hyperparameter tuning*.
- It delivers high precision and a strong ROC AUC score. (**0.928** for Precision-Recall Curve and **0.920** for ROC Curve)
- The top features influencing churn are 'high_support_calls', 'low_spender', and 'high_payment_delay'.

**Usage**
This project includes a Flask application, app.py, which allows users to interact with the machine learning model for customer churn prediction. Here's a step-by-step guide on how to pull and use the Flask application: 
1. Clone the Repository: \
git clone https://github.com/fausstoo/Telecom-Customer-Churn-Prediction.git 
2. Install Dependencies: \
pip install -r requirements.txt
3. Start the Application: \
python app.py
4. Access the Application: \
http://127.0.0.1:5000/predictdata
5. Input Customer Data: \
On the home page, you'll find a form where you can input customer data. Fill out the form with relevant customer information, such as age, support calls, payment delays, and spending.
6. Get Predictions: \
After submitting the data, the application will utilize the trained machine learning model to predict customer churn. The prediction result will be displayed on the screen.
