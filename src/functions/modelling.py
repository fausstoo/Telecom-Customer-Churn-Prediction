# Analysis libraries
import numpy as np
import pandas as pd
import pickle

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Data Splitting and Modeling
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
import joblib




#---------------------------------------------------------------------------
#                   TRAIN AND SAVE ALGORITHMS FUNCTION                     |
#---------------------------------------------------------------------------
def train_algorithms(models, X_train, y_train, X_test, y_test):
    for model_name, model in models.items():
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Save model
        joblib.dump(model, f"../models/{model_name}_joblib")
        
        # Training set performance
        model_train_accuracy_score = accuracy_score(y_train, y_train_pred)
        model_train_f1_score = f1_score(y_train, y_train_pred, average='weighted')
        model_train_recall_score = recall_score(y_train, y_train_pred)
        model_train_roc_auc_score = roc_auc_score(y_train, y_train_pred)
        
        # Test set performance
        model_test_accuracy_score = accuracy_score(y_test, y_test_pred)
        model_test_f1_score = f1_score(y_test, y_test_pred, average='weighted')
        model_test_recall_score = recall_score(y_test, y_test_pred)
        model_test_roc_auc_score = roc_auc_score(y_test, y_test_pred)
        
        # Print results
        print(model_name)
        
        print("Model performance for Training set")
        print("- Accuracy Score: {:.4f}".format(model_train_accuracy_score))
        print("- F1 Score: {:.4f}".format(model_train_f1_score))
        print("- Recall Score: {:.4f}".format(model_train_recall_score))
        print("- ROC & AUC Score: {:.4f}".format(model_train_roc_auc_score))
        
        print("-" * 35)
        
        print("Model performance for Test set")
        print("- Accuracy Score: {:.4f}".format(model_test_accuracy_score))
        print("- F1 Score: {:.4f}".format(model_test_f1_score))
        print("- Recall Score: {:.4f}".format(model_test_recall_score))
        print("- ROC & AUC Score: {:.4f}".format(model_test_roc_auc_score))

        print("=" * 35)