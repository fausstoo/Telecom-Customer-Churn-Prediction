import sys
sys.path.append('../src/functions')

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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, auc, roc_auc_score, roc_curve, make_scorer, precision_recall_curve
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
import joblib




#---------------------------------------------------------------------------
#                   TRAIN AND SAVE ALGORITHMS FUNCTION                     |
#---------------------------------------------------------------------------
def train_and_evaluate_best_model(X_train, y_train, X_test, y_test, param_grid, models):
    model_list = []
    accuracy_list = []
    f1_list = []
    recall_list = []
    roc_auc_list = []

    best_model = None
    best_mean_accuracy = 0.0

    for model_name, model in models.items():
        # Hyperparameter tuning using RandomizedSearchCV
        if model_name in param_grid:
            param_dist = param_grid[model_name]
            random_search = RandomizedSearchCV(
                model, param_distributions=param_dist, n_iter=10, cv=5, n_jobs=-1, random_state=42, scoring=make_scorer(accuracy_score)
            )
            random_search.fit(X_train, y_train)
            best_estimator = random_search.best_estimator_
        else:
            best_estimator = model

        # Cross-validation
        scores = cross_val_score(best_estimator, X_train, y_train, cv=5, scoring="accuracy")
        mean_accuracy = scores.mean()

        # Check if this model has the best mean accuracy so far
        if mean_accuracy > best_mean_accuracy:
            best_mean_accuracy = mean_accuracy
            best_model = best_estimator

        # Train the best model
        best_estimator.fit(X_train, y_train)

    # Now you have the best model based on mean accuracy
    if best_model is not None:
        # Make predictions
        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)

        # Evaluate Train and Test sets
        # Train set
        accuracy_train = accuracy_score(y_train, y_train_pred)
        precision_score_train = precision_score(y_train, y_train_pred)
        recall_train = recall_score(y_train, y_train_pred)
        roc_auc_train = roc_auc_score(y_train, y_train_pred)

        # Test set
        accuracy_test = accuracy_score(y_test, y_test_pred)
        precision_score_test = precision_score(y_test, y_test_pred)
        recall_test = recall_score(y_test, y_test_pred)
        roc_auc_test = roc_auc_score(y_test, y_test_pred)

        # Best model name
        best_model_name = best_model.__class__.__name__

        # Save the best model to a file
        best_model_filename = f"../Models/{best_model_name}.pkl"
        joblib.dump(best_model, best_model_filename)

        # Print results for the best model
        print("Best Model: {}".format(best_model_name))
        print("Mean Cross-Validation Accuracy: {:.4f}".format(best_mean_accuracy))

        print("Model performance for Training set")
        print("- Accuracy Score: {:.4f}".format(accuracy_train))
        print("- Precision Score: {:.4f}".format(precision_score_train))
        print("- Recall Score: {:.4f}".format(recall_train))
        print("- ROC & AUC Score: {:.4f}".format(roc_auc_train))

        print("-" * 35)

        print("Model performance for Test set")
        print("- Accuracy Score: {:.4f}".format(accuracy_test))
        print("- Precision Score: {:.4f}".format(precision_score_test))
        print("- Recall Score: {:.4f}".format(recall_test))
        print("- ROC & AUC Score: {:.4f}".format(roc_auc_test))
        
        
        

#---------------------------------------------------------------------------
#                           MODEL EVALUATION                               |
#---------------------------------------------------------------------------
def evaluate_model(true, predicted):
    accuracy = accuracy_score(true, predicted)
    precision_s = precision_score(true, predicted)
    recall = recall_score(true, predicted)
    roc_auc = roc_auc_score(true, predicted)
    
    return accuracy, precision_s, recall, roc_auc