import os
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
from sklearn.metrics import (
                            accuracy_score,
                            precision_score,
                            recall_score,
                            roc_auc_score,
                            make_scorer
                            )
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
import joblib

from src.exception import CustomException
from src.logger import logging

#---------------------------------------------------------------------------
#                 TRAINING, EVALUATION AND SAVE BEST MODEL                 |
#---------------------------------------------------------------------------
def select_best_model(X_train, y_train, X_test, y_test, param_grid, models):
    model_list = []
    accuracy_list = []
    f1_list = []
    recall_list = []
    roc_auc_list = []

    best_model = None
    best_mean_accuracy = 0.0
    try:
        
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

            logging.info("Cross-validation started...")
            # Cross-validation
            scores = cross_val_score(best_estimator, X_train, y_train, cv=5, scoring="precision")
            mean_accuracy = scores.mean()
            
            
            logging.info("Selecting the best model based on precision score")
            # Check if this model has the best mean accuracy so far
            if mean_accuracy > best_mean_accuracy:
                best_mean_accuracy = mean_accuracy
                best_model = best_estimator
            logging.info("Best model finded")

            logging.info("Training the best model...")
            # Train the best model
            best_estimator.fit(X_train, y_train)
            logging.info("Model training finished")

        # Now you have the best model based on precision
        if best_model is not None:
            
            logging.info("Data prediction started...")
            # Make predictions
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)
            logging.info("Data prediction finished")

            
            logging.info("Model evaluation started...")
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
            
            logging.info("Model evaluation finished")
            
            # Best model name
            best_model_name = best_model.__class__.__name__
            
            #logging.info("Saving the best model...")
            ## Save the best model to a file
            #best_model_filename = "/artifacts/models/best_model.pkl"
            #os.makedirs(os.path.dirname(best_model_filename), exist_ok=True)  # Create the directory if it doesn't exist
            #joblib.dump(best_model, best_model_filename)  # Save the model to the specified file
            #logging.info("Saving process completed")
            
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
    
    except Exception as e:
        raise CustomException(e,sys)    
        
    print(best_mean_accuracy)
    return best_model
        
    
    