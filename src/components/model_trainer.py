import os 
import sys
from dataclasses import dataclass

# ML algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Model evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from src.functions.modeling import select_best_model

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = "./artifacts/best_model.pkl"
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting trainin and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            
            models = {
                "Decission Tree Classifier": DecisionTreeClassifier(),
                "Random Forest Classifier": RandomForestClassifier(),
                "XGBClassifier": XGBClassifier()
            }
            
            param_grid = {
                "Decission Tree Classifier": {
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                },
                "Random Forest Classifier": {
                    "n_estimators": [50, 100, 150],
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                },
                "XGBClassifier": {
                    "n_estimators": [50, 100, 150],
                    "max_depth": [3, 4, 5],
                    "learning_rate": [0.01, 0.1, 0.2],
                }
            }
            
            logging.info("Best model selection started...")            
            best_model = select_best_model(X_train=X_train,
                                           y_train=y_train,
                                           X_test=X_test,
                                           y_test=y_test,
                                           param_grid=param_grid,
                                           models=models)
            logging.info("Best model selection finished")
            
            logging.info("Saving the best model...")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info(f"Model saved at {self.model_trainer_config.trained_model_file_path}")
        
        except Exception as e:
            raise CustomException(e, sys)