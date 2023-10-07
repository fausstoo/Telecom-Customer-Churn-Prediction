import os
import sys
from dataclasses import dataclass
sys.path.append('/data/')

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.functions.modeling import *
from src.functions.null_imputation import NullImputer
from src.functions.feature_engineer import *
from src.utils import save_object

from src.exception import CustomException
from src.logger import logging



@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = "./artifacts/preprocessor.pkl"
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
        
    def get_data_transformer_object(self):
        """
        This functions is responsible of the data transformation"""
        
        try:
            df_c = pd.read_pickle("/data/raw/customer_churn_dataset.pkl")
            columns = [column for column in df_c.columns if column != 'Churn']
                            
            whole_pipeline = Pipeline(
                steps=[
                    ("Null Imputation", NullImputer()),
                    ("Age Binning", AgeBinning()),
                    ("Binary Features", BinaryFeatures()),
                    ("Interaction Features", InteractionFeatures()),
                    ("Feature Scaler", FeatureScaler()),
                    ("Drop Unnecessary Columns", DropColumns())
                ]
            )
            
            logging.info("Pipeline completed")
            
            preprocessor = ColumnTransformer(
                [
                    ("whole_pipeline", whole_pipeline, columns)
                ]
            )

            return preprocessor
        
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_pickle(train_path)
            test_df = pd.read_pickle(test_path)
            
            logging.info("Train and test data reading completed")
            
            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()
            
            target_column = "Churn"
            
            input_feature_train_df = train_df.drop(columns=[target_column], axis=1)
            target_feature_train_df = train_df[target_column]
            
            input_feature_test_df = test_df.drop(columns=[target_column], axis=1)
            target_feature_test_df = test_df[target_column]
            
            logging.info("Applying preprocessing object on training and test data frame...")
            
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.fit_transform(input_feature_test_df)
            
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]
            
            logging.info("Saving preprocessing object...")
            
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            
        except Exception as e:
            raise CustomException(e,sys)