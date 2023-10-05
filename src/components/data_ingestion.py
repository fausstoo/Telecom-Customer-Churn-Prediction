import os
import sys

sys.path.append('../../data')

from src.exception import CustomException
from src.logger import logging

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass



@dataclass
class DataIngestionConfig:
    raw_data_path: str = "./data/raw/customer_churn_dataset.pkl"
    train_data_path: str = "./data/raw/customer_churn_dataset-train.pkl"
    test_data_path: str = "./data/raw/customer_churn_dataset-test.pkl"
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info("Data ingestion process initiated")
        
        try:
            df = pd.read_pickle("/data/raw/customer_churn_dataset.pkl")
            logging.info("Dataset readed as Data Frame")
            
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            
            df.to_pickle(self.ingestion_config.raw_data_path)
    
            logging.info("Data splitting initiated")
            train_set, test_set =  train_test_split(df, test_size=0.3, random_state=42)
          
            train_set.to_pickle(self.ingestion_config.train_data_path)
            test_set.to_pickle(self.ingestion_config.test_data_path)
            
            logging.info("Data ingestion completed")
            
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
            
        except Exception as e:
            raise CustomException(e, sys)
        
        
if __name__=="__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)
    
    model_trainer = ModelTrainer()
    
    print(model_trainer.initiate_model_trainer(train_arr,test_arr))