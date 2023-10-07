import sys

import pandas as pd

from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self, features):
        try:
            model_path = "./artifacts/Models/RandomForestClassifier.pkl"
            preprocessor_path = "./artifacts/preprocessor.pkl"
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            
            return preds

        except Exception as e:
            raise CustomException(e, sys)
        
class CustomData:
    def __init__(self,
       CustomerID: float,
       Age: float,
       Gender: object,
       Tenure: float,
       Usage_Frequency: float,
       Support_Calls: float,
       Payment_Delay: float,
       Subscription_Type: object,
       Contract_Length: object,
       Total_Spend: float,
       Last_Interaction: float
    ):
        self.CustomerID = CustomerID
        self.Age = Age
        self.Gender = Gender
        self.Tenure = Tenure
        self.Usage_Frequency = Usage_Frequency       
        self.Support_Calls = Support_Calls
        self.Payment_Delay = Payment_Delay
        self.Subscription_Type = Subscription_Type       
        self.Contract_Length = Contract_Length
        self.Total_Spend = Total_Spend
        self.Last_Interaction = Last_Interaction
        
        
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                'CustomerID': [self.CustomerID],       
                'Age': [self.Age],
                'Gender': [self.Gender],
                'Tenure': [self.Tenure],
                'Usage_Frequency': [self.Usage_Frequency],
                'Support_Calls': [self.Support_Calls],
                'Payment_Delay': [self.Payment_Delay],
                'Subscription_Type': [self.Subscription_Type],
                'Contract_Length': [self.Contract_Length],
                'Total_Spend': [self.Total_Spend],
                'Last_Interaction': [self.Last_Interaction]
            }
            
            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e, sys)