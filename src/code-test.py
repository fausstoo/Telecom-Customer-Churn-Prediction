import pickle


import pandas as pd
df_c = pd.read_pickle("../data/processed/modeling_df.pkl")
import numpy as np

from src.functions.feature_engineer import FeatureEngineer
from src.functions.null_imputation import NullImputer
from src.utils import save_object
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

val_data = pd.read_pickle("../data/raw/customer_churn_dataset-test.pkl")

# Load the preprocessor
with open('../artifacts/flask/flask_preprocessor.pkl', 'rb') as preprocessor_file:
    preprocessor = pickle.load(preprocessor_file)


# Load the preprocessor
with open('../artifacts/best_model.pkl', 'rb') as preprocessor_file:
    model = pickle.load(preprocessor_file)
    
from src.exception import CustomException
from src.logger import logging


processed_df = pd.read_pickle("../data/processed/modeling_df.pkl")

# Make predictions on the preprocessed validation data
data_prep = preprocessor.transform(val_data)
predictions = model.predict(data_prep)