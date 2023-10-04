
import sys
sys.path.append('../src/functions')

# Analysis libraries
import numpy as np
import pandas as pd
import pickle

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin


#-----------------------------------------------------------------
#                         AGE BINNING                            |
#-----------------------------------------------------------------

class AgeBinning(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Define the bins and labels for age categories
        bins = [0, 29, 49, float('inf')]
        labels = ['Young Adults', 'Middle-age Adults', 'Seniors']

        # Create a new column 'age_category' with the age categories
        X['age_category'] = pd.cut(X['Age'], bins=bins, labels=labels, right=False)

        return X

#-----------------------------------------------------------------
#                      BINARY FEATURES                           |
#-----------------------------------------------------------------
class BinaryFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # HIGH SUPPORT CALL feature
        X['high_support_calls'] = np.where(X['Support Calls'] > 4, 1, 0)

        # HIGH PAYMENT DELAY feature
        X['high_payment_delay'] = np.where(X['Payment Delay'] > 20, 1, 0)

        # LOW SPENDER feature
        X['low_spender'] = np.where(X['Total Spend'] > 450, 0, 1)

        # GENDER TRANSFORMATION
        X['Gender'] = X['Gender'].apply(lambda x: 1 if x == 'Female' else 0)

        # MONTHLY CONTRACT feature
        X['monthly_contract'] = X['Contract Length'].apply(lambda x: 1 if x == 'Monthly' else 0)

        # LOW INTERACTION CUST feature
        X['low_interaction_cust'] = X['Last Interaction'].apply(lambda x: 1 if x > 14 else 0 )
        
        return X
    

#-----------------------------------------------------------------
#                    INTERACTION FEATURES                        |
#-----------------------------------------------------------------
class InteractionFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # AGE SUPPORT CALLS INTERACTION feature
        X['age_support_calls_interaction'] = X['Age'] * X['Support Calls']
        
        # PAYMENT DELAY TO TOTAL SPENT RATIO feature
        X['payment_delay_to_total_spent_ratio'] = X['Payment Delay'] / X['Total Spend']
        
        return X
    


#-----------------------------------------------------------------
#                        FEATURE SCALER                          |
#-----------------------------------------------------------------
class FeatureScaler(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Create a StandardScaler instance
        std_scaler = StandardScaler()
        # Standardize the 'payment_delay_to_total_spent_ratio' feature
        X['payment_delay_to_total_spent_ratio_standardized'] = std_scaler.fit_transform(X[['payment_delay_to_total_spent_ratio']])
        
        return X



#-----------------------------------------------------------------
#                    DROP UNNECESSARY COLUMNS                    |
#-----------------------------------------------------------------
class DropColumns(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.drop(columns=['CustomerID', 'Age','Tenure', 'Usage Frequency',
                    'Support Calls', 'Payment Delay',
                    'Subscription Type', 'Contract Length',
                    'Total Spend', 'Last Interaction', 'payment_delay_to_total_spent_ratio_standardized']) 

        return X