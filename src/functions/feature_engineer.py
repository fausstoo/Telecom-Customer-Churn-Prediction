
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

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Age Binning
        bins = [0, 29, 49, float('inf')]
        labels = ['Young Adults', 'Middle-age Adults', 'Seniors']
        X['age_category'] = pd.cut(X['Age'], bins=bins, labels=labels, right=False)
        category_mapping = {'Young Adults': 1, 'Middle-age Adults': 2, 'Seniors': 3}
        X['age_category'] = X['age_category'].map(category_mapping)

        # Binary Features
        X['high_support_calls'] = np.where(X['Support Calls'] > 4, 1, 0)
        X['high_payment_delay'] = np.where(X['Payment Delay'] > 20, 1, 0)
        X['low_spender'] = np.where(X['Total Spend'] > 450, 0, 1)
        X['Gender'] = X['Gender'].apply(lambda x: 1 if x == 'Female' else 0)
        X['monthly_contract'] = X['Contract Length'].apply(lambda x: 1 if x == 'Monthly' else 0)
        X['low_interaction_cust'] = X['Last Interaction'].apply(lambda x: 1 if x > 14 else 0)

        # Interaction Features
        X['age_support_calls_interaction'] = X['Age'] * X['Support Calls']
        X['payment_delay_to_total_spent_ratio'] = X['Payment Delay'] / X['Total Spend']

        # Drop Unnecessary Columns
        X = X.drop(columns=['Age', 'Tenure', 'Usage Frequency',
                            'Support Calls', 'Payment Delay',
                            'Subscription Type', 'Contract Length',
                            'Total Spend', 'Last Interaction'])

        return X