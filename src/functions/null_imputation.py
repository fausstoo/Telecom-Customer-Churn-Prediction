import pickle 
from sklearn.base import BaseEstimator, TransformerMixin

class NullImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
         # Filter out continuous columns with cardinality > 9
        continuous_cols = X.select_dtypes(exclude=['object'])
        selected_cont_cols = continuous_cols.columns[continuous_cols.nunique() > 9]

        # Filter out categorical columns with cardinality < 9
        categorical_cols = X.select_dtypes(include=['number', 'object'])
        selecte_cat_cols = categorical_cols.columns[categorical_cols.nunique() < 9]

        # Impute missing values in continuous columns with median
        X[selected_cont_cols] = X[selected_cont_cols].fillna(X[selected_cont_cols].median())

        # Impute missing values in categorical columns with mode
        for col in selecte_cat_cols:
            mode_val = X[col].mode().iloc[0]
            X[col] = X[col].fillna(mode_val)

        return X