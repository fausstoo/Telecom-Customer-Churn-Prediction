import pickle 
from sklearn.ensemble import IsolationForest
from scipy.stats import zscore
from sklearn.ensemble import GradientBoostingRegressor

#----------------------------------------------------------------------------------
#                           z-Score Outlier Removal                               |
#----------------------------------------------------------------------------------
def outliers_zscore(df, threshold=None, path=None):
    # Separate continuous features
    continuous_features = df.select_dtypes(include=['float64', 'int64']).columns

    # Calculate Z-scores for continuous features
    z_scores = np.abs(zscore(df[continuous_features]))

    # Create a mask for outlier detection
    outlier_mask = (z_scores < threshold).all(axis=1)

    # Filter the DataFrame to remove outliers
    df_filtered = df[outlier_mask]
    
    # Save the new data frame
    if path:
        with open(path, 'wb') as file:
            pickle.dump(df_filtered, file)
        
    return df_filtered

#-----------------------------------------------------------------------------------
#                                IQR Outlier Removal                               |
#-----------------------------------------------------------------------------------
def outliers_iqr(df, multiplier=None, path=None):
    # Separate continuous features
    continuous_features = df.select_dtypes(include=['float64', 'int64']).columns
    
    # Calculate Q1 and Q3 for continuous features
    Q1 = df[continuous_features].quantile(0.25)
    Q3 = df[continuous_features].quantile(0.75)
    
    # Calculate IQR for continuous features
    IQR = Q3 - Q1
    
    # Create a mask for outlier detection
    outlier_mask = ~((df[continuous_features] < (Q1 - multiplier * IQR)) | (df[continuous_features] > (Q3 + multiplier * IQR))).any(axis=1)
    
    # Filter the DataFrame to remove outliers
    df_filtered = df[outlier_mask]
    
    # Save the new data frame
    if path:
        with open(path, 'wb') as file:
            pickle.dump(df_filtered, file)
    
    return df_filtered




#-----------------------------------------------------------------------------------
#                        IsolationForest Outlier Removal                           |
#-----------------------------------------------------------------------------------
def remove_outliers_isolation_forest(df, contamination=None, random_state=None, path=None):
    # Separate continuous features
    continuous_features = df.select_dtypes(include=['float64', 'int64']).columns
    
    # Create a DataFrame with only continuous features
    df_continuous = df[continuous_features]
    
    # Initialize and fit the Isolation Forest model
    model = IsolationForest(contamination=contamination, random_state=random_state, max_samples=0.35)
    model.fit(df_continuous.values)
    
    # Predict outliers
    outlier_mask = model.predict(df_continuous.values) == 1
    
    # Filter the DataFrame to remove outliers
    df_filtered = df[outlier_mask]

    # Save the new data frame
    if path:
        with open(path, 'wb') as file:
            pickle.dump(df_filtered, file)
    
    return df_filtered

#-----------------------------------------------------------------------------------
#                   GradientBoostingRegressor Outlier Removal                      |
#-----------------------------------------------------------------------------------
def remove_outliers_gradient_boost(df, threshold=0.9, random_state=50, path=None):
    # Separate continuous features
    continuous_features = df.select_dtypes(include=['float64', 'int64']).columns
    
    # Create a DataFrame with only continuous features
    df_continuous = df[continuous_features]
    
    # Initialize and fit the Gradient Boosting Regressor model
    model = GradientBoostingRegressor(loss='squared_error', random_state=random_state)
    model.fit(df_continuous, df_continuous.index)
    
    # Calculate the absolute residuals
    residuals = np.abs(model.predict(df_continuous) - df_continuous.index)
    
    # Calculate the threshold for outlier detection
    threshold_value = np.percentile(residuals, threshold * 100)
    
    # Create a mask for outlier detection
    outlier_mask = residuals <= threshold_value
    
    # Filter the DataFrame to remove outliers
    df_filtered = df[outlier_mask]

    # Save the new data frame
    if path:
        with open(path, 'wb') as file:
            pickle.dump(df_filtered, file)
    
    return df_filtered