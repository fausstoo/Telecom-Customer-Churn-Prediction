import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#---------------------------------------------------
#       Tabular Report Continuous Features         |
#---------------------------------------------------
def tr_continuous_f(data_frame):
    # Filter out non-continuous variables
    numeric_df = data_frame.select_dtypes(include='number')

    # Calculate statistics using vectorized operations
    report_df = pd.DataFrame({
        'Feature': numeric_df.columns,
        'Count': numeric_df.count(),
        'Miss %': (numeric_df.isnull().sum() / len(numeric_df)) * 100,
        'Card.': numeric_df.nunique(),
        'Min': numeric_df.min(),
        '1st QRT': numeric_df.quantile(0.25),
        'Mean': numeric_df.mean(),
        'Median': numeric_df.median(),
        '3rd QTR': numeric_df.quantile(0.75),
        'Max': numeric_df.max(),
        'STD Dev': numeric_df.std()
    }).reset_index(drop=True)

    return report_df



#---------------------------------------------------
#      Tabular Report Categorical Features         |
#---------------------------------------------------

def tr_categorical_f(data_frame):
 # Filter out categorical variables
    cat_df = data_frame.select_dtypes(include='object')

    # Calculate statistics using vectorized operations
    mode_value = cat_df.mode().iloc[0]
    mode_frequency = cat_df.apply(lambda col: col.value_counts().max())
    mode_percentage = (mode_frequency / len(cat_df)) * 100

    # Find the second mode and its frequency
    def find_second_mode_freq(col):
        value_counts = col.value_counts()
        max_freq = value_counts.max()
        second_mode_freq = value_counts[value_counts < max_freq].max()  # Find the second highest frequency
        return second_mode_freq

    def find_second_mode_value(col):
        value_counts = col.value_counts()
        max_freq = value_counts.max()
        second_mode_values = value_counts[value_counts < max_freq].index.tolist()  # Find all values with the second highest frequency
        return second_mode_values

    second_mode_frequency = cat_df.apply(find_second_mode_freq)
    second_mode_value = cat_df.apply(find_second_mode_value)
    second_mode_percentage = (second_mode_frequency / len(cat_df)) * 100

    # Calculate other statistics
    count = cat_df.count()
    missing_percentage = round((cat_df.isnull().sum() / count) * 100, 3)
    cardinality = cat_df.nunique()

    # Create the final DataFrame from the calculated statistics
    report_df = pd.DataFrame({
        'Feature': cat_df.columns,
        'Count': count,
        'Miss %': missing_percentage,
        'Card.': cardinality,
        'Mode': mode_value,
        'Mode Freq': mode_frequency,
        'Mode %': mode_percentage,
        '2nd Mode': second_mode_value,
        '2nd Mode Freq': second_mode_frequency,
        '2nd Mode %': second_mode_percentage
    })

    return report_df




