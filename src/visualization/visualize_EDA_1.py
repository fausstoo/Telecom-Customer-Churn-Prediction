import sys
sys.path.append('../src/functions')

# EDA funcitons
from tabular_report_functions import *
from plot_functions import *

# Analysis libraries
import numpy as np
import pandas as pd

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Null values imputation
from null_imputation import impute_null_data

# Data
train_df = pd.read_csv("../data/raw/customer_churn_dataset-training-master.csv")
test_df = pd.read_csv("../data/raw/customer_churn_dataset-testing-master.csv")

df = pd.concat([train_df, test_df])


#---------------------------------------------------
#                 Plot Settings                    |
#---------------------------------------------------
plt.style.use("seaborn-v0_8-deep")
plt.rcParams["figure.figsize"] = [12, 5]
plt.rcParams["figure.dpi"] = 200



#---------------------------------------------------
#         Box Plots Continuous Features            |
#---------------------------------------------------

# Print Grouped Box Plots
grouped_boxplots(df)

# Save Plots


#---------------------------------------------------
#          Histograms Continuous Features          |
#---------------------------------------------------



#---------------------------------------------------------
#     Single Stacked Bar Plot Categorical Features       |
#---------------------------------------------------------



#---------------------------------------------------------
#     Get Stacked Bar Plots Categorical Features         |
#--------------------------------------------------------- 



#---------------------------------------------------------
#             Get Bar Plots of Null Values               |
#--------------------------------------------------------- 
