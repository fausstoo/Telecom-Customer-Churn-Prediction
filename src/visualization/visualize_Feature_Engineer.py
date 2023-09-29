
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append('../functions/')

# EDA functions
from plot_functions import *


# Import data
df = pd.read_pickle("../../data/processed/features_df.pkl")

#-------------------------------------------------------------------
#                        CORRELATION HEATMAP                       |
#-------------------------------------------------------------------

# Run both line of code at the same time
# Creat plot heat map variable
corr_heatmap = plot_correlation_heatmap(df)

# Save png heat map plot
save_corr_heatmap(corr_heatmap, "../../reports/figures/Feature Engineer")

