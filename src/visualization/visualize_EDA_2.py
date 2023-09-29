
import os
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append('../functions/')

# EDA functions
from plot_functions import *

# Analysis libraries
import numpy as np
import pandas as pd

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Import data
df = pd.read_pickle("../../data/processed/imputed_df.pkl")


#---------------------------------------------------
#                 Plot Settings                    |
#---------------------------------------------------
plt.style.use("seaborn-v0_8-deep")
plt.rcParams["figure.figsize"] = [10, 8]
plt.rcParams["figure.dpi"] = 200


#-------------------------------------------------------------------
#             BIVARIATE ANALYSIS CLASSIFICATION TASK               |
#-------------------------------------------------------------------
#               GROUPED HISTOGRAMS vs TARGET LABEL                 |
#-------------------------------------------------------------------

# Create the plot
hists = biv_grouped_histograms(df, 'Churn')

# Save the plot
save_biv_grouped_histograms(hists, "../../reports/figures/EDA-2")

#-------------------------------------------------------------------
#                GROUPED BOX PLOTS vs TARGET LABEL                 |
#-------------------------------------------------------------------

# Create and saved box plots  
biv_grouped_boxplots(df, 'Churn', "../../reports/figures/EDA-2")

#-------------------------------------------------------------------
#                GROUPED BAR PLOTS vs TARGET LABEL                 |
#-------------------------------------------------------------------

# Create and save bar plots
barplots = biv_grouped_barplots(df, 'Churn', "../../reports/figures/EDA-2")


#-------------------------------------------------------------------
#                      MULTIVARIATE ANALYSIS                       |
#-------------------------------------------------------------------
#                        CORRELATION HEATMAP                       |
#-------------------------------------------------------------------


# Run both line of code at the same time
# Creat plot heat map variable
corr_heatmap = plot_correlation_heatmap(df)

# Save png heat map plot
save_corr_heatmap(corr_heatmap, "../../reports/figures/EDA-2")


#-------------------------------------------------------------------
#                        CORRELATION PAIRPLOT                       |
#-------------------------------------------------------------------

# Create and save a pairplot with all features
plot_pairplot(df, 'Churn', "../../reports/figures/EDA-2")