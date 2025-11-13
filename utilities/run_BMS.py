#### New! Written by Kate ####
#### This runs Bayesian Model Selection on the F values of sdcm and pdcm model ####
#### This compares the two model and provide the one with the best performance ####

import pandas as pd
import numpy as np
from utilities.python_BMS import bms 

# Load the Excel files
sdcm_df = pd.read_excel("sdcm_F_values.xlsx")
pdcm_df = pd.read_excel("pdcm_F_values.xlsx")

# Combine and sort the normalized F values
combined_df = pd.concat([
    sdcm_df[['subject_idx', 'model', 'F_value_normalized']],
    pdcm_df[['subject_idx', 'model', 'F_value_normalized']]
], ignore_index=True)

# Pivot to form the LME matrix (log model evidence): rows = subjects, columns = models
lme_matrix = combined_df.pivot_table(
    index='subject_idx',
    columns='model',
    values='F_value_normalized'
)

# Convert to numpy array
lme_array = lme_matrix.to_numpy()

# Run BMS
alpha, exp_r, xp, pxp, bor = bms(lme_array)

# Print results
print("Model Probabilities (alpha):", alpha)                    # probability of each model being the best model across the group， range from 0 to 1
print("Expected Model Frequencies (exp_r):", exp_r)             # expected model frequency eg [0.7, 0.3] 70% of subjects are best fited by model 1
print("Exceedance Probabilities (xp):", xp)                     # exceedence probability eg[0.95, 0.05] model 1 is 95% more frequent than model 2
print("Protected Exceedance Probabilities (pxp):", pxp)         # corrected version of exceedence probaility, takes chance into account
print("Bayes Omnibus Risk (bor):", bor)                         # probability that there are no difference, if this value is high, difference might be due to chance

# Summary interpretation
best_model_idx = np.argmax(pxp)
print("\n=== Summary ===")
print(f"Based on protected exceedance probabilities, Model {best_model_idx + 1} is most likely to be the best model across the group.")
print(f"Bayes Omnibus Risk (BOR) = {bor:.4f} suggests that the probability all models are equally frequent is {bor:.4%}.")
