# Get percentiles

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# Load data and specify epochs and percentiles
data_fname = "/Volumes/WilliamX6/discrete_maze/results/spike_shuffle_50_151_5.pkl"
# data_fname = "/Volumes/WilliamX6/discrete_maze/results/test.pkl"
if 'df' not in locals():
    df = pd.read_pickle(data_fname)
epochs = ["optionMade", "optionsOn"]
percentiles = [95]

# Specify the columns to be used
gridscore_cols = ["Standard grid scores, optionMade",
                  "Elliptical grid scores, optionMade",
                  "Standard grid scores, optionsOn",
                  "Elliptical grid scores, optionsOn"]

permute_cols = ["Permuted standard grid scores, optionMade", 
                "Permuted elliptical grid scores, optionMade",
                "Permuted standard grid scores, optionsOn",
                "Permuted elliptical grid scores, optionsOn"]

# Clean df
def replace_none_with_nan(item):
    if isinstance(item, list):
        return [replace_none_with_nan(sub_item) for sub_item in item]
    elif item is None:
        return np.nan
    else:
        return item
    
def replace_nan_with_zero(item):
    if isinstance(item, list):
        return [replace_nan_with_zero(sub_item) for sub_item in item]
    elif np.isnan(item):
        return 0
    else:
        return item

for column in (gridscore_cols + permute_cols):
    df[column] = df[column].apply(replace_none_with_nan)
    if column in permute_cols:
        df[column] = df[column].apply(replace_nan_with_zero)

# Get percentiles
def get_percentile(perm_list, percentile=95):
    perm_array = np.asarray(perm_list)
    return np.percentile(perm_array, percentile, axis=0)

for pct in percentiles:
     for perm_col, col in zip(permute_cols, gridscore_cols):
        col_name = str(pct) + "th pctile, " + col[0].lower() + col[1:]
        df[col_name] = df[perm_col].apply(get_percentile, percentile=percentiles[0])

def get_streak(data, pct_array):
    if isinstance(data[0], list):  # Check if the first element is a list, indicating a list of lists
        perm_array = np.asarray(data)

        streak_list = []
        for perm_row in perm_array:
            if len(perm_row) != len(pct_array):
                raise ValueError(f"Row length {len(perm_row)} does not match pct_array length {len(pct_array)}")
            
            comparison = np.greater(perm_row, pct_array)
            streak = 0
            max_streak = 0
            for item in comparison:
                if item:
                    streak += 1
                    if streak > max_streak:
                        max_streak = streak
                else:
                    streak = 0
            streak_list.append(max_streak)
        return streak_list
    else:  # Assume data is a single list of floats
        if len(data) != len(pct_array):
            raise ValueError(f"Data length {len(data)} does not match pct_array length {len(pct_array)}")
        
        comparison = np.greater(data, pct_array)
        streak = 0
        max_streak = 0
        for item in comparison:
            if item:
                streak += 1
                if streak > max_streak:
                    max_streak = streak
            else:
                streak = 0
        return max_streak


for pct in percentiles:
    for perm_col, col in zip(permute_cols, gridscore_cols):
        perm_timepoints = "Perm timepoints above " + str(pct) + \
                          "th pctile, " + col[0].lower() + col[1:]
        real_timepoints = "Real timepoints above " + str(pct) + \
                          "th pctile, " + col[0].lower() + col[1:]
        perm_timepoints_pctile = str(pct) + "th pctile of timepoints \
                                 above " + str(pct) + "th pctile, " \
                                 + col[0].lower() + col[1:]
        pct_col = str(pct) + "th pctile, " + col[0].lower() + col[1:]
        comparison = "Significance, " + col[0].lower() + col[1:]
        
        # Apply the get_streak function row-wise
        df[perm_timepoints] = df.apply(lambda x: get_streak(x[perm_col], 
                                       x[pct_col]), axis=1)
        df[real_timepoints] = df.apply(lambda x: get_streak(x[col],                                                
                                       x[pct_col]), axis=1)
        df[perm_timepoints_pctile] = df[perm_timepoints].apply(lambda x: 
                                                                   np.round(np.percentile(x, pct)))
                                                                   
        

        df[comparison] = df[real_timepoints] >= df[perm_timepoints_pctile]
        df[comparison] = df.apply(lambda x: x[comparison] if np.sum(np.isnan(x[col])) < len(x[col])/2 else False, axis=1)
        print(f"{comparison}: {df[comparison].sum()}")
        
        

        