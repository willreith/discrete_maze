#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Produce ratemap-permuted autocorrelograms for specified time bins
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import seaborn as sb
from misc.ImportDm import epochs
from misc import Utils, ImportDm
from functools import partial
import multiprocessing
import random

random.seed(0)


# Set analysis parameters
timestamps = {'optionMade':100, 'optionsOn':130}

manhattan_scaling = 0.25
diag_scaling      = 0

bin_start    = 55
bin_finish   = 150
bin_interval = 10
n_bins       = len(range(bin_start, bin_finish, bin_interval))



def func(area, 
         cell, 
         manhattan_scaling=manhattan_scaling,
         diag_scaling=diag_scaling,
         save=False,
         bin_start=bin_start,
         bin_finish=bin_finish,
         bin_interval=bin_interval,
         n_bins=n_bins):
    
    for epoch in epochs:

        # Load in the data

        (y,             # firing rate in 10 ms increments +/- 1 second from trial epoch
         distChange,    # For this step, what is the new distance to target compared to the previous step (+1 = moved towards target)
         currAngle,     # Angle between current location and target
         hd,            # What direction did they move (north south east west)
         numsteps,      # In this trial, how many steps have they taken?
         perfTrials,    # Was this trial perfect? I.e. all steps were towards the target
         startAngle,    # Starting angle to the target
         currDist,      # Current distance to the target
         from_x,         # X-coordinate of state they just moved from
         from_y,         # Y-coordinate of ....
         to_x,           # X-coordinate of state they have just chosen
         to_y,           # Y-coordinate of ...
         ) = ImportDm.getData(area, cell, epoch)

        # Only use trials where they moved towards the target
        m = distChange == 1
        y = y[m]
        state_x = np.array(to_x[m], dtype=int, copy=True)
        state_y = np.array(to_y[m], dtype=int, copy=True)

        # We count the number of state visits in each half of the session in order to perform a median split
        frs    = np.zeros((np.max(state_x) + 1, np.max(state_y) + 1, len(y[0])))
        counts = np.zeros((np.max(state_x) + 1, np.max(state_y) + 1, len(y[0])))
        n      = len(state_x)

        for i, (xx, yy) in enumerate(zip(state_x, state_y)):
            counts[xx, yy] += 1
            frs[xx, yy]    += y[i]

        counts = counts[..., timestamps[epoch]]
        frs    = frs[..., timestamps[epoch]]
        
        ac_rows = 2*counts.shape[0]-1
        ac_cols = 2*counts.shape[1]-1

        nPerm = 100
        acs   = np.empty((ac_rows*nPerm, ac_cols))
        
        for k in range(nPerm):
            
            # Get FR at relevant time stamp
            curr_counts = np.copy(counts)
            curr_frs    = np.copy(frs)
            
            curr_frs, curr_counts = curr_frs.flatten(), curr_counts.flatten()
            arr_len               = curr_frs.size            
            perm                  = np.random.permutation(arr_len)
            curr_frs, curr_counts = curr_frs[perm], curr_counts[perm]
            curr_frs, curr_counts = np.reshape(curr_frs, frs.shape), np.reshape(curr_counts, counts.shape)
            
            # We smooth with a different weighting on diagonal vs manhattan distances
            diag_scaling      = 0.25
            manhattan_scaling = 0.5
            origcounts, origfrs = np.copy(counts), np.copy(frs)          
            
            for i in range(len(counts)):
                for j in range(len(counts[0])):
                    if i>0 and j>0:
                        curr_counts[i,j] += origcounts[i-1,j-1] * diag_scaling
                        curr_frs[i,j] += origfrs[i-1,j-1] * diag_scaling
                    if i>0:
                        curr_counts[i,j] += origcounts[i-1,j] * manhattan_scaling
                        curr_frs[i,j] += origfrs[i-1,j] * manhattan_scaling
                    if i>0 and j<len(curr_counts[0])-1:
                        curr_counts[i,j] += origcounts[i-1,j+1] * diag_scaling
                        curr_frs[i,j] += origfrs[i-1,j+1] * diag_scaling
                    if i<len(curr_counts)-1 and j<len(curr_counts[0])-1:
                        curr_counts[i,j] += origcounts[i+1,j+1] * diag_scaling
                        curr_frs[i,j] += origfrs[i+1,j+1] * diag_scaling
                    if i<len(curr_counts)-1:
                        curr_counts[i,j] += origcounts[i+1,j] * manhattan_scaling
                        curr_frs[i,j] += origfrs[i+1,j] * manhattan_scaling
                    if i<len(curr_counts)-1 and j>0:
                        curr_counts[i,j] += origcounts[i+1,j-1] * diag_scaling
                        curr_frs[i,j] += origfrs[i+1,j-1] * diag_scaling
                    if j<len(curr_counts[0])-1:
                        curr_counts[i,j] += origcounts[i,j+1] * manhattan_scaling
                        curr_frs[i,j] += origfrs[i,j+1] * manhattan_scaling
                    if j>0:
                        curr_counts[i,j] += origcounts[i,j-1] * manhattan_scaling
                        curr_frs[i,j] += origfrs[i,j-1] * manhattan_scaling
    
            curr_frs /= curr_counts
    
            #%% Cross correlation
            try:
                ac = scipy.signal.correlate2d(curr_frs, curr_frs)
                
                ac_max = max(ac)
                ac     = ac/ac_max
                
            except IndexError:
                print('error numero dos')
                
                
            acs[k * ac_rows:(k + 1) * ac_rows, 0:ac_cols] = ac
    
        if save:
            fname = f"/Volumes/WilliamX6/DiscreteMaze/analysis_to/rate_map_shuffle/man{man}diag{diag}/perm/bins{n_bins}/ac_{epoch}_{area}_{cell}.csv"
            np.savetxt(fname, acs)


if __name__ == '__main__':
    for area in ImportDm.areas[::-1]:
        for cell in range(ImportDm.n[area]):
            func(area, cell)

# if __name__ == '__main__':
#     parallel = True
#     for area in ImportDm.areas[::-1]:
#         if parallel:
#             f = partial(func, area)
#             pool = multiprocessing.Pool()
#             pool.map(f, range(ImportDm.n[area]))
#         else:
#             for cell in range(ImportDm.n[area]):
#                 func(area, cell)

