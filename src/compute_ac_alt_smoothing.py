
"""
Module to clean, smooth and spatially autocorrelate firing rate data.
"""


import numpy as np
from skimage.measure import label, regionprops
from scipy.signal import correlate2d
from scipy.ndimage import convolve
from src import ImportDm


def clean_data(area, cell, epoch):
    """
    Index data by trials where monkey moved towards target.

    Arguments:
        area (str): "Area32", "dACC", or "OFC"
        cell (int): cell number in area
        epoch (str): "optionsOn" or "optionMade"

    Returns:
        firing_rates (np.array): firing rate on each trial
        states (np.array): state on each trial
    """

    assert area in ["Area32", "dACC", "OFC"]
    assert isinstance(cell, int)
    assert epoch in ["optionsOn", "optionMade"]

    (firing_rates,   # Firing rate in 10 ms increments +/- 1 second from trial epoch
     dist_change,    # New distance to target compared to previous step (+1 = moved towards target)
     to_x,           # X-coordinate of state they have just chosen
     to_y,           # Y-coordinate of ...
    ) = ImportDm.getData(area, cell, epoch)

    # Only use trials where they moved towards the target
    good_trials = dist_change == 1
    firing_rates = firing_rates[good_trials]
    state_x = np.array(to_x[good_trials], dtype=int, copy=True)
    state_y = np.array(to_y[good_trials], dtype=int, copy=True)
    states = np.array(list(zip(state_x, state_y)))
    assert len(firing_rates) == len(states)

    return firing_rates, states


def count_states(firing_rates, states, timepoints):
    """
    Count number of visits to each state and attach firing rates.

    Arguments:
        firing_rates (np.array): array of firing rates by trial
        states (np.array): array of states by trial in (x,y) coordinates
        timepoints (list): list of time points to perform computation

    Returns:
        firing_rates_by_state (np.array): array of cumulative firing rate in each state
        state_counts (np.array): array counting the number of times each state has been visited
    """
    print(firing_rates.shape)
    dim_x = np.max(states[:, 0])
    dim_y = np.max(states[:, 1])
    state_counts = np.zeros((dim_x + 1, dim_y + 1, len(firing_rates[0])))
    firing_rates_by_state = np.zeros((dim_x + 1, dim_y + 1, \
                                      len(firing_rates[0])))

    for trial, (x_loc, y_loc) in enumerate(states):
        state_counts[x_loc, y_loc, :] += 1
        firing_rates_by_state[x_loc, y_loc, :] += firing_rates[trial]
    assert np.sum(state_counts[...,0]) == firing_rates.shape[0]

    firing_rates_by_state = firing_rates_by_state[..., timepoints]
    state_counts = state_counts[..., timepoints]
    assert firing_rates_by_state.shape == state_counts.shape

    return firing_rates_by_state, state_counts


def smooth_firing_rates(firing_rates_by_state, state_counts,
                        manhattan_scaling, diagonal_scaling
                        ):
    """
    Spatially smooth firing rate input.

    Arguments:
        mean_firing_rates (np.array): spatial array of mean firing rates
        manhattan_scaling (float): number in [0, 1] that determines degree of manhattan scaling
        diagonal_scaling (float): number in [0, 1] that determines degree of diagonal scaling

    Returns:
        smoothed_data (np.array): Spatially smoothed firing rates
    """
    # FIXME: write this in a more elegant fashion

    # Define the convolution kernel for smoothing
    kernel = np.array([
        [diagonal_scaling, manhattan_scaling, diagonal_scaling],
        [manhattan_scaling, 1, manhattan_scaling],
        [diagonal_scaling, manhattan_scaling, diagonal_scaling]
    ])

    kernel = kernel[:, :, None]

    # Perform convolution
    smoothed_counts = convolve(state_counts, kernel, mode='mirror', cval=0)
    smoothed_frs = convolve(firing_rates_by_state * state_counts, kernel, mode='mirror', cval=0)

    # Calculate the smoothed firing rates
    smoothed_firing_rates = smoothed_frs / smoothed_counts

    return smoothed_firing_rates

    """
    counts, frs = np.copy(state_counts), np.copy(firing_rates_by_state)

    for i in range(len(counts)):
        for j in range(len(counts[0])):
            if i>0 and j>0:
                counts[i,j] += state_counts[i-1,j-1] * diagonal_scaling
                frs[i,j] += firing_rates_by_state[i-1,j-1] * diagonal_scaling
            if i>0:
                counts[i,j] += state_counts[i-1,j] * manhattan_scaling
                frs[i,j] += firing_rates_by_state[i-1,j] * manhattan_scaling
            if i>0 and j<len(counts[0])-1:
                counts[i,j] += state_counts[i-1,j+1] * diagonal_scaling
                frs[i,j] += firing_rates_by_state[i-1,j+1] * diagonal_scaling
            if i<len(counts)-1 and j<len(counts[0])-1:
                counts[i,j] += state_counts[i+1,j+1] * diagonal_scaling
                frs[i,j] += firing_rates_by_state[i+1,j+1] * diagonal_scaling
            if i<len(counts)-1:
                counts[i,j] += state_counts[i+1,j] * manhattan_scaling
                frs[i,j] += firing_rates_by_state[i+1,j] * manhattan_scaling
            if i<len(counts)-1 and j>0:
                counts[i,j] += state_counts[i+1,j-1] * diagonal_scaling
                frs[i,j] += firing_rates_by_state[i+1,j-1] * diagonal_scaling
            if j<len(counts[0])-1:
                counts[i,j] += state_counts[i,j+1] * manhattan_scaling
                frs[i,j] += firing_rates_by_state[i,j+1] * manhattan_scaling
            if j>0:
                counts[i,j] += state_counts[i,j-1] * manhattan_scaling
                frs[i,j] += firing_rates_by_state[i,j-1] * manhattan_scaling
    frs /= counts

    return frs"""


def compute_autocorrelogram(smoothed_data):
    """
    Take smoothed firing rates and return spatial autocorrelogram(s).

    Arguments:
        smoothed_data (np.array): smoothed firing rates

    Returns:
        ac (np.array): normalised autocorrelogram
    """
    x_dim, y_dim = smoothed_data.shape[0]*2 - 1, smoothed_data.shape[1]*2 - 1
    if len(smoothed_data.shape) > 2:
        n_timepoints = smoothed_data.shape[-1]
    else:
        n_timepoints = 1
    autocorrelograms = np.zeros((x_dim, y_dim, n_timepoints))
    autocorrelograms.squeeze()

    for time_point in range(n_timepoints):
        firing_rate = smoothed_data[..., time_point]
        ac_init = correlate2d(firing_rate, firing_rate)
        ac_max = np.max(ac_init)
        autocorr = ac_init / ac_max
        autocorrelograms[..., time_point] = autocorr

    return autocorrelograms


def make_autocorrelogram(area, cell, epoch, timepoints = range(55,150,10),
                         scaling = (0.50, 0.25)
                         ):
    """
    Combine function to clean + smooth FRs and compute ACs.

    Arguments:
        area (str): "Area32", "dACC", or "OFC"
        cell (int): cell number in area
        epoch (str): "optionsOn" or "optionMade"
        timepoints (list): list of time points to perform computation
        scaling (tuple): scaling factors in [0, 1] written as (manhattan, diagonal)

    Returns:
        autocorrelograms (np.array): x-by-y-by-t array of normalised spatial 
                                     autocorrelation values
    """
    assert [0 <= scale <= 1 for scale in scaling]

    firing_rates, states = clean_data(area, cell, epoch)
    firing_rates_by_state, state_counts = count_states(firing_rates, states,
                                                       timepoints)
    smoothed_data = smooth_firing_rates(firing_rates_by_state, state_counts,
                                        scaling[0], scaling[1])
    autocorrelograms = compute_autocorrelogram(smoothed_data)
    assert autocorrelograms.shape[-1] == len(timepoints)

    return autocorrelograms

# TODO: add ratemap permutation with parameters for n_perm
# TODO: create module to calculate grid score (circular + elliptical)
# TODO: add function to integrate grid score calculation

def clean_ac(ac, threshold = 0.1):
    "Clean AC; remove inf/nan and set low correlation to 0."
    ac_clean = np.copy(ac)
    ac_clean[np.isnan(ac) | np.isinf(ac)] = 0
    ac_clean[ac < threshold] = 0
    return ac_clean

def find_central_peak(ac_clean):
    """Find diameter of central peak of autocorrelogram"""
    centre = [(ac_clean.shape[0]-1)/2, (ac_clean.shape[1]-1)/2]
    labeled_ac = label(ac_clean > 0)
    props = regionprops(labeled_ac)
    centroids = np.array([prop.centroid for prop in props])
    if centroids.size == 0:
        return None, None

    for i, centroid in enumerate(centroids):
        if np.allclose(centroid, centre):
            return centre, props[i]
    
    return None, None

    #centre_centroid = props[np.where(centroids==centre)[0][0]]
    #return centre, centre_centroid