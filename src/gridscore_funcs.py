"""
Circular grid scoring approach based on
https://github.com/google-deepmind/grid-cells/blob/master/scores.py
"""

import numpy as np
import scipy.signal

from src import sac_funcs as sf
from src import elliptical_funcs as ef
from src import permute_funcs as pf
from src import all_funcs

def circle_mask(size, radius, in_val=1, out_val=0):
    """
    Make circular masks.
    
    Args:
        size (tuple): two-tuple specifying size of field 
        radius (float): radius of circle
        in_val (float): value for elements within radius
        out_val (float): value for elements outside of radius

    Returns:
        (np.array) Boolean array specifying a circle.
    """
    x_dist = np.arange(-(size[0]+1)/2 + 1, (size[0]+1)/2)
    x_dist = np.expand_dims(x_dist, 0)
    x_dist = x_dist.repeat(size[1],0)
    y_dist = np.arange(-(size[1]+1)/2 + 1, (size[1]+1)/2)
    y_dist = np.expand_dims(y_dist, 0)
    y_dist = y_dist.repeat(size[0],0)
    eu_dist = np.sqrt(x_dist.T**2 + y_dist**2)
    circ = np.less_equal(eu_dist, radius)
    vfunc = np.vectorize(lambda b: b and in_val or out_val)
    return vfunc(circ)


def get_ring_mask(mask_min, mask_max, centre_rad, size=(23,17)):
    """
    Get ring mask by removing small from large circle mask.

    Args:
        mask_min (float): inner radius of ring
        mask_max (float): outer radius of ring
        centre_rad (float): radius of central peak, to be removed
        size (tuple): two-tuple specifying size of field

    Returns:
        (np.array) Boolean numpy array specifying a ring mask
    """
    if mask_min is None or mask_max is None:
        return None
    return circle_mask(size, mask_max) * (1 - circle_mask(size, mask_min)) \
            * (1- circle_mask(size, centre_rad))
  

def get_all_ring_masks(mask_params, centre_rad):
    """
    Get multiple ring masks.

    Args:
        mask_params (* by 2 np.array of floats): ordered pairs of values 
            specifying inner and outer radii of ring masks
        centre_rad (float): radius of central peak for given mask params

    Return:
        (list) List of np.arrays specifying ring masks.
    """
    return [
        get_ring_mask(mask_min, mask_max, centre_rad)
        if mask_min != np.inf and mask_max != np.inf else None
        for mask_min, mask_max in mask_params
    ]


def grid_score_60(corr, min_max=True):
    """
    Get hexagonal (circular) grid score.
    """
    if min_max:
        return np.minimum(corr[60], corr[120]) - np.maximum(
                          corr[30], np.maximum(corr[90], corr[150]))
    else:
        return (corr[60] + corr[120]) / 2 - (corr[30] + corr[90] + corr[150]) / 3


def rotate_sacs(sac, corr_angles=[30, 45, 60, 90, 120, 135, 150]):
    """
    Rotates spatial autocorrelograms.

    Args:
        sac (np.array): 2D spatial autocorrelogram
        corr_angles (list(int)): rotation angles in degrees

    Returns:
        (list) List of np.arrays of rotated spatial autocorrelograms.
    """
    return [
        scipy.ndimage.rotate(sac, angle, reshape=False)
        for angle in corr_angles
    ]


def get_grid_score_for_mask(sac,
                            rotated_sacs,
                            mask,
                            corr_angles=[30, 45, 60, 90, 120, 135, 150]):
    """
    Get grid score for given sac and mask.

    Args:
        sac (np.array): 
    """

    masked_sac = sac * mask # mask sac
    ring_area = np.sum(mask) # get total area of ring
    masked_sac_mean = np.sum(masked_sac) / ring_area # get mean value on ring
    masked_sac_centered = (masked_sac - masked_sac_mean) * mask
    variance = np.sum(masked_sac_centered**2) / ring_area + 1e-5
    corrs = dict()
    for angle, rotated_sac in zip(corr_angles, rotated_sacs):
        masked_rotated_sac = (rotated_sac - masked_sac_mean) * mask
        cross_prod = np.sum(masked_sac_centered * masked_rotated_sac) / ring_area
        corrs[angle] = cross_prod / variance
    return grid_score_60(corrs)

def get_grid_score(sac, masks):
    rotated_sacs = rotate_sacs(sac, corr_angles=[30, 45, 60, 90, 120, 135, 150])
    assert sac is not None
    scores = [
        get_grid_score_for_mask(sac, rotated_sacs, mask) 
        if mask is not None else None for mask in masks
    ]
    if len(scores) != 0:
        scores_60 = np.asarray(scores)
        max_60_ind = np.argmax(scores_60)
        return scores_60[max_60_ind] 
    else:
        return np.nan


def get_all_grid_scores(sacs, mask_params, radii):    
    curr_sacs = np.copy(sacs)
    curr_sacs = np.squeeze(curr_sacs)
    masks = map(get_all_ring_masks, mask_params, radii)
    if curr_sacs.ndim > 2:
        curr_sacs = np.moveaxis(curr_sacs, 2, 0)
        return [
            get_grid_score(ind_sac, mask) 
            for ind_sac, mask in zip(curr_sacs, masks)
        ]
    else:
        return get_grid_score(curr_sacs, masks)


def gridscore(area,
              cell,
              epoch,
              elliptical=False,
              timepoints=range(50,151,10),
              outer_threshold=0.1,
              ):
    raw_frs, dist_change, to_x, to_y = all_funcs.load_data(area, cell, epoch, timepoints=timepoints)
    firing_rates, states = all_funcs.clean_data(raw_frs, dist_change, to_x, to_y)
    fr_by_states, state_counts = all_funcs.bin_data(firing_rates, states)
    smoothed_data = all_funcs.manual_smoother(fr_by_states, state_counts)
    sacs = all_funcs.make_sacs(smoothed_data)
    
    if not elliptical:
        cleaned_sacs = all_funcs.clean_sacs(sacs)
        centre, radii = all_funcs.find_peaks(cleaned_sacs)
        mask_parameters = all_funcs.get_mask_parameters(centre, radii)
        return get_all_grid_scores(sacs, mask_parameters, radii)
    
    circularised_sacs = all_funcs.circularise_sacs(sacs, outer_threshold=0.1)
    cleaned_sacs = all_funcs.clean_sacs(circularised_sacs)
    centre, radii = all_funcs.find_peaks(cleaned_sacs)
    mask_parameters = all_funcs.get_mask_parameters(centre, radii)
    return get_all_grid_scores(circularised_sacs, mask_parameters, radii)

def perm_gridscore(area,
                   cell,
                   epoch,
                   timepoints=range(50,151,10),
                   num_permutations=500,
                   elliptical=False,
                   outer_threshold=0.1,
                   ):
    perm_gen = all_funcs.get_permuted_sacs(area,
                                    cell,
                                    epoch,
                                    timepoints=timepoints,
                                    elliptical=elliptical,
                                    num_permutations=num_permutations,
                                    outer_threshold=outer_threshold,
                                    )

    for _ in range(num_permutations):
        curr_gen = next(perm_gen)
        perm_sacs, mask_params, radii = curr_gen[0], curr_gen[1], curr_gen[2]
        yield get_all_grid_scores(perm_sacs, mask_params, radii)