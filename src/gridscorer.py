"""
Circular grid scoring approach based on
https://github.com/google-deepmind/grid-cells/blob/master/scores.py
"""

import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import seaborn as sb

def circle_mask(size, radius, in_val=1, out_val=0):
    """
    Make ring masks.
    
    Arguments:
      size: 
      radius:
      in_val:
      out_val:
    """
    x_dist = np.arange(-(size[0]+1)/2 + 1, (size[0]+1)/2)
    x_dist = np.expand_dims(x_dist, 0)
    x_dist = x_dist.repeat(size[1],0)
    y_dist = np.arange(-(size[1]+1)/2 + 1, (size[1]+1)/2)
    y_dist = np.expand_dims(y_dist, 0)
    y_dist = y_dist.repeat(size[0],0)
    eu_dist = np.sqrt(x_dist.T**2 + y_dist**2)
    ############################################################################
    circ = np.less_equal(eu_dist, np.round(radius)) # CHANGED TO ROUNDING RADIUS
    ############################################################################
    vfunc = np.vectorize(lambda b: b and in_val or out_val)
    return vfunc(circ)

class GridScorer(object):
  """Class for scoring autocorrelograms."""

  def __init__(self, ac, mask_parameters, size=(23,17), min_max=True):
    """Scoring ratemaps given trajectories.

    Args:
      nbins: Number of bins per dimension in the ratemap.
      coords_range: Environment coordinates range.
      mask_parameters: parameters for the masks that analyze the angular
        autocorrelation of the 2D autocorrelation.
      min_max: Correction.
    """
    self.ac = ac
    self._x_size, self._y_size = size
    self._size = size
    self._min_max = min_max
    self._corr_angles = [30, 45, 60, 90, 120, 135, 150]
    # Create all masks
    self._masks = [self._get_ring_mask(mask_min, mask_max) 
                   if mask_min != np.inf and mask_max != np.inf else None
                   for mask_min, mask_max in mask_parameters]
    self._mask_param = mask_parameters
    # Mask for hiding the parts of the SAC that are never used
    self._plotting_sac_mask = circle_mask(
        self._size,
        int(np.min(((self._x_size + 1)/2, (self._y_size + 1)/2))),
        in_val=1.0,
        out_val=np.nan)

  def _get_ring_mask(self, mask_min, mask_max):
    if mask_min is None or mask_max is None:
      return None
    
    sb.heatmap((circle_mask(self._size, mask_max) *
            (1 - circle_mask(self._size, mask_min))), square=True, cmap='rainbow')
    return (circle_mask(self._size, mask_max) *
            (1 - circle_mask(self._size, mask_min)))

  def _grid_score_60(self, corr):
    if self._min_max:
      return np.minimum(corr[60], corr[120]) - np.maximum(
          corr[30], np.maximum(corr[90], corr[150]))
    else:
      return (corr[60] + corr[120]) / 2 - (corr[30] + corr[90] + corr[150]) / 3

  def _grid_score_90(self, corr):
    """"""
    return corr[90] - (corr[45] + corr[135]) / 2

  def set_ac(self, ac):
    """Set spatial autocorrelogram."""
    self.ac = ac

  def get_ac(self):
    return self.ac
  
  def _rotated_acs(self, ac, angles):
    return [
        scipy.ndimage.rotate(ac, angle, reshape=False)
        for angle in angles
    ]
  
  def _get_grid_scores_for_mask(self, ac, rotated_acs, mask):
    """Calculate Pearson correlations of area inside mask at corr_angles."""
    masked_ac = ac * mask
    ring_area = np.sum(mask)
    # Calculate dc on the ring area
    masked_ac_mean = np.sum(masked_ac) / ring_area
    # Center the sac values inside the ring
    masked_ac_centered = (masked_ac - masked_ac_mean) * mask
    variance = np.sum(masked_ac_centered**2) / ring_area + 1e-5
    corrs = dict()
    for angle, rotated_ac in zip(self._corr_angles, rotated_acs):
      masked_rotated_ac = (rotated_ac - masked_ac_mean) * mask
      cross_prod = np.sum(masked_ac_centered * masked_rotated_ac) / ring_area
      corrs[angle] = cross_prod / variance
    return self._grid_score_60(corrs), self._grid_score_90(corrs)

  def _get_score(self, ac=None):
    """Get summary of scrores for grid cells."""
    if ac is None:
      ac = self.ac
    rotated_acs = self._rotated_acs(ac, self._corr_angles)
    self.rotated_acs = rotated_acs

    scores = [
        self._get_grid_scores_for_mask(ac, rotated_acs, mask)
        for mask in self._masks
    ]
    if len(scores) != 0:
      scores_60, scores_90 = map(np.asarray, zip(*scores)) 
      max_60_ind = np.argmax(scores_60)
      max_90_ind = np.argmax(scores_90)
      return scores_60[max_60_ind] #scores_90[max_90_ind],
    else:
      return np.nan

      # return scores_60[max_60_ind] #scores_90[max_90_ind],
            #self._mask_param[max_60_ind], self._mask_param[max_90_ind])

  def get_all_scores(self):
    """"""
    if any(mask is None for mask in self._masks):
      self.g_scores = np.nan
      return
    else:
      curr_ac = self.ac
      curr_ac = np.squeeze(curr_ac)
      if curr_ac.ndim > 2:
        acs = np.moveaxis(curr_ac, 2, 0)
        scores = [self._get_score(ac=np.array(ind_ac)) for ind_ac in acs]
        self.g_scores = scores
      else:
        self.g_scores = self._get_score(ac=curr_ac)