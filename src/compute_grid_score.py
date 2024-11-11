"""
Script to compute circular and elliptical grid scores with expanding ellipses
"""

"""
We need functions that:

1. Cleans and defines central peak

2. 
"""

import math
import numpy as np
from skimage.measure import label, regionprops
from skimage.transform import rotate
from scipy.spatial.distance import cdist

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
        return None, None, None

    centre_centroid = props[np.where(centroids==centre)[0][0]]
    return centre, centre_centroid, centroids

def compute_distances(centre, centroids, min_dist):
    dists_to_centre = np.linalg.norm(centroids - centre, axis=1)
    surrounding_peaks = centroids[dists_to_centre >= min_dist]
    return surrounding_peaks, dists_to_centre

def compute_orientation(nearest_peaks, centre):
    angles = np.arctan2(nearest_peaks[:,0] - centre[0],
                        nearest_peaks[:,1] - centre[1])
    
    return angles#np.median(np.degrees(angles))
    
def grid_autocorr(ac, angles):
    ac_rot = [rotate(ac, angle) for angle in angles]
    

def circle_mask(size, radius, in_val=1, out_val=0):
    """
    Make circular masks with centre removed.
    
    Arguments:
    """
    x_dist = np.arange(-(size[0]+1)/2 + 1, (size[0]+1)/2)
    print(x_dist)
    x_dist = np.expand_dims(x_dist, 0)
    x_dist = x_dist.repeat(size[1],0)
    y_dist = np.arange(-(size[1]+1)/2 + 1, (size[1]+1)/2)
    y_dist = np.expand_dims(y_dist, 0)
    y_dist = y_dist.repeat(size[0],0)
    eu_dist = np.sqrt(x_dist.T**2 + y_dist**2)
    circ = np.less_equal(eu_dist, radius)
    vfunc = np.vectorize(lambda b: b and in_val or out_val)
    return vfunc(circ)

