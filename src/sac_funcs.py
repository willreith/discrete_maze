import numpy as np
import seaborn as sb

from scipy.ndimage import convolve
from scipy.signal import correlate2d
from skimage.measure import label, regionprops
from src import import_data
from src import elliptical_funcs as ef


def load_data(area, cell, epoch, timepoints=range(50,151,10)):
    """Load firing rate, distance change and location data"""
    (raw_frs,  
     dist_change,    
     to_x,           
     to_y,           
     ) = import_data.get_data(area, cell, epoch)
    return raw_frs[..., timepoints], dist_change, to_x, to_y


def clean_data(raw_frs, dist_change, to_x, to_y):
    good_trials = dist_change == 1
    firing_rates = raw_frs[good_trials]
    state_x = np.array(to_x[good_trials], dtype=int, copy=True)
    state_y = np.array(to_y[good_trials], dtype=int, copy=True)
    states = np.array(list(zip(state_x, state_y)))
    return firing_rates, states


def bin_data(firing_rates, states):
    """Attach firing rates to states on x by y grid"""
    dim_x = np.max(states[:, 0]) + 1
    dim_y = np.max(states[:, 1]) + 1
    state_counts = np.zeros((dim_x, dim_y, len(firing_rates[0])))
    firing_rates_by_state = np.zeros((dim_x, dim_y, len(firing_rates[0])))

    for trial, (x_loc, y_loc) in enumerate(states):
        state_counts[x_loc, y_loc, :] += 1
        firing_rates_by_state[x_loc, y_loc, :] += firing_rates[trial]
    
    return firing_rates_by_state, state_counts


def kernel_smoother(firing_rates_by_state, state_counts, scaling=(0.5, 0.25)):
    manhattan_scaling = scaling[0]
    diagonal_scaling = scaling[1]

    kernel = np.array([
                [diagonal_scaling, manhattan_scaling, diagonal_scaling],
                [manhattan_scaling, 1, manhattan_scaling],
                [diagonal_scaling, manhattan_scaling, diagonal_scaling]
            ])
    
    kernel = kernel[:, :, None]
    smoothed_counts = convolve(state_counts, kernel, mode='constant', cval=0)
    smoothed_frs = convolve(firing_rates_by_state * state_counts, kernel,
                            mode='constant', cval=0) #FIXME: change from constant to mirror
    return smoothed_frs / smoothed_counts


def manual_smoother(firing_rates_by_state, state_counts, scaling=(0.5, 0.25)):
    manhattan_scaling = scaling[0]
    diagonal_scaling = scaling[1]

    counts = np.copy(state_counts)
    frs = np.copy(firing_rates_by_state)

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
    return frs / counts
            

def make_sacs(smoothed_data): 
    dim1, dim2 = (x*2 - 1 for x in smoothed_data.shape[:2])
    dim3 = smoothed_data.shape[2]
    sacs = np.empty((dim1, dim2, dim3))
    for bin in range(dim3):
        curr_fr = smoothed_data[..., bin]
        ac_init = correlate2d(curr_fr, curr_fr)
        ac_max = np.max(ac_init)
        sacs[..., bin] = ac_init / ac_max
    return sacs


def clean_sacs(sacs, threshold=0.4):
    n_bins = sacs.shape[2]
    cleaned_sacs = np.copy(sacs)

    for bin in range(n_bins):
        curr_ac = cleaned_sacs[..., bin]
        curr_ac[np.isnan(curr_ac)] = 0
        curr_ac[np.isinf(curr_ac)] = 0
        curr_ac[curr_ac < threshold] = 0
        cleaned_sacs[..., bin] = curr_ac
    return cleaned_sacs


def find_peaks(clean_sacs):
    n_bins = clean_sacs.shape[2]
    radii = np.empty((n_bins, 1))
    labeled_sac = label(clean_sacs > 0)
    centre = ((clean_sacs.shape[0]-1)/2, (clean_sacs.shape[1]-1)/2)
    for bin in range(n_bins):
        props = regionprops(labeled_sac[..., bin])
        centroids = np.array([prop.centroid for prop in props])
        diameters = [prop.equivalent_diameter_area for prop in props]
        centre_index = int(len(centroids)/2 + 0.5) - 1
        radii[bin] = diameters[centre_index] / 2
    return centre, radii


def get_mask_parameters(centre, radii, step_size=0.45):
    n_bins = radii.shape[0]
    max_rad = np.min(centre) + step_size - 1e-5
    mask_parameters = []
    
    min_dist = [rad * 1.5 for rad in radii]
    expanding_radii = [
        np.arange(min_rad, max_rad - rad, step_size)
        for min_rad, rad in zip(min_dist, radii)
    ]
    
    for bin in range(n_bins):
        if expanding_radii[bin].size == 0:
            mask_parameters.append([(np.inf, np.inf)])
        else:
            centre_rad = radii[bin]
            curr_parameters = [(rad - centre_rad, rad + centre_rad)
                               for rad in expanding_radii[bin]]
            filtered_parameters = [param for param in curr_parameters
                                   if param[0] >= centre_rad]
            mask_parameters.append(filtered_parameters)
    return mask_parameters


def get_sacs(area,
             cell,
             epoch,
             timepoints=range(50,151,10),
             inner_threshold=0.4,
             elliptical=False,
             outer_threshold=0.1,
             ):
    raw_frs, dist_change, to_x, to_y = load_data(area, cell, epoch, timepoints=timepoints)
    firing_rates, states = clean_data(raw_frs, dist_change, to_x, to_y)
    fr_by_states, state_counts = bin_data(firing_rates, states)
    smoothed_data = manual_smoother(fr_by_states, state_counts)
    
    sacs = make_sacs(smoothed_data)
    if elliptical:
        sacs = ef.circularise_sacs(sacs, outer_threshold=outer_threshold)
    
    cleaned_sacs = clean_sacs(sacs, threshold=inner_threshold)
    centre, radii = find_peaks(cleaned_sacs)
    mask_parameters = get_mask_parameters(centre, radii)
    return sacs, mask_parameters, radii

def get_sacs_from_frs_and_states(firing_rates,
                                 states,
                                 inner_threshold=0.4,
                                 elliptical=False,
                                 outer_threshold=0.1,
                                 ):
    smoothed_data = manual_smoother(firing_rates, states)

    sacs = make_sacs(smoothed_data)
    if elliptical:
        sacs = ef.circularise_sacs(sacs)

    cleaned_sacs = clean_sacs(sacs)
    centre, radii = find_peaks(cleaned_sacs)
    mask_parameters = get_mask_parameters(centre, radii)
    return sacs, mask_parameters, radii


def visualise_sacs(sacs, index=None):
    if index is None and np.ndim(sacs) == 3:
        index = sacs.shape[2] // 2
        sb.heatmap(sacs[..., index], square=True)
    elif index is None and np.ndim(sacs) == 2:
        sb.heatmap(sacs, square=True)
    else:
        sb.heatmap(sacs[..., index], square=True)
    return


def load_clean_bin_data(area, cell, epoch, timepoints=range(50,151,10)):
    raw_frs, dist_change, to_x, to_y = load_data(area, cell, epoch, timepoints=timepoints)
    firing_rates, states = clean_data(raw_frs, dist_change, to_x, to_y)
    fr_by_states, state_counts = bin_data(firing_rates, states)
    return fr_by_states, state_counts

area = "Area32"
cell = 33
epoch = "optionMade"

def main(area, cell, epoch, timepoints=range(50,151,10)):
    if __name__ == '__main__':
        return get_sacs(area, cell, epoch, timepoints=timepoints)
    