import numpy as np
import pandas as pd
import seaborn as sb

from scipy.ndimage import convolve
from scipy.signal import correlate2d
from skimage.measure import label, regionprops
from src import ImportDm, import_sessions

def load_data(area, cell, epoch, timepoints=range(50,151,10)):
    """Load firing rate, distance change and location data"""
    (raw_frs,  
     dist_change,    
     to_x,           
     to_y,           
     ) = ImportDm.getData(area, cell, epoch)
    return raw_frs[..., timepoints], dist_change, to_x, to_y


def clean_data(raw_frs, dist_change, to_x, to_y):
    good_trials = dist_change == 1
    firing_rates = raw_frs[good_trials]
    state_x = np.array(to_x[good_trials], dtype=int, copy=True)
    state_y = np.array(to_y[good_trials], dtype=int, copy=True)
    states = np.array(list(zip(state_x, state_y)))
    return firing_rates, states


def clean_states(to_x, to_y):
    state_x = np.array(to_x, dtype=int, copy=True)
    state_y = np.array(to_y, dtype=int, copy=True)
    states = np.array(list(zip(state_x, state_y)))
    return states


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

    #FIXME: ensure smoothing is applied to each slice independently

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
        sacs = circularise_sacs(sacs, outer_threshold=outer_threshold)
    
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
        sacs = circularise_sacs(sacs)

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

######################### Elliptical functions #################################

import scipy.ndimage

from skimage.measure import label, regionprops
from sunpy.image.transform import affine_transform

def remove_low_corr(sacs, threshold=0.1):
    """Remove low autocorrelation values from spatial autocorrelograms."""
    cleaned_sacs = np.copy(sacs)
    cleaned_sacs[cleaned_sacs < threshold] = 0
    return cleaned_sacs


def get_peak_properties(sac, threshold=0.1):
    """Get peak properties of a spatial autocorrelogram."""
    clean_sac = remove_low_corr(sac, threshold=threshold)
    centre = ((clean_sac.shape[0] - 1)/2, (clean_sac.shape[1] - 1)/2)
    labeled_sac = label(clean_sac > 0)
    props = regionprops(labeled_sac)
    centroids = np.array([prop.centroid for prop in props])
    centre_index = np.argmin(np.linalg.norm(centroids - centre, axis=1))
    radii = props[centre_index].equivalent_diameter_area / 2
    return np.array(centre), radii, centroids, props


def get_all_peak_properties(sacs, threshold=0.1):
    """Get peak properties of multiple spatial autocorrelograms."""
    if sacs.ndim == 2:
        sacs = np.expand_dims(sacs, axis=2)
    all_centroids = []
    all_radii = []
    all_props = []
    for sac_idx in range(sacs.shape[2]):
        cen, rad, centroids, props = get_peak_properties(sacs[..., sac_idx],
                                                         threshold=threshold)
        all_centroids.append(centroids)
        all_radii.append(rad)
        all_props.append(props)
    return cen, all_centroids, all_radii, all_props


def get_nearest_peaks(centroids, centre):
    """Return two-tuple of sorted peak dist and loc for a single SAC."""
    curr_centroids = np.copy(centroids)
    indices_to_remove = np.where((curr_centroids == centre).all(axis=1))
    curr_centroids = np.delete(curr_centroids, indices_to_remove, axis=0)
    if len(curr_centroids) < 2:
        nearest_peak_distances = (None, None)
        nearest_centroids = (None, None)
    else:
        distances = np.linalg.norm(curr_centroids - centre, axis=1)
        peaks_by_dist = np.argsort(distances)
        nearest_peak_distances = distances[peaks_by_dist]
        nearest_centroids = curr_centroids[peaks_by_dist]
    return nearest_peak_distances, nearest_centroids


def get_all_nearest_peaks(all_centroids, centre):
    """Return two-tuples of sorted peak dist and loc for multiple SACs."""
    all_nearest_dist = []
    all_nearest_peaks = []
    for centroids in all_centroids:
        nearest_dist, nearest_peak = get_nearest_peaks(centroids, centre)
        all_nearest_dist.append(nearest_dist)
        all_nearest_peaks.append(nearest_peak)
    return all_nearest_dist, all_nearest_peaks


def get_furthest_peak(centroids_by_dist):
    """Return furthest peak (of max 6) from centre for a single SAC."""
    if centroids_by_dist is None:
        return None
    if len(centroids_by_dist) > 6:
        furthest_peak = centroids_by_dist[5]
    else:
        furthest_peak = centroids_by_dist[-1]
    return furthest_peak


def get_all_furthest_peaks(nearest_centroids):
    """Return furthest peak (of max 6) from centre for multiple SACs."""
    return [get_furthest_peak(nc) for nc in nearest_centroids]


def get_rotation_offset(furthest_peak, centre):
    """Calculate angle of major ellipsis axis from y-axis."""
    if furthest_peak is None:
        return None
    x_len = furthest_peak[1] - centre[1]
    y_len = furthest_peak[0] - centre[0]
    offset_maj = np.degrees(np.arctan2(x_len, y_len))
    if offset_maj < -90:
        offset_maj = 180 + offset_maj
    elif offset_maj > 90:
        offset_maj = -180 + offset_maj
    return offset_maj


def get_all_rotation_offsets(furthest_peaks, centre):
    """Calculate angle of major ellipsis axis from y-axis for multiple SACs."""
    return [get_rotation_offset(fp, centre) for fp in furthest_peaks]


def rotate_sac(sac, angle):
    """Rotate a single spatial autocorrelogram."""
    rotated_sac = np.copy(sac)
    if angle is not None:
        return scipy.ndimage.rotate(rotated_sac, -angle, reshape=False)
    else:
        return sac


def rotate_all_sacs(sacs, angles):
    """Rotate multiple spatial autocorrelograms."""
    if sacs.ndim == 2:
        sacs = np.expand_dims(sacs, axis=2)
    rotated_sacs = np.empty_like(sacs)
    for idx in range(sacs.shape[2]):
        rotated_sacs[..., idx] = rotate_sac(sacs[..., idx], angles[idx])
    return rotated_sacs


def check_rotated_sac_peak(rotated_sac, 
                           orig_centroids_by_dist,
                           new_threshold=0.1,
                           ):
    """Check whether # peaks in rotated SAC matches original #."""
    orig_peak_no = len(orig_centroids_by_dist)
    centre, _, all_centroids, _ = get_peak_properties(rotated_sac,
                                                      threshold=new_threshold)
    _, nearest_centroids = get_nearest_peaks(all_centroids, centre)
    rotated_peak_no = len(nearest_centroids)
    return orig_peak_no <= rotated_peak_no

def check_rotated_sac_peaks(rotated_sacs, 
                            orig_centroids_by_dist, 
                            new_threshold=0.05,
                            ):
    """Check whether # peaks in rotated SACs matches original #."""
    if rotated_sacs.ndim == 2:
        rotated_sacs = np.expand_dims(rotated_sacs, axis=2)
    return [check_rotated_sac_peak(rotated_sacs[..., idx], 
                                   orig_centroids_by_dist[idx],
                                   new_threshold=new_threshold)
            for idx in range(rotated_sacs.shape[2])]


def compute_closest_peaks(sacs, threshold=0.1):
    """Compute closest peaks for all SACs."""
    centre, all_centroids, _, _ = get_all_peak_properties(sacs)
    _, nearest_peaks = get_all_nearest_peaks(all_centroids, centre, threshold=threshold)
    return nearest_peaks


def get_major_axis(furthest_peak, centre):
    """Return angle and length of major axis of rotated SAC."""
    major_axis_length = np.abs(furthest_peak[0] - centre[0])
    if major_axis_length == 0:
        return None
    if np.isnan(major_axis_length) or np.isinf(major_axis_length):
        return None
    if major_axis_length is None:
        return None
    

def get_single_closest_peak(nearest_peaks):
    """Return nearest peak from centre for SAC."""
    return nearest_peaks[0]


def get_closest_peak_angle(nearest_peak, furthest_peak, centre):
    """Return angle between nearest and furthest peaks."""
    furthest_vector = furthest_peak - centre
    nearest_vector = nearest_peak - centre
    if furthest_vector[0] == nearest_vector[0] and furthest_vector[1] == nearest_vector[1]:
        return None
    if np.isnan(furthest_vector).any() or np.isnan(nearest_vector).any():
        return None
    if np.isinf(furthest_vector).any() or np.isinf(nearest_vector).any():
        return None
    if furthest_vector is None or nearest_vector is None:
        return None
    angle = abs(np.arccos(np.dot(furthest_vector, nearest_vector) /
                                 (np.linalg.norm(furthest_vector) *
                                  np.linalg.norm(nearest_vector))))
    if angle > np.pi:
        return np.pi*2 - angle
    return angle


def calculate_minor_axis_length(major_axis_len, angle, nearest_peak, centre):
    """Return length of minor axis of rotated SAC."""
    nearest_peak_dist = np.linalg.norm(nearest_peak - centre)
    numerator = (np.sin(angle)**2)
    denominator = nearest_peak_dist**(-2) - (np.cos(angle)**2 / major_axis_len**2)
    if denominator == 0 or numerator == 0:
        return None
    if np.isnan(denominator) or np.isnan(numerator) or np.isinf(denominator) or np.isinf(numerator):
        return None
    if denominator is None or numerator is None:
        return None
    minor_axis_length = np.sqrt(numerator / denominator)                        
    return np.abs(minor_axis_length)


def calculate_all_minor_axis_lengths(major_axis_lengths, angles, nearest_peaks, centre):
    """Return length of minor axis for multiple rotated SACs."""
    return [calculate_minor_axis_length(mal, angle, np, centre) 
            for mal, angle, np in zip(major_axis_lengths, angles, nearest_peaks)]


def get_scaling_factor(major_axis_length, minor_axis_length):
    """Return scaling factor for rescaling SAC."""
    return major_axis_length / minor_axis_length


def scale_sac(rotated_sac, scaling_factor):
    """Scale rotated SAC."""
    curr_sac = np.copy(rotated_sac)
    curr_sac.squeeze()
    scale_matrix = np.array([[1, 0], [0, scaling_factor]])
    scaled_sac = affine_transform(curr_sac, scale_matrix, missing=0.0, recenter=True)
    return scaled_sac


def circularise_sacs(sacs, outer_threshold=0.1):
    """Return circularised SACs."""
    if sacs.ndim == 2:
        sacs = np.expand_dims(sacs, axis=2)

    final_sacs = np.empty_like(sacs)

    centre, all_centroids, _, _ = get_all_peak_properties(sacs)
    _, nearest_peaks = get_all_nearest_peaks(all_centroids, centre)
    furthest_peaks = get_all_furthest_peaks(nearest_peaks)
    rotation_offsets = get_all_rotation_offsets(furthest_peaks, centre)
    rotated_sacs = rotate_all_sacs(sacs, rotation_offsets)
    
    peak_check = check_rotated_sac_peaks(rotated_sacs, 
                                         nearest_peaks, 
                                         new_threshold=outer_threshold,
                                         )

    for index, check in enumerate(peak_check):
        if not check:
            final_sacs[..., index] = sacs[..., index]
            continue
        
        curr_sac = rotated_sacs[..., index]
        _, _, curr_peaks, _ = get_peak_properties(curr_sac, threshold=outer_threshold)
        _, curr_nearest_peaks = get_nearest_peaks(curr_peaks, centre)
        closest_peak = get_single_closest_peak(curr_nearest_peaks)
        furthest_peak = get_furthest_peak(curr_nearest_peaks)

        if furthest_peak is None:
            final_sacs[..., index] = sacs[..., index]
            continue

        major_axis_len = get_major_axis(furthest_peak, centre)
        if major_axis_len is None: 
            final_sacs[..., index] = sacs[..., index]
            continue

        angle = get_closest_peak_angle(closest_peak, furthest_peak, centre)
        if angle is None:
            final_sacs[..., index] = sacs[..., index]
            continue

        minor_axis_len = calculate_minor_axis_length(major_axis_len, 
                                                     angle, 
                                                     closest_peak, 
                                                     centre,
                                                     )
        if minor_axis_len is None:
            final_sacs[..., index] = sacs[..., index]
            continue

        scaling_factor = get_scaling_factor(major_axis_len, minor_axis_len)

        if scaling_factor <= 2 and not np.isnan(scaling_factor):
            final_sacs[..., index] = scale_sac(curr_sac, scaling_factor)
        else:
            final_sacs[..., index] = sacs[..., index]

    return final_sacs


######################### Permutation functions #################################

def get_spike_permute_key(shape=(12,9), num_permutations=500):
    total_elements = shape[0] * shape[1]
    perm_range = np.arange(total_elements)
    #return (np.random.permutation(perm_range).reshape(shape) 
    #        for _ in range(num_permutations))
    perm_key = np.empty((total_elements, num_permutations))
    for idx in range(num_permutations):
        perm_key[..., idx] = np.random.permutation(perm_range)
    return perm_key.astype(int)


def spike_permute(fr_maps, state_counts, num_permutations=500):
    assert fr_maps.shape == state_counts.shape
    firing_rate_maps = np.copy(fr_maps)
    states = np.copy(state_counts)

    if firing_rate_maps.ndim == 2:
        firing_rate_maps = firing_rate_maps.reshape(firing_rate_maps.shape[0], 
                                                    firing_rate_maps.shape[1], 
                                                    1)
    
    n_timepoints = firing_rate_maps.shape[2]
    shape = firing_rate_maps.shape
    flat_frs = np.reshape(firing_rate_maps,
                          (shape[0]*shape[1], n_timepoints))
    perm_keys = get_spike_permute_key(shape[:2], num_permutations)
    
    permuted_frs = np.empty((shape[0], shape[1], shape[2], num_permutations)) 
    permuted_state_counts = np.empty((shape[0], shape[1], shape[2], num_permutations))
    
    for idx in range(num_permutations):
        permute = perm_keys[..., idx]
        for timepoint in range(shape[2]):
            permuted_flat_frs = flat_frs[..., timepoint][permute]
            permuted_frs[..., timepoint, idx] = permuted_flat_frs.reshape(shape[:2])
            permuted_state_counts[..., timepoint, idx] = states[..., timepoint].flatten()[permute].reshape(shape[:2])
        assert(permuted_frs.shape == permuted_state_counts.shape)
    return permuted_frs, permuted_state_counts


def get_session_permute_key(num_permutations=36, n_sess=37):
    
    key_array = np.zeros((n_sess, n_sess))
    
    for row, _ in enumerate(key_array):
        for col, _ in enumerate(key_array):
            key_array[row, col] = row + col
            if key_array[row, col] >= n_sess:
                key_array[row, col] -= n_sess
                
    return key_array[:,1:num_permutations+1]


def session_permute(num_permutations = 36,
                    session_info = '/Volumes/WilliamX6/discrete_maze/data/session_info.csv', 
                    timepoints=range(50,151,10),
                    epoch='optionMade',
                    inner_threshold=0.4,
                    elliptical=False,
                    outer_threshold=0.1, 
                    ):
    
    sessions = import_sessions.get_session_info(session_info=session_info)
    session_table = pd.read_csv(session_info, header=None, names=None)
    session_table["Session permuted standard grid scores"] = np.empty((len(session_table), 0)).tolist()
    session_table["Session permuted elliptical grid scores"] = np.empty((len(session_table), 0)).tolist()

    perm_key = get_session_permute_key(num_permutations=num_permutations)

    for k in range(num_permutations):
        print(f"Permutation {k+1} of {num_permutations}")
        for orig, perm in enumerate(perm_key[:, k]):
            orig_sess = sessions[orig]
            perm_sess = sessions[perm]
            
            # Get minimum trajectory length and crop permutation session trajectory
            traj_len = min(orig_sess['traj_len'], perm_sess['traj_len'])
            perm_sess['traj'] = perm_sess['traj'][0:traj_len]
            traj = perm_sess['traj']
            traj = np.array(traj)
            x_states = traj[:, 0]
            y_states = traj[:, 1]
            state_list = np.array(list(zip(x_states, y_states)))
            
            # Get cells, truncate and smooth FRs, and calculate autocorrelogram
            cells = orig_sess['cells']
                
            for cell in cells:
                
                area = session_table[1][cell]
                cell_in_area = session_table[2][cell]

                orig_frs, dist_change = ImportDm.getFR(area, cell_in_area, epoch)
                orig_frs = orig_frs[dist_change == 1]  # Shape is trials x timepoints
                if orig_frs.shape[0] > traj_len:
                    orig_frs = orig_frs[:traj_len, :]  # Now minimum trajectory length x timepoints
                if len(state_list) > orig_frs.shape[0]:
                    state_list = state_list[:orig_frs.shape[0], :]
                orig_frs, states = bin_data(orig_frs, state_list)
                orig_frs, states = orig_frs[..., timepoints], states[..., timepoints]

                sacs, mask_param, radii = get_sacs_from_frs_and_states(orig_frs,
                                                    states,
                                                    inner_threshold=inner_threshold,
                                                    elliptical=False,
                                                    outer_threshold=outer_threshold
                                                    )

                grid_score = get_all_grid_scores(sacs, mask_param, radii)
                session_table.loc[cell, "Session permuted standard grid scores"].append(grid_score)

                print(min(sacs.flatten()), max(sacs.flatten()))

                if elliptical:
                    try:
                        circularised_sacs = circularise_sacs(sacs, outer_threshold=0.1)
                        cleaned_sacs = clean_sacs(circularised_sacs)
                        centre, radii = find_peaks(cleaned_sacs)
                        mask_param = get_mask_parameters(centre, radii)
                        elliptical_grid_score = get_all_grid_scores(circularised_sacs, mask_param, radii)
                        session_table.loc[cell, "Session permuted elliptical grid scores"].append(elliptical_grid_score)
                    except:
                        session_table.loc[cell, "Session permuted elliptical grid scores"].append(grid_score)

    return session_table


def time_permute():
    pass


def block_time_permute():
    pass

    
def get_permuted_sacs(area,
                      cell,
                      epoch,
                      timepoints=range(50,151,10),
                      permute_func=spike_permute,
                      num_permutations=500,
                      inner_threshold=0.4,
                      elliptical=False,
                      outer_threshold=0.1,
                      **kwargs):
    fr_maps, state_counts = load_clean_bin_data(area, cell, epoch, timepoints=timepoints)
    perm_frs, perm_states = permute_func(fr_maps,
                                         state_counts,
                                         num_permutations=num_permutations,
                                        )

    sacs = []
    mask_params = []
    radii = []

    for perm in range(num_permutations):
        curr_sacs, curr_mask_param, curr_radii = get_sacs_from_frs_and_states(perm_frs[..., perm],
                                                                              perm_states[..., perm],
                                                                              inner_threshold=inner_threshold,
                                                                              elliptical=elliptical,
                                                                              outer_threshold=outer_threshold,
                                                                             ) 
        sacs.append(curr_sacs)
        mask_params.append(curr_mask_param)
        radii.append(curr_radii)
    return sacs, mask_params, radii


######################### Gridscoring functions #################################
        
"""
Circular grid scoring approach based on
https://github.com/google-deepmind/grid-cells/blob/master/scores.py
"""

import scipy.signal

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
        if scores_60[max_60_ind] is not None:
            return scores_60[max_60_ind] 
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
    raw_frs, dist_change, to_x, to_y = load_data(area, cell, epoch, timepoints=timepoints)
    firing_rates, states = clean_data(raw_frs, dist_change, to_x, to_y)
    fr_by_states, state_counts = bin_data(firing_rates, states)
    smoothed_data = manual_smoother(fr_by_states, state_counts)
    sacs = make_sacs(smoothed_data)
    
    print(f"{area} {cell} {epoch} {elliptical}, raw")

    if not elliptical:
        cleaned_sacs = clean_sacs(sacs)
        centre, radii = find_peaks(cleaned_sacs)
        mask_parameters = get_mask_parameters(centre, radii)
        return get_all_grid_scores(sacs, mask_parameters, radii)
    
    circularised_sacs = circularise_sacs(sacs, outer_threshold=0.1)
    cleaned_sacs = clean_sacs(circularised_sacs)
    centre, radii = find_peaks(cleaned_sacs)
    mask_parameters = get_mask_parameters(centre, radii)
    return get_all_grid_scores(circularised_sacs, mask_parameters, radii)

def perm_gridscore(area,
                   cell,
                   epoch,
                   timepoints=range(50,151,10),
                   num_permutations=500,
                   elliptical=False,
                   outer_threshold=0.1,
                   ):
    sacs, mask_params, radii = get_permuted_sacs(area,
                                                 cell,
                                                 epoch,
                                                 timepoints=timepoints,
                                                 elliptical=elliptical,
                                                 num_permutations=num_permutations,
                                                 outer_threshold=outer_threshold,
                                                 )
    all_grids_scores = []

    print(f"{area} {cell} {epoch} {elliptical}, perm")

    for perm in range(num_permutations):
        grid_scores = get_all_grid_scores(sacs[perm],
                                          mask_params[perm],
                                          radii[perm],
                                         )
        all_grids_scores.append(grid_scores)
    return all_grids_scores
