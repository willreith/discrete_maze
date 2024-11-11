import numpy as np
import seaborn as sb
import scipy.ndimage

from skimage.measure import label, regionprops
from sunpy.image.transform import affine_transform

from src import sac_funcs as sf

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
    return np.abs(furthest_peak[0] - centre[0])


def get_all_major_axes(furthest_peak, centre):
    """Return angle and length of major axis for multiple rotated SACs."""
    return [get_major_axis(fp, centre) for fp in furthest_peak]


def get_single_closest_peak(nearest_peaks):
    """Return nearest peak from centre for SAC."""
    return nearest_peaks[0]


def get_all_single_closest_peak(nearest_peaks, centre):
    """Return nearest peak from centre for multiple SACs."""
    return [get_single_closest_peak(np) for np in nearest_peaks]


def get_closest_peak_angle(nearest_peak, furthest_peak, centre):
    """Return angle between nearest and furthest peaks."""
    furthest_vector = furthest_peak - centre
    nearest_vector = nearest_peak - centre
    if furthest_vector[0] == nearest_vector[0] and furthest_vector[1] == nearest_vector[1]:
        return None
    angle = abs(np.arccos(np.dot(furthest_vector, nearest_vector) /
                                 (np.linalg.norm(furthest_vector) *
                                  np.linalg.norm(nearest_vector))))
    if angle > np.pi:
        return np.pi*2 - angle
    return angle


def get_all_closest_peak_angles(nearest_peaks, furthest_peaks, centre):
    """Return angle between nearest and furthest peaks for multiple SACs."""
    return [get_closest_peak_angle(np, fp, centre) 
            for np, fp in zip(nearest_peaks, furthest_peaks)]


def calculate_minor_axis_length(major_axis_len, angle, nearest_peak, centre):
    """Return length of minor axis of rotated SAC."""
    nearest_peak_dist = np.linalg.norm(nearest_peak - centre)
    numerator = (np.sin(angle)**2)
    denominator = nearest_peak_dist**(-2) - (np.cos(angle)**2 / major_axis_len**2)
    if denominator == 0 or numerator == 0:
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


def get_all_scaling_factors(major_axis_lengths, minor_axis_lengths):
    """Return scaling factor for multiple SACs."""
    return [get_scaling_factor(mal, mil) 
            for mal, mil in zip(major_axis_lengths, minor_axis_lengths)]


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


