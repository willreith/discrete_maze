import numpy as np

from src import sac_funcs as sf

def get_spike_permute_key(shape=(12,9), num_permutations=500):
    total_elements = shape[0] * shape[1]
    perm_range = np.arange(total_elements)
    #return (np.random.permutation(perm_range).reshape(shape) 
    #        for _ in range(num_permutations))
    for _ in range(num_permutations):
        yield np.random.permutation(perm_range)


def spike_permute(fr_maps, state_counts, num_permutations=500):
    assert fr_maps.shape == state_counts.shape
    firing_rate_maps = np.copy(fr_maps)
    states = np.copy(state_counts)

    if firing_rate_maps.ndim == 2:
        firing_rate_maps = firing_rate_maps.reshape(firing_rate_maps.shape[0], 
                                                    firing_rate_maps.shape[1], 
                                                    1)
    
    if states.ndim == 3:
        states = states[..., 0].squeeze()
    n_timepoints = firing_rate_maps.shape[2]
    shape = firing_rate_maps.shape
    flat_frs = np.reshape(firing_rate_maps,
                          (shape[0]*shape[1], n_timepoints))
    perm_key = get_spike_permute_key(shape[:2], num_permutations)
    
    for _ in range(num_permutations):
        permuted_frs = np.empty((shape[0], shape[1], n_timepoints))
        permute = next(perm_key)
        for timepoint in range(shape[2]):
            permuted_flat_frs = flat_frs[..., timepoint][permute]
            permuted_frs[..., timepoint] = permuted_flat_frs.reshape(shape[:2])
        permuted_state_counts = states.flatten()[permute].reshape(shape[:2])
        permuted_state_counts = np.repeat(states[..., np.newaxis], n_timepoints, axis=2)
        assert(permuted_frs.shape == permuted_state_counts.shape)
        yield permuted_frs, permuted_state_counts


def session_permute(sacs, ordered=True):
    pass


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
    fr_maps, state_counts = sf.load_clean_bin_data(area, cell, epoch, timepoints=timepoints)
    perm_frs_and_states = permute_func(fr_maps,
                                       state_counts,
                                       num_permutations=num_permutations,
                                       )

    for _ in range(num_permutations):
        perm_frs, perm_states = next(perm_frs_and_states)
        sacs, mask_param, radii = sf.get_sacs_from_frs_and_states(perm_frs,
                                                                  perm_states,
                                                                  inner_threshold=inner_threshold,
                                                                  elliptical=elliptical,
                                                                  outer_threshold=outer_threshold,
                                                                 ) 
        yield sacs, mask_param, radii