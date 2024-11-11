import numpy as np
from scipy.ndimage import convolve
from scipy.signal import correlate2d
from skimage.measure import label, regionprops
from src import ImportDm

class DataLoader:
    """Class to load data."""
    def __init__(self, area, cell, epoch):        
        self.area = area
        self.cell = cell
        self.epoch = epoch
        self.firing_rates = None

    def load_data(self):
        """Load firing rate, distance change and location data"""
        (self.firing_rates,  
        self.dist_change,    
        self.to_x,           
        self.to_y,           
        ) = ImportDm.getData(self.area, self.cell, self.epoch)


class DataCleaner:
    """Class to clean data"""
    def __init__(self, data):
        """Takes a DataLoader object as argument."""
        self.data = data 

    def clean_data(self):
        """Cleans firing rate and state data"""
        good_trials = self.data.dist_change == 1
        self.firing_rates = self.data.firing_rates[good_trials]
        state_x = np.array(self.data.to_x[good_trials], dtype=int, copy=True)
        state_y = np.array(self.data.to_y[good_trials], dtype=int, copy=True)
        self.states = np.array(list(zip(state_x, state_y)))


class Permuter:
    def __init__(self):
        pass


class DataBinner:
    def __init__(self, data):
        """Takes clean data (DataCleaner or Permuter class) object"""
        self.data = data

    def bin_data(self):
        """Bins data by state"""
        dim_x = np.max(self.data.states[:, 0])
        dim_y = np.max(self.data.states[:, 1])
        state_counts = np.zeros((dim_x + 1, dim_y + 1, 
                                 len(self.data.firing_rates[0])))
        firing_rates_by_state = np.zeros((dim_x + 1, dim_y + 1, \
                                        len(self.data.firing_rates[0])))

        for trial, (x_loc, y_loc) in enumerate(self.data.states):
            state_counts[x_loc, y_loc, :] += 1
            firing_rates_by_state[x_loc, y_loc, :] += \
                    self.data.firing_rates[trial]

        self.firing_rates_by_state = firing_rates_by_state
        self.state_counts = state_counts


class DataSmoother:
    def __init__(self, binned_data, scaling=(0.5, 0.25), smoothing_type="kernel"):
        """Takes a DataBinner object"""
        self.data = binned_data
        self.manhattan_scaling = scaling[0]
        self.diagonal_scaling = scaling[1]
        self.smoothing_type = smoothing_type

    def smooth(self):
        """Manual or kernel smooth data defined by scaling parameters."""
        if self.smoothing_type == "kernel":
            kernel = np.array([
                [self.diagonal_scaling, self.manhattan_scaling, self.diagonal_scaling],
                [self.manhattan_scaling, 1, self.manhattan_scaling],
                [self.diagonal_scaling, self.manhattan_scaling, self.diagonal_scaling]
            ])

            kernel = kernel[:, :, None]
            smoothed_counts = convolve(self.data.state_counts, kernel, mode='constant', # FIXME: change constant to mirror?
                                    cval=0)
            smoothed_frs = convolve(self.data.firing_rates_by_state *                   # FIXME: change constant to mirror?
                                    self.data.state_counts, kernel, mode='constant', cval=0)

            # Calculate the smoothed firing rates
            self.firing_rates = smoothed_frs / smoothed_counts

        elif self.smoothing_type == "manual":
            counts = np.copy(self.data.state_counts)
            frs = np.copy(self.data.firing_rates_by_state)

            for i in range(len(counts)):
                for j in range(len(counts[0])):
                    if i>0 and j>0:
                        counts[i,j] += self.data.state_counts[i-1,j-1] * self.diagonal_scaling
                        frs[i,j] += self.data.firing_rates_by_state[i-1,j-1] * self.diagonal_scaling
                    if i>0:
                        counts[i,j] += self.data.state_counts[i-1,j] * self.manhattan_scaling
                        frs[i,j] += self.data.firing_rates_by_state[i-1,j] * self.manhattan_scaling
                    if i>0 and j<len(counts[0])-1:
                        counts[i,j] += self.data.state_counts[i-1,j+1] * self.diagonal_scaling
                        frs[i,j] += self.data.firing_rates_by_state[i-1,j+1] * self.diagonal_scaling
                    if i<len(counts)-1 and j<len(counts[0])-1:
                        counts[i,j] += self.data.state_counts[i+1,j+1] * self.diagonal_scaling
                        frs[i,j] += self.data.firing_rates_by_state[i+1,j+1] * self.diagonal_scaling
                    if i<len(counts)-1:
                        counts[i,j] += self.data.state_counts[i+1,j] * self.manhattan_scaling
                        frs[i,j] += self.data.firing_rates_by_state[i+1,j] * self.manhattan_scaling
                    if i<len(counts)-1 and j>0:
                        counts[i,j] += self.data.state_counts[i+1,j-1] * self.diagonal_scaling
                        frs[i,j] += self.data.firing_rates_by_state[i+1,j-1] * self.diagonal_scaling
                    if j<len(counts[0])-1:
                        counts[i,j] += self.data.state_counts[i,j+1] * self.manhattan_scaling
                        frs[i,j] += self.data.firing_rates_by_state[i,j+1] * self.manhattan_scaling
                    if j>0:
                        counts[i,j] += self.data.state_counts[i,j-1] * self.manhattan_scaling
                        frs[i,j] += self.data.firing_rates_by_state[i,j-1] * self.manhattan_scaling
            frs /= counts
            self.firing_rates = frs



class Autocorrelograms:
    """Class to make autocorrelograms from binned firing rate data."""
    def __init__(self, smoothed_data, bins=range(50,151,10)):
        self.bins = bins
        self.firing_rates = smoothed_data.firing_rates[..., bins]
        self.dim1, self.dim2 = (x*2 - 1 for x in self.firing_rates.shape[:2])

    def autocorrelate(self):
        """Autocorrelate smoothed firing rates"""
        self.autocorrelograms = np.empty((self.dim1, self.dim2, len(self.bins)))
        for time_index in range(len(self.bins)):
            curr_fr = self.firing_rates[..., time_index]
            ac_init = correlate2d(curr_fr, curr_fr)
            ac_max = np.max(ac_init)
            self.autocorrelograms[..., time_index] = ac_init / ac_max
            

class CleanAutocorrelogram:
    """Class to clean autocorellograms (remove nan, inf, low corr)."""
    def __init__(self, autocorrelograms, threshold = 0.4):
        self.threshold = threshold
        self.autocorrelograms = np.copy(autocorrelograms.autocorrelograms)
        self.bins = len(autocorrelograms.bins)

    def clean_autocorrelograms(self):
        """Clean ACs by setting nan, inf, subthreshold values to 0"""
        for bin in range(self.bins):
            curr_ac = self.autocorrelograms[..., bin]
            curr_ac[np.isnan(curr_ac)] = 0
            curr_ac[np.isinf(curr_ac)] = 0
            curr_ac[curr_ac < self.threshold] = 0
            self.autocorrelograms[..., bin] = curr_ac
            

class PeakFinder:
    """Class to get radius of central peak."""
    def __init__(self, clean_autocorrelograms):
        self.autocorrelograms = clean_autocorrelograms.autocorrelograms
        self.bins = clean_autocorrelograms.bins
        self.centre = ((self.autocorrelograms.shape[0]-1) / 2, 
                       (self.autocorrelograms.shape[1]-1) / 2)
        
    def find_radius(self):
        """Find radius of central peak"""
        self.radii = np.empty((self.bins,1))
        self.region_properties = []
        labeled_ac = label(self.autocorrelograms > 0)
        for bin in range(self.bins):
            props = regionprops(labeled_ac[..., bin])
            centroids = np.array([prop.centroid for prop in props])
            diameters = [prop.equivalent_diameter_area for prop in props]
            centre_index = int(len(centroids)/2 + 0.5) - 1
            self.radii[bin] = diameters[centre_index] / 2
            self.region_properties.append(props)


class MaskParameters:
    """Class for getting radii for circular masks."""
    def __init__(self, peak_data, step_size = 0.45):
        self.centre = peak_data.centre
        self.max_radius = np.min(self.centre)
        self.bins = peak_data.bins
        self.radii = peak_data.radii
        self.step_size = step_size

    def make_parameters(self):
        """Get all radii of circular masks"""
        self.min_dist = [rad * 1.5 for rad in self.radii]
        self.expanding_radii = [np.arange(min_rad, self.max_radius - rad, 
                                          self.step_size) 
                                for min_rad, rad in zip(self.min_dist,
                                                        self.radii)]

        self.mask_parameters = []
        for bin in range(self.bins):
            if self.expanding_radii[bin].size == 0:
                self.mask_parameters.append([(np.inf, np.inf)])
            else:
                centre_rad = self.radii[bin]
                curr_parameters = [(rad - centre_rad, rad + centre_rad) 
                                   for rad in self.expanding_radii[bin]]
                filtered_parameters = [param for param in curr_parameters
                                       if param[0] >= centre_rad]

                self.mask_parameters.append(filtered_parameters)
