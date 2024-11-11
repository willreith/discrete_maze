import numpy as np
from src import compute_ac

class autocorrelogram():
    def __init__(self, area, cell, epoch):
        self.area = area
        self.cell = cell
        self.epoch = epoch
        self.timepoints = range(55,150,10)
        self.manhattan_smoothing = 0.5
        self.diagonal_smoothing = 0.25
        self.ac = compute_ac.make_autocorrelogram(self.area, self.cell, 
                                                  self.epoch, self.timepoints,
                                                  (self.manhattan_smoothing,
                                                  self.diagonal_smoothing))
        self.ac_clean = None

    def update_ac(self):
        self.ac = compute_ac.make_autocorrelogram(self.area, self.cell, 
                                                  self.epoch, self.timepoints,
                                                  self.manhattan_smoothing,
                                                  self.diagonal_smoothing)

    def set_smoothing(self, scaling):        
        """Set smoothing based on two-tuple smoothing params."""
        assert(0 <= scale <= 1 for scale in scaling)
        self.manhattan_smoothing = scaling[0]
        self.diagonal_smoothing = scaling[1]
        self.update_ac(self)

    def set_timepoints(self, timepoints):
        """Set timepoints to compute AC by range specifying indices."""
        self.timepoints = timepoints
        self.update_ac(self)

    def clean_ac(self):
        self.ac_clean = compute_ac.clean_ac(self.ac)

    def find_central_peak(self):
        if self.ac_clean is None:
            self.clean_ac()
        # Assuming you need to handle centroids for each slice
        self.centroids = self.ac_clean
        return self.centroids


    """
    def clean_ac(self, threshold = 0.1):
        if self.ac.ndim > 2:
            self.ac = np.moveaxis(self.ac, 2, 0)
            self.ac_clean = [compute_ac.clean_ac(ind_ac, threshold=threshold)
                             for ind_ac in self.ac]
        else:
            self.ac_clean = compute_ac.clean_ac(self.ac, threshold=threshold)
        #return self.clean_ac

    def find_central_peak(self):
        if self.ac_clean is None:
            self.clean_ac()
        elif isinstance(self.ac_clean, list):
            save_acs = np.empty((self.size[0], self.size[1], len(self.ac_clean)))
            save_acs = np.empty((self))
        self.centre, self.centroid = compute_ac.find_central_peak(self.ac_clean)
        
"""