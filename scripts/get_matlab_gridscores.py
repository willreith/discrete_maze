import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import src.sac_funcs as sf
import src.permute_funcs as pf
import matlab.engine

epochs = ['optionMade', 'optionsOn']
timestamps = {'optionMade':100, 'optionsOn':130}

class SAC():
    def __init__(self, area, cell, epoch, timepoints=range(50, 151, 10)):
        self.area = area
        self.cell = cell
        self.epoch = epoch
        self.gridscores = {}
        self.timepoints=timepoints
        
    def get_sac(self):
        self.sacs, _, _ = sf.get_sacs(self.area, self.cell, self.epoch, self.timepoints)
        
    def get_gridscore(self):
        pass
        # matlab engine (?)
    
    def get_perm_gridscores(self):
        pass
        # matlab engine (?)
    
    def save(self, fname):
        pass
    
    
test_cell = SAC('Area32', 1, 'optionMade')
test_cell.get_sac()
test_sac = test_cell.sacs[..., 5].tolist()
    
eng = matlab.engine.start_matlab()

matlab_array = matlab.double(test_sac)
output = eng.compute_grid_score(matlab_array, "langston", 0.4)

eng.quit()
    
    
"""
start matlab engine


for epoch in epochs:
    for cell in cells:
        data = import_data.get_data()
        


"""