import pandas as pd
import numpy as np
from src import ImportDm
import warnings

warnings.filterwarnings('ignore')

def get_sessions(session_info='/Volumes/WilliamX6/discrete_maze/data/session_info.csv'):
    
    unique_sesh = []
    unique_cell = []
    unique_area = []
    unique_date = []
    
    sesh = pd.read_csv(session_info, header=None, names=None)
    
    for i, c in sesh.iterrows():
        if c[3] not in unique_date:
            unique_sesh.append(c[0])
            unique_area.append(c[1])
            unique_cell.append(c[2])
            unique_date.append(c[3])
            
    unique_sesh = np.array(unique_sesh)
    unique_cell = np.array(unique_cell)
    unique_area = np.array(unique_area)
    unique_date.sort()
    unique_date = np.array(unique_date)
    
    sessions = {}
    
    for i, date in enumerate(unique_date):
        
        sesh_dict = {}
        sesh_dict['date'] = int(date)
        
        idx = sesh.index[sesh[3] == date]
        curr_sesh = sesh.loc[idx]
        cells = curr_sesh[0]
        sesh_dict['cells'] = list(cells)
        
        areas = curr_sesh[1].unique()
        
        for area in areas:
            
            area_idx = curr_sesh.index[curr_sesh[1] == area]
            curr_area = curr_sesh.loc[area_idx]
            area_cells = curr_area[2]
            sesh_dict[area] = list(area_cells)
            
        sessions[i] = sesh_dict
                     
    return sessions


def get_traj(area, cell):
    
    (dist_change,    # For this step, what is the new distance to target compared to the previous step (+1 = moved towards target)
     currAngle,     # Angle between current location and target
     hd,            # What direction did they move (north south east west)
     numsteps,      # In this trial, how many steps have they taken?
     perfTrials,    # Was this trial perfect? I.e. all steps were towards the target
     startAngle,    # Starting angle to the target
     currDist,      # Current distance to the target
     from_x,         # X-coordinate of state they just moved from
     from_y,         # Y-coordinate of ....
     to_x,           # X-coordinate of state they have just chosen
     to_y,           # Y-coordinate of ...
     ) = ImportDm.getBehavData(area, cell)
    
    m = dist_change == 1
    to_x = to_x[m]
    to_y = to_y[m]
    traj = list(zip(from_x, from_y))
            
    return traj   
        

def get_session_info(session_info='/Volumes/WilliamX6/discrete_maze/data/session_info.csv'):

    sessions = get_sessions(session_info)
            
    for sess in sessions.keys():
        curr_sess = sessions[sess]
        area = list(curr_sess.keys())[2]
        cell = curr_sess[area][0]
        
        curr_traj = get_traj(area, cell)
        traj_len = len(curr_traj)
        
        sessions[sess]['traj'] = curr_traj
        sessions[sess]['traj_len'] = traj_len
        
    return sessions    

if __name__ == '__main__':
    sessions = get_session_info()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    