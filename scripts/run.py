import numpy as np
import pandas as pd
import multiprocessing
import time

from functools import partial

from src import ImportDm
from src import all_funcs as af

areas = "Area32", "dACC", "OFC"
cells = {'Area32':92,'dACC':162, 'OFC':143}
epochs=("optionMade", "optionsOn")
timepoints = range(50,151,5)

df = pd.read_csv("../data/session_info.csv", names=["Global cell no.", "Area", "Area cell no.", "Date"])
new_columns = ["Standard grid scores, optionMade", 
               "Elliptical grid scores, optionMade", 
               "Permuted standard grid scores, optionMade", 
               "Permuted elliptical grid scores, optionMade",
                "Standard grid scores, optionsOn",
                "Elliptical grid scores, optionsOn",
                "Permuted standard grid scores, optionsOn",
                "Permuted elliptical grid scores, optionsOn",
               ]
for col in new_columns:
    df[col] = np.nan
    df[col] = df[col].astype(object)

def get_gridscores(area="Area32", cell=33, epochs=epochs):
    start_time = time.time()
    exists = False
    results = {}
    for epoch in epochs:
        curr_index = df[(df["Area"] == area) & (df["Area cell no."] == cell)].index[0]
        circ = af.gridscore(area, cell, epoch, timepoints=timepoints)   
        ellip = af.gridscore(area, cell, epoch, elliptical=True, timepoints=timepoints)
        perm_circ = af.perm_gridscore(area, cell, epoch, timepoints=timepoints)
        perm_ellip = af.perm_gridscore(area, cell, epoch, elliptical=True, timepoints=timepoints)
        if exists:
            results[curr_index].update({
                8: circ,
                9: ellip,
                10: perm_circ,
                11: perm_ellip
            })
        else:
            results[curr_index] = {
                4: circ,
                5: ellip,
                6: perm_circ,
                7: perm_ellip
            }
        exists = True
    print(f"Time taken: {time.time() - start_time}")
    return results


def update_dataframe(results):
    for index, values in results.items():
        for col_idx, value in values.items():
            df.iat[index, col_idx] = value

if __name__ == '__main__':
    parallel = True
    results = []

    for area in ImportDm.areas[::-1]:
        print(area)
        if parallel:
            f = partial(get_gridscores, area)
            with multiprocessing.Pool(6) as pool:
                result_list = pool.map(f, range(cells[area]))            
                pool.close()
                pool.join()
                results.extend(result_list)
        else:
            for cell in range(ImportDm.n[area]):
                result = get_gridscores(area, cell, epochs)
                results.append(result)

    for result in results:
        update_dataframe(result)