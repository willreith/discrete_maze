import numpy as np
import pandas as pd

from src import all_funcs as af

df = af.session_permute(num_permutations=1, elliptical=True, timepoints=range(50,155,5))