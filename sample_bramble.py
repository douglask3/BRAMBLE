import matplotlib.pyplot as plt

import numpy  as np
import pandas as pd
import seaborn as sns

import math
import numbers

import pytensor.tensor as tt
import pymc  as pm
import arviz as az

from pdb import set_trace

import sys
sys.path.append('model/')
from BRAMBLE import BRAMBLE
from fake_data_simulator import *






file = 'data/SH_allometry_2023.csv'
df_measurements = pd.read_csv(file)
df_measurements['height'] = df_measurements[['h1', 'h2', 'h3']].mean(axis=1)
df_measurements['h_std'] = df_measurements[['h1', 'h2', 'h3']].std(axis=1)
df_measurements['diameter'] = df_measurements[['d1', 'd2', 'd3']].mean(axis=1)
df_measurements['d_std'] = df_measurements[['d1', 'd2', 'd3']].std(axis=1)
    
#df_measurements['Y_fake'] = fake_data_simulator(df_measurements)

params = {}
model = BRAMBLE(params, inference=False)
out = model.compute(heights=df_measurements['height'], widths=df_measurements['diameter'], rings=None, stems=None)

set_trace()
