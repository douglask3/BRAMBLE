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

def MaxEnt(Y, fx):
    fx = 1 - tt.exp(-fx)
    Y = 1 - tt.exp(-Y)
    return Y*tt.log(fx) + (1.0-Y)*tt.log((1-fx))



def train_pymc(df, varnames = ['height', 'diameter'], catogory = 'species'):
    # Define the number of predictor variables
    X = df[varnames].values/100.0  # Predictor variables (height and diameter)
    species = df[catogory].values       # Species labels
    species_idx = df[catogory].apply(lambda x: np.where(unique_species == x)[0][0]).values
    nvars = X.shape[1]

    # Map each species name to a unique index for model use
    unique_species = species.unique()
    species_idx = df['species'].apply(lambda x: np.where(unique_species == x)[0][0]).values
    biomass_threshold = df['Y_fake'].max()*2
    # In the PyMC model
    with pm.Model() as max_ent_model:
        # Priors for hierarchical parameters
        mu_betas = pm.Normal('mu_betas', mu=0, sigma=1, shape=nvars)
        sigma_betas = pm.HalfNormal('sigma_betas', sigma=1, shape=nvars)
        mu_beta0 = pm.Normal('mu_beta0', mu=0, sigma=1)
        sigma_beta0 = pm.HalfNormal('sigma_beta0', sigma=1)
        
        # Species-specific parameters
        betas = pm.LogNormal('betas', mu=mu_betas, sigma=sigma_betas, 
                             shape=(len(unique_species), nvars))
        beta0 = pm.Normal('beta0', mu=mu_beta0, sigma=sigma_beta0, shape=len(unique_species))
        
        # Set up params dictionary for BRAMBLE instance
        params = {'betas': betas, 'beta0': beta0}
            
        # Create the BRAMBLE instance with species-specific parameters
        allometry_model = BRAMBLE(params, inference = True)
    
        # Compute predictions using the multi-species method
        prediction = allometry_model.predict_multi_species(X, species_idx)
        penalty = pm.Potential('penalty', 
                               -0.1 * pm.math.sum(pm.math.switch(prediction > biomass_threshold,
                                                                 prediction - biomass_threshold,
                                                                 0)))   
        # Define likelihood
        error = pm.DensityDist("error", prediction,
                               logp=MaxEnt, 
                               observed=df['Y_fake'])
    
        # Sampling
        #start = pm.find_MAP()
        trace = pm.sample(1000, return_inferencedata=True, chains=2, cores=2)

    return trace

if __name__ == "__main__":
    file = 'data/SH_allometry_2023.csv'
    df_measurements = pd.read_csv(file)
    df_measurements['height'] = df_measurements[['h1', 'h2', 'h3']].mean(axis=1)
    df_measurements['h_std'] = df_measurements[['h1', 'h2', 'h3']].std(axis=1)
    df_measurements['diameter'] = df_measurements[['d1', 'd2', 'd3']].mean(axis=1)
    df_measurements['d_std'] = df_measurements[['d1', 'd2', 'd3']].std(axis=1)
    
    df_measurements['Y_fake'] = fake_data_simulator(df_measurements)


