import matplotlib.pyplot as plt

import numpy  as np
import pandas as pd
import arviz as az
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



def train_pymc(df, Y_varnameX_varnames = ['height', 'diameter'], catogory = 'species'):
    # Define the number of predictor variables
    X = df[varnames].values/100.0  # Predictor variables (height and diameter)
    species = df[catogory].values       # Species labels
    species_idx = df[catogory].apply(lambda x: np.where(unique_species == x)[0][0]).values
    nvars = X.shape[1]

    # Map each species name to a unique index for model use
    unique_species = df[catogory].unique()
    species_idx = species.apply(lambda x: np.where(unique_species == x)[0][0]).values
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

    
    file = 'data/shrewberry_hill.csv'
    #file = 'data/
    df = pd.read_csv(file)

    model = BRAMBLE(params={}, inference=True)
    
    data = {
        'heights': df['height mean'].to_numpy()/100.0,      # can contain np.nan
        'widths': df['diameter mean (field measurement'].to_numpy()/100.0,        # can contain np.nan
        #'rings': df['Tree rings'].to_numpy(),
        'carbon_mass': df['total shrub  Carbon estimate (kg)'].to_numpy()  
            # must be provided (np.nan ok for some rows)
    }
    
    bramble = BRAMBLE(params={}, inference=True)
    pymc_model = bramble.build_pymc_model(data)

    # sample
    with pymc_model:
        trace = pm.sample(
            draws=400, tune=400,
            target_accept=0.9,
            chains=2)
        ppc = pm.sample_posterior_predictive(trace, var_names=["carbon_obs"])
    az.to_netcdf(trace, "bramble_trace.nc") 


    import matplotlib.pyplot as plt
    import numpy as np

    # Suppose df['carbon mass'] holds observed values (with NaN for missing)
    observed = data['carbon_mass']
    
    # Posterior predictive draws for carbon
    # Shape: (n_draws, n_obs)
    
    y_pred = ppc.posterior_predictive["carbon_obs"].values.T
    
    y_pred = y_pred.reshape(y_pred.shape[0], -1)
    # Mask rows with missing observed carbon
    mask = ~np.isnan(observed)
    obs = observed[mask]
    
    pred = y_pred[mask, :]   # shape (n_draws, n_masked_obs)
    
    # Plot
    plt.figure(figsize=(6,6))
    #set_trace()
    plt.scatter(obs, pred.mean(axis=1), alpha=0.6, label="Posterior mean prediction")
    
    # Add uncertainty: 5–95% intervals for each obs
    low, high = np.percentile(pred, [5, 95], axis=1)
    plt.errorbar(obs, pred.mean(axis=1),
                 yerr=[pred.mean(axis=1)-low, high-pred.mean(axis=1)],
                 fmt="o", color="tab:blue", alpha=0.4)
    
    # 1:1 line
    #lims = [min(obs.min(), pred.mean(axis=1).min()), max(obs.max(), pred.mean(axis=1).max())]
    #plt.plot(lims, lims, "k--", alpha=0.7)
    
    plt.xlabel("Observed carbon mass")
    plt.ylabel("Predicted carbon mass")
    plt.legend()
    plt.title("Observed vs Posterior Predictive Carbon")
    plt.show()
