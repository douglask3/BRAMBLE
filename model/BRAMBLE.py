import numpy  as np
import pandas as pd

import math
import numbers

import pytensor.tensor as tt
import pymc  as pm

from pdb import set_trace

class BRAMBLE:
    def __init__(self, params, inference=False):
        if inference:
            self.numPCK = __import__('pytensor').tensor
        else:
            self.numPCK = __import__('numpy')
        
        self.params = params

    def predict_single_species(self, X, betas, beta0, apply_exp_transform = True):
        """
        Predict biomass for a batch of inputs (e.g., for multiple samples or species).
        
        Parameters
        ----------
        X : np.ndarray
            Input matrix (N, M), where rows are samples and columns are features.
        betas : np.ndarray
            Array of coefficients for each species (N, M).
        beta0 : np.ndarray
            Array of intercepts for each species (N,).
        apply_exp_transform : bool
            Whether to apply exponential transformation.

        Returns
        -------
        np.ndarray
            Predicted values for each input in X.
        """
        pred = self.numPCK.sum(X * betas, axis=1) + beta0
        if apply_exp_transform:
            pred = self.numPCK.exp(pred)
        return pred
    
    def predict_multi_species(self, X, species_idx, apply_exp_transform=True):
        """
        Predict biomass for multiple species using species-specific parameters.
        
        Parameters
        ----------
        X : np.ndarray
            Input matrix (N, M) of features.
        species_idx : np.ndarray
            Indices indicating the species of each sample.
        apply_exp_transform : bool
            Whether to apply exponential transformation.

        Returns
        -------
        np.ndarray
            Predictions for each row in X.
        """
        # Gather species-specific parameters based on indices
        if isinstance(self.params['betas'], dict):
            betas_species = np.array([self.params['betas'][id] for id in species_idx])
            beta0_species = np.array([self.params['beta0'][id] for id in species_idx])
        else:
            betas_species = self.params['betas'][species_idx]
            beta0_species = self.params['beta0'][species_idx]
            
        # Call predict_single_species for batch processing
        return self.predict_single_species(X, betas_species, beta0_species, apply_exp_transform=apply_exp_transform)

