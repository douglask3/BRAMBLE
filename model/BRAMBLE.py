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


if __name__ == "__main__":
    ## EXAMPLE
    import matplotlib.pyplot as plt

    # Define parameters for the BRAMBLE model
    params = {
        'betas': np.array([[0.1, 0.05], [0.15, 0.03], [0.2, 0.1]]),  # coefficients for 3 species and 2 features
        'beta0': np.array([1.0, 1.2, 1.5])  # intercepts for each species
    }

    # Initialize the BRAMBLE model with the parameters
    bramble_model = BRAMBLE(params)

    # Example input data: 10 samples, 2 features (e.g., height and diameter), with species indices
    X = np.random.rand(10, 2) * [100, 50]  # Random data for features
    species_idx = np.random.choice([0, 1, 2], size=10)  # Random species assignments for each sample

    # Make predictions
    predictions = bramble_model.predict_multi_species(X, species_idx)

    # Define species colors for the scatter plot
    species_colors = {0: 'blue', 1: 'green', 2: 'purple'}
    species_labels = {0: 'Species 0', 1: 'Species 1', 2: 'Species 2'}

    # Plotting each feature against biomass, colored by species
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for species in np.unique(species_idx):
        # Scatter plot for Feature 1 vs Biomass for each species
        axs[0].scatter(
            X[species_idx == species, 0], predictions[species_idx == species], 
            label=species_labels[species], color=species_colors[species], alpha=0.7, edgecolor='k'
        )
        axs[0].set_xlabel("Feature 1 (e.g., Diameter)")
        axs[0].set_ylabel("Predicted Biomass")
        axs[0].set_title("Feature 1 vs Predicted Biomass")
        axs[0].set_yscale("log")
        axs[0].grid(True)

        # Scatter plot for Feature 2 vs Biomass for each species
        axs[1].scatter(
            X[species_idx == species, 1], predictions[species_idx == species], 
            label=species_labels[species], color=species_colors[species], alpha=0.7, edgecolor='k'
        )
        axs[1].set_xlabel("Feature 2 (e.g., Height)")
        axs[1].set_title("Feature 2 vs Predicted Biomass")
        axs[1].set_yscale("log")
        axs[1].grid(True)

    # Adding legend to both subplots
    axs[0].legend(title="Species")
    axs[1].legend(title="Species")

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()
