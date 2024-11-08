import numpy  as np
import pandas as pd
#import sys
#sys.path.append('model/')
#sys.path.append('.')
from BRAMBLE import BRAMBLE


def fake_data_simulator(df_measurements):
    # Step 1: Identify unique species and assign random parameters
    unique_species = df_measurements['species'].unique()
    species_params = {'betas': {}, 'beta0': {}}

    for species in unique_species:
        # Assign random parameters to each species
        species_params['betas'][species] = np.random.lognormal(mean=[0.1, 0.075], sigma=0.3, size=2)/3.0  # random betas
        species_params['beta0'][species] = np.random.lognormal(mean=0.01, sigma=0.2)  # random intercept
    
    # Step 2: Initialize the ExpAllometry model with all species parameters
    allometry_model = BRAMBLE(species_params)

    # Step 3: Prepare input arrays for prediction
    X = df_measurements[['height', 'diameter']].values/100.0  # Predictor variables (height and diameter)
    species = df_measurements['species'].values       # Species labels
    species_idx = df_measurements['species'].apply(lambda x: np.where(unique_species == x)[0][0]).values

    # Step 4: Generate fake biomass predictions using the multi-species prediction method
    Y_fake = allometry_model.predict_multi_species(X, species, apply_exp_transform=True)

    # Step 5: Add random noise to simulate measurement error
    Y_fake += np.random.lognormal(mean=0.0, sigma=1.0, size=Y_fake.shape)

    # Step 6: Add the generated fake data to the DataFrame
    return Y_fake
    


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    # Create a sample DataFrame with random species, height, and diameter
    num_samples = 50
    np.random.seed(42)
    data = {
        'species': np.random.choice(['species_1', 'species_2', 'species_3'], num_samples),
        'height': np.random.uniform(100, 300, num_samples),  # in cm
        'diameter': np.random.uniform(10, 50, num_samples)   # in cm
    }
    df_measurements = pd.DataFrame(data)

    # Generate fake biomass data
    df_measurements['Y_fake'] = fake_data_simulator(df_measurements)

    # Plot the fake data
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Plot Feature 1 (height) vs Y_fake, colored by species
    for species in df_measurements['species'].unique():
        subset = df_measurements[df_measurements['species'] == species]
        ax[0].scatter(subset['height'], subset['Y_fake'], label=species, alpha=0.7)
    ax[0].set_xlabel("Height (cm)")
    ax[0].set_ylabel("Fake Biomass")
    ax[0].set_yscale("log")
    ax[0].set_title("Height vs Fake Biomass")
    ax[0].legend()

    # Plot Feature 2 (diameter) vs Y_fake, colored by species
    for species in df_measurements['species'].unique():
        subset = df_measurements[df_measurements['species'] == species]
        ax[1].scatter(subset['diameter'], subset['Y_fake'], label=species, alpha=0.7)
    ax[1].set_xlabel("Diameter (cm)")
    ax[1].set_yscale("log")
    ax[1].set_title("Diameter vs Fake Biomass")
    ax[1].legend()

    plt.tight_layout()
    plt.show()
