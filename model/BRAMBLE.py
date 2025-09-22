"""BRAMBLE: Bayesian/Routine Allocation Model for Biomass, Liberties for shrubs

This single-file implements a flexible BRAMBLE class that can be used in two modes:
 - deterministic / forward (numpy) mode: BRAMBLE(params, inference=False)
 - probabilistic (PyMC/pytensor) mode: BRAMBLE(params, inference=True)

Features
- Accepts any combination of inputs (height, crown diameter, ring count, stems, density,
  wet/dry biomass, nitrogen or carbon) and computes missing quantities using mechanistic
  relationships parameterised by `params`.
- Implements a simple hierarchical structure for inference (hyperpriors for coefficient
  groups). The PyMC model builder returns a pm.Model for sampling.
- Prediction utilities that work with either numpy arrays (forward) or pytensor tensors
  (inside the PyMC model).

Notes
- The geometric shape used when both height and crown diameter are available is a sphere
  approximation (user requested). If only one of height/width provided the code uses that
  as the effective diameter.
- The relationships are intentionally simple and parameter-driven so you can later
  replace them with more complex ecophysiological models.

"""
from pdb import set_trace
import numpy as np
import numbers
from typing import Optional, Dict, Any

# Optional imports for probabilistic mode
try:
    import pytensor.tensor as tt
    import pymc as pm
except Exception:
    tt = None
    pm = None


class BRAMBLE:
    """Flexible biomass -> N -> C pipeline for shrubs.

    Usage examples (deterministic):
        params = {}  # see defaults below
        model = BRAMBLE(params, inference=False)
        out = model.compute(heights=h_arr, widths=w_arr, rings=r_arr, stems=s_arr)

    For Bayesian inference mode:
        model = BRAMBLE(params, inference=True)
        with model.build_pymc_model(data_dict) as pymc_model:
            trace = pm.sample(...)

    The compute() method will calculate all missing intermediate variables if possible.
    """

    def __init__(self, params: Optional[Dict[str, float]] = None, inference: bool = False):
        self.inference = bool(inference)
        self.numPCK = None
        if self.inference:
            if tt is None or pm is None:
                raise ImportError("pymc and pytensor are required for inference=True")
            self.numPCK = tt
        else:
            self.numPCK = np

        # Default parameters (physically plausible defaults; override with params)
        defaults = {
            # density model: density (kg / m3) = rho0 + rho_rings * rings + rho_stems * stems
            'rho0': 30.0,  # base wood density (kg/m3)
            'rho_rings': 2.0,  # per ring increment
            'rho_stems': 20.0,  # per additional stem increment

            # moisture fraction to convert wet->dry (wet_mass * (1 - moisture_frac) = dry_mass)
            'moisture_mean': 0.5,  # typical moisture fraction (50% water)

            # nitrogen fraction of dry mass (mass N / mass dry)
            'N_frac_mean': 0.01,  # 1% N by dry mass

            # carbon fraction baseline and dependence on N
            'C_base': 0.45,  # baseline carbon fraction
            'C_N_coeff': 2.0,  # multiplier on nitrogen fraction to shift carbon fraction

            # coefficient priors (used only if building a pymc model)
            'beta_prior_sd': 1.0,

            # footprint multiplier default
            'footprint_multiplier': 1.0,
        }

        self.params = defaults
        if params:
            self.params.update(params)

    # ----------------------------- helpers ---------------------------------
    def _as_array(self, x):
        """Convert numeric or sequence to numpy array (or tensor) consistent with mode."""
        npk = self.numPCK
        if x is None:
            return None
        if self.inference:
            # expect pytensor compatible objects (tt.as_tensor_variable)
            try:
                return tt.as_tensor_variable(x)
            except Exception:
                return tt.as_tensor_variable(np.asarray(x))
        else:
            return np.asarray(x)

    def _stack(self, *arrays):
        """Stack horizontally for prediction depending on numeric package."""
        npk = self.numPCK
        if self.inference:
            return tt.concatenate([a.reshape((-1, 1)) if a.ndim == 1 else a for a in arrays], axis=1)
        else:
            arrays2 = [np.asarray(a) for a in arrays]
            return np.hstack([a.reshape((-1, 1)) if a.ndim == 1 else a for a in arrays2])

    # ------------------------ geometric / allometric ------------------------
    def effective_diameter(self, heights=None, widths=None):
        """Compute an effective diameter from available inputs.

        Rule used: if both present: effective_d = mean([height, crown_diameter]).
                   if only one present use that one as diameter.
        Returns an array/tensor of diameters.
        """
        h = heights
        w = widths
        if h is None and w is None:
            return None
        if self.inference:
            # pytensor handling
            if h is None:
                return tt.as_tensor_variable(w)
            if w is None:
                return tt.as_tensor_variable(h)
            return 0.5 * (tt.as_tensor_variable(h) + tt.as_tensor_variable(w)) * 2.0 / 2.0  # retains shape
        else:
            if h is None:
                return np.asarray(w)
            if w is None:
                return np.asarray(h)
            return (np.asarray(h) + np.asarray(w)) / 2.0

    def sphere_volume_from_diameter(self, diam):
        """Compute sphere volume from diameter (m).

        Volume = 4/3 * pi * r^3, r = diam/2
        """
        if diam is None:
            return None
        npk = self.numPCK
        if self.inference:
            r = diam / 2.0
            return (4.0 / 3.0) * npk.pi * r ** 3
        else:
            r = diam / 2.0
            return (4.0 / 3.0) * npk.pi * r ** 3

    # ------------------------ process chain -------------------------------
    def compute_density(self, rings=None, stems=None, density=None):
        """Compute or accept wood density (kg/m3). Linear relation with rings & stems.

        density = rho0 + rho_rings * rings + rho_stems * stems
        If density provided, return it unchanged.
        """
        if density is not None:
            return density
        rho0 = self.params['rho0']
        rho_rings = self.params['rho_rings']
        rho_stems = self.params['rho_stems']
        if self.inference:
            rings_t = tt.as_tensor_variable(rings) if rings is not None else 0.0
            stems_t = tt.as_tensor_variable(stems) if stems is not None else 0.0
            return rho0 + rho_rings * rings_t + rho_stems * stems_t
        else:
            rings_a = np.asarray(rings) if rings is not None else 0.0
            stems_a = np.asarray(stems) if stems is not None else 0.0
            return rho0 + rho_rings * rings_a + rho_stems * stems_a

    def compute_wet_biomass(self, volume=None, density=None, wet_biomass=None, footprint_multiplier=None):
        """Compute wet biomass (kg) from volume (m3) and density (kg/m3).
        Optionally accept wet_biomass directly.
        footprint_multiplier scales the per-unit plant mass (useful if you pass per-plant vol and want per-area)
        """
        if wet_biomass is not None:
            return wet_biomass
        if footprint_multiplier is None:
            footprint_multiplier = self.params.get('footprint_multiplier', 1.0)
        if self.inference:
            vol_t = tt.as_tensor_variable(volume)
            rho_t = tt.as_tensor_variable(density)
            return vol_t * rho_t * footprint_multiplier
        else:
            return np.asarray(volume) * np.asarray(density) * float(footprint_multiplier)

    def compute_dry_biomass(self, wet_biomass=None, dry_biomass=None, moisture_frac=None):
        """Dry = wet * (1 - moisture_frac)."""
        if dry_biomass is not None:
            return dry_biomass
        if moisture_frac is None:
            moisture_frac = self.params.get('moisture_mean', 0.5)
        if self.inference:
            return tt.as_tensor_variable(wet_biomass) * (1.0 - moisture_frac)
        else:
            return np.asarray(wet_biomass) * (1.0 - moisture_frac)

    def compute_nitrogen(self, dry_biomass=None, N_mass=None, N_frac=None):
        """Compute nitrogen mass (kg N) from dry biomass and nitrogen fraction.

        If N_frac is not provided use N_frac_mean parameter.
        """
        if N_mass is not None:
            return N_mass
        if N_frac is None:
            N_frac = self.params.get('N_frac_mean', 0.01)
        if self.inference:
            return tt.as_tensor_variable(dry_biomass) * N_frac
        else:
            return np.asarray(dry_biomass) * float(N_frac)

    def compute_carbon(self, dry_biomass=None, nitrogen_mass=None, carbon_mass=None, C_base=None, C_N_coeff=None):
        """Compute carbon mass from dry biomass and nitrogen mass.

        Simple rule: carbon fraction = C_base + C_N_coeff * (N_frac)
        carbon_mass = dry_biomass * carbon_fraction
        """
        if carbon_mass is not None:
            return carbon_mass
        if C_base is None:
            C_base = self.params.get('C_base', 0.45)
        if C_N_coeff is None:
            C_N_coeff = self.params.get('C_N_coeff', 2.0)

        if nitrogen_mass is None:
            # if nitrogen mass not provided try to compute N_frac from params
            N_frac = self.params.get('N_frac_mean', 0.01)
            nitrogen_mass = dry_biomass * N_frac

        # N_frac = nitrogen_mass / dry_biomass (handle divide by zero)
        if self.inference:
            nfrac = tt.switch(tt.eq(dry_biomass, 0), 0.0, nitrogen_mass / dry_biomass)
            cfrac = C_base + C_N_coeff * nfrac
            return dry_biomass * cfrac
        else:
            dry_b = np.asarray(dry_biomass)
            with np.errstate(divide='ignore', invalid='ignore'):
                nfrac = np.where(dry_b == 0, 0.0, np.asarray(nitrogen_mass) / dry_b)
            cfrac = C_base + C_N_coeff * nfrac
            return dry_b * cfrac

    # ------------------------ prediction utilities -------------------------
    def predict_multi_species(self, X, betas, beta0):
        """Generic linear predictor X @ betas + beta0 supporting numpy or pytensor.

        X: shape (n, p)
        betas: length p
        beta0: scalar or length n
        """
        if self.inference:
            return tt.dot(X, betas) + beta0
        else:
            return np.dot(X, betas) + beta0

    # ------------------------ user-facing compute -------------------------
    def compute(self,
                heights=None,
                widths=None,
                footprints=None,
                rings=None,
                stems=None,
                density=None,
                wet_biomass=None,
                dry_biomass=None,
                nitrogen_mass=None,
                carbon_mass=None,
                # optional flags to return intermediate outputs
                return_all=True,
                **kw):
        """High-level forward computation. Accepts any subset of inputs and computes others.

        Returns a dict with keys: diameter, volume, density, wet_biomass, dry_biomass,
        nitrogen_mass, carbon_mass (only those that could be computed will be present)
        """
        # handle arrays/tensors
        diam = self.effective_diameter(heights, widths)
        vol = self.sphere_volume_from_diameter(diam) if diam is not None else None

        dens = self.compute_density(rings=rings, stems=stems, density=density)
        wet = self.compute_wet_biomass(volume=vol, density=dens, wet_biomass=wet_biomass,
                                       footprint_multiplier=footprints or self.params.get('footprint_multiplier'))
        dry = self.compute_dry_biomass(wet_biomass=wet, dry_biomass=dry_biomass)
        N_mass = self.compute_nitrogen(dry_biomass=dry, N_mass=nitrogen_mass)
        C_mass = self.compute_carbon(dry_biomass=dry, nitrogen_mass=N_mass, carbon_mass=carbon_mass)

        out = {
            'diameter': diam,
            'volume': vol,
            'density': dens,
            'wet_biomass': wet,
            'dry_biomass': dry,
            'nitrogen_mass': N_mass,
            'carbon_mass': C_mass,
        }

        if not return_all:
            # only return the end point that user explicitly provided or asked for
            return out['carbon_mass']
        return out

    def build_pymc_model(self, data: Dict[str, Any], group_key: Optional[str] = None, fixed_params: Optional[Dict[str, Any]] = None):
        """PyMC model with constrained densities and truncated carbon likelihood."""
    
        if not self.inference:
            raise RuntimeError("build_pymc_model requires BRAMBLE(..., inference=True)")
        if pm is None:
            raise ImportError("pymc is required to build a probabilistic model")
    
        fixed_params = fixed_params or {}
        C_arr = np.asarray(data['carbon_mass']) if 'carbon_mass' in data else None
        if C_arr is None:
            raise ValueError('Observed carbon (carbon_mass) must be provided (np.nan allowed).')
        n = len(C_arr)
    
        H_arr = np.asarray(data.get('heights', np.full(n, np.nan)))
        W_arr = np.asarray(data.get('widths', np.full(n, np.nan)))
        rings_arr = np.asarray(data.get('rings', np.full(n, np.nan)))
        stems_arr = np.asarray(data.get('stems', np.full(n, np.nan)))
        footprints_arr = np.asarray(data.get('footprints', np.full(n, np.nan)))
    
        with pm.Model() as model:
            # ----------------- Priors for mechanistic coefficients -----------------
            beta_rho0 = fixed_params.get('rho0', pm.Normal('beta_rho0', mu=self.params['rho0'], sigma=self.params['beta_prior_sd']))
            beta_rings = fixed_params.get('rho_rings', pm.Normal('beta_rings', mu=self.params['rho_rings'], sigma=self.params['beta_prior_sd']))
            beta_stems = fixed_params.get('rho_stems', pm.Normal('beta_stems', mu=self.params['rho_stems'], sigma=self.params['beta_prior_sd']))
            C_base = fixed_params.get('C_base', pm.Normal('C_base', mu=self.params['C_base'], sigma=0.05))
            C_N_coeff = fixed_params.get('C_N_coeff', pm.Normal('C_N_coeff', mu=self.params['C_N_coeff'], sigma=1.0))
            moisture = fixed_params.get('moisture_mean', pm.Beta('moisture', alpha=2, beta=2))
            N_frac = fixed_params.get('N_frac_mean', pm.Beta('N_frac', alpha=2, beta=200))
    
            # ----------------- Observed inputs (with missing support) -----------------
            def obs_normal(name, arr):
                mu = float(np.nanmean(arr)) if np.isfinite(np.nanmean(arr)) else 0.0
                sigma = float(np.nanstd(arr)) if np.isfinite(np.nanstd(arr)) and np.nanstd(arr) > 0 else max(1.0, abs(mu)*0.5)
                return pm.Normal(name, mu=mu, sigma=sigma, observed=arr)
    
            H_rv = obs_normal('H_obs', H_arr) if not np.all(np.isnan(H_arr)) else None
            W_rv = obs_normal('W_obs', W_arr) if not np.all(np.isnan(W_arr)) else None
            rings_rv = obs_normal('rings_obs', rings_arr) if not np.all(np.isnan(rings_arr)) else None
            stems_rv = obs_normal('stems_obs', stems_arr) if not np.all(np.isnan(stems_arr)) else None
            footprints_rv = obs_normal('fp_obs', footprints_arr) if not np.all(np.isnan(footprints_arr)) else None
    
            # ----------------- Diameter & Volume -----------------
            if H_rv is not None and W_rv is not None:
                diam_t = 0.5 * (tt.as_tensor_variable(H_rv) + tt.as_tensor_variable(W_rv))
            elif H_rv is not None:
                diam_t = tt.as_tensor_variable(H_rv)
            elif W_rv is not None:
                diam_t = tt.as_tensor_variable(W_rv)
            else:
                raise ValueError('At least one of heights or widths must be supplied.')
    
            vol_t = (4.0/3.0) * tt.pi * (diam_t/2.0)**3
    
            # ----------------- Density (strictly positive) -----------------
            r_t = tt.as_tensor_variable(rings_rv) if rings_rv is not None else tt.zeros(n)
            s_t = tt.as_tensor_variable(stems_rv) if stems_rv is not None else tt.zeros(n)
            density_mu = beta_rho0 + beta_rings * r_t + beta_stems * s_t
            density_t = pm.Lognormal('density', mu=tt.log(density_mu), sigma=0.2, shape=n)
    
            # ----------------- Wet & Dry Biomass -----------------
            fp_val = tt.as_tensor_variable(footprints_rv) if footprints_rv is not None else float(self.params.get('footprint_multiplier', 1.0))
            wet_t = vol_t * density_t * fp_val
            dry_t = wet_t * (1.0 - (moisture if isinstance(moisture, (int,float)) else tt.as_tensor_variable(moisture)))
    
            # ----------------- Nitrogen -----------------
            N_mass_t = dry_t * (N_frac if isinstance(N_frac, (int,float)) else tt.as_tensor_variable(N_frac))
            nfrac_t = tt.switch(tt.eq(dry_t, 0), 0.0, N_mass_t/dry_t)
    
            # ----------------- Carbon fraction -----------------
            C_frac_t = (C_base if isinstance(C_base,(int,float)) else C_base) + \
                       (C_N_coeff if isinstance(C_N_coeff,(int,float)) else C_N_coeff) * nfrac_t
            # clip to physically meaningful range
            C_frac_t = tt.clip(C_frac_t, 0.01, 0.99)
            C_t = dry_t * C_frac_t
    
            # ----------------- Likelihood: Observed carbon (truncated >=0) -----------------
            sigma_obs = pm.HalfNormal('sigma_obs', sigma=5.0)
            
            pm.Lognormal('carbon_obs', mu=tt.log(C_t), sigma=sigma_obs, observed=C_arr)
    
        return model
# End of file

