# BRAMBLE – Bayesian Relationships in Allometric Models of Biomass with Laplace and Entropy.


# Quick Start:

In command line:
```
uv venv
source .venv/bin/activate
```
```
uv sync
```
```
pip install git+https://github.com/MAMBO-Horizon-WP4/BRAMBLE.git
```
```
cd BRAMBLE
jupyter notebook development_notebook.ipynb
```

Reusable version of Bayesian Relationships in Allometric Models of Biomass with Laplace and Entropy.

This is a fork of an early work-in-progress. Please look back for updates or visit the original project

# Introduction

BRAMBLE is a reusable Bayesian framework for fitting and applying shrub allometric models.

It uses field measurements (e.g., height, diameter, and destructively sampled biomass) to infer a probabilistic parameterisation of simple allometric relationships.

Once trained, the model can estimate biomass from height and diameter alone — returning a full probability distribution rather than a single deterministic value.

---

# Why BRAMBLE?

Shrub biomass estimation is often limited by:

* Small destructive sampling datasets
* Structural variability across species and sites
* Missing measurements in field surveys
* Overconfident deterministic allometric fits

BRAMBLE addresses these issues by:

* Using Bayesian inference to estimate parameter uncertainty
* Propagating uncertainty into biomass predictions
* Allowing flexible model structures
* Supporting inference of missing variables (development version)

---

# Model Overview

At its core, BRAMBLE fits a simple allometric relationship of the form:

Biomass = f(height, diameter; θ)

where parameters θ are inferred using Bayesian methods.

## Key Features

* Posterior distributions for all model parameters
* Probabilistic biomass predictions
* Uncertainty propagation from data to predictions
* Extensible to alternative allometric functional forms
* Designed for shrub systems but adaptable

---

# Development Notebook

Most implementation details and methodological development are documented in:
`development_notebook.ipynb`


## Get started

Create python environment

```
uv venv
source .venv/bin/activate
```

Install project plus dependencies with uv

```
uv sync
```

Install from git using pip 

```
pip install git+https://github.com/MAMBO-Horizon-WP4/BRAMBLE.git
```

# Development Status

BRAMBLE is under active development.

The current development version is available at:

https://github.com/douglask3/BRAMBLE

Important notes:

* Documentation is ongoing
* Interfaces may change
* Not recommended for non-expert users at this stage

The development version supports:

* Training on alternative allometric measurements
* Flexible model structures
* Inference of missing predictor variables (not only biomass)
* Expanded parameterisations

---

