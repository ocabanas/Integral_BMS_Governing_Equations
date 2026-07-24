# Integral BMS: Governing Equations — Code and Data

![GitHub repo size](https://img.shields.io/github/repo-size/ocabanas/Integral_BMS_Governing_Equations)
![GitHub last commit](https://img.shields.io/github/last-commit/ocabanas/Integral_BMS_Governing_Equations)
![GitHub issues](https://img.shields.io/github/issues/ocabanas/Integral_BMS_Governing_Equations)
![GitHub Downloads (all assets, all releases)](https://img.shields.io/github/downloads/ocabanas/Integral_BMS_Governing_Equations/total)

This repository contains the code and datasets supporting the research paper:

**"Integral Bayesian symbolic regression for optimal discovery of governing equations from scarce and noisy data"**

---

# 🧬 Overview

We introduce **Integral Bayesian Machine Scientist (I-BMS)**, a modification of the Bayesian Machine Scientist designed to discover governing equations from scarce and noisy data. Instead of evaluating differential equations directly, I-BMS evaluates their **integral form**, enabling robust posterior estimation under noise.

This repository includes:

- Implementation of the **Integral BMS (I-BMS)** algorithm
- Baseline comparisons: **BMS** and **SINDy**
- Benchmark dynamical systems:
  - Logistic growth (1D)
  - Lotka–Volterra (2D)
  - Physics-informed bacterial growth model
- Synthetic noisy datasets
- Experimental bacterial growth datasets
- Scripts for training and evaluation

---

# 📦 Installation

```bash
git clone https://github.com/ocabanas/Integral_BMS_Governing_Equations.git
cd Integral_BMS_Governing_Equations
pip install -r requirements.txt
```

Main dependencies:

- Python ≥ 3.9
- numpy
- pandas
- matplotlib
- scipy
- pynumdiff

---

## 📂 Directory and File Description

### `Bacteria/`
Contains the bacterial growth datasets and experiments used to evaluate the physics-informed version of I-BMS.

| Path | Description |
|------|-------------|
| `Full_data_lin_term_prod_2025_03_27-06_00_44/` | Full processed bacterial growth dataset used in the experiments. |
| `microbial_growth_full*.csv` | Experimental microbial growth measurements. |
| `Train_test_data_lin_term_com2025_03_11-11_21_44/` | Preprocessed train/test splits used for benchmarking. |
| `ms_bacteris_v2_extended_datasets.py` | Main script for symbolic regression on the complete bacterial datasets. |

### `BIC_validation/`
Notebooks used to evaluate the Bayesian Information Criterion (BIC) employed for model selection.

| Path | Description |
|------|-------------|
| `Benchmark.ipynb` | Benchmark comparison of candidate symbolic models. |
| `BIC_model_comp.ipynb` | Comparison of models using the BIC score. |
| `GenerateModels.ipynb` | Generation of candidate symbolic equations for BIC evaluation. |

### `Gelman-Rubin/`
Utilities for assessing MCMC convergence.

| Path | Description |
|------|-------------|
| `Gelman-Rubin.ipynb` | Computes and visualizes the Gelman–Rubin convergence diagnostic. |
| `MCMC_sampling.py` | MCMC sampling of a dataset. |

### `I-BMS-1d/`
Implementation of the Integral Bayesian Machine Scientist for one-dimensional dynamical systems.

| Path | Description |
|------|-------------|
| `machinescientist_ode.py` | Helper functions to use the 1D I-BMS algorithm. |
| `mcmc_ode.py` | I-BMS model class and MCMC sampler. |
| `parallel_ode.py` | Parallel execution functions. |

### `I-BMS-2d/`
Implementation of I-BMS for coupled two-dimensional systems.

| Path | Description |
|------|-------------|
| `machinescientist_ode.py` | Helper functions to use the 2D I-BMS algorithm. |
| `mcmc_ode.py` | MCMC sampler for coupled ODE systems. |
| `parallel_ode.py` | Parallel execution functions. |

### `I-BMS-constrained/`
Constrained version of I-BMS with additional structural restrictions.

| Path | Description |
|------|-------------|
| `machinescientist_ode.py` | Helper functions to use the I-BMS constrained. |
| `mcmc_ode.py` | MCMC sampler. |
| `parallel_ode.py` | Parallel execution functions. |

### `Logistic/`
Experiments on the one-dimensional logistic growth equation.

| Path | Description |
|------|-------------|
| `ms_logistic_v1_fit.py` | Symbolic regression with original BMS. |
| `ms_logistic_v1_ode.py` | Symbolic regression with the I-BMS. |
| `learnability.ipynb` | Analysis of learnability. |
| `learnability.pkl` | Cached learnability results. |
| `detection_accuracy.ipynb` | Detection accuracy analysis. |
| `detection_length.pkl` | Cached equation-length experiments. |
| `detection_sigma.pkl` | Cached noise-level experiments. |
| `exhaustive_linear_MS.py` | Exhaustive baseline search over linear models. |
| `noise_data/` | Synthetic noisy datasets. |
| `sample_noise.py` | Noise dataset sampling. |

### `Lotka-Volterra/`
Experiments on the Lotka–Volterra predator–prey system.

| Path | Description |
|------|-------------|
| `ms_LV_v1_fit.py` | Symbolic regression with original BMS. |
| `ms_LV_v1_ode.py` | Symbolic regression with I-BMS. |
| `learnability.py` | Learnability computations. |
| `learnability.pkl` | Cached learnability results. |
| `detection_accuracy.ipynb` | Detection accuracy experiments. |
| `detection_length.pkl` | Cached equation-length experiments. |
| `detection_sigma.pkl` | Cached noise-level experiments. |
| `exhaustive_linear_ms.py` | Exhaustive linear symbolic regression baseline. |
| `noise_data/` | Synthetic noisy datasets. |
| `sample_noise.py` | Sample noise datasets. |
| `slurm.py` | HPC execution utilities. |

### `rguimera-machine-scientist/`
Original Bayesian Machine Scientist implementation from Guimerà *et al.*, used as the baseline for comparison.

| Path | Description |
|------|-------------|
| `machinescientist.py` | Main Bayesian Machine Scientist implementation. |
| `mcmc.py` | MCMC sampler. |
| `parallel.py` | Parallel execution. |

### `SINDy_implementation/`
Reference implementation of Sparse Identification of Nonlinear Dynamics (SINDy).

| Path | Description |
|------|-------------|
| `EnsembleWeakSINDy.ipynb` | Ensemble Weak-SINDy experiments. |
| `Z-SINDy-strogatz.ipynb` | SINDy benchmark on Strogatz systems. |
| `requirements.txt` | Python dependencies. |
| `requirements_pysindy.txt` | Dependencies for the PySINDy implementation. |

### `strogatz/`
Benchmark experiments using dynamical systems from the Strogatz collection.

| Path | Description |
|------|-------------|
| `datasets/` | Benchmark datasets. |
| `datasets.ipynb` | Dataset generation notebook. |
| `MCMC_sampling.py` | MCMC experiments. |
| `ms_LV_v1_fit.py` | Symbolic regression experiments. |
| `Plot_results.ipynb` | Visualization of benchmark results. |
| `PYSR_sampling.py` | PySR learning  experiments. |
| `True_models_I-BMS.ipynb` | Recovery of ground-truth models using I-BMS. |
| `True_models_PYSR_BMS.ipynb` | Comparison of I-BMS, Bayesian Machine Scientist and PySR. |
| `slurm.py` | HPC execution utilities. |

---

# 🚀 Getting Started

## 1. Import I-BMS

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from copy import deepcopy

from importlib.machinery import SourceFileLoader

script_dir = os.path.dirname(os.path.abspath(__file__))
relative_module_path = "../I-BMS-2d/parallel_ode.py"
path = os.path.join(script_dir, relative_module_path)

ms = SourceFileLoader("ms", path).load_module()
```

## 2. Load Prior

```python
path = os.path.join(script_dir, "../I-BMS-2d/Prior/")
sys.path.append(path)

from fit_prior import read_prior_par

prior = read_prior_par(
'../I-BMS-2d/Prior/final_prior_param_sq.named_equations.nv2.np8.2016-09-09.dat'
)
```

## 3. Read and Prepare Data
Prepare your dataset as Pandas DataFrames:

- data: time series of variables
- dxdt_data: numerical derivatives

```python
data = pd.DataFrame()  # columns: t, x, y, ...
dxdt_data = pd.DataFrame()  # columns: x, y, ... (derivatives)
```
### Multiple Datasets

If using multiple datasets, organize them as dictionaries:
```python
data = {
    'data1': dataframe1,
    'data2': dataframe2,
}

dxdt_data = {
    'data1': dx_dataframe1,
    'data2': dx_dataframe2,
}
```
### Special Case: 2D Systems

For 2D systems, define swapped datasets for the second variable:
```python
data_y = pd.DataFrame()   # swap column variable labels (x ↔ y)
dydt_data = pd.DataFrame() # swap column variable labels (x ↔ y)
```
## 4. Initialize MCMC

Set up the parallel tempering MCMC:
```python
mcmc_resets = 2
mcmc_steps = 3000
XLABS = ['x','y']
params = 8

pms_x = ms.Parallel(
        Ts,
        variables=XLABS,
        parameters=["a%d" % i for i in range(params)],
        x=data,
        dx=dxdt_data,
        prior_par=prior,
    )
```
### For 2D systems, initialize the second model:
```python
pms_y = ms.Parallel(
    Ts,
    variables=XLABS,
    parameters=["a%d" % i for i in range(params)],
    x=data_y,
    dx=dydt_data,
    prior_par=prior,
)
```
And Couple the Models Across Temperatures
```python
  for temp in pms_x.trees.keys():
      pms_x.trees[temp].fy = pms_y.trees[temp]
      pms_y.trees[temp].fy = pms_x.trees[temp]
      # print('refit')
      pms_x.trees[temp].get_bic(reset=True, fit=True)
      pms_x.trees[temp].get_energy(bic=True, reset=True)
  pms_x.t1 = pms_x.trees[str(min(Ts))]
  pms_y.t1 = pms_y.trees[str(min(Ts))]
```

## 5. Run I-BMS
Execute the MCMC sampling with parallel tempering:
```python
for i in range(0, mcmc_steps):
    pms_x.mcmc_step()
    pms_y.mcmc_step()
    # Attempt to swap two randomly selected consecutive temps
    ET1, ET2 = (
            pms_x.tree_swap()
        )
```
In a 2D system, swap the layers in the second component
```python
        if ET1 != None:
            t1 = pms_y.trees[ET1]
            t2 = pms_y.trees[ET2]
            BT1, BT2 = t1.BT, t2.BT
            pms_y.trees[ET1] = t2
            pms_y.trees[ET2] = t1
            t1.BT = BT2
            t2.BT = BT1
            
            pms_x.trees[ET1].fy = pms_y.trees[ET1]
            pms_y.trees[ET1].fy = pms_x.trees[ET1]

            pms_x.trees[ET2].fy = pms_y.trees[ET2]
            pms_y.trees[ET2].fy = pms_x.trees[ET2]
            pms_y.t1 = pms_y.trees[pms_y.Ts[0]]
```

## 6. Retrieve Best Model
After sampling, extract the best models:
```python
print('Best model X:', model_x)
print('Best model Y:', model_y)
print('Description length:', model_x.E)
```

---

# 📊 Output

The algorithm returns:

- Symbolic governing equations
- Posterior description length
- Coupled system model
- MCMC trace of description length

---

# 📈 Benchmarks Included

| System | Dimensions | Data Type | Noise |
|--------|------------|-----------|------|
| Logistic | 1D | Synthetic | ✔ |
| Lotka–Volterra | 2D | Synthetic | ✔ |
| Bacterial Respiration | 2D | Synthetic | ✔ |
| Bar Magnets | 2D | Synthetic | ✔ |
| Glider | 2D | Synthetic | ✔ |
| Shear Flow | 2D | Synthetic | ✔ |
| Lotka-Volterra (2) | 2D | Synthetic | ✔ |
| Predator-Prey | 2D | Synthetic | ✔ |
| Van der Pol Osc. | 2D | Synthetic | ✔ |
