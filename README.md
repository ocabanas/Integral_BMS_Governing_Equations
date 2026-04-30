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

⚠️ **Note:**  
This branch contains **research code** used for the paper.  
A **production-ready implementation** will be released in a separate branch.

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

# 📁 Repository Structure

## Datasets

```
Logistic/noise_data/
Lotka_Volterra/noise_data/
Bacteria/Train_test_data_*/
```

## Source Code

```
Logistic/rguimera-machinescientist/         # 1D Integral BMS
Lotka-Volterra/rguimera-machinescientist/   # 2D Integral BMS
Bacteria/rguimera-machinescientist/         # Physics-informed I-BMS
```

## Results

```
Logistic/noise_data_ode/
Logistic/noise_data_fit/
Logistic/noise_data_smooth/

Lotka_Volterra/noise_data_ode/      (available on demand)
Lotka_Volterra/noise_data_fit/
Lotka_Volterra/noise_data_smooth/
Lotka_Volterra/results/             (available on demand)
```

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

```python
data = pd.DataFrame() # columns t (time), and compontents as variable names ej: 'x':x,'y':y

dxdt_data = pd.DataFrame() # compontents numerical derivatives as variable names ej: 'x':dxdt,'y':dydt

# In the case of multiple datasets, generate a dictionary of dataframes where the keys identify the dataset

data = {'data1': dataframe1,
'data2':dataframe2,...
}
dxdt_data = {'data1': dx_dataframe1,
'data2':dx_dataframe2,...
}

```

## 4. Initialize MCMC

```python
mcmc_resets = 2
mcmc_steps = 3000
XLABS = ['x','y']
params = 8

pms_x = Parallel(
Ts,
x,
dx,
XLABS,
parameters,
prior
)

pms_y = Parallel(
Ts,
x,
dx,
XLABS,
parameters,
prior
)
```

## 5. Run I-BMS

```python
for i in range(1, mcmc_steps + 1):
    pms_x.mcmc_step()
    pms_y.mcmc_step()
    pms_x.tree_swap()
```

## 6. Retrieve Best Model

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

---

# 📄 Citation

```bibtex
@article{integral_bms_2025,
  title={Integral Bayesian symbolic regression for optimal discovery
         of governing equations from scarce and noisy data},
  author={...},
  year={2025}
}
```
