# Integral BMS: Governing Equations - Code and Data  

![GitHub repo size](https://img.shields.io/github/repo-size/ocabanas/Integral_BMS_Governing_Equations)
![GitHub last commit](https://img.shields.io/github/last-commit/ocabanas/Integral_BMS_Governing_Equations)
![GitHub issues](https://img.shields.io/github/issues/ocabanas/Integral_BMS_Governing_Equations)
![GitHub Downloads (all assets, all releases)](https://img.shields.io/github/downloads/ocabanas/Integral_BMS_Governing_Equations/total)


This repository contains the code and data supporting the research paper:  
**"Integral Bayesian symbolic regression for optimal discovery
of governing equations from scarce and noisy data"**.

## 🧬 Overview

We introduce a modification of the Bayesian Machine Scientist to learn governing equations from scarce and noisy data. We use the same algorithm as in the BMS, but we employ the integrated version of the differential equation to quantify the poserior of an expression.

This repository includes:
- The full implementation of the I-BMS approach for 1D Logistic model, 2D Lotka-Volterra and physics-informed approach.
- Scripts to learn models from data: I-BMS, BMS and SINDy
- Noise datasets
- Bacterial growth datasets

## 📁 Repository Structure
Noise datasets
```
Logistic/noise_data/
Lotka_Volterra/noise_data/
Bacteria/Train_test_data_lin_term_com2025_03_11-11_21_44/
```
Source code
```
Logistic/rguimera-machinescientist/         (1D Integral BMS)
Lotka-Volterra/rguimera-machinescientist/   (2D Integral BMS)
Bacteria/rguimera-machinescientist/         (Physics Informed 1D Integral BMS)

```
Model resuls
```
Logistic/noise_data_ode/
Logistic/noise_data_fit/
Logistic/noise_data_smooth/
Lotka_Volterra/noise_data_ode/      (available on demand)
Lotka_Volterra/noise_data_fit/
Lotka_Volterra/noise_data_smooth/
Lotka_Volterra/results/             (available on demand)

```
