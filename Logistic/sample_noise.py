import warnings
import gc
from IPython.display import display
import os
import sys


sigmas = [
    0.005,
    0.006,
    0.007,
    0.008,
    0.009,
    0.01,
    0.02,
    0.03,
    0.04,
    0.05,
    0.06,
    0.07,
    0.08,
    0.09,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
]
for i in range(0, 40):
    for sigma in sigmas:
        os.system(f"python3 ms_logistic_v1_fit.py -f {sigma}_{i}.pkl -s True")
        os.system(f"python3 ms_logistic_v1_ode.py -f {sigma}_{i}.pkl")
        os.system(f"python3 exhaustive_linear_MS.py -f {sigma}_{i}.pkl")


for i in range(0, 40):
    for n in list(range(10, 91, 10)) + list(range(100, 1001, 100)):
        os.system(f"python3 exhaustive_linear_MS.py -f {i}_{n}.pkl")
