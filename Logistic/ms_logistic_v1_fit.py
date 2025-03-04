import pandas as pd
import numpy as np
import warnings
import gc
from copy import deepcopy, copy
from IPython.display import display
from datetime import datetime
import pickle
import os
import random
from math import ceil, sqrt
from scipy.optimize import curve_fit
import sys
import pynumdiff
import matplotlib.pyplot as plt

# Import Machine Scientist
from importlib.machinery import SourceFileLoader
# Get the absolute path of the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
# Define the relative path to the module
relative_module_path = "rguimera-machine-scientist/machinescientist_fit.py"
# Get the full absolute path
path = os.path.join(script_dir, relative_module_path)
ms = SourceFileLoader("ms", path).load_module()


import sys, getopt


def main(argv):
    inputfile = ""
    outputfile = ""
    try:
        opts, args = getopt.getopt(argv, "h:f:s:", ["file=", "smooth="])
    except getopt.GetoptError:
        print("test.py -s <state>")
        sys.exit(2)
    print(opts, args)
    for opt, arg in opts:
        if opt == "-h":
            print("test.py -i <state>")
            sys.exit()
        elif opt in ("-f", "--file"):
            file = arg
        elif opt in ("-s", "--smooth"):
            smooth = eval(arg)
    return file, smooth


file, smooth = main(sys.argv[1:])
print("parsed args:", file, smooth, type(smooth))

data = pd.read_pickle("noise_data/" + file)
print(file[:-4], smooth)

x = {}
y = {}

# Derivate 3point
if not smooth:
    x["A0"] = deepcopy(data)
    B = x["A0"].B.values
    h = x["A0"].t.to_numpy()[1] - x["A0"].t.to_numpy()[0]
    y["A0"] = pd.Series(
        [(B[1] - B[0]) / h]
        + [(B[i + 1] - B[i - 1]) / (2 * h) for i in range(1, len(B) - 1)]
        + [(B[len(B) - 1] - B[len(B) - 2]) / h]
    )
    append = ""
# Smooth derivate
if smooth:
    x["A0"] = deepcopy(data)
    par = [2, 21, 21]
    h = x["A0"].t.to_numpy()[1] - x["A0"].t.to_numpy()[0]
    x_hat, dxdt_hat = pynumdiff.linear_model.polydiff(
        x["A0"].B.to_numpy(), h, par, options=None
    )
    y["A0"] = pd.Series(dxdt_hat)
    append = "_smooth"

mcmc_resets = 2
mcmc_steps = 3000
XLABS = ["B"]
params = 8


best_model, dls = ms.machinescientist_to_folder(
    x, y, XLABS=XLABS, n_params=params, resets=mcmc_resets, steps_prod=mcmc_steps
)
with open(f"./noise_data_res_fit{append}/{file[:-4]}.pkl", "wb") as f:
    # A new file will be created
    pickle.dump(best_model, f)
plt.plot(dls)
plt.savefig(f"./noise_data_res_fit{append}/{file[:-4]}_dl.pdf", format="pdf")
plt.clf()
print()
print(file, best_model, best_model.E)
print(best_model.par_values)
print("end main")
