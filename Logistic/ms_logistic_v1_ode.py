import pandas as pd
import numpy as np
import sys
import warnings
from copy import deepcopy, copy
from IPython.display import display
from datetime import datetime
import pickle
import os
import random
from math import ceil, sqrt
from scipy.optimize import curve_fit
import sys
import matplotlib.pyplot as plt

# Import Machine Scientist
from importlib.machinery import SourceFileLoader
# Get the absolute path of the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
# Define the relative path to the module
relative_module_path = "rguimera-machine-scientist/machinescientist_ode.py"
path = os.path.join(script_dir, relative_module_path)
ms = SourceFileLoader("ms", path).load_module()
#
# Read data
import sys, getopt


def main(argv):
    inputfile = ""
    outputfile = ""
    try:
        opts, args = getopt.getopt(argv, "h:f:", ["file="])
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
    return file


file = main(sys.argv[1:])
print("parsed args:", file)


data = pd.read_pickle("noise_data/" + file)

x = {}
y = {}

y["A0"] = pd.Series(deepcopy(data["B"]))
x["A0"] = deepcopy(data)

mcmc_resets = 2
mcmc_steps = 3000
XLABS = ["B"]
params = 8

best_model, dls = ms.machinescientist_to_folder(
    x, y, XLABS=XLABS, n_params=params, resets=mcmc_resets, steps_prod=mcmc_steps
)
with open(f"./noise_data_res_ODE/{file[:-4]}.pkl", "wb") as f:
    # A new file will be created
    pickle.dump(best_model, f)
plt.plot(dls)
plt.savefig(f"./noise_data_res_ODE/{file[:-4]}_dl.pdf", format="pdf")
plt.clf()
print()
print(file, best_model, best_model.E)
print(best_model.par_values)
print(best_model.x0)
print(len(best_model.x0), len(best_model.fit_par))
print("end main")
