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
relative_module_path = "../rguimera-machine-scientist/parallel.py"
# Get the full absolute path
path = os.path.join(script_dir, relative_module_path)
ms = SourceFileLoader("ms", path).load_module()

#Import prior
path = os.path.join(script_dir, "../rguimera-machine-scientist/Prior/")
sys.path.append(path)
from fit_prior import read_prior_par
prior = read_prior_par('../rguimera-machine-scientist/Prior/final_prior_param_sq.named_equations.nv1.np8.2017-10-18 18:07:35.261518.dat')


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

data = pd.read_pickle(file)
print(file[:-4], smooth)

x = {}
y = {}

# Derivate 3point
if not smooth:
    x["d0"] = deepcopy(data)
    B = x["d0"].B.values
    h = x["d0"].t.to_numpy()[1] - x["d0"].t.to_numpy()[0]
    #Here x_hat_finite = x
    x_hat_finite, dxdt_hat_finite = pynumdiff.finite_difference._finite_difference.second_order(B, h)
    y["d0"] = pd.Series(dxdt_hat_finite)
    append = ""
# Smooth derivate
if smooth:
    x["d0"] = deepcopy(data)
    par = [2, 21, 21]
    h = x["d0"].t.to_numpy()[1] - x["d0"].t.to_numpy()[0]
    x_hat, dxdt_hat = pynumdiff.linear_model.polydiff(
        x["d0"].B.to_numpy(), h, par, options=None
    )
    y["d0"] = pd.Series(dxdt_hat)
    append = "_smooth"

print('dX', y)


mcmc_resets = 2
mcmc_steps = 3000
XLABS = ["B"]
params = 8

mdl = np.inf
best_model = None
dls = []

for r in range(mcmc_resets):

    pms = ms.Parallel(
        Ts = [1] + [1.04**k for k in range(1, 20)],
        variables=XLABS,
        parameters=['a%d' % i for i in range(params)],
        prior_par=prior,
        x=x,
        y=y)
    for step in range(mcmc_steps):
        pms.mcmc_step()
        pms.tree_swap()
        if pms.t1.E < mdl:
            mdl = deepcopy(pms.t1.E)
            best_model = deepcopy(pms.t1)

        dls.append(pms.t1.E)
        

file_name = os.path.basename(file)
with open(f"./noise_data_res_fit{append}/{file_name[:-4]}.pkl", "wb") as f:
    # A new file will be created
    pickle.dump(best_model, f)
plt.plot(dls)
plt.savefig(f"./noise_data_res_fit{append}/{file_name[:-4]}_dl.pdf", format="pdf")
plt.clf()
print()
print(file, best_model, best_model.E)
print(best_model.par_values)
print("end main")
