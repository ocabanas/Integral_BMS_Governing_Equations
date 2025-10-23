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
from ipywidgets import IntProgress
import pynumdiff
import time

src = os.path.dirname('/export/home/oriolca/Integral_BMS_Governing_Equations/I-BMS-1d/')
sys.path.append(src)
from mcmc_ode import *
from parallel_ode import *

path = os.path.join(src, "Prior/")
sys.path.append(path)
from fit_prior import read_prior_par

priors = {
    "v1_p8": f"Prior/final_prior_param_sq.named_equations.nv1.np8.2017-10-18 18:07:35.261518.dat",
    "v2_p3": f"Prior/final_prior_param_sq.named_equations.nv2.np3.2016-09-09 18:49:42.927679.dat",
    "v2_p4": f"Prior/final_prior_param_sq.named_equations.nv2.np4.2016-09-09 18:49:43.056910.dat",
    "v2_p8": f"Prior/final_prior_param_sq.named_equations.nv2.np8.2016-09-09 18:49:42.800618.dat",
}

"""
# Import Machine Scientist
from importlib.machinery import SourceFileLoader
# Get the absolute path of the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
# Define the relative path to the module
relative_module_path = "I-BMS_1d/machinescientist_ode.py"
path = os.path.join(script_dir, relative_module_path)
ms = SourceFileLoader("ms", path).load_module()
"""
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


data = pd.read_pickle(file)

x={'d0':deepcopy(data)}
B = x["d0"].B.to_numpy()
h = x["d0"].t.to_numpy()[1] - x["d0"].t.to_numpy()[0]
par = [2, 21, 21]
x_hat, dxdt_hat = pynumdiff.linear_model.polydiff(
        B, h, par, options=None
    )
dx={'d0': [pd.DataFrame(data={'B': x_hat}) , pd.Series(dxdt_hat) ]}
y={}
y["d0"] = pd.Series(B)

mcmc_resets = 2
mcmc_steps = 3000
XLABS = ["B"]
n_params = 8

Ts=[1] + [1.04**k for k in range(1, 20)]

path = os.path.join(src, priors[f"v{len(XLABS)}_p{str(n_params)}"])
prior_par = read_prior_par(path)

del prior_par["Nopi_abs"]
del prior_par["Nopi2_abs"]
del prior_par["Nopi_tan"]
del prior_par["Nopi2_tan"]
del prior_par["Nopi_sinh"]
del prior_par["Nopi2_sinh"]
del prior_par["Nopi_cosh"]
del prior_par["Nopi2_cosh"]
del prior_par["Nopi_tanh"]
del prior_par["Nopi2_tanh"]
OPS = {
	'sin': 1,
	'cos': 1,
	#'tan': 1,
	"exp": 1,
	#'log': 1,
	#'sinh' : 1,
	#'cosh' : 1,
	#'tanh' : 1,
	"pow2": 1,
	"pow3": 1,
	#'sqrt' : 1,
	#'fac' : 1,
	"-": 1,
	"+": 2,
	"*": 2,
	"/": 2,
	"**": 2,
}
description_lengths, mdl, mdl_model = [], np.inf, None
# Start some MCMC
runs = 0
while runs < mcmc_resets:
	try:  # Sometimes a NaN error appears. Therefore we forget the current MCMC and start again.
		# Initialize the parallel machine scientist
		pms = Parallel(
			Ts,
			ops=OPS,
			variables=XLABS,
			parameters=["a%d" % i for i in range(n_params)],
			x=x,
            y=y,
			dx=dx,
			prior_par=prior_par,
		)
		# MCMC
		mc_start = time.time()
		fbar = IntProgress(min=0, max=mcmc_steps, description='Running:')
		for i in range(1, mcmc_steps + 1):
			start = time.time()
			# MCMC update
			pms.mcmc_step()  # MCMC step within each T
			"""
			if abs(pms.t1.E - pms.t1.get_energy(bic=True, reset=True)[0]) > 1.0e-6:
				print("Reset energy")
				for tree in pms.trees.values():
					tree.get_energy(bic=True, reset=True)
			"""
			pms.tree_swap()  # Attempt to swap two randomly selected consecutive temps

			description_lengths.append(copy(pms.t1.E))
			# Add the description length to the trace
			# description_lengths.append(pms.t1.E)
			# Check if this is the MDL expression so far
			if pms.t1.E < mdl:
				mdl, mdl_model = copy(pms.t1.E), deepcopy(pms.t1)
			# Save step of model
		runs += 1
	except Exception as e:
		print("Error during MCMC evolution:")
		print(e)
		print(traceback.format_exc())
		print("Current model", pms.t1)
		print("Current energy", pms.t1.E)
		print("Restarting MCMC")
		q
file_name = os.path.basename(file)
with open(f"./noise_data_res_ODE/{file_name[:-4]}.pkl", "wb") as f:
    # A new file will be created
    pickle.dump(mdl_model, f)
plt.plot(description_lengths)
plt.savefig(f"./noise_data_res_ODE/{file_name[:-4]}_dl.pdf", format="pdf")
plt.clf()
print()
print(file, mdl_model, mdl_model.E)
print(mdl_model.par_values)
print("end main")
