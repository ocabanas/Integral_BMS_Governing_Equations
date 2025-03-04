import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
from numpy.random import normal

import sys
import gc
import pandas as pd
import numpy as np

from copy import deepcopy
from itertools import combinations
from scipy.stats import multinomial
import warnings
import pynumdiff
import pickle

warnings.filterwarnings("ignore")

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


# ------------------------------------------------------------------------------
# Get all possible linear expressions with n terms
# ------------------------------------------------------------------------------
def get_expressions_n(the_vars, n=2):
    if n == 0:
        return [("_a0_", 1)]
    groups = combinations(the_vars, n)
    all_exp = []
    for vs in groups:
        expres, npar = "(_a0_ + (_a1_ * %s))" % vs[0], 2
        for nv, v in enumerate(vs[1:]):
            expres = "(%s + (_a%d_ * %s))" % (expres, nv + 2, v)
            npar += 1
        all_exp.append((expres, npar))
    return all_exp


def get_expressions_no_cte(the_vars, n=2):
    if n == 0:
        return [("_a0_", 1)]
    groups = combinations(the_vars, n)
    all_exp = []
    for vs in groups:
        expres, npar = "(_a0_ * %s)" % vs[0], 1
        for nv, v in enumerate(vs[1:]):
            expres = "(%s + (_a%d_ * %s))" % (expres, nv + 1, v)
            npar += 1
        all_exp.append((expres, npar))
    return all_exp


# ------------------------------------------------------------------------------
# Get all possible linear expressions
# ------------------------------------------------------------------------------
def get_expressions(the_vars, nmax=None):
    all_exp = []
    if nmax == None:
        nmax = len(the_vars)
    for n in range(nmax + 1):
        all_exp += get_expressions_n(the_vars, n=n)
        all_exp += get_expressions_no_cte(the_vars, n=n)
    return all_exp


# Import Machine Scientist ODE
from importlib.machinery import SourceFileLoader

# Get the absolute path of the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
# Define the relative path to the module
relative_module_path = "rguimera-machine-scientist/machinescientist_ode.py"
path = os.path.join(script_dir, relative_module_path)
ms = SourceFileLoader("ms", path).load_module()

# Import Machine Scientist FIT
from importlib.machinery import SourceFileLoader

# Get the absolute path of the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
# Define the relative path to the module
relative_module_path = "rguimera-machine-scientist/machinescientist_fit.py"
path = os.path.join(script_dir, relative_module_path)
ms_fit = SourceFileLoader("ms_fit", path).load_module()

# data=pd.read_pickle(f'noise_data/{file}')
data = pd.read_pickle(f"noise_data_sigma0.05/{file}")

x = {}
y = {}

y["A0"] = pd.Series(deepcopy(data["B"]))
x["A0"] = deepcopy(data)


par = [2, 21, 21]
h = data.t.to_numpy()[1] - data.t.to_numpy()[0]
x_hat, dxdt_hat = pynumdiff.linear_model.polydiff(
    data.B.to_numpy(), h, par, options=None
)
deriv = pd.Series(dxdt_hat)

x_hat, dxdt_hat_finite = pynumdiff.finite_difference._finite_difference.first_order(
    data.B.to_numpy(), h
)
deriv_finite = pd.Series(dxdt_hat_finite)

x["A0"]["dx"] = deriv

models = list(set(get_expressions(["B", "(pow2(B))", "(pow3(B))", "(pow2(pow2(B)))"])))

mdl = np.inf
mdl_model = None

mdl_fit = np.inf
mdl_model_fit = None

mdl_smooth = np.inf
mdl_model_smooth = None


for model in models:
    ms_model = ms.from_string_model(x, y, model[0], 1, 8, ["B"], silence=True)

    ms_model_fit = ms_fit.from_string_model(
        x, {"A0": deriv_finite}, model[0], 1, 8, ["B"], silence=True
    )

    ms_model_smooth = ms_fit.from_string_model(
        x, {"A0": deriv}, model[0], 1, 8, ["B"], silence=True
    )

    if ms_model.E < mdl:
        mdl_model = deepcopy(ms_model)
        mdl = deepcopy(ms_model.E)
    if ms_model_fit.E < mdl_fit:
        mdl_model_fit = deepcopy(ms_model_fit)
        mdl_fit = deepcopy(ms_model_fit.E)
    if ms_model_smooth.E < mdl_smooth:
        mdl_model_smooth = deepcopy(ms_model_smooth)
        mdl_smooth = deepcopy(ms_model_smooth.E)
with open(f"./noise_data_res_exhaustive_new/BMS_{file[:-4]}.pkl", "wb") as f:
    # A new file will be created
    pickle.dump(mdl_model, f)
with open(f"./noise_data_res_exhaustive_new/BMS_fit_{file[:-4]}.pkl", "wb") as f:
    # A new file will be created
    pickle.dump(mdl_model_fit, f)
with open(f"./noise_data_res_exhaustive_new/BMS_smooth_{file[:-4]}.pkl", "wb") as f:
    # A new file will be created
    pickle.dump(mdl_model_smooth, f)
print(mdl_model, mdl_model_fit, mdl_model_smooth)
