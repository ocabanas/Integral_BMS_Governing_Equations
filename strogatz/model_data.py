import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import pandas as pd
import numpy as np
import sys
from copy import deepcopy,copy
from datetime import datetime
import pickle
import sys
import matplotlib.pyplot as plt
import pynumdiff
import traceback
import time
# Import Machine Scientist
from importlib.machinery import SourceFileLoader
# Get the absolute path of the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
# Define the relative path to the module
relative_module_path = "../I-BMS-2d/parallel_ode.py"
path = os.path.join(script_dir, relative_module_path)
ms = SourceFileLoader("ms", path).load_module()

#Import prior
path = os.path.join(script_dir, "../I-BMS-2d/Prior/")
sys.path.append(path)
from fit_prior import read_prior_par
prior = read_prior_par('../I-BMS-2d/Prior/final_prior_param_sq.named_equations.nv2.np8.2016-09-09 18:49:42.800618.dat')


# Read data
import sys, getopt

def main(argv):
    inputfile = ''
    outputfile = ''
    try:
        opts, args = getopt.getopt(argv,"h:f:",["file="])
    except getopt.GetoptError:
        print('test.py -s <state>')
        sys.exit(2)
    print(opts,args)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -i <state>')
            sys.exit()
        elif opt in ("-f", "--file"):
            file = arg
    return file

file=main(sys.argv[1:])

data=pd.read_csv(file)

x={}
y={}
dx = {}

dy = {}

x['d0']=deepcopy(data)
y['d0']=deepcopy(data)
y['d0'].x=deepcopy(x['d0'].y)
y['d0'].y=deepcopy(x['d0'].x)

h = x['d0'].t.to_numpy()[1] - x['d0'].t.to_numpy()[0]

par = [2, 21, 21]
x_hat, dxdt_hat = pynumdiff.linear_model.polydiff(
        x['d0'].x, h, par, options=None)
y_hat, dydt_hat = pynumdiff.linear_model.polydiff(
        x['d0'].y, h, par, options=None)

dx['d0'] = [ pd.DataFrame(data={'x':x_hat,'y':y_hat}),
            pd.DataFrame(data={'x':dxdt_hat,'y':dydt_hat})]

dy['d0'] = [ pd.DataFrame(data={'x':y_hat,'y':x_hat}),
            pd.DataFrame(data={'x':dydt_hat,'y':dxdt_hat})]


mcmc_resets = 2
mcmc_steps = 4000
XLABS = ['x','y']
params = 8
print(x)
print(y)

description_lengths, mdl, mdl_x, mdl_y, mdl_model_x, mdl_model_y = (
        [],
        np.inf,
        np.inf,
        np.inf,
        None,
        None,
    )

del prior["Nopi_abs"]
del prior["Nopi2_abs"]
del prior["Nopi_sin"]
del prior["Nopi2_sin"]
del prior["Nopi_cos"]
del prior["Nopi2_cos"]
del prior["Nopi_tan"]
del prior["Nopi2_tan"]
del prior["Nopi_sinh"]
del prior["Nopi2_sinh"]
del prior["Nopi_cosh"]
del prior["Nopi2_cosh"]
del prior["Nopi_tanh"]
del prior["Nopi2_tanh"]
OPS = {
    #'sin': 1,
    #'cos': 1,
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
#stderr_fileno = sys.stderr
#sys.stderr = open(os.devnull, 'w')
from multiprocessing import Pool

def run_instance(_):
    dl, i_mdl, model_x, model_y = [], np.inf, None, None
    Ts=[1] + [1.04**k for k in range(1, 40,2)]
    pms_x = ms.Parallel(
        Ts,
        ops=OPS,
        variables=XLABS,
        parameters=["a%d" % i for i in range(params)],
        x=x,
        dx=dx,
        prior_par=prior,
    )
    pms_y = ms.Parallel(
        Ts,
        ops=OPS,
        variables=XLABS,
        parameters=["a%d" % i for i in range(params)],
        x=y,
        dx=dy,
        prior_par=prior,
    )
    print("setting f-g links")
    for temp in pms_x.trees.keys():
        pms_x.trees[temp].fy = pms_y.trees[temp]
        pms_y.trees[temp].fy = pms_x.trees[temp]
        # print('refit')
        pms_x.trees[temp].get_bic(reset=True, fit=True)
        pms_x.trees[temp].get_energy(bic=True, reset=True)
    pms_x.t1 = pms_x.trees[str(min(Ts))]
    pms_y.t1 = pms_y.trees[str(min(Ts))]
    print('Initial MCMC model x:',pms_x.t1,pms_x.t1.E)
    print('Initial MCMC model y:',pms_y.t1,pms_y.t1.E)
    stderr_fileno = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    mc_start = time.time()
    for i in range(1, mcmc_steps + 1):
        start = time.time()
        # MCMC update
        pms_x.mcmc_step()  # MCMC step within each T
        pms_y.mcmc_step()
        ET1, ET2 = (
            pms_x.tree_swap()
        )  # Attempt to swap two randomly selected consecutive temps

        if ET1 != None:
            t1 = pms_y.trees[ET1]
            t2 = pms_y.trees[ET2]
            BT1, BT2 = t1.BT, t2.BT
            pms_y.trees[ET1] = t2
            pms_y.trees[ET2] = t1
            t1.BT = BT2
            t2.BT = BT1
            pms_x.trees[ET1].get_bic(reset=True, fit=True)
            pms_x.trees[ET1].get_energy(bic=False, reset=True)
            pms_y.t1 = pms_y.trees[str(min(Ts))]

        dl.append(copy(pms_x.t1.E ))
        # Add the description length to the trace
        # description_lengths.append(pms.t1.E)
        # Check if this is the MDL expression so far
        if pms_x.t1.E < i_mdl:
            # if pms.t1.E==float('NaN'): print('NaN in best model mdl')
            i_mdl = copy(pms_x.t1.E)
            model_x = deepcopy(pms_x.t1)
            model_y = deepcopy(pms_y.t1)
    return dl,i_mdl,model_x,model_y
    
with Pool(processes=2, maxtasksperchild=1) as pool:
    results = pool.map(run_instance, range(2))
for result in results:
    #dl, , smooth_E, combo_x, combo_y = result
    print('parallel res:',result[1:])
    description_lengths.append(result[0])
    if result[1] <mdl:
        description_lengths.append(result[0])
        mdl=copy(result[1])
        mdl_model_x = deepcopy(result[2])
        mdl_model_y = deepcopy(result[3])

file_name = os.path.basename(file)
with open(f'./models/IBMS_{file_name}', 'wb') as f:
    # A new file will be created
    pickle.dump({'x':mdl_model_x,'y':mdl_model_y}, f)
for r in description_lengths:
    plt.plot(r)
plt.yscale('symlog')
plt.savefig(f'./models/IBMS_{file_name[:-4]}_dl.pdf',format='pdf')
plt.clf()

from zsindy.ml_module import ZSindy
import pysindy


X = data[['x','y']].to_numpy()

t = data['t'].to_numpy()

x0 = X[0]
poly_degree = 5

# Zsindy parameters
e_ensemble_trials = 10
z_ensemble_trials = e_ensemble_trials
rho = 0.00001 # Resolution ¿error? hyperparameter
zsindy_num_terms = 5
lmbda = 1e5
varnames = ['x','y']

## Z-Sindy
zmodel = ZSindy(poly_degree=poly_degree,
                variable_names=varnames)

zmodel.fit(X, t)
print('end main')