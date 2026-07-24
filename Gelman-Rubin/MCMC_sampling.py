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
import matplotlib
matplotlib.use("Agg")
import pynumdiff
import traceback
import time
from sympy.core.cache import clear_cache
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

file='../Lotka-Volterra/noise_data/1.0_0.csv'

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


mcmc_resets = 5
mcmc_steps = 10000
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
del prior["Nopi_tan"]
del prior["Nopi2_tan"]
del prior["Nopi_sinh"]
del prior["Nopi2_sinh"]
del prior["Nopi_cosh"]
del prior["Nopi2_cosh"]
del prior["Nopi_tanh"]
del prior["Nopi2_tanh"]
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
clear_cache()
from multiprocessing import Pool
start = time.time()
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
    description_lengths.append([])
    for i in range(mcmc_steps):
        # MCMC update
        pms_x.mcmc_step()  # MCMC step within each T
        pms_y.mcmc_step()
        ET1, ET2 = (
            pms_x.tree_swap()
        )  # Attempt to swap two randomly selected consecutive temps

        if ET1 != None:
            """
            print('Test couplings. Updating:',ET1,ET2)
            print('X ET1', pms_x.trees[ET1].fy)
            print('X ET2', pms_x.trees[ET2].fy)
            """
            t1 = pms_y.trees[ET1]
            t2 = pms_y.trees[ET2]
            #print(t1)
            #print(t2)
            BT1, BT2 = t1.BT, t2.BT
            pms_y.trees[ET1] = t2
            pms_y.trees[ET2] = t1
            #print(pms_y.trees[ET1])
            #print(pms_y.trees[ET2])
            t1.BT = BT2
            t2.BT = BT1
            
            pms_x.trees[ET1].fy = pms_y.trees[ET1]
            pms_y.trees[ET1].fy = pms_x.trees[ET1]

            pms_x.trees[ET2].fy = pms_y.trees[ET2]
            pms_y.trees[ET2].fy = pms_x.trees[ET2]
            pms_y.t1 = pms_y.trees[pms_y.Ts[0]]
            """
            print('XX ET1', pms_x.trees[ET1].fy)
            print('XX ET2', pms_x.trees[ET2].fy)
            
            print('Energy:', pms_x.trees[ET1].E)
            pms_x.trees[ET1].get_bic(reset=True, fit=True)
            pms_x.trees[ET1].get_energy(bic=True, reset=True)
            print('Energy (Upd.):', pms_x.trees[ET1].E)
            
            print('AFTER UPDATE')
            print('T1', pms_x.trees[ET1],'<-->',pms_x.trees[ET1].fy,pms_x.trees[ET1].E)
            print('T2', pms_x.trees[ET2],'<-->',pms_x.trees[ET2].fy,pms_x.trees[ET2].E)
            print('#######################')
            """
        dl.append(copy(pms_x.t1.E ))
        # Add the description length to the trace
        # description_lengths.append(pms.t1.E)
        # Check if this is the MDL expression so far
        if pms_x.t1.E < i_mdl:
            # if pms.t1.E==float('NaN'): print('NaN in best model mdl')
            print('New best model?', pms_x.t1.E , i_mdl,_)
            i_mdl = deepcopy(pms_x.t1.E)
            model_x = deepcopy(pms_x.t1)
            model_y = deepcopy(pms_y.t1)
            print('New best model', model_x, model_y, i_mdl, pms_x.t1.E,_)
        if i%50==0:
            print(f'Sampled model (Step:{i})', pms_x.t1, pms_y.t1,pms_x.t1.E)
            file_name = os.path.basename(file)
            with open(f'{file_name[:-4]}_IBMSv2_mdl_run_{_}_1.pkl', 'wb') as f:
                # A new file will be created
                pickle.dump({'x':model_x,'y':model_y}, f)
            with open(f'{file_name[:-4]}_IBMSv2_pt_run_{_}_1.pkl', 'wb') as f:
                # A new file will be created
                pickle.dump({'x':pms_x,'y':pms_y}, f)
            pd.Series(dl).to_csv(f"dl_evol_run_{_}.csv", index=False, header=False)
            plt.plot(dl)
            plt.yscale('symlog')
            plt.savefig(f'{file_name[:-4]}_IBMSv2_dl_run_{_}_1.pdf',format='pdf')
            plt.clf()
        clear_cache()
    return dl,i_mdl,model_x,model_y
   
with Pool(processes=mcmc_resets, maxtasksperchild=1) as pool:
    results = pool.map(run_instance, range(mcmc_resets))
    for result in results:
        #for instance in range(mcmc_resets):
        #result = run_instance(instance)
        #dl, , smooth_E, combo_x, combo_y = result
        print('parallel res:',result[1:])
        description_lengths.append(result[0])
        if result[1] <mdl:
            mdl=copy(result[1])
            mdl_model_x = deepcopy(result[2])
            mdl_model_y = deepcopy(result[3])

file_name = os.path.basename(file)
print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
print('END')
print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')

with open(f'{file_name[:-4]}_IBMSv2_1.pkl', 'wb') as f:
    # A new file will be created
    pickle.dump({'x':mdl_model_x,'y':mdl_model_y}, f)
for r in description_lengths:
    plt.plot(r)
plt.yscale('symlog')
plt.savefig(f'{file_name[:-4]}_IBMSv2_1.pdf',format='pdf')
plt.clf()


end = time.time()

elapsed = end - start
elapsed_hours = elapsed / 3600
print(f"Elapsed time: {elapsed_hours:.2f} hours")
