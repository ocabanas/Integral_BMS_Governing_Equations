import pandas as pd
import numpy as np
import sys
import warnings
import gc
from copy import deepcopy,copy
from datetime import datetime
import pickle
import os
import random
from math import ceil,sqrt
from scipy.optimize import curve_fit
# Catch stout
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
prior = read_prior_par('../rguimera-machine-scientist/Prior/final_prior_param_sq.named_equations.nv2.np8.2016-09-09 18:49:42.800618.dat')

import sys, getopt

def main(argv):
    inputfile = ''
    outputfile = ''
    try:
        opts, args = getopt.getopt(argv,"h:f:s:",["file=",'smooth='])
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
        elif opt in ("-s", "--smooth"):
            smooth = eval(arg)
    return file,smooth

file,smooth=main(sys.argv[1:])
print('parsed args:', file,smooth,type(smooth))
#file=f'{sigma}_{d}.pkl'

data=pd.read_csv(file)
print(file[:-4], smooth)

#####################################
#  x
#####################################
x={}
y={}

# Derivate 3point
if not smooth:
    x['d0']=deepcopy(data)
    B=x['d0'].x.values
    h=x['d0'].t.to_numpy()[1]-x['d0'].t.to_numpy()[0]
    x_hat_finite, dxdt_hat_finite = pynumdiff.finite_difference._finite_difference.second_order(B, h)
    y['d0']=pd.Series(dxdt_hat_finite)
    append=''
#Smooth derivate
if smooth:
    
    par = [2,21,21]
    h=data.t.to_numpy()[1]-data.t.to_numpy()[0]
    x_hat, dxdt_hat = pynumdiff.linear_model.polydiff(data.x.to_numpy(), h, par, options=None)
    y_hat, dydt_hat = pynumdiff.linear_model.polydiff(data.y.to_numpy(), h, par, options=None)
    # Setting smoothed deriv
    y['d0']=pd.Series(dxdt_hat)
    # Setting smoothed states
    x['d0']=deepcopy(pd.DataFrame(data={'x':x_hat,'y':y_hat}))
    append='_smooth'

mcmc_resets = 2
mcmc_steps = 3000
XLABS = ['x','y']
params = 8


mdl_x = np.inf
best_model_x = None
dlsx = []

for r in range(mcmc_resets):

    pms_x = ms.Parallel(
        Ts = [1] + [1.04**k for k in range(1, 20)],
        variables=XLABS,
        parameters=['a%d' % i for i in range(params)],
        prior_par=prior,
        x=x,
        y=y)
    for step in range(mcmc_steps):
        pms_x.mcmc_step()
        pms_x.tree_swap()
        if pms_x.t1.E < mdl_x:
            mdl_x = deepcopy(pms_x.t1.E)
            best_model_x = deepcopy(pms_x.t1)

        dlsx.append(pms_x.t1.E)

del pms_x
#####################################
#  y
#####################################
x={}
y={}

# Derivate 3point
if not smooth:
    x['d0']=deepcopy(data)
    B=x['d0'].y.values
    h=x['d0'].t.to_numpy()[1]-x['d0'].t.to_numpy()[0]
    y_hat_finite, dydt_hat_finite = pynumdiff.finite_difference._finite_difference.second_order(B, h)
    y['d0']=pd.Series(dydt_hat_finite)
    append=''
#Smooth derivate
if smooth:
    par = [2,21,21]
    h=data.t.to_numpy()[1]-data.t.to_numpy()[0]
    x_hat, dxdt_hat = pynumdiff.linear_model.polydiff(data.x.to_numpy(), h, par, options=None)
    y_hat, dydt_hat = pynumdiff.linear_model.polydiff(data.y.to_numpy(), h, par, options=None)
    # Setting smoothed deriv
    y['d0']=pd.Series(dydt_hat)
    # Setting smoothed states
    x['d0']=deepcopy(pd.DataFrame(data={'x':x_hat,'y':y_hat}))

mcmc_resets = 2
mcmc_steps = 3000
XLABS = ['x','y']
params = 8


mdl_y = np.inf
best_model_y = None
dlsy = []

for r in range(mcmc_resets):

    pms_y = ms.Parallel(
        Ts = [1] + [1.04**k for k in range(1, 20)],
        variables=XLABS,
        parameters=['a%d' % i for i in range(params)],
        prior_par=prior,
        x=x,
        y=y)
    for step in range(mcmc_steps):
        pms_y.mcmc_step()
        pms_y.tree_swap()
        if pms_y.t1.E < mdl_y:
            mdl_y = deepcopy(pms_y.t1.E)
            best_model_y = deepcopy(pms_y.t1)

        dlsy.append(pms_y.t1.E)

del pms_y
print(best_model_x,best_model_y)
#######################################
file_name = os.path.basename(file)
with open(f'./noise_data_fit{append}/{file_name[:-4]}.pkl', 'wb') as f:
    # A new file will be created
    pickle.dump({'x':best_model_x,'y':best_model_y}, f)
plt.plot(dlsx)
plt.plot(dlsy)
plt.savefig(f'./noise_data_fit{append}/{file_name[:-4]}_dl.pdf',format='pdf')
plt.clf()
del data,x,y,h,best_model_x,best_model_y,dls
if smooth:
    del x_hat,dxdt_hat
else:
    del B
gc.collect()
print('end main')