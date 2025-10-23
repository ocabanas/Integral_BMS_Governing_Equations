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
import warnings
#warnings.filterwarnings("ignore")
#warnings.filterwarnings("ignore", category=RuntimeWarning)
stderr_fileno = sys.stderr
sys.stderr = open(os.devnull, 'w')

src = os.path.dirname('/export/home/oriolca/Integral_BMS_Governing_Equations/I-BMS/')
sys.path.append(src)
from mcmc_ode import *
from parallel_ode import *

path = os.path.join(src, "Prior/")
sys.path.append(path)
from fit_prior import read_prior_par

priors = {
    "v2_p3": f"Prior/final_prior_param_sq.named_equations.nv2.np3.2016-09-09 18:49:42.927679.dat",
    "v2_p4": f"Prior/final_prior_param_sq.named_equations.nv2.np4.2016-09-09 18:49:43.056910.dat",
    "v2_p8": f"Prior/final_prior_param_sq.named_equations.nv2.np8.2016-09-09 18:49:42.800618.dat",
}

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

print(data)

x = data[['x','y']]
t = pd.Series(data[['t']].t)

dt = t.to_numpy()[1] - t.to_numpy()[0]

dx = pd.DataFrame(data = {'x': np.gradient(data["x"], dt), 'y': np.gradient(data["y"], dt)})

print(dx)

XLABS = ['x','y']
n_params = 8

path = os.path.join(src, priors[f"v{len(XLABS)}_p{str(n_params)}"])
prior_par = read_prior_par(path)

Ts=[1] + [1.04**k for k in range(1, 40,2)]

del prior_par['Nopi_abs']
del prior_par['Nopi2_abs']
del prior_par['Nopi_tan']
del prior_par['Nopi2_tan']
del prior_par['Nopi_sinh']
del prior_par['Nopi2_sinh']
del prior_par['Nopi_cosh']
del prior_par['Nopi2_cosh']
del prior_par['Nopi_tanh']
del prior_par['Nopi2_tanh']


OPS = {
    "sin": 1,
    "cos": 1,
    "exp": 1,
    "pow2": 1,
    "pow3": 1,
    "-": 1,
    "+": 2,
    "*": 2,
    "/": 2,
    "**": 2,
}
import time

start = time.time()
mdl = np.inf
mdl_x = None
mdl_y = None
dl=[[],[]]
for resets in range(0,2):
    pms_x = Parallel(
        Ts,
        ops=OPS,
        variables=XLABS,
        parameters=["a%d" % i for i in range(n_params)],
        x=x, t=t, dx=dx,
        prior_par=prior_par,
    )
    
    print(pms_x.t1)
    print(pms_x.t1.E)
    
    pms_y = Parallel(
        Ts,
        ops=OPS,
        variables=XLABS,
        parameters=["a%d" % i for i in range(n_params)],
        x=x, t=t, dx=dx,
        prior_par=prior_par,
    )
    print(pms_y.t1)
    print(pms_y.t1.E)
    
    for temp in pms_x.Ts:
        print(pms_x.trees[temp],pms_y.trees[temp])
        print(pms_x.trees[temp].E,pms_y.trees[temp].E)
    
    couplings=[pms_x,pms_y]
    pms_x.set_couplings(couplings)
    
    
    for temp in pms_x.Ts:
        print(pms_x.trees[temp],pms_y.trees[temp])
        print(pms_x.trees[temp].E,pms_y.trees[temp].E)
    
    for step in range(4000):
        pms_x.mcmc_step()
        pms_y.mcmc_step()
        pms_x.tree_swap()
        print('###########################')
        if pms_x.t1.E < mdl:
            mdl_x = deepcopy(pms_x.t1)
            mdl_y = deepcopy(pms_y.t1)
            mdl = deepcopy(pms_x.t1.E)
    dl[resets].append(pms_x.t1.E)
print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
print('END')
print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')

with open(f'models/{file[:-4]}_IBMS.pkl', 'wb') as f:
    # A new file will be created
    pickle.dump({'x':mdl_x,'y':mdl_y}, f)
for r in dl:
    plt.plot(r)
plt.yscale('symlog')
plt.savefig(f'models/{file[:-4]}_IBMS.pdf',format='pdf')
plt.clf()

end = datetime.now()

elapsed = end - start
elapsed_hours = elapsed.total_seconds() / 3600
print(f"Elapsed time: {elapsed_hours:.2f} hours")
