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


# Import Machine Scientist
from importlib.machinery import SourceFileLoader
# Get the absolute path of the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
# Define the relative path to the module
relative_module_path = "rguimera-machine-scientist/machinescientist_ode.py"
path = os.path.join(script_dir, relative_module_path)
ms = SourceFileLoader("ms", path).load_module()


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

data=pd.read_csv('noise_data/'+file)

x={}
y={}

x['A0']=deepcopy(data)
y['A0']=deepcopy(data)
y['A0'].x=deepcopy(x['A0'].y)
y['A0'].y=deepcopy(x['A0'].x)
y['A0'].dx=deepcopy(x['A0'].dy)
y['A0'].dy=deepcopy(x['A0'].dx)
mcmc_resets = 2
mcmc_steps = 4000
XLABS = ['x','y']
params = 8
print(x)
print(y)
best_model_x,best_model_y,dls = ms.machinescientist_to_folder(x,y,
                       XLABS=XLABS,n_params=params,
                       resets=mcmc_resets,
                       steps_prod=mcmc_steps,
                        folder=file
                     )
with open(f'./llac_{file}', 'wb') as f:
    # A new file will be created
    pickle.dump({'x':best_model_x,'y':best_model_y}, f)
for r in dls:
    plt.plot(r)
plt.yscale('symlog')
plt.savefig(f'./results/{file[:-4]}_dl.pdf',format='pdf')
plt.clf()
print('end main')

