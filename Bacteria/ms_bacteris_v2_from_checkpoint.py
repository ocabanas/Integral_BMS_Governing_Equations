import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")
from copy import deepcopy, copy
from datetime import datetime
import pickle
from sklearn.model_selection import train_test_split
import sys, getopt
import matplotlib.pyplot as plt
import pynumdiff
import time
# Import Machine Scientist
from importlib.machinery import SourceFileLoader

# Get the absolute path of the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
# Define the relative path to the module
relative_module_path = "rguimera-machine-scientist-constrained/machinescientist_ode.py"
path = os.path.join(script_dir, relative_module_path)
ms = SourceFileLoader("ms", path).load_module()


# Read data folder + checkpoint
# Read mdl
folder_name='Full_data_lin_term_prod_2025_03_20-06_17_27/'
checkpoint=1000
with open(f"./{folder_name}checkpoint_run_0_step_{checkpoint}.pkl", "rb") as file:
    # A new file will be created
    ms_ckp = pickle.load(file)
pms = ms_ckp['object']

with open(f"./{folder_name}mdl.pkl", "rb") as file:
    # A new file will be created
    ms_mdl = pickle.load(file)

print('checkpoint model',ms_ckp['object'].t1)
print('mdl model',ms_mdl)

description_lengths=[]
mdl = copy(ms_mdl.E)
mdl_model = deepcopy(ms_mdl)
# Loop until end
for i in range(checkpoint,5000):
    #MCMC update and PT
    pms.mcmc_step()  # MCMC step within each T

    if abs(pms.t1.E - pms.t1.get_energy(bic=True, reset=True)[0]) > 1.0e-6:
        print("Reset energy")
        for tree in pms.trees.values():
            tree.get_energy(bic=True, reset=True)
    ET1, ET2 = (
        pms.tree_swap()
    )  # Attempt to swap two randomly selected consecutive temps
    description_lengths.append(copy(pms.t1.E))
    
    # mdl update
    if pms.t1.E < mdl:
        # if pms.t1.E==float('NaN'): print('NaN in best model mdl')
        mdl, mdl_model = copy(pms.t1.E), deepcopy(pms.t1)
        with open(f"./{folder}mdl_ckp.pkl", "wb") as file:
            # A new file will be created
            pickle.dump(pms.t1, file)


with open(f"./{folder_name}model_mdl_ckp.pkl", "wb") as file:
    # A new file will be created
    pickle.dump(best_model, file)


plt.plot(dls)
plt.savefig(f"./{folder_name}model_mdl_dl_ckp.pdf", format="pdf")
plt.clf()
print()
print(file, best_model, best_model.E)
print(best_model.par_values)
print(best_model.x0)
print(len(best_model.x0), len(best_model.fit_par))


# fig_dl.savefig(f'./{folder_name}/1description_length_B.pdf',format='pdf')
file1.write(f"Best model: {best_model}\n")
file1.write(f"DL: {best_model.E}\n")
file1.write(f"Latex: {best_model.latex()}\n")
file1.write(f"Parameters: {best_model.par_values}\n")
file1.write("###################### \n")
