import pandas as pd
import numpy as np
import sys
import warnings
warnings.filterwarnings('ignore')
from copy import deepcopy,copy
from datetime import datetime
import pickle
import os

# Import Machine Scientist
from importlib.machinery import SourceFileLoader
# Get the absolute path of the script's directory
script_dir = os.getcwd()
# Define the relative path to the module
relative_module_path = "rguimera-machine-scientist-constrained/machinescientist_ode.py"
path = os.path.join(script_dir, relative_module_path)
ms_ode = SourceFileLoader("ms_ode", path).load_module()


folder_name = 'Train_test_data_lin_term_com2025_03_11-11_21_44'

with open(f'./{folder_name}/x.pkl', 'rb') as file:
    # A new file will be created
    x = pickle.load(file)
    
with open(f'./{folder_name}/y.pkl', 'rb') as file:
    # A new file will be created
    y = pickle.load(file)
"""
with open(f'./{folder_name}/x_test.pkl', 'rb') as file:
    # A new file will be created
    x=pickle.load(file)
    
with open(f'./{folder_name}/y_test.pkl', 'rb') as file:
    # A new file will be created
    y=pickle.load(file)
"""

# In[5]:


string ='((_a1_ * B) - (pow2(B) * (_a1_ / _a2_)))'  #Logistic
string ='((_a0_ * B) - (_a0_ * (B * ((_a1_ / B) ** _a2_))))' #Generalized Logistic mdel
#string = '(_a0_ * (B * (log((_a3_ / B)))))' #Gompertz
bms_fulldata_new = ms_ode.from_string_model(x,y,string,1,8,['B'],silence=True)

print(bms_fulldata_new)
print(bms_fulldata_new.E)
cols=bms_fulldata_new.x.keys()
initial_fit_pars=deepcopy(bms_fulldata_new.par_values)
f_name=f'{folder_name}/genlogistic_train1.pkl'
file = open(f_name,'wb')
pickle.dump(bms_fulldata_new,file)
file.close()
print(bms_fulldata_new.E)
old_energy = deepcopy(bms_fulldata_new.E)
initial_fit_pars=deepcopy(bms_fulldata_new.par_values)
for o_col in cols:
    for n_col in cols:
        #print(o_col,n_col)
        #test_model=deepcopy(bms_fulldata_new)
        bms_fulldata_new.fit_par={}
        p = deepcopy(bms_fulldata_new.par_values[n_col])
        bms_fulldata_new.par_values[n_col]=deepcopy(initial_fit_pars[o_col])
        bms_fulldata_new.get_bic(reset=True,fit=True)
        bms_fulldata_new.get_energy(reset=True)
        if bms_fulldata_new.E<old_energy:
            print('Better fit. Update model.',bms_fulldata_new.E)
            file = open(f_name,'wb')
            pickle.dump(bms_fulldata_new,file)
            file.close()
            old_energy=deepcopy(bms_fulldata_new.E)
        else:
            # Not improving energy. Setting previorus parameters
            bms_fulldata_new.par_values[n_col]=deepcopy(p)