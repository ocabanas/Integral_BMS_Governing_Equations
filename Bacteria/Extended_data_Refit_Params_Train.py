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
ms = SourceFileLoader("ms", path).load_module()


folder_name = 'Train_test_data_lin_term_com2025_03_11-11_21_44'
with open(f'./{folder_name}/x.pkl', 'rb') as file:
    # A new file will be created
    x = pickle.load(file)
    
with open(f'./{folder_name}/y.pkl', 'rb') as file:
    # A new file will be created
    y = pickle.load(file)


#f_name_train='Full_data_lin_term_com_2025_03_27-06_00_27'
#f_name_train='Full_data_lin_term_com_2025_03_20-06_17_10'
f_name_train='Full_data_lin_term_prod_2025_03_27-06_00_44'
try:
    file = open(f'./{f_name_train}/model_mdl.pkl','rb')
except:
    file = open(f'./{f_name_train}/mdl.pkl','rb')
print(file)
bms_fulldata = pickle.load(file)
file.close()

bms_fulldata.get_bic(reset=True,fit=True)
bms_fulldata.get_energy(reset=True)

print('BMS Train',bms_fulldata.E,bms_fulldata.EB,bms_fulldata.EP)

############
#Generate new model fromstring

string = f'{bms_fulldata.constraint[0]}{bms_fulldata.pr(show_pow=True)}{bms_fulldata.constraint[1]}'
print(string)
bms_fulldata_new = ms.from_string_model(x, y,string , 1, 8, ['B'])
#bms_fulldata_new.fit_par={}
#bms_fulldata_new.x0[str(bms_fulldata_new)]=deepcopy(bms_fulldata.x0[str(bms_fulldata)])
#bms_fulldata_new.par_values=deepcopy(bms_fulldata.par_values)
#bms_fulldata_new.get_bic(reset=True,fit=False)
#bms_fulldata_new.get_energy(reset=True)
print('New Train (recovered pars)',bms_fulldata_new.E,bms_fulldata_new.EB,bms_fulldata_new.EP)
#bms_fulldata_new=deepcopy(bms_fulldata)
print(bms_fulldata_new.E)

print('types',bms_fulldata_new.E,type(bms_fulldata_new.E))
print(float(bms_fulldata_new.E))

bms_fulldata_new.fit_par={}
bms_fulldata.par_values['D Mannose_10']={'_a0_': 1., '_a2_': 1., '_a4_': 1., '_a6_': 1.}
bms_fulldata_new.x0[str(bms_fulldata_new)]=deepcopy(bms_fulldata.x0[str(bms_fulldata)])
bms_fulldata_new.par_values=deepcopy(bms_fulldata.par_values)
bms_fulldata_new.get_bic(reset=True,fit=True,verbose=True)
bms_fulldata_new.get_energy(reset=True)
print('New Train (initi refit pars)',bms_fulldata_new.E,bms_fulldata_new.EB,bms_fulldata_new.EP)

if np.isinf(float(bms_fulldata_new.E)):
    print('Error in fitting pars')
    print('SSE',bms_fulldata_new.sse)
    quit()

old_par_values=deepcopy(bms_fulldata_new.par_values)
old_cols=list(old_par_values.keys())


f_name=f'{f_name_train}/mdl_refit_train2.pkl'
file = open(f_name,'wb')
pickle.dump(bms_fulldata_new,file)
file.close()
print('Energy before refitting:',bms_fulldata_new.E)
old_energy = deepcopy(bms_fulldata_new.E)

for o_col in old_cols:
    for n_col in old_cols:
        #print(o_col,n_col)
        #test_model=deepcopy(bms_fulldata_new)
        bms_fulldata_new.fit_par={}
        p=deepcopy(bms_fulldata_new.par_values[n_col])
        bms_fulldata_new.par_values[n_col]=deepcopy(old_par_values[o_col])
        bms_fulldata_new.get_bic(reset=True,fit=True)
        bms_fulldata_new.get_energy(reset=True)
        if bms_fulldata_new.E<old_energy and not np.isinf(float(bms_fulldata_new.E)):
            print('Better fit. Update model.',bms_fulldata_new.E)
            file = open(f_name,'wb')
            pickle.dump(bms_fulldata_new,file)
            file.close()
            old_energy=deepcopy(bms_fulldata_new.E)
        else:
            # Not improving energy. Setting previorus parameters
            bms_fulldata_new.par_values[n_col]=deepcopy(p)


