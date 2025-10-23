import pandas as pd
import numpy as np
import sys
import warnings
warnings.filterwarnings('ignore')
from copy import deepcopy,copy
from datetime import datetime
import pickle
import os
#import pynumdiff

# Import Machine Scientist
from importlib.machinery import SourceFileLoader
# Get the absolute path of the script's directory
script_dir = os.getcwd()
# Define the relative path to the module
relative_module_path = "rguimera-machine-scientist-constrained/machinescientist_ode.py"
path = os.path.join(script_dir, relative_module_path)
ms = SourceFileLoader("ms", path).load_module()


folder_name='Train_test_data_lin_term_com2025_03_11-11_21_44'
"""
x={}
y={}

# Dataset growth:
data=pd.read_pickle('microbial_growth_full3.pkl')
data['t'] = data['t'].shift(3)
data.drop(index=data.index[:3], axis=0, inplace=True)
data.reset_index(inplace=True)
with open(f'./{folder_name}/train_test_0.pkl', 'rb') as file:
    # A new file will be created
    test_labels=pickle.load(file)['test']
for col in test_labels:
    No=data[col].to_numpy()[0]
    y[col]=pd.Series(deepcopy(data[col]))
    par = [2,21,21]
    h=data.t.to_numpy()[1]-data.t.to_numpy()[0]
    x_hat, dxdt_hat = pynumdiff.linear_model.polydiff(data[col].to_numpy(), h, par, options=None)
    x[col]=deepcopy(data[['t',col]].rename(columns={col:'B'}))
    x[col]['dx']=dxdt_hat

# Dataset growth 6:
data=pd.read_csv('microbial_growth_full6.csv')
csv_cols=data.columns.to_numpy()
data=data.rename(columns={csv_cols[0]:'t'})
data['t'] = data['t'].shift(4)
data.drop(index=data.index[:4], axis=0, inplace=True)
data.reset_index(inplace=True)
with open(f'./{folder_name}/train_test_6.pkl', 'rb') as file:
    # A new file will be created
    test_labels=pickle.load(file)['test']
for col in test_labels:
    No=data[col].to_numpy()[0]
    y[col+'_6']=pd.Series(deepcopy(data[col]))
    par = [2,21,21]
    h=data.t.to_numpy()[1]-data.t.to_numpy()[0]
    x_hat, dxdt_hat = pynumdiff.linear_model.polydiff(data[col].to_numpy(), h, par, options=None)
    x[col+'_6']=deepcopy(data[['t',col]].rename(columns={col:'B'}))
    x[col+'_6']['dx']=dxdt_hat

# Dataset growth 10:
data=pd.read_csv('microbial_growth_full10.csv')
csv_cols=data.columns.to_numpy()
data=data.rename(columns={csv_cols[0]:'t'})
data['t'] = data['t'].shift(4)
data.drop(index=data.index[:4], axis=0, inplace=True)
def convert_to_hours(time_str):
    parts = time_str.split(':')
    
    # If format is hh:mm:ss
    if len(parts) == 3:
        hours, minutes, seconds = map(int, parts)
    # If format is mm:ss (assume hours = 0)
    elif len(parts) == 2:
        hours = 0
        minutes, seconds = map(int, parts)
    else:
        return np.nan  # Handle unexpected cases
    print(parts)
    return hours + minutes / 60. + seconds / 3600.  # Convert to hours
data['t'] = data['t'].apply(lambda x: convert_to_hours(x))
data.reset_index(inplace=True)
with open(f'./{folder_name}/train_test_10.pkl', 'rb') as file:
    # A new file will be created
    test_labels=pickle.load(file)['test']
for col in test_labels:
    No=data[col].to_numpy()[0]
    y[col+'_10']=pd.Series(deepcopy(data[col]))
    par = [2,21,21]
    h=data.t.to_numpy()[1]-data.t.to_numpy()[0]
    x_hat, dxdt_hat = pynumdiff.linear_model.polydiff(data[col].to_numpy(), h, par, options=None)
    x[col+'_10']=deepcopy(data[['t',col]].rename(columns={col:'B'}))
    x[col+'_10']['dx']=dxdt_hat

# Dataset growth 12:
data=pd.read_csv('microbial_growth_full12.csv')
csv_cols=data.columns.to_numpy()
data=data.rename(columns={csv_cols[0]:'t'})
data['t'] = data['t'].shift(4)
data.drop(index=data.index[:4], axis=0, inplace=True)
data.reset_index(inplace=True)
with open(f'./{folder_name}/train_test_12.pkl', 'rb') as file:
    # A new file will be created
    test_labels=pickle.load(file)['test']
for col in test_labels:
    No=data[col].to_numpy()[0]
    y[col+'_12']=pd.Series(deepcopy(data[col]))
    par = [2,21,21]
    h=data.t.to_numpy()[1]-data.t.to_numpy()[0]
    x_hat, dxdt_hat = pynumdiff.linear_model.polydiff(data[col].to_numpy(), h, par, options=None)
    x[col+'_12']=deepcopy(data[['t',col]].rename(columns={col:'B'}))
    x[col+'_12']['dx']=dxdt_hat

# Dataset growth 18:
data=pd.read_csv('microbial_growth_full18.csv',header=1)
csv_cols=data.columns.to_numpy()
data=data.rename(columns={csv_cols[0]:'t'})
data['t'] = data['t'].shift(4)
data.drop(index=data.index[:4], axis=0, inplace=True)
data.reset_index(inplace=True)
with open(f'./{folder_name}/train_test_18.pkl', 'rb') as file:
    # A new file will be created
    test_labels=pickle.load(file)['test']
for col in test_labels:
    No=data[col].to_numpy()[0]
    y[col+'_18']=pd.Series(deepcopy(data[col]))
    par = [2,21,21]
    h=data.t.to_numpy()[1]-data.t.to_numpy()[0]
    x_hat, dxdt_hat = pynumdiff.linear_model.polydiff(data[col].to_numpy(), h, par, options=None)
    x[col+'_18']=deepcopy(data[['t',col]].rename(columns={col:'B'}))
    x[col+'_18']['dx']=dxdt_hat

# Dataset growth 19:
data=pd.read_csv('microbial_growth_full19.csv')
csv_cols=data.columns.to_numpy()
data=data.rename(columns={csv_cols[0]:'t'})
data['t'] = data['t'].shift(4)
data.drop(index=data.index[:4], axis=0, inplace=True)
data.reset_index(inplace=True)
with open(f'./{folder_name}/train_test_19.pkl', 'rb') as file:
    # A new file will be created
    test_labels=pickle.load(file)['test']
for col in test_labels:
    No=data[col].to_numpy()[0]
    y[col+'_19']=pd.Series(deepcopy(data[col]))
    par = [2,21,21]
    h=data.t.to_numpy()[1]-data.t.to_numpy()[0]
    x_hat, dxdt_hat = pynumdiff.linear_model.polydiff(data[col].to_numpy(), h, par, options=None)
    x[col+'_19']=deepcopy(data[['t',col]].rename(columns={col:'B'}))
    x[col+'_19']['dx']=dxdt_hat
"""

with open(f'./{folder_name}/x_test.pkl', 'rb') as file:
    # A new file will be created
    x=pickle.load(file)
    
with open(f'./{folder_name}/y_test.pkl', 'rb') as file:
    # A new file will be created
    y=pickle.load(file)


# In[5]:


#f_name_train='Full_data_lin_term_com_2025_03_27-06_00_27'
f_name_train='Full_data_lin_term_prod_2025_03_27-06_00_44'
try:
    file = open(f'./{f_name_train}/mdl_refit_train1.pkl','rb')
except:
    pass
print(file)
bms_fulldata = pickle.load(file)
print(bms_fulldata)
print(bms_fulldata.E)

old_par_values=deepcopy(bms_fulldata.par_values)
old_cols=list(old_par_values.keys())

bms_fulldata_new=deepcopy(bms_fulldata)
bms_fulldata_new.x=deepcopy(x)
bms_fulldata_new.y=deepcopy(y)
bms_fulldata_new.fit_par={}
new_pars={}
new_cols=deepcopy(list(bms_fulldata_new.x.keys()))
for col in new_cols:
    if col in old_cols:
        new_pars[col]=deepcopy(old_par_values[col])
    else:
        new_pars[col]=deepcopy(old_par_values[old_cols[0]])
bms_fulldata_new.par_values=new_pars
bms_fulldata_new.get_bic(reset=True,fit=True)
bms_fulldata_new.get_energy(reset=True)
initial_fit_pars=deepcopy(bms_fulldata_new.par_values)
f_name=f'{f_name_train}/mdl_refit_test1.pkl'
file = open(f_name,'wb')
pickle.dump(bms_fulldata_new,file)
file.close()
print(bms_fulldata_new.E)
old_energy = deepcopy(bms_fulldata_new.E)
for o_col in old_cols:
    for n_col in new_cols:
        #print(o_col,n_col)
        #test_model=deepcopy(bms_fulldata_new)
        bms_fulldata_new.fit_par={}
        p = deepcopy(bms_fulldata_new.par_values[n_col])
        bms_fulldata_new.par_values[n_col]=deepcopy(old_par_values[o_col])
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


# In[ ]:




