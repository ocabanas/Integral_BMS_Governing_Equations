import pandas as pd
import numpy as np
import sys
import warnings
warnings.filterwarnings('ignore')
from copy import deepcopy,copy
from datetime import datetime
import pickle
import os
import scipy
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

# Import Machine Scientist
from importlib.machinery import SourceFileLoader
# Get the absolute path of the script's directory
script_dir = os.getcwd()
# Define the relative path to the module
relative_module_path = "rguimera-machine-scientist-constrained/machinescientist_ode.py"
path = os.path.join(script_dir, relative_module_path)
ms_ode = SourceFileLoader("ms_ode", path).load_module()


folder_name = 'Train_test_data_lin_term_com2025_03_11-11_21_44'
"""
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

def RK_BMS(model,y0,h,steps,col):
    res=[y0]
    for i in range(steps-1):
        k1 = h*model.predict({col:pd.DataFrame(data={'B':[res[-1]]})})[col].to_numpy()[0]
        k2 = h*model.predict({col:pd.DataFrame(data={'B':[res[-1]+k1/2.]})})[col].to_numpy()[0]
        k3 = h*model.predict({col:pd.DataFrame(data={'B':[res[-1]+k2/2.]})})[col].to_numpy()[0]
        k4 = h*model.predict({col:pd.DataFrame(data={'B':[res[-1]+k3]})})[col].to_numpy()[0]

        # Calculate new x and y
        res.append(res[-1] + 1./6*(k1+2*k2+2*k3+k4))
        """x = x + dx
        f=model.predict({col:pd.DataFrame(data={'B':[res[-1]]})})[col].to_numpy()[0]
        res.append(res[-1]+h*f)"""
    return res



# Uncomment to fit logistic growth

string ='((_a1_ * B) + ((pow2(B) * _a2_) + _a0_))' #Logistic

def loss_ode(param):
    model=[]
    for value in x[col].B.to_numpy():
        model.append(value*param[0]+value*value*param[1]+param[2])
    return np.sum(np.square(np.subtract(model,x[col].dx.to_numpy())))
def ode0(y,t,a,b,c):
    return y*a+y*y*b+c


bms_fulldata_new = ms_ode.from_string_model(x,y,string,1,8,['B'],silence=True)
print(bms_fulldata_new)
print(bms_fulldata_new.E)
print(bms_fulldata_new.sse)

# Fit process
for col in list(x.keys()):
    
    if np.isinf(bms_fulldata_new.sse[col]) or (bms_fulldata_new.par_values[col]['_a1_']<0. and bms_fulldata_new.par_values[col]['_a2_']>0.):
        
        fit = scipy.optimize.minimize(loss_ode,[1.,1.,1.],bounds=((0.,10.),(-10.,0.),(0.,x[col].B.to_numpy())).x
        print(fit)
        
        """
        if bms_fulldata_new.par_values[col]['_a1_']<0. and bms_fulldata_new.par_values[col]['_a2_']>0.:
            print('inverted parabola?')
            plt.plot(x[col].B.to_numpy(),x[col].dx.to_numpy(),label='ground truth')
            plt.plot(x[col].B.to_numpy(),ode0(x[col].B.to_numpy(),x[col].t.to_numpy(),
                                              bms_fulldata_new.par_values[col]['_a1_'],
                                                bms_fulldata_new.par_values[col]['_a2_'],
                                                bms_fulldata_new.par_values[col]['_a0_']),
                                              label='model')
            plt.plot(x[col].B.to_numpy(),ode0(x[col].B.to_numpy(),x[col].t.to_numpy(),*fit),
                                              label='new fit model')
            plt.show()
        """
            
        bms_fulldata_new.par_values[col]['_a1_'] = deepcopy(fit[0])
        bms_fulldata_new.par_values[col]['_a2_'] = deepcopy(fit[1])
        bms_fulldata_new.par_values[col]['_a0_'] = deepcopy(fit[2])
        bms_fulldata_new.x0[str(bms_fulldata_new)][col]=deepcopy(x[col].B.to_numpy()[0])
        
        bms_fulldata_new.get_bic(reset=True,fit=False,verbose=True)
        
        if np.isinf(bms_fulldata_new.sse[col]):
            success=False
            init_points=np.linspace(x[col].B.to_numpy()[0],1.,1000)
            for i in init_points:
                bms_fulldata_new.x0[str(bms_fulldata_new)][col]=deepcopy(i)
                bms_fulldata_new.get_bic(reset=True,fit=False,verbose=True)
                print(i,bms_fulldata_new.sse[col])
                h=x[col].t[1]-x[col].t[0]
                int0 = RK_BMS(bms_fulldata_new,i,h,len(y[col]),col)
                if not np.isinf(bms_fulldata_new.sse[col]) and not np.isnan(sum(int0)) and not np.isinf(sum(int0)):
                    print('break!',bms_fulldata_new.sse[col],sum(int0))
                    success=True
                    break
            if success==None:
                print('sse not successful',bms_fulldata_new.sse[col])
                plt.plot(x[col].B.to_numpy(),x[col].dx.to_numpy(),label='ground truth')
                plt.plot(x[col].B.to_numpy(),ode0(x[col].B.to_numpy(),x[col].t.to_numpy(),*fit),label='model')
                plt.show()


# Assign parameters and evaluate energy

bms_fulldata_new.fit_par={}
bms_fulldata_new.get_bic(reset=True,fit=False,verbose=True)
bms_fulldata_new.get_energy(reset=True)

print(bms_fulldata_new)
print(bms_fulldata_new.E)
print(bms_fulldata_new.sse)



f_name=f'{folder_name}/logistic_test_fit_t.pkl'
file = open(f_name,'wb')
pickle.dump(bms_fulldata_new,file)
file.close()