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

string = '((B * _a0_) + (B * (_a1_ * ((B / _a2_) ** _a3_))))'

def loss_ode(param):
    model=[]
    for value in x[col].B.to_numpy():
        model.append(ode0(value,1,param[0],param[1],param[2],param[3]))
    return np.sum(np.square(np.subtract(model,x[col].dx.to_numpy())))
def ode0(y,t,a,b,c,d):
    #return c + y/(a*b)*(1.-(y/d)**(b))
    return y*a-y*c*(y/d)**(b)
def ode1(y,a,b,c,d):
    #return c + y/(a*b)*(1.-(y/d)**(b))
    return y*a-y*c*(y/d)**(b)


bms_fulldata_new = ms_ode.from_string_model(x,y,string,1,8,['B'],silence=True)
print(bms_fulldata_new)
print(bms_fulldata_new.E)
print(bms_fulldata_new.sse)

# Fit process
for col in list(x.keys()):
        
        

        fit1,pcov = scipy.optimize.curve_fit(ode1, x[col].B.to_numpy(), x[col].dx.to_numpy(),maxfev=1000000)

        fit = scipy.optimize.minimize(loss_ode,fit1).x
        print(fit)
        
        print(fit1)
        plt.plot(x[col].B.to_numpy(),x[col].dx.to_numpy(),label='ground truth')
        plt.plot(x[col].B.to_numpy(),ode0(x[col].B.to_numpy(),x[col].t.to_numpy(),*fit),
                                          label='new fit model')
        plt.plot(x[col].B.to_numpy(),ode1(x[col].B.to_numpy(),*fit1),
                                          label='new fit model1')
        plt.draw()

        bms_fulldata_new.par_values[col]['_a0_']=fit1[0]
        bms_fulldata_new.par_values[col]['_a1_']=-fit1[2]
        bms_fulldata_new.par_values[col]['_a2_']=fit1[3]
        bms_fulldata_new.par_values[col]['_a3_']=fit1[1]
        bms_fulldata_new.x0[str(bms_fulldata_new)][col]=deepcopy(x[col].B.to_numpy()[0])
            
        bms_fulldata_new.get_bic(reset=True,fit=False,verbose=True)
        
        if np.isinf(bms_fulldata_new.sse[col]):
            success=False
            init_points=np.linspace(1e-4,1.,1000)
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
            if success==False:
                predictions=bms_fulldata_new.predict(x,verbose=True)
                print('sse not successful',bms_fulldata_new.sse[col])
                plt.plot(x[col].B.to_numpy(),x[col].dx.to_numpy(),label='ground truth')
                plt.plot(x[col].B.to_numpy(),ode0(x[col].B.to_numpy(),x[col].t.to_numpy(),*fit),label='model')
                plt.plot(x[col].B.to_numpy(),predictions[col].to_numpy(),label='BMS')
                plt.legend()
                plt.show()
                h=x[col].t[1]-x[col].t[0]
                y0=deepcopy(x[col].B.to_numpy()[0])
                int1=RK_BMS(bms_fulldata_new,y0,h,len(y[col]),col)
                plt.plot(x[col].t.to_numpy(),x[col].B.to_numpy(),label='data')
                plt.plot(x[col].t.to_numpy(),int1,label='BMS')
                plt.show()



# Assign parameters and evaluate energy

bms_fulldata_new.fit_par={}
bms_fulldata_new.get_bic(reset=True,fit=True,verbose=True)
bms_fulldata_new.get_energy(reset=True)

print(bms_fulldata_new)
print(bms_fulldata_new.E)
print(bms_fulldata_new.sse)



f_name=f'{folder_name}/thetalog_train_fit_t.pkl'
file = open(f_name,'wb')
pickle.dump(bms_fulldata_new,file)
file.close()