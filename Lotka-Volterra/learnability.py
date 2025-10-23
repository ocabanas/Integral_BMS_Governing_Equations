
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
# Imports
import pandas as pd
import numpy as np
import sys
import warnings
from copy import deepcopy,copy
from datetime import datetime
import pickle
import os
#from sympy import sympify,latex,Float
import random
from math import ceil,sqrt
#import seaborn as sbrn
from scipy.optimize import curve_fit
from scipy.stats import bootstrap
# Catch stout
#from io import StringIO 
import sys
import traceback
import gc

# Import Machine Scientist
from importlib.machinery import SourceFileLoader
# Get the absolute path of the script's directory
script_dir = os.getcwd()
# Define the relative path to the module
relative_module_path = "rguimera-machine-scientist/machinescientist_ode.py"
path = os.path.join(script_dir, relative_module_path)
ms_ode = SourceFileLoader("ms_ode", path).load_module()

# Import Machine Scientist (fit)
relative_module_path = "rguimera-machine-scientist/machinescientist_fit.py"
path = os.path.join(script_dir, relative_module_path)
ms_fit = SourceFileLoader("ms_fit", path).load_module()


# In[2]:


#INTEGRATOR ODE
def ode(y,model_x,model_y):
    #k1
    input_x=pd.DataFrame(data=[y],columns=['x','y'])
    input_y=pd.DataFrame(data=[y[::-1]],columns=['x','y'])
    k1 = np.array([h*model_x.predict({'A0':input_x})['A0'].iloc[0],h*model_y.predict({'A0':input_y})['A0'].iloc[0]])
    #k2
    input_x=pd.DataFrame(data=[y+k1/2.],columns=['x','y'])
    input_y=pd.DataFrame(data=[(y+k1/2.)[::-1]],columns=['x','y'])
    k2 = np.array([h*model_x.predict({'A0':input_x})['A0'].iloc[0],h*model_y.predict({'A0':input_y})['A0'].iloc[0]])
    #k3
    input_x=pd.DataFrame(data=[y+k2/2.],columns=['x','y'])
    input_y=pd.DataFrame(data=[(y+k2/2.)[::-1]],columns=['x','y'])
    k3 = np.array([h*model_x.predict({'A0':input_x})['A0'].iloc[0],h*model_y.predict({'A0':input_y})['A0'].iloc[0]])
    #k4
    input_x=pd.DataFrame(data=[y+k3],columns=['x','y'])
    input_y=pd.DataFrame(data=[(y+k3)[::-1]],columns=['x','y'])
    k4 = np.array([h*model_x.predict({'A0':input_x})['A0'].iloc[0],h*model_y.predict({'A0':input_y})['A0'].iloc[0]])
    return [1./6.*(k1[0]+2.*k2[0]+2.*k3[0]+k4[0]),1./6.*(k1[1]+2.*k2[1]+2.*k3[1]+k4[1])]
def RK45(y0,h,t_eval,model_x,model_y):
    x=y0[0]
    y=y0[1]

    t_x=[y0[0]]
    t_y=[y0[1]]
    for i in range(len(t_eval)-1):
        diff=ode([x,y],model_x,model_y)
        x=x+diff[0]
        y=y+diff[1]
        t_x.append(x)
        t_y.append(y)
    return [t_x,t_y]

#def initial_values(x,x0):
#    sol=RK45()
##################################################################################
#INTEGRATOR FIT
def ode_fit(y,model_x,model_y):
    #k1
    input_x=pd.DataFrame(data=[y],columns=['x','y'])
    input_y=pd.DataFrame(data=[y],columns=['x','y'])
    k1 = np.array([h*model_x.predict({'A0':input_x})['A0'].iloc[0],h*model_y.predict({'A0':input_y})['A0'].iloc[0]])
    #k2
    input_x=pd.DataFrame(data=[y+k1/2.],columns=['x','y'])
    input_y=pd.DataFrame(data=[(y+k1/2.)],columns=['x','y'])
    k2 = np.array([h*model_x.predict({'A0':input_x})['A0'].iloc[0],h*model_y.predict({'A0':input_y})['A0'].iloc[0]])
    #k3
    input_x=pd.DataFrame(data=[y+k2/2.],columns=['x','y'])
    input_y=pd.DataFrame(data=[(y+k2/2.)],columns=['x','y'])
    k3 = np.array([h*model_x.predict({'A0':input_x})['A0'].iloc[0],h*model_y.predict({'A0':input_y})['A0'].iloc[0]])
    #k4
    input_x=pd.DataFrame(data=[y+k3],columns=['x','y'])
    input_y=pd.DataFrame(data=[(y+k3)],columns=['x','y'])
    k4 = np.array([h*model_x.predict({'A0':input_x})['A0'].iloc[0],h*model_y.predict({'A0':input_y})['A0'].iloc[0]])
    return [1./6.*(k1[0]+2.*k2[0]+2.*k3[0]+k4[0]),1./6.*(k1[1]+2.*k2[1]+2.*k3[1]+k4[1])]
def RK45_fit(y0,h,t_eval,model_x,model_y):
    x=y0[0]
    y=y0[1]

    t_x=[y0[0]]
    t_y=[y0[1]]
    for i in range(len(t_eval)-1):
        diff=ode_fit([x,y],model_x,model_y)
        x=x+diff[0]
        y=y+diff[1]
        t_x.append(x)
        t_y.append(y)
    return [t_x,t_y]
def func(x,x0,y0):
    sol=RK45_fit([x0,y0],h,t_eval,model_to_fit_x,model_to_fit_y)
    return np.concatenate(sol)

#h=(time[1]-time[0])
#sol=Euler(ode,y0=[x0,y0], h=h ,t_eval=time)


# In[ ]:


# Loop over data files

files=[f for f in os.listdir('noise_data/')]
#print(files)
sigmas=[0.1,0.5,1.,1.5,2.,2.5,3.,3.5,4.,4.5,5.,5.5,6.]
#sigmas=[2.,2.5,3.,3.5,4.]

s_array=[]
ode_lernability=[]
ode_lernability_err=[]
fit_lernability=[]
fit_lernability_err=[]
smooth_lernability=[]
smooth_lernability_err=[]

#true_best_error_ode_final=[]
data_best_error_ode_final=[]
data_best_error_ode_final_err=[]

#true_best_error_fit_final=[]
data_best_error_fit_final=[]
data_best_error_fit_final_err=[]

#true_best_error_smth_final=[]
data_best_error_smth_final=[]
data_best_error_smth_final_err=[]

corpus_true=[('(x * (_a0_ + (_a1_ * y)))','(x * (_a0_ + (_a1_ * y)))'),
             ('(((_a7_ * y) + _a0_) * x)','(((y * (_a6_ + (_a0_ * x))) + _a1_) * x)'),
             ('((_a6_ * (y + _a0_)) * x)','(((_a2_ * (y + _a1_)) * x) + _a0_)'),
             ('(((y + _a6_) * (_a5_ * x)) + _a1_)','(x * (_a2_ + (_a7_ * y)))'),
             ('(((_a5_ + y) * _a2_) * x)','(((y * _a6_) + _a3_) * x)'),
             ('((_a3_ * (y + _a1_)) * x)','(x * ((y * _a2_) + _a3_))'),
             ('(_a4_ * (x * (_a7_ + y)))','(x * (_a2_ * ((_a2_ + y) / _a7_)))'),
             ('((y + _a2_) * (_a6_ * x))','((_a0_ + (y * _a1_)) * (x + _a4_))'),
             ('(((y * _a6_) + _a3_) * (x * _a3_))','(((_a3_ + y) / _a1_) * x)'),
             ('(((y + _a6_) * _a7_) * x)','((_a4_ + (y / _a1_)) * x)'),
             ('((x * _a1_) * (y + _a2_))','(((y / _a6_) + _a4_) * x)'),
             ('(((_a5_ + y) * _a7_) * x)','((((_a7_ ** 2) * y) + _a4_) * x)'),
             ('(((_a0_ * y) + _a3_) * x)','((x / _a6_) * (_a5_ + y))'),
            ]

corpus_true_fits=[ ('(x * (_a0_ + (_a1_ * y)))', '(y * (_a0_ + (_a1_ * x)))')
]

for sigma in sigmas:
    count_ode=0
    l_ode=0
    count_fit=0
    l_fit=0
    count_smth=0
    l_smth=0
    
    #true_best_error_ode=[]
    data_best_error_ode=[]
    
    #true_best_error_fit=[]
    data_best_error_fit=[]
    
    #true_best_error_smth=[]
    data_best_error_smth=[]
    
    print(sigma)
    
    for d in range(0,40):
        file=f'{sigma}_{d}.csv'
        #Lernability for ODE
        print(file)
        try:
            # True Model ODEint
            data=pd.read_csv('noise_data/'+file)
            x={}
            y={}
            
            x['A0']=deepcopy(data)
            y['A0']=deepcopy(data)
            y['A0'].x,y['A0'].y=y['A0'].y,y['A0'].x
            y['A0'].dx,y['A0'].dy=y['A0'].dy,y['A0'].dx
            XLABS = ['x','y']
            params = 8
            list_dl_trues=[]
            true_x=ms_ode.from_string_model(x,{'A0':10.},'((_a0_ * x) + (_a1_ * (x * y)))',2,8,['x','y'],silence=True)
            true_y=ms_ode.from_string_model(y,{'A0':5.},'((_a0_ * x) + (_a1_ * (x * y)))',2,8,['x','y'],silence=True)
            true_x.fy,true_y.fy=true_y,true_x
            true_x.x0_guess={'A0':10.}
            true_y.x0_guess={'A0':5.}
            true_x.par_values['A0']={'_a0_': 0.1, '_a1_': -0.02}
            true_y.par_values['A0']={'_a0_': -0.4, '_a1_': 0.02}
            true_x.x0={str(true_x):{str(true_x):{'A0':10.}}}
            true_y.x0={str(true_x):{str(true_x):{'A0':5.}}}
            true_x.fit_par={str(true_x):{}}
            true_y.fit_par={str(true_x):{}}
            #true_x.fit_par={str(true_x):{str(true_x):{'A0':{'_a0_': 0.1, '_a1_': -0.02}}}}
            #true_y.fit_par={str(true_x):{str(true_x):{'A0':{'_a0_': -0.4, '_a1_': 0.02}}}}
            true_x.get_bic(reset=True,fit=True)
            true_x.get_energy(reset=True)
            list_dl_trues.append((true_x,true_y,true_x.E+true_y.EP))
            for a,b in corpus_true:
                true_x=ms_ode.from_string_model(x,{'A0':10.},a,2,8,['x','y'],silence=True)
                true_y=ms_ode.from_string_model(y,{'A0':5.},b,2,8,['x','y'],silence=True)
                true_x.fy,true_y.fy=true_y,true_x
                print('Get bic corpus')
                true_x.get_bic(reset=True,fit=True)
                true_x.get_energy(reset=True)
                #print('------')
                #print(a,true_x.par_values)
                #print(b,true_y.par_values)
                #print(true_x.E+true_y.EP)
                #print('------')
                list_dl_trues.append((a,b,true_x.E+true_y.EP))
            
            true_model = min(list_dl_trues, key=lambda x: x[2])
            #print(true_model)
            #Load sampled model
            print(f'./noise_data_ode/llac_{file[:-3]}csv')
            with open(f'./noise_data_ode/llac_{file[:-3]}pkl', 'rb') as f:
                # A new file will be created
                #print()
                best_model=pd.read_pickle(f)
                #print('x0',best_model['x'].x0)
            try:
                with open(f'./results/exh_ode_mdl{file[:-4]}.pkl', 'rb') as f:
                    # A new file will be created
                    bms_exh=pickle.load(f)
                if best_model['x'].E+best_model['y'].EP>bms_exh['x'].E+bms_exh['y'].EP:
                    print('Updating mdl ode model with exhaustive model:',best_model,bms_exh)
                    del best_model
                    best_model = deepcopy(bms_exh)
                    del bms_exh
            except:
                pass
            
            deriv_ode_x= best_model['x'].predict(best_model['x'].x)['A0']
            deriv_ode_y= best_model['y'].predict(best_model['y'].x)['A0']
            
            #plt.scatter(best_model['x'].x['A0']['x'].to_numpy(),deriv_ode_x,c='red',marker='+',label='d ode x')
            #plt.scatter(best_model['y'].x['A0']['x'].to_numpy(),deriv_ode_y,c='red',marker='^',label='d ode y')
            
            h=(data.t.to_numpy()[1]-data.t.to_numpy()[0])
            sol=RK45([best_model['x'].x0[str(best_model['x'])][str(best_model['y'])]['A0'],
                      best_model['y'].x0[str(best_model['y'])][str(best_model['x'])]['A0']], h ,data.t.to_numpy(),best_model['x'],best_model['y'])

            int_best=np.concatenate(sol)
            int_data=np.concatenate([data['x'].to_numpy(),data['y'].to_numpy()])
            
            #true_best_err=np.sqrt(np.mean((int_true-int_best)**2))
            data_best_err=np.sqrt(np.mean((int_data-int_best)**2))
            
            if not np.isinf(int_best[-1]):
                #true_best_error_ode.append(true_best_err)
                data_best_error_ode.append(data_best_err)
            else:
                print('RMSE error in ODE:')
                print('ODE***********************')#,str(true_model),best_model)
                print('ODE',file)
                print('Best:',e_best,best_model['x'],best_model['y'])
                print('Best:',best_model['x'].par_values,best_model['y'].par_values)
                print('True:',e_true,true_x,true_y)
                print('True:',true_x.par_values,true_y.par_values)
                print('**************************')
            if data_best_err>10.:
                print('Check BMS model ODE:',file,str(best_model))
                print('ODE***********************')#,str(true_model),best_model)
                print('ODE',file)
                print('Best:',e_best,best_model['x'],best_model['y'])
                print('Best:',best_model['x'].par_values,best_model['y'].par_values)
                print('True:',e_true,true_x,true_y)
                print('True:',true_x.par_values,true_y.par_values)
                print('**************************')
            
            e_true=true_model[2]
            e_best=best_model['x'].E+best_model['y'].EP
            if e_best>=e_true:
                l_ode+=1
                #print('lernable')
            elif np.isclose(np.float64(e_best),np.float64(e_true), rtol=1e-05, atol=1e-08):
                l_ode+=1
            else:
                print('ODE***********************')#,str(true_model),best_model)
                print('ODE',file)
                print('Best:',e_best,best_model['x'],best_model['y'])
                print('Best:',best_model['x'].par_values,best_model['y'].par_values)
                print('True:',e_true,true_x,true_y)
                print('True:',true_x.par_values,true_y.par_values)
                print('**************************')
                """
                plt.show()
                plt.scatter(data.t.to_numpy(),data['x'].to_numpy(),c='red',marker='+',label='data x')
                plt.scatter(data.t.to_numpy(),data['y'].to_numpy(),c='red',marker='x',label='data y')
                plt.scatter(data.t.to_numpy(),sol[0],c='blue',marker='+',label='mdl x')
                plt.scatter(data.t.to_numpy(),sol[1],c='blue',marker='x',label='mdl y')
                # Integration of the true model
                true_x=ms_ode.from_string_model(x,{'A0':10.},'((_a0_ * x) + (_a1_ * (x * y)))',2,8,['x','y'],silence=True)
                true_y=ms_ode.from_string_model(y,{'A0':5.},'((_a0_ * x) + (_a1_ * (x * y)))',2,8,['x','y'],silence=True)
                true_x.fy,true_y.fy=true_y,true_x
                true_x.par_values['A0']={'_a0_': 0.1, '_a1_': -0.02}
                true_y.par_values['A0']={'_a0_': -0.4, '_a1_': 0.02}
                true_x.x0={str(true_x):{str(true_x):{'A0':10.}}}
                true_y.x0={str(true_x):{str(true_x):{'A0':5.}}}
                true_x.fit_par={str(true_x):{}}
                true_y.fit_par={str(true_x):{}}
                true_x.get_bic(reset=True,fit=True,verbose=True)
                true_x.get_energy(reset=True)
                sol=RK45([true_x.x0[str(true_x)][str(true_y)]['A0'],
                      true_y.x0[str(true_y)][str(true_x)]['A0']], h ,data.t.to_numpy(),true_x,true_y)
                plt.scatter(data.t.to_numpy(),sol[0],c='green',marker='+',label='true x')
                plt.scatter(data.t.to_numpy(),sol[1],c='green',marker='x',label='true y')
                plt.legend(loc='best')
                plt.show()
                plt.close()
                print('best x',best_model['x'],'E',best_model['x'].E,'bic',best_model['x'].bic,'Prior',best_model['x'].EP,
                      'sse',best_model['x'].sse)
                print('best y',best_model['y'],'E',best_model['y'].E,'bic',best_model['y'].bic,'Prior',best_model['y'].EP,
                      'sse',best_model['y'].sse)
                print('true x',true_x,'E',true_x.E,'bic',true_x.bic,'Prior',true_x.EP,
                      'sse',true_x.sse)
                print('true y',true_y,'E',true_y.E,'bic',true_y.bic,'Prior',true_y.EP,
                      'sse',true_y.sse)
                """
            count_ode+=1
            #print('ODE',file,'true',true_model.E,'mdl',best_model.E)
            del data,x,y,true_x,true_y,best_model,sol
        except Exception as e:
            print('Error in ODE:',e)
            print(traceback.format_exc())
            pass
            
        try:
            # True Model Fit deriv
            data=pd.read_csv('./noise_data/'+file)
            x={}
            y={}
            with open(f'./noise_data_fit/{file[:-4]}.pkl', 'rb') as f:
                # A new file will be created
                best_model=pickle.load(f)

            try:
                with open(f'./results/exh_fit_mdl_{file[:-4]}.pkl', 'rb') as f:
                    # A new file will be created
                    bms_exh=pickle.load(f)
                if best_model_x.E+best_model_y.E>bms_exh.E+bms_exh.E:
                    print('Updating mdl ode model with exhaustive model:',best_model,bms_exh)
                    del best_model
                    best_model = deepcopy(bms_exh)
                    del bms_exh
            except:
                pass
            
            #print(best_model)
            best_model_x=best_model['x']
            
            x=best_model_x.x
            y=best_model_x.y
            XLABS = ['x','y']
            params = 8
            
            list_dl_trues=[ms_fit.from_string_model(x,y,str_model[0],1,8,XLABS,silence=True) for str_model in corpus_true_fits]
            true_model_x = min(list_dl_trues, key=lambda x: x.E)

            best_model_y=best_model['y']
            
            x=best_model_y.x
            y=best_model_y.y
            XLABS = ['x','y']
            params = 8
            
            list_dl_trues=[ms_fit.from_string_model(x,y,str_model[1],1,8,XLABS,silence=True) for str_model in corpus_true_fits]
            true_model_y = min(list_dl_trues, key=lambda x: x.E)
            
            deriv_fit_x= best_model['x'].predict(best_model['x'].x)['A0']
            deriv_fit_y= best_model['y'].predict(best_model['y'].x)['A0']
            #plt.scatter(best_model['x'].x['A0']['x'].to_numpy(),deriv_fit_x,c='blue',marker='+',label='d fit x')
            #plt.scatter(best_model['y'].x['A0']['y'].to_numpy(),deriv_fit_y,c='blue',marker='^',label='d fit y')

            model_to_fit_x=best_model_x
            model_to_fit_y=best_model_y
            t_eval=data.t.to_numpy()
            h=(data.t.to_numpy()[1]-data.t.to_numpy()[0])
            int_data=np.concatenate([data['x'].to_numpy(),data['y'].to_numpy()])
            popt,pcov=curve_fit(func, [0.], int_data, p0=[5.,10.])

            sol=RK45_fit(popt, h ,data.t.to_numpy(),best_model_x,best_model_y)

            int_best=np.concatenate(sol)
            
            #true_best_err=np.sqrt(np.mean((int_true-int_best)**2))
            data_best_err=np.sqrt(np.mean((int_data-int_best)**2))
            
            if not np.isinf(int_best[-1]):
                
                #true_best_error_fit.append(true_best_err)
                data_best_error_fit.append(data_best_err)
            else:
                print('RMSE error in FIT:')
                print('FIT***************')#,true_model,best_model)
                print('Best:',e_best,best_model['x'],best_model['y'])
                print('Best:',best_model['x'].par_values,best_model['y'].par_values)
                print('True:',e_true,true_model_x,true_model_y)
                print('True:',true_model_x.par_values,true_model_y.par_values)
                print('***********************')
            if data_best_err>10.:
                print('Check BMS model FIT:',file,str(best_model))
                print('FIT***************')#,true_model,best_model)
                print('Best:',e_best,best_model['x'],best_model['y'])
                print('Best:',best_model['x'].par_values,best_model['y'].par_values)
                print('True:',e_true,true_model_x,true_model_y)
                print('True:',true_model_x.par_values,true_model_y.par_values)
                print('***********************')
                
            

            e_true=true_model_x.E+true_model_y.E
            e_best=best_model_x.E+best_model_y.E
            if e_best>=e_true:
                l_fit+=1
            elif np.isclose(np.float64(e_best),np.float64(e_true), rtol=1e-05, atol=1e-08):
                l_fit+=1
            else:
                print('FIT***************')#,true_model,best_model)
                print('Best:',e_best,best_model['x'],best_model['y'])
                print('Best:',best_model['x'].par_values,best_model['y'].par_values)
                print('True:',e_true,true_model_x,true_model_y)
                print('True:',true_model_x.par_values,true_model_y.par_values)
                print('***********************')
            count_fit+=1
            #print('FIT',file,'true',true_model.E,'mdl',best_model.E)
            del data,x,y,true_model_x,true_model_y,best_model,sol
        except Exception as e:
            print('Error in FIT:',e)
            print(traceback.format_exc())
            #print(true_best_err,type(true_best_err))
            #print(data_best_err,type(data_best_err))
            pass
            
            
        try:
            # True Model Fit deriv smooth
            data=pd.read_csv('noise_data/'+file)
            x={}
            y={}
            with open(f'./noise_data_fit_smooth/{file[:-4]}.pkl', 'rb') as f:
                # A new file will be created
                best_model=pickle.load(f)

            try:
                with open(f'./results/exh_smooth_mdl_{file[:-4]}.pkl', 'rb') as f:
                    # A new file will be created
                    bms_exh=pickle.load(f)
                if best_model_x.E+best_model_y.E>bms_exh.E+bms_exh.E:
                    print('Updating mdl ode model with exhaustive model:',best_model,bms_exh)
                    del best_model
                    best_model = deepcopy(bms_exh)
                    del bms_exh
            except:
                pass
            
            best_model_x=best_model['x']
            x=best_model_x.x
            y=best_model_x.y
            XLABS = ['x','y']
            params = 8
            
            list_dl_trues=[ms_fit.from_string_model(x,y,str_model[0],1,8,XLABS,silence=True) for str_model in corpus_true_fits]
            true_model_x = min(list_dl_trues, key=lambda x: x.E)

            best_model_y=best_model['y']
            
            x=best_model_y.x
            y=best_model_y.y
            XLABS = ['x','y']
            params = 8
            
            list_dl_trues=[ms_fit.from_string_model(x,y,str_model[1],1,8,XLABS,silence=True) for str_model in corpus_true_fits]
            true_model_y = min(list_dl_trues, key=lambda x: x.E)
            
            deriv_smooth_x= best_model['x'].predict(best_model['x'].x)['A0']
            deriv_smooth_y= best_model['y'].predict(best_model['y'].x)['A0']
            deriv_smooth_x_true= true_model_x.predict(best_model['x'].x)['A0']
            deriv_smooth_y_true= true_model_y.predict(best_model['y'].x)['A0']
            #plt.scatter(best_model['x'].x['A0']['x'].to_numpy(),deriv_smooth_x,c='green',marker='+',label='d smo x')
            #plt.scatter(best_model['y'].x['A0']['y'].to_numpy(),deriv_smooth_y,c='green',marker='^',label='d smo y')
            
            #plt.scatter(best_model['x'].x['A0']['x'].to_numpy(),deriv_smooth_x_true,c='green',marker='1',label='smo x true')
            #plt.scatter(best_model['y'].x['A0']['y'].to_numpy(),deriv_smooth_y_true,c='green',marker='2',label='smo y true')
            
            model_to_fit_x=best_model_x
            model_to_fit_y=best_model_y
            t_eval=data.t.to_numpy()
            h=(data.t.to_numpy()[1]-data.t.to_numpy()[0])
            int_data=np.concatenate([data['x'].to_numpy(),data['y'].to_numpy()])
            popt,pcov=curve_fit(func, [0.], int_data, p0=[5.,10.])

            #print('initial fit values',popt)
            
            sol=RK45_fit(popt, h ,data.t.to_numpy(),best_model_x,best_model_y)

            int_best=np.concatenate(sol)
            #print(int_best)
            #true_best_err=np.sqrt(np.mean((int_true-int_best)**2))
            data_best_err=np.sqrt(np.mean((int_data-int_best)**2))
            #print('Data Best Err',data_best_err)
            
            if not np.isinf(int_best[-1]) and not np.isnan(int_best[-1]):
                
                #true_best_error_smth.append(true_best_err)
                data_best_error_smth.append(data_best_err)
            else:
                print('RMSE error in SMTH:')
                print('SMTH*******************')#,true_model,best_model)
                print('SMOOTH',sigma)
                print('Best:',e_best,best_model['x'],best_model['y'])
                print('Best:',best_model['x'].par_values,best_model['y'].par_values)
                print('True:',e_true,true_model_x,true_model_y)
                print('True:',true_model_x.par_values,true_model_y.par_values)
                print('***********************')
            if data_best_err>10.:
                print('Check BMS model SMTH:',file,str(best_model))
                print('SMTH*******************')#,true_model,best_model)
                print('SMOOTH',sigma)
                print('Best:',e_best,best_model['x'],best_model['y'])
                print('Best:',best_model['x'].par_values,best_model['y'].par_values)
                print('True:',e_true,true_model_x,true_model_y)
                print('True:',true_model_x.par_values,true_model_y.par_values)
                print('***********************')
            #Load sampled model
            
            """
            h=(data.t.to_numpy()[1]-data.t.to_numpy()[0])
            sol=RK45_fit(ode_fit,[9.833530 , 5.090359], h ,data.t.to_numpy(),best_model['x'],best_model['y'])
            plt.plot(sol[0])
            plt.plot(data.x.to_numpy())
            plt.plot(sol[1])
            plt.plot(data.y.to_numpy())
            plt.title('smooth')
            plt.show()
            plt.cla()
            plt.clf()
            plt.plot(best_model_x.y['A0'].to_numpy(),label='dx')
            plt.plot(best_model_y.y['A0'].to_numpy(),label='dy')
            plt.plot(best_model_x.predict(best_model_x.x)['A0'],label='mdx')
            plt.plot(best_model_y.predict(best_model_y.x)['A0'],label='mdy')
            plt.legend()
            plt.show()
            """
            e_true=true_model_x.E+true_model_y.E
            e_best=best_model_x.E+best_model_y.E
            if e_best>=e_true:
                l_smth+=1
            elif np.isclose(np.float64(e_best),np.float64(e_true), rtol=1e-05, atol=1e-08):
                l_smth+=1
            else:
                print('SMTH*******************')#,true_model,best_model)
                print('SMOOTH',sigma)
                print('Best:',e_best,best_model['x'],best_model['y'])
                print('Best:',best_model['x'].par_values,best_model['y'].par_values)
                print('True:',e_true,true_model_x,true_model_y)
                print('True:',true_model_x.par_values,true_model_y.par_values)
                print('***********************')
                
                #print(best_model.par_values)
            count_smth+=1
            #print('SMTH',file,'true',true_model.E,'mdl',best_model.E)
            del data,x,y,true_model_x,true_model_y,best_model_x,sol
            gc.collect()
            #print(best_model['x'])
            del best_model['x']
            #print(best_model)
        except Exception as e:
            print('Error in SMOOTH:',e)
            print(traceback.format_exc())
            #print(true_best_err,type(true_best_err))
            #print(data_best_err,type(data_best_err))
            
            pass
        #plt.legend(loc='best')
        #plt.show()
        
            
    # appending lernability fracction
    s_array.append(sigma)
    ode_lernability.append(np.divide(float(l_ode),float(count_ode)))
    ode_lernability_err.append(bootstrap(([1]*l_ode+[0]*(count_ode-l_ode),), np.mean, confidence_level=0.95, method='percentile').standard_error)
    fit_lernability.append(np.divide(float(l_fit),float(count_fit)))
    fit_lernability_err.append(bootstrap(([1]*l_fit+[0]*(count_fit-l_fit),), np.mean, confidence_level=0.95, method='percentile').standard_error)
    smooth_lernability.append(np.divide(float(l_smth),float(count_smth)))
    smooth_lernability_err.append(bootstrap(([1]*l_smth+[0]*(count_smth-l_smth),), np.mean, confidence_level=0.95, method='percentile').standard_error)
    
    #true_best_error_ode_final.append(np.mean(true_best_error_ode))
    data_best_error_ode_final.append(np.mean(data_best_error_ode))
    data_best_error_ode_final_err.append(np.std(data_best_error_ode,ddof=1)/np.sqrt(np.size(data_best_error_ode)))

    #true_best_error_fit_final.append(np.mean(true_best_error_fit))
    data_best_error_fit_final.append(np.mean(data_best_error_fit))
    data_best_error_fit_final_err.append(np.std(data_best_error_fit,ddof=1)/np.sqrt(np.size(data_best_error_fit)))

    #true_best_error_smth_final.append(np.mean(true_best_error_smth))
    data_best_error_smth_final.append(np.mean(data_best_error_smth))
    data_best_error_smth_final_err.append(np.std(data_best_error_smth,ddof=1)/np.sqrt(np.size(data_best_error_smth)))
    
    
# In[ ]:


data_store={
    's_array':s_array,
    'ode_lernability':ode_lernability,
    'ode_lernability_err':ode_lernability_err,
    'fit_lernability':fit_lernability,
    'fit_lernability_err':fit_lernability_err,
    'smooth_lernability':smooth_lernability,
    'smooth_lernability_err':smooth_lernability_err,
    'data_best_error_ode_final':data_best_error_ode_final,
    'data_best_error_fit_final':data_best_error_fit_final,
    'data_best_error_smth_final':data_best_error_smth_final,
    'data_best_error_ode_final_err':data_best_error_ode_final_err,
    'data_best_error_fit_final_err':data_best_error_fit_final_err,
    'data_best_error_smth_final_err':data_best_error_smth_final_err,
}
with open('learnability.pkl','wb') as file:
    pickle.dump(data_store,file=file)


# In[ ]:




