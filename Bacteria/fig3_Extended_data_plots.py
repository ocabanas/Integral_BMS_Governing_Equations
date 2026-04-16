#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import sys
import warnings
import gc
warnings.filterwarnings('ignore')
gc.disable()
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from copy import deepcopy,copy
from ipywidgets import IntProgress
from itertools import chain
from IPython.display import display
from datetime import datetime
import pickle
import os
from sympy import sympify,latex,Float,simplify
import random
from math import ceil,sqrt
import seaborn as sbrn
from scipy.optimize import curve_fit
import pynumdiff
# Catch stout
from io import StringIO 
import sys
from contextlib import redirect_stdout
# Since the 'user' column do not have relevant information will not be read

# Import Machine Scientist
from importlib.machinery import SourceFileLoader
# Get the absolute path of the script's directory
script_dir = os.getcwd()
# Define the relative path to the module
relative_module_path = "rguimera-machine-scientist-constrained/machinescientist_ode.py"
path = os.path.join(script_dir, relative_module_path)
ms = SourceFileLoader("ms", path).load_module()


# Load Arial Font
try:
    from matplotlib import font_manager
    font_manager.fontManager.addfont('/usr/share/fonts/truetype/msttcorefonts/Arial.ttf')
    prop = font_manager.FontProperties(fname='/usr/share/fonts/truetype/msttcorefonts/Arial.ttf')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = prop.get_name()
except Exception as e:
    print('Could not load Arial font')
    print(e)

plt.rcParams['axes.labelsize'] = 10
plt.rcParams['font.size'] = 10
#plt.rcParams["font.family"] = "Arial"
colors={
    'data': 'grey',
    'gt':'k',
    'fd':'#7fc97f',
    'sd':'#fdc086',
    'I-BMS':'#bdc9e1',
    'FD-BMS':'#74a9cf',
    'SD-BMS':'#0570b0',
    'WS':'#fdae6b',
    'IS':'#e6550d',
#    'A-BMS':
    'Log':'#b96902',
    'Gom':'#9a0eea'
}

import sys, getopt

# Loading out-of-sample data
f_name='Train_test_data_lin_term_com2025_03_11-11_21_44/x_test.pkl'
file = open(f_name,'rb')
test_x = pickle.load(file)
print('test labels',test_x.keys())
print(test_x['H3'])

f_name='Train_test_data_lin_term_com2025_03_11-11_21_44/y_test.pkl'
file = open(f_name,'rb')
test_y = pickle.load(file)

f_name='Full_data_lin_term_prod_2025_03_27-06_00_44/mdl_refit_test1.pkl'
file = open(f_name,'rb')
test_bms = pickle.load(file)
print('Test model',test_bms,test_bms.E)
#test_col1= 'H12'
#test_col2= 'C5'
print(test_bms.latex())

f_name='Train_test_data_lin_term_com2025_03_11-11_21_44/x.pkl'
file = open(f_name,'rb')
train_x = pickle.load(file)
print(train_x.keys())

f_name='Train_test_data_lin_term_com2025_03_11-11_21_44/y.pkl'
file = open(f_name,'rb')
train_y = pickle.load(file)

f_name='Full_data_lin_term_prod_2025_03_27-06_00_44/mdl_refit_train1.pkl'
file = open(f_name,'rb')
train_bms = pickle.load(file)
print('Train model', train_bms, train_bms.E)

#train_col1= 'C1'
#train_col2= 'A6'

    
train_cols=list(train_x.keys())
train_col1,train_col2 = 'B5','Adenine_10'
test_cols=list(test_x.keys())
test_col1,test_col2 = 'R.S.6 Citrate, 0.3%_6','R.S.19 D-Glucose_19'


# Logistic model
logistic_string ='((_a1_ * B) + (pow2(B) * _a2_))))'
f_name='Train_test_data_lin_term_com2025_03_11-11_21_44/logistic_train_fit_t.pkl'
file = open(f_name,'rb')
train_logistic_model = pickle.load(file)
file.close()
print('Logistic train',train_logistic_model.E)
f_name='Train_test_data_lin_term_com2025_03_11-11_21_44/logistic_test_fit_t.pkl'
file = open(f_name,'rb')
test_logistic_model = pickle.load(file)
file.close()
print('Logistic test',test_logistic_model.E)
#print(logistic_model.lambdify(verbose=True))


# In[4]:


# Gompertz model
gompertz_string = '(_a0_ * (_a2_ * (B * (_a1_ - ((B / _a3_) ** (_a4_ / _a2_))))))'
gompertz_string = '(_a0_ * (B * (_a1_ + ((B / _a3_) ** _a2_))))'
gompertz_string = '(_a0_ * (B * (log((B / _a3_)))))'
f_name='Train_test_data_lin_term_com2025_03_11-11_21_44/gompertz_train_fit_t.pkl'
file = open(f_name,'rb')
train_gompertz_model = pickle.load(file)
file.close()
print('gom train',train_gompertz_model,train_gompertz_model.E)
f_name='Train_test_data_lin_term_com2025_03_11-11_21_44/gompertz_test_fit_t.pkl'
file = open(f_name,'rb')
test_gompertz_model = pickle.load(file)
file.close()
print('gom test',test_gompertz_model,test_gompertz_model.E)


# Plotting B(t) real and predicted
def euler_BMS(model,y0,h,steps,col):
    res=[y0]
    for i in range(steps-1):
        f=model.predict({col:pd.DataFrame(data={'B':[res[-1]]})})[col].to_numpy()[0]
        res.append(res[-1]+h*f)
    return res

def trap_BMS(model,y0,h,steps,col):
    res=[y0]
    for i in range(steps-1):
        f=model.predict({col:pd.DataFrame(data={'B':[res[-1]]})})[col].to_numpy()[0]
        y1=res[-1]+h*f
        f1=model.predict({col:pd.DataFrame(data={'B':[y1]})})[col].to_numpy()[0]
        res.append(res[-1]+0.5*h*(f+f1))
    return res

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



train_error_model ={'I-BMS':[],
             'Logistic':[],
             'Gompertz':[]}

test_error_model ={'I-BMS':[],
             'Logistic':[],
             'Gompertz':[]}
for col in train_cols :
    # Integrate ODE with BMS expression
    h=train_x[col].t[1]-train_x[col].t[0]
    y0=train_bms.x0[str(train_bms)][col]
    train_ode_fulldata=RK_BMS(train_bms,y0,h,len(train_y[col]),col)
    train_error_model['I-BMS'].append(np.sqrt(np.mean((train_x[col].B.to_numpy() - train_ode_fulldata)**2)))
    
    y0=train_logistic_model.x0[str(train_logistic_model)][col]
    train_ode_log=RK_BMS(train_logistic_model,y0,h,len(train_y[col]),col)
    train_error_model['Logistic'].append(np.sqrt(np.mean((train_x[col].B.to_numpy() - train_ode_log)**2)))
    
    y0=train_gompertz_model.x0[str(train_gompertz_model)][col]
    train_ode_gom=RK_BMS(train_gompertz_model,y0,h,len(train_y[col]),col)
    train_error_model['Gompertz'].append(np.sqrt(np.mean((train_x[col].B.to_numpy() - train_ode_gom)**2)))

for col in test_cols :
    # Integrate ODE with BMS expression
    h=test_x[col].t[1]-test_x[col].t[0]
    y0=test_bms.x0[str(test_bms)][col]
    test_ode_fulldata=RK_BMS(test_bms,y0,h,len(test_y[col]),col)
    test_error_model['I-BMS'].append(np.sqrt(np.mean((test_x[col].B.to_numpy() - test_ode_fulldata)**2)))
    
    y0=test_logistic_model.x0[str(test_logistic_model)][col]
    test_ode_log=RK_BMS(test_logistic_model,y0,h,len(test_y[col]),col)
    test_error_model['Logistic'].append(np.sqrt(np.mean((test_x[col].B.to_numpy() - test_ode_log)**2)))
    
    y0=test_gompertz_model.x0[str(test_gompertz_model)][col]
    test_ode_gom=RK_BMS(test_gompertz_model,y0,h,len(test_y[col]),col)
    test_error_model['Gompertz'].append(np.sqrt(np.mean((test_x[col].B.to_numpy() - test_ode_gom)**2)))



import pylustrator
pylustrator.start()
import matplotlib.gridspec as gridspec

plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

fig = plt.figure(figsize=(18.3/2.54,20/2.54))

gs = gridspec.GridSpec(4, 3, wspace=0.4, hspace=0.3)  # Adjust spacing if needed

#ax = fig.add_subplot(gs[i, j]) #Row, column 

ax0 = fig.add_subplot(gs[0, 0]) #Row, column 
ax1 = fig.add_subplot(gs[0, 1]) #Row, column 
ax2 = fig.add_subplot(gs[0, 2]) #Row, column 
ax3 = fig.add_subplot(gs[1, 0]) #Row, column 
ax4 = fig.add_subplot(gs[1, 1]) #Row, column 
ax5 = fig.add_subplot(gs[1, 2]) #Row, column 
#ax4 = fig.add_subplot(third_row[0])
#ax5 = fig.add_subplot(third_row[1])
# 


h=train_x[train_col1].t[1]-train_x[train_col1].t[0]
y0=train_bms.x0[str(train_bms)][train_col1]
train_ode_fulldata=RK_BMS(train_bms,y0,h,len(train_y[train_col1]),train_col1)

y0=train_logistic_model.x0[str(train_logistic_model)][train_col1]
train_ode_log=RK_BMS(train_logistic_model,y0,h,len(train_y[train_col1]),train_col1)

y0=train_gompertz_model.x0[str(train_gompertz_model)][train_col1]
train_ode_gom=RK_BMS(train_gompertz_model,y0,h,len(train_y[train_col1]),train_col1)


train_predictions_fulldata=train_bms.predict(train_x,verbose=True)
train_predictions_log=train_logistic_model.predict(train_x,verbose=True)
train_predictions_gom=train_gompertz_model.predict(train_x,verbose=True)

ax0.scatter(train_x[train_col1].t.to_numpy(),train_y[train_col1],marker='.',color=colors['data'],label='data')
ax0.plot(train_x[train_col1].t.to_numpy(),train_ode_fulldata,color=colors['I-BMS'],label='I-BMS Aggegated')
ax0.plot(train_x[train_col1].t.to_numpy(),train_ode_log,color=colors['Log'],label='Logaritmic')
ax0.plot(train_x[train_col1].t.to_numpy(),train_ode_gom,color=colors['Gom'],label='Gompertz')

#ax0.set_xlabel('time(h)')
ax0.set_ylabel('OD(600 nm)')
ax0.set_box_aspect(1)
#ax0.legend(loc='best',frameon=False,fontsize=7)
ax1.scatter(train_y[train_col1].to_numpy(),train_x[train_col1].dx.to_numpy(),marker='.',color=colors['data'],label='data')
ax1.plot(train_y[train_col1].to_numpy(),train_predictions_fulldata[train_col1].to_numpy(),color=colors['I-BMS'],label='I-BMS')
ax1.plot(train_y[train_col1].to_numpy(),train_predictions_log[train_col1].to_numpy(),color=colors['Log'],label='Logistic')
ax1.plot(train_y[train_col1].to_numpy(),train_predictions_gom[train_col1].to_numpy(),color=colors['Gom'],label='Gompertz')
#ax1.set_xlabel('time(h)')
ax1.set_ylabel('Derivative')
ax1.set_box_aspect(1)
#ax2.legend(loc='best',frameon=False,fontsize=7)
#


y0=train_bms.x0[str(train_bms)][train_col2]
h=train_x[train_col2].t[1]-train_x[train_col2].t[0]
train_ode_fulldata=RK_BMS(train_bms,y0,h,len(train_y[train_col2]),train_col2)

y0=train_logistic_model.x0[str(train_logistic_model)][train_col2]
train_ode_log=RK_BMS(train_logistic_model,y0,h,len(train_y[train_col2]),train_col2)

y0=train_gompertz_model.x0[str(train_gompertz_model)][train_col2]
train_ode_gom=RK_BMS(train_gompertz_model,y0,h,len(train_y[train_col2]),train_col2)
#ode3=RK_BMS(train_model,y0,h,len(new_y[col]),col)
#if col=='C1': print(ode)
ax3.scatter(train_x[train_col2].t.to_numpy(),train_y[train_col2],marker='o',color=colors['data'],label='data')
ax3.plot(train_x[train_col2].t.to_numpy(),train_ode_fulldata,color=colors['I-BMS'],label='I-BMS')
ax3.plot(train_x[train_col2].t.to_numpy(),train_ode_log,color=colors['Log'],label='Logaritmic')
ax3.plot(train_x[train_col2].t.to_numpy(),train_ode_gom,color=colors['Gom'],label='Gompertz')
#ax3.set_xlabel('time(h)')
ax3.set_ylabel('OD(600 nm)')
ax3.set_box_aspect(1)
#ax1.legend(loc='best',frameon=False,fontsize=7)

ax4.scatter(train_y[train_col2].to_numpy(),train_x[train_col2].dx.to_numpy(),marker='.',color=colors['data'],label='data')
ax4.plot(train_y[train_col2].to_numpy(),train_predictions_fulldata[train_col2].to_numpy(),color=colors['I-BMS'],label='I-BMS')
ax4.plot(train_y[train_col2].to_numpy(),train_predictions_log[train_col2].to_numpy(),color=colors['Log'],label='Logistic')
ax4.plot(train_y[train_col2].to_numpy(),train_predictions_gom[train_col2].to_numpy(),color=colors['Gom'],label='Gompertz')
#ax4.set_xlabel('time(h)')
ax4.set_ylabel('Derivative')
ax4.set_box_aspect(1)
#ax3.legend(loc='best',frameon=False,fontsize=7)
train_mean_errors={key:np.mean(np.array(train_error_model[key])/np.array(train_error_model['I-BMS'])) for key in ['I-BMS','Logistic','Gompertz']}
print(train_mean_errors)
train_err_mean_errors={key:np.std(np.array(train_error_model[key])/np.array(train_error_model['I-BMS'])) / np.sqrt(len(train_error_model[key])) for key in ['I-BMS','Logistic','Gompertz']}
ax2.bar([0,1,2],list(train_mean_errors.values()),yerr=list(train_err_mean_errors.values()),tick_label=['I-BMS','Logistic','Gompertz'],
        color=[colors['I-BMS'],colors['Log'],colors['Gom']])
ax2.set_ylabel('RMSE(model)/RMSE(IBMS)')
#ax2.set_xticklabels(['I-BMS','Logistic','Gompertz'],rotation=60, ha='right')
ax2.set_xticklabels([])
#ax2.set_box_aspect(1)

ax5.scatter([0,1,2],[train_bms.E,train_logistic_model.E,train_gompertz_model.E],
        color=[colors['I-BMS'],colors['Log'],colors['Gom']])
#ax5.set_xticks([0,1,2], ['I-BMS','Logistic','Gompertz'], rotation=60)
ax5.set_ylabel('Description Length')
#ax5.set_xticklabels(['I-BMS','Logistic','Gompertz'],rotation=60, ha='right')
ax5.set_xticklabels([])
ax5.set_xlim([-0.1,2.1])
ax5.ticklabel_format(axis='y', style='scientific',useOffset=True, scilimits=(0, 0))
#ax5.set_box_aspect(1)
print('after legends')

#############################################################################
####################3
#####################
###########################
ax6 = fig.add_subplot(gs[2, 0]) #Row, column 
ax7 = fig.add_subplot(gs[2, 1]) #Row, column 
ax8 = fig.add_subplot(gs[2, 2]) #Row, column 
ax9 = fig.add_subplot(gs[3, 0]) #Row, column 
ax10 = fig.add_subplot(gs[3, 1]) #Row, column 
ax11 = fig.add_subplot(gs[3, 2]) #Row, column 
#ax4 = fig.add_subplot(third_row[0])
#ax5 = fig.add_subplot(third_row[1])
# 


h=test_x[test_col1].t[1]-test_x[test_col1].t[0]
y0=test_bms.x0[str(test_bms)][test_col1]
test_ode_fulldata=RK_BMS(test_bms,y0,h,len(test_y[test_col1]),test_col1)

y0=test_logistic_model.x0[str(test_logistic_model)][test_col1]
test_ode_log=RK_BMS(test_logistic_model,y0,h,len(test_y[test_col1]),test_col1)

y0=test_gompertz_model.x0[str(test_gompertz_model)][test_col1]
test_ode_gom=RK_BMS(test_gompertz_model,y0,h,len(test_y[test_col1]),test_col1)


test_predictions_fulldata=test_bms.predict(test_x,verbose=True)
test_predictions_log=test_logistic_model.predict(test_x,verbose=True)
test_predictions_gom=test_gompertz_model.predict(test_x,verbose=True)

ax6.scatter(test_x[test_col1].t.to_numpy(),test_y[test_col1],marker='.',color=colors['data'],label='data')
ax6.plot(test_x[test_col1].t.to_numpy(),test_ode_fulldata,color=colors['I-BMS'],label='I-BMS Aggegated')
ax6.plot(test_x[test_col1].t.to_numpy(),test_ode_log,color=colors['Log'],label='Logaritmic')
ax6.plot(test_x[test_col1].t.to_numpy(),test_ode_gom,color=colors['Gom'],label='Gompertz')
#ax6.set_xlabel('time(h)')
ax6.set_ylabel('OD(600 nm)')
ax6.set_box_aspect(1)
#ax0.legend(loc='best',frameon=False,fontsize=7)
#handles,labels=ax0.get_legend_handles_labels()
ax7.scatter(test_y[test_col1].to_numpy(),test_x[test_col1].dx.to_numpy(),marker='.',color=colors['data'],label='data')
ax7.plot(test_y[test_col1].to_numpy(),test_predictions_fulldata[test_col1].to_numpy(),color=colors['I-BMS'],label='I-BMS')
ax7.plot(test_y[test_col1].to_numpy(),test_predictions_log[test_col1].to_numpy(),color=colors['Log'],label='Logistic')
ax7.plot(test_y[test_col1].to_numpy(),test_predictions_gom[test_col1].to_numpy(),color=colors['Gom'],label='Gompertz')
#ax7.set_xlabel('time(h)')
ax7.set_ylabel('Derivative')
#ax7.set_box_aspect(1)
#ax2.legend(loc='best',frameon=False,fontsize=7)
#


y0=test_bms.x0[str(test_bms)][test_col2]
h=test_x[test_col2].t[1]-test_x[test_col2].t[0]
test_ode_fulldata=RK_BMS(test_bms,y0,h,len(test_y[test_col2]),test_col2)

y0=test_logistic_model.x0[str(test_logistic_model)][test_col2]
test_ode_log=RK_BMS(test_logistic_model,y0,h,len(test_y[test_col2]),test_col2)

y0=test_gompertz_model.x0[str(test_gompertz_model)][test_col2]
test_ode_gom=RK_BMS(test_gompertz_model,y0,h,len(test_y[test_col2]),test_col2)

#ode3=RK_BMS(test_model,y0,h,len(new_y[col]),col)
#if col=='C1': print(ode)
ax9.scatter(test_x[test_col2].t.to_numpy(),test_y[test_col2],marker='o',color=colors['data'],label='data')
ax9.plot(test_x[test_col2].t.to_numpy(),test_ode_fulldata,color=colors['I-BMS'],label='I-BMS')
ax9.plot(test_x[test_col2].t.to_numpy(),test_ode_log,color=colors['Log'],label='Logaritmic')
ax9.plot(test_x[test_col2].t.to_numpy(),test_ode_gom,color=colors['Gom'],label='Gompertz')
ax9.set_xlabel('time(h)')
ax9.set_ylabel('OD(600 nm)')
ax9.set_box_aspect(1)
#ax1.legend(loc='best',frameon=False,fontsize=7)

ax10.scatter(test_y[test_col2].to_numpy(),test_x[test_col2].dx.to_numpy(),marker='.',color=colors['data'],label='data')
ax10.plot(test_y[test_col2].to_numpy(),test_predictions_fulldata[test_col2].to_numpy(),color=colors['I-BMS'],label='I-BMS')
ax10.plot(test_y[test_col2].to_numpy(),test_predictions_log[test_col2].to_numpy(),color=colors['Log'],label='Logistic')
ax10.plot(test_y[test_col2].to_numpy(),test_predictions_gom[test_col2].to_numpy(),color=colors['Gom'],label='Gompertz')
ax10.set_xlabel('OD(600 nm)')
ax10.set_ylabel('Derivative')
ax10.set_box_aspect(1)
#handles,labels=ax3.get_legend_handles_labels()
#ax3.legend(loc='best',frameon=False,fontsize=7)
test_mean_errors={key:np.mean(np.array(test_error_model[key])/np.array(test_error_model['I-BMS'])) for key in ['I-BMS','Logistic','Gompertz']}
print(test_mean_errors)
test_err_mean_errors={key:np.std(np.array(test_error_model[key])/np.array(test_error_model['I-BMS'])) / np.sqrt(len(test_error_model[key])) for key in ['I-BMS','Logistic','Gompertz']}
ax8.bar([0,1,2],list(test_mean_errors.values()),yerr=list(test_err_mean_errors.values()),tick_label=['I-BMS','Logistic','Gompertz'],
        color=[colors['I-BMS'],colors['Log'],colors['Gom']])
ax8.set_ylabel('RMSE(model)/RMSE(IBMS)')
ax8.set_xticklabels(['I-BMS','Logistic','Gompertz'],rotation=60, ha='right')
#ax8.set_box_aspect(1)

ax11.scatter([0,1,2],[test_bms.E,test_logistic_model.E,test_gompertz_model.E],
        color=[colors['I-BMS'],colors['Log'],colors['Gom']])
print([test_bms.E,test_logistic_model.E,test_gompertz_model.E])
#ax5.set_xticks([0,1,2], ['I-BMS','Logistic','Gompertz'], rotation=60)
ax11.set_ylabel('Description Length')
ax11.set_xticklabels(['I-BMS','Logistic','Gompertz'],rotation=60, ha='right')
ax11.ticklabel_format(axis='y', style='scientific',useOffset=True, scilimits=(0, 0))
ax11.set_xlim([-0.1,2.1])
#ax11.set_box_aspect(1)
ax10.legend(frameon=False, ncols=4)

print(fig.get_constrained_layout_pads())
#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
getattr(plt.figure(1), '_pylustrator_init', lambda: ...)()
plt.figure(1).axes[0].text(0.09, 0.87, 'A', transform=plt.figure(1).axes[0].transAxes, fontsize=14., weight='bold')  # id=plt.figure(1).axes[0].texts[0].new
plt.figure(1).axes[1].text(0.09, 0.87, 'E', transform=plt.figure(1).axes[1].transAxes, fontsize=14., weight='bold')  # id=plt.figure(1).axes[1].texts[0].new
plt.figure(1).axes[2].set(position=[0.712, 0.5186, 0.1638, 0.3614])
plt.figure(1).axes[2].text(0.09, 0.87, 'I', transform=plt.figure(1).axes[2].transAxes, fontsize=14., weight='bold')  # id=plt.figure(1).axes[2].texts[0].new
plt.figure(1).axes[3].text(0.09, 0.87, 'B', transform=plt.figure(1).axes[3].transAxes, fontsize=14., weight='bold')  # id=plt.figure(1).axes[3].texts[0].new
plt.figure(1).axes[4].text(0.09, 0.87, 'F', transform=plt.figure(1).axes[4].transAxes, fontsize=14., weight='bold')  # id=plt.figure(1).axes[4].texts[0].new
plt.figure(1).axes[5].set(position=[0.9814, 0.5186, 0.1637, 0.3612], xticks=[0., 1., 2.], xticklabels=['', '', ''])
plt.figure(1).axes[5].text(0.09, 0.87, 'J', transform=plt.figure(1).axes[5].transAxes, fontsize=14., weight='bold')  # id=plt.figure(1).axes[5].texts[0].new
plt.figure(1).axes[5].get_yaxis().get_label().set(position=(350.4, 0.5))
plt.figure(1).axes[6].set(position=[0.141, 0.3143, 0.172, 0.1571])
plt.figure(1).axes[6].text(0.09, 0.87, 'C', transform=plt.figure(1).axes[6].transAxes, fontsize=14., weight='bold')  # id=plt.figure(1).axes[6].texts[0].new
plt.figure(1).axes[7].text(0.09, 0.87, 'G', transform=plt.figure(1).axes[7].transAxes, fontsize=14., weight='bold')  # id=plt.figure(1).axes[7].texts[0].new
plt.figure(1).axes[8].set(position=[0.7161, 0.11, 0.1638, 0.3614])
plt.figure(1).axes[8].text(0.09, 0.87, 'K', transform=plt.figure(1).axes[8].transAxes, fontsize=14., weight='bold')  # id=plt.figure(1).axes[8].texts[0].new
plt.figure(1).axes[9].set(position=[0.141, 0.11, 0.172, 0.1571])
plt.figure(1).axes[9].text(0.09, 0.87, 'D', transform=plt.figure(1).axes[9].transAxes, fontsize=14., weight='bold')  # id=plt.figure(1).axes[9].texts[0].new
plt.figure(1).axes[9].get_yaxis().get_label().set(position=(31.26, 0.5))
plt.figure(1).axes[10].legend(loc=(-0.4245, -0.8093), frameon=False, ncols=4)
plt.figure(1).axes[10].text(0.09, 0.87, 'H', transform=plt.figure(1).axes[10].transAxes, fontsize=14., weight='bold')  # id=plt.figure(1).axes[10].texts[0].new
plt.figure(1).axes[11].set(position=[0.9815, 0.11, 0.1637, 0.3612], xticks=[0., 1., 2.], xticklabels=['I-BMS', 'Logistic', 'Gompertz'])
plt.figure(1).axes[11].text(0.09, 0.87, 'L', transform=plt.figure(1).axes[11].transAxes, fontsize=14., weight='bold')  # id=plt.figure(1).axes[11].texts[0].new
plt.figure(1).text(0.0281, 0.6755, 'Train', transform=plt.figure(1).transFigure, fontsize=14., weight='bold', rotation=90.)  # id=plt.figure(1).texts[0].new
plt.figure(1).text(0.0281, 0.2691, 'Test', transform=plt.figure(1).transFigure, fontsize=14., weight='bold', rotation=90.)  # id=plt.figure(1).texts[1].new
#% end: automatic generated code from pylustrator
plt.show()
fig.savefig(filename=f'fig3_bacterial_growth.pdf',dpi=300,format='pdf',bbox_inches='tight',pad_inches=0.1)




