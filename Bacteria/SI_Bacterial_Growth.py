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
import math
# Since the 'user' column do not have relevant information will not be read

# Import Machine Scientist
from importlib.machinery import SourceFileLoader
# Get the absolute path of the script's directory
script_dir = os.getcwd()
# Define the relative path to the module
relative_module_path = "rguimera-machine-scientist-constrained/machinescientist_ode.py"
path = os.path.join(script_dir, relative_module_path)
ms = SourceFileLoader("ms", path).load_module()

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

# SI plots for training:
f_name='Train_test_data_lin_term_com2025_03_11-11_21_44/x.pkl'
file = open(f_name,'rb')
x = pickle.load(file)
print(x.keys())

f_name='Train_test_data_lin_term_com2025_03_11-11_21_44/y.pkl'
file = open(f_name,'rb')
y = pickle.load(file)

f_name='Full_data_lin_term_prod_2025_03_27-06_00_44/mdl_refit_train.pkl'
file = open(f_name,'rb')
bms = pickle.load(file)
print('Train model', bms, bms.E)

f_name='Train_test_data_lin_term_com2025_03_11-11_21_44/logistic_train_fit_t.pkl'
file = open(f_name,'rb')
logistic_model = pickle.load(file)
file.close()
print('Logistic train',logistic_model.E)

f_name='Train_test_data_lin_term_com2025_03_11-11_21_44/gompertz_train.pkl'
file = open(f_name,'rb')
gompertz_model = pickle.load(file)
file.close()
print('gom train',gompertz_model,gompertz_model.E)

fold='train'

"""

# SI plots for training:
f_name='Train_test_data_lin_term_com2025_03_11-11_21_44/x_test.pkl'
file = open(f_name,'rb')
x = pickle.load(file)
print(x.keys())

f_name='Train_test_data_lin_term_com2025_03_11-11_21_44/y_test.pkl'
file = open(f_name,'rb')
y = pickle.load(file)

f_name='Full_data_lin_term_prod_2025_03_27-06_00_44/mdl_refit_test.pkl'
file = open(f_name,'rb')
bms = pickle.load(file)
print('Train model', bms, bms.E)

f_name='Train_test_data_lin_term_com2025_03_11-11_21_44/logistic_test_fit_t.pkl'
file = open(f_name,'rb')
logistic_model = pickle.load(file)
file.close()
print('Logistic train',logistic_model.E)

f_name='Train_test_data_lin_term_com2025_03_11-11_21_44/gompertz_test_fit_t.pkl'
file = open(f_name,'rb')
gompertz_model = pickle.load(file)
file.close()
print('gom train',gompertz_model,gompertz_model.E)

fold='test'
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





import matplotlib.gridspec as gridspec

plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

ncols=6
nrows=8
count=0
b=False
for fig_count in range(10):
    fig = plt.figure(figsize=(15, 20))
    
    gs = gridspec.GridSpec(nrows, ncols, wspace=0.3, hspace=0.2)  # Adjust spacing if needed
    
    colums=list(x.keys())
    
    for i in range(nrows):
        for j in range(ncols):
            try:
                col1=colums[count]
            except:
                b=True
                break
            ax0 = fig.add_subplot(gs[i, j])
            
            h=x[col1].t[1]-x[col1].t[0]
            y0=bms.x0[str(bms)][col1]
            ode_fulldata=RK_BMS(bms,y0,h,len(y[col1]),col1)
            
            y0=logistic_model.x0[str(logistic_model)][col1]
            ode_log=RK_BMS(logistic_model,y0,h,len(y[col1]),col1)
            
            y0=gompertz_model.x0[str(gompertz_model)][col1]
            ode_gom=RK_BMS(gompertz_model,y0,h,len(y[col1]),col1)
            if j==0:
                ax0.set_ylabel('OD(600 nm)')
            if i==nrows-1:
                ax0.set_xlabel('Time(h)')
            ax0.set_ylim([0.,1.2])
            ax0.set_xlim([0.,35.])
            ax0.set_box_aspect(1)
            ax0.scatter(x[col1].t.to_numpy(),y[col1],marker='.',color=colors['data'],label='data')
            ax0.plot(x[col1].t.to_numpy(),ode_fulldata,color=colors['I-BMS'],label='I-BMS Aggegated')
            ax0.plot(x[col1].t.to_numpy(),ode_log,color=colors['Log'],label='Logaritmic')
            ax0.plot(x[col1].t.to_numpy(),ode_gom,color=colors['Gom'],label='Gompertz')
            count+=1
            print(count,col1)
        if b:
            break
            
    
    print('saving fig')
    fig.savefig(fname=f'SI_{fold}_integrated_curves{fig_count}.pdf',dpi=300,format='pdf',bbox_inches='tight')
    if b:
        break
print('Integrate',len(x),count)
count=0
b=False
for fig_count in range(10):
    fig1 = plt.figure(figsize=(15, 20))
    
    gs = gridspec.GridSpec(nrows, ncols, wspace=0.3, hspace=0.2)  # Adjust spacing if needed
    
    colums=list(x.keys())
    
    predictions_fulldata=bms.predict(x,verbose=True)
    predictions_log=logistic_model.predict(x,verbose=True)
    predictions_gom=gompertz_model.predict(x,verbose=True)
    for i in range(nrows):
        for j in range(ncols):
            try:
                col1=colums[count]
            except:
                b=True
                break
                
            ax1 = fig1.add_subplot(gs[i, j])
            ax1.scatter(y[col1].to_numpy(),x[col1].dx.to_numpy(),marker='.',color=colors['data'],label='data')
            ax1.plot(y[col1].to_numpy(),predictions_fulldata[col1].to_numpy(),color=colors['I-BMS'],label='I-BMS')
            ax1.plot(y[col1].to_numpy(),predictions_log[col1].to_numpy(),color=colors['Log'],label='Logistic')
            ax1.plot(y[col1].to_numpy(),predictions_gom[col1].to_numpy(),color=colors['Gom'],label='Gompertz')
            #ax1.set_xlabel('time(h)')
            
            ax1.set_box_aspect(1)
            if j==0:
                ax1.set_ylabel('Derivative')
            if i==nrows-1:
                ax1.set_xlabel('OD(600 nm)')
            #ax1.set_ylim([-0.01,0.12])
            #ax1.set_xlim([0.,1.2])
            count+=1
            print(count)
        if b:
            break
            
    
    print('saving fig')
    fig1.savefig(fname=f'SI_{fold}_derivative_curves{fig_count}.pdf',dpi=300,format='pdf',bbox_inches='tight')
    if b:
        break
print('Derivate',len(x),count)




