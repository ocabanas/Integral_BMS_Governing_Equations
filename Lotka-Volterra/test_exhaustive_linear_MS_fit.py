#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
from numpy.random import normal

import sys
import gc
import pandas as pd
import numpy as np

from copy import copy,deepcopy
from itertools import combinations
from scipy.stats import multinomial
import warnings
import pynumdiff
import itertools
import pickle
import pysindy as ps
warnings.filterwarnings("ignore")
# Memory debug
from pympler import asizeof
from pympler import tracker
from pympler.process import ProcessMemoryInfo
from pympler import refbrowser
import referrers
from pympler import summary, muppy
import objgraph
all_objects = muppy.get_objects()
sum1 = summary.summarize(all_objects)
#tr = tracker.SummaryTracker()

def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1000.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1000.0
    return "%.1f %s%s" % (num, 'Yi', suffix)
def output_function(o):
    return str(type(o))
# ------------------------------------------------------------------------------
# Get all possible linear expressions with n terms
# ------------------------------------------------------------------------------
def get_expressions_n(the_vars, n=2):
    if n == 0:
        return [('_a0_', 1)]
    groups = combinations(the_vars, n)
    all_exp = []
    for vs in groups:
        expres, npar = '(_a0_ + (_a1_ * %s))' % vs[0], 2
        for nv, v in enumerate(vs[1:]):
            expres = '(%s + (_a%d_ * %s))' % (expres, nv+2, v)
            npar += 1
        all_exp.append((expres, npar))
    return all_exp
def get_expressions_no_cte(the_vars, n=2):
    if n == 0:
        return [('_a0_', 1)]
    groups = combinations(the_vars, n)
    all_exp = []
    #print(groups)
    for vs in groups:
        #print(vs)
        expres, npar = '(_a0_ * %s)' % vs[0], 1
        for nv, v in enumerate(vs[1:]):
            expres = '(%s + (_a%d_ * %s))' % (expres, nv+1, v)
            npar += 1
        all_exp.append((expres, npar))
    #print(all_exp)
    return all_exp
# ------------------------------------------------------------------------------
# Get all possible linear expressions
# ------------------------------------------------------------------------------
def get_expressions(the_vars, nmax=None):
    all_exp = []
    if nmax == None:
        nmax = len(the_vars)
    for n in range(nmax+1):
        all_exp += get_expressions_n(the_vars, n=n)
        all_exp += get_expressions_no_cte(the_vars, n=n)
    return all_exp


# Import Machine Scientist
from importlib.machinery import SourceFileLoader
path = '/export/home/oriolca/BMS_ODE/Lotka_Volterra/rguimera-machine-scientist/machinescientist_fit.py'
ms = SourceFileLoader('ms', path).load_module()

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

data=pd.read_pickle(f'noise_data/{file}')

x={}
fit_x={}
fit_y={}
smooth_x={}
smooth_y={}

x['A0']=deepcopy(data)
B=x['A0'].x.values
h=x['A0'].t.to_numpy()[1]-x['A0'].t.to_numpy()[0]
fit_x['A0']=pd.Series([(B[1]-B[0])/h]+[(B[i+1]-B[i-1])/(2*h) for i in range(1,len(B)-1)]+[(B[len(B)-1]-B[len(B)-2])/h])
B=x['A0'].y.values
h=x['A0'].t.to_numpy()[1]-x['A0'].t.to_numpy()[0]
fit_y['A0']=pd.Series([(B[1]-B[0])/h]+[(B[i+1]-B[i-1])/(2*h) for i in range(1,len(B)-1)]+[(B[len(B)-1]-B[len(B)-2])/h])
smooth_x['A0']=deepcopy(data['dx'])
smooth_y['A0']=deepcopy(data['dy'])

terms=['x', '(pow2(x))',
         'y', '(pow2(y))',
         '(x * y)', '(x * (pow2(y)))',
         '((pow2(x)) * y)']

models = list(set(get_expressions(terms,len(terms))))

### DL for true model:
true_fits=('((_a0_ * x) + (_a1_ * (y * x)))', '((_a0_ * y) + (_a1_ * (y * x)))')

true_fit_x = ms.Tree(variables=['x','y'],
        parameters=['a%d' % i for i in range(8)],
        x=x,y=fit_x,from_string=true_fits[0])
true_fit_y = ms.Tree(variables=['x','y'],
        parameters=['a%d' % i for i in range(8)],
        x=x,y=fit_y,from_string=true_fits[1])
true_fit_x.EP=np.log((1.-(2./3.))*np.exp(-np.log(2./3.)))
true_fit_y.EP=np.log((1.-(2./3.))*np.exp(-np.log(2./3.)))
true_fit_x.E=true_fit_x.EB + true_fit_x.EP
true_fit_y.E=true_fit_y.EB + true_fit_y.EP



fit_mdl=np.inf
fit_visited_dl=[]

smooth_mdl=np.inf
smooth_visited_dl=[]
print(file)
count=0
init_mem=muppy.filter(muppy.get_objects(), Type=str)
my_types = muppy.filter(all_objects, Type=type)
init_obj=muppy.get_objects()
#print(my_types)

df_sindy = pd.DataFrame(data={'x':data['x'].to_numpy(),'y':data['y'].to_numpy()})
optimizer = ps.STLSQ(threshold=0.009)
model = ps.SINDy(feature_names=['x','y'],optimizer=optimizer,feature_library=ps.PolynomialLibrary(degree=5))
model.fit(df_sindy, t=data.t.to_numpy())
model.print()
for combo in itertools.product(models, repeat=2):
    #print(model[0])
    o1 = muppy.get_objects()
    #ms_model_x=ms.Tree(x=x,y=y,variables=['B'], parameters=[f'_a{i}_' for i in range(model[1])],from_string=model[0])
    fit_pms_x = ms.Tree(variables=['x','y'],
                parameters=['a%d' % i for i in range(8)],
                x=x,y=fit_x,from_string=combo[0][0])
    fit_pms_y = ms.Tree(variables=['x','y'],
        parameters=['a%d' % i for i in range(8)],
        x=x,y=fit_y,from_string=combo[1][0])
    
    fit_pms_x.EP=np.log((1.-(2./3.))*np.exp(-np.log(2./3.)*(combo[0][1]-1)))
    fit_pms_y.EP=np.log((1.-(2./3.))*np.exp(-np.log(2./3.)*(combo[0][1]-1)))
    fit_pms_x.E=fit_pms_x.EB + fit_pms_x.EP
    fit_pms_y.E=fit_pms_y.EB + fit_pms_y.EP
    
    fit_visited_dl.append(fit_pms_x.E + fit_pms_y.EP)
    if (fit_pms_x.E + fit_pms_y.EP) <fit_mdl:
        fit_mdl=copy(fit_pms_x.E + fit_pms_y.EP)
        print('New MDL fit',fit_mdl,fit_pms_x,fit_pms_y,'####################')
        print(file)
        with open(f'./exhaustive_results/fit_mdl_{file[:-4]}.pkl', 'wb') as file1:
            # A new file will be created
            pickle.dump({'x':fit_pms_x,'y':fit_pms_y}, file1)
    ########################3
    # Smooth model
    smooth_pms_x = ms.Tree(variables=['x','y'],
                parameters=['a%d' % i for i in range(8)],
                x=x,y=smooth_x,from_string=combo[0][0])
    smooth_pms_y = ms.Tree(variables=['x','y'],
        parameters=['a%d' % i for i in range(8)],
        x=x,y=smooth_y,from_string=combo[1][0])
    
    smooth_pms_x.EP=np.log((1.-(2./3.))*np.exp(-np.log(2./3.)*(combo[0][1]-1)))
    smooth_pms_y.EP=np.log((1.-(2./3.))*np.exp(-np.log(2./3.)*(combo[0][1]-1)))
    smooth_pms_x.E=smooth_pms_x.EB + smooth_pms_x.EP
    smooth_pms_y.E=smooth_pms_y.EB + smooth_pms_y.EP
    
    print('Now','true x',true_fit_x.E,'mdl x',fit_pms_x.E ,'smooth',smooth_pms_x.E,'%%%%%%%%%%%%%%%%%%%%%%%',end='\r')
    smooth_visited_dl.append(smooth_pms_x.E + smooth_pms_y.EP)
    if (smooth_pms_x.E + smooth_pms_y.EP) <smooth_mdl:
        smooth_mdl=copy(smooth_pms_x.E + smooth_pms_y.EP)
        print('New MDL smooth',smooth_mdl,smooth_pms_x,smooth_pms_y,'####################')
        print(file)
        with open(f'./exhaustive_results/smooth_mdl_{file[:-4]}.pkl', 'wb') as file2:
            # A new file will be created
            pickle.dump({'x':smooth_pms_x,'y':smooth_pms_y}, file2)
    """str_true='((_a0_ * x) + (_a1_ * (x * y)))'
    if str(pms_x)==str_true and str(pms_y)==str_true:
        with open(f'./exhaustive_results/true_model.pkl', 'wb') as file:
            # A new file will be created
            pickle.dump({'x':pms_x,'y':pms_y}, file)"""
    del fit_pms_x, fit_pms_y,smooth_pms_x, smooth_pms_y
    #count+=1
    #o2 = muppy.get_objects()
    # New strings:
    final_mem=muppy.filter(muppy.get_objects(), Type=str)
    #final_obj=muppy.get_objects()
    for item in list(set(final_mem+init_mem)):
        if init_mem.count(item)!=final_mem.count(item):
            print('#####################')
            print(item)
            print(init_mem.count(item),final_mem.count(item))
            objgraph.show_backrefs([item], filename=f'backrefs/{item}.png')
            print('#####################')
    """print(type(final_obj),type(init_obj))
    print(type(final_obj+init_obj))
    for item in list(set(''.join(map(str, final_obj))+''.join(map(str, init_obj)))):
        if ''.join(map(str, init_obj)).count(item)!=''.join(map(str, final_obj)):
            print('#####################')
            print(item)
            print(''.join(map(str, init_obj)).count(item),''.join(map(str, final_obj)).count(item))
            for item1 in final_obj+init_obj:
                if str(item1)==item:
                    print('found item')
                    cb = refbrowser.ConsoleBrowser(item1,str_func=output_function)
                    cb.print_tree()
                    break
            print('#####################')"""
    """o1_ids = {id(obj) for obj in o1}
    o2_ids = {id(obj): obj for obj in o2}
    diff = [obj for obj_id, obj in o2_ids.items() if obj_id not in o1_ids]
    
    summary.print_(summary.get_diff(summary.summarize(o1), summary.summarize(o2)), limit=2)
    
    for obj in diff:
        print(
            referrers.get_referrer_graph(
                obj,
                exclude_object_ids=[id(o1), id(o2), id(diff), id(o2_ids)]
            )
        )"""
        
    exit(1)
    init_mem=muppy.filter(muppy.get_objects(), Type=str)
    if count%100==0:
        """for name, size in sorted(((name, asizeof.asizeof(value)) for name, value in list(
                          locals().items())), key= lambda x: -x[1])[:10]:
            print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))"""
        #tr.print_diff()
        print()
        sum2 = summary.summarize(muppy.get_objects())
        diff = summary.get_diff(sum1, sum2)
        summary.print_(diff)
        print(ProcessMemoryInfo().__dict__)
with open(f'./exhaustive_results/fit_dl_list{file[:-4]}.pkl', 'wb') as file3:
    # A new file will be created
    pickle.dump(fit_visited_dl, file3)

with open(f'./exhaustive_results/smooth_dl_list{file[:-4]}.pkl', 'wb') as file4:
    # A new file will be created
    pickle.dump(smooth_visited_dl, file4)