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
import itertools
import pickle
warnings.filterwarnings("ignore")
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
import mcmc_ode as ms_ode
import mcmc_fit as ms_fit
import fit_prior
prior=fit_prior.read_prior_par('final_prior_param_sq.named_equations.nv2.np8.2016-09-09 18:49:42.800618.dat')

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

data=pd.read_csv(file)

x={}
y={}

x['A0']=deepcopy(data)

fit_x={}
fit_y={}
smooth_x={}
smooth_y={}

B=x['A0'].x.values
h=x['A0'].t.to_numpy()[1]-x['A0'].t.to_numpy()[0]
fit_x['A0']=pd.Series([(B[1]-B[0])/h]+[(B[i+1]-B[i-1])/(2*h) for i in range(1,len(B)-1)]+[(B[len(B)-1]-B[len(B)-2])/h])
B=x['A0'].y.values
h=x['A0'].t.to_numpy()[1]-x['A0'].t.to_numpy()[0]
fit_y['A0']=pd.Series([(B[1]-B[0])/h]+[(B[i+1]-B[i-1])/(2*h) for i in range(1,len(B)-1)]+[(B[len(B)-1]-B[len(B)-2])/h])
smooth_x['A0']=deepcopy(data['dx'])
smooth_y['A0']=deepcopy(data['dy'])



y['A0']=deepcopy(data)
y['A0'].x=deepcopy(x['A0'].y)
y['A0'].y=deepcopy(x['A0'].x)
y['A0'].dx=deepcopy(x['A0'].dy)
y['A0'].dy=deepcopy(x['A0'].dx)

terms=['x', '(pow2(x))',
         'y', '(pow2(y))',
         '(x * y)', '(x * (pow2(y)))',
         '((pow2(x)) * y)']

models = list(set(get_expressions(terms,4)))

print(x)
print(y)


mdl=np.inf
visited_dl=[]

fit_mdl=np.inf
fit_visited_dl=[]

smooth_mdl=np.inf
smooth_visited_dl=[]

#ODE BMS###############################################################
true=('((_a0_ * x) + (_a1_ * (y * x)))', '((_a0_ * x) + (_a1_ * (y * x)))')
pms_x = ms_ode.Tree(variables=['x','y'],
        parameters=['a%d' % i for i in range(8)],
        x=x,from_string=true[0],prior_par=prior)
pms_y = ms_ode.Tree(variables=['x','y'],
        parameters=['a%d' % i for i in range(8)],
        x=y,from_string=true[1],prior_par=prior)
pms_x.fy=pms_y
pms_y.fy=pms_x
pms_x.get_bic(reset=True, fit=True)
pms_x.get_energy(bic=True, reset=True)
visited_dl.append(pms_x.E + pms_y.EP)
if (pms_x.E + pms_y.EP) <mdl:
    mdl=copy(pms_x.E + pms_y.EP)
    mdl_exp_ode=f'{pms_x}____________{pms_y}'
    print('New MDL ODE',mdl,pms_x,pms_y,'####################')
    with open(f'./exh_ode_mdl{tail[:-4]}.pkl', 'wb') as file1:
        # A new file will be created
        pickle.dump({'x':pms_x,'y':pms_y}, file1)
#FIT BMS###############################################################
true_fits=('((_a0_ * x) + (_a1_ * (y * x)))', '((_a0_ * y) + (_a1_ * (y * x)))')
fit_pms_x = ms_fit.Tree(variables=['x','y'],
        parameters=['a%d' % i for i in range(8)],
        x=x,y=fit_x,from_string=true_fits[0],prior_par=prior)
fit_pms_y = ms_fit.Tree(variables=['x','y'],
        parameters=['a%d' % i for i in range(8)],
        x=x,y=fit_y,from_string=true_fits[1],prior_par=prior)


fit_visited_dl.append(fit_pms_x.E + fit_pms_y.E)
if (fit_pms_x.E + fit_pms_y.E) <fit_mdl:
    fit_mdl=copy(fit_pms_x.E + fit_pms_y.E)
    mdl_exp_fit=f'{fit_pms_x}____________{fit_pms_y}'
    print('New MDL fit',fit_mdl,fit_pms_x,fit_pms_y,'####################')
    print(file)
    with open(f'./exh_fit_mdl_{tail[:-4]}.pkl', 'wb') as file1:
        # A new file will be created
        pickle.dump({'x':fit_pms_x,'y':fit_pms_y}, file1)
#SMOOTH BMS#############################################################
smooth_pms_x = ms_fit.Tree(variables=['x','y'],
        parameters=['a%d' % i for i in range(8)],
        x=x,y=smooth_x,from_string=true_fits[0],prior_par=prior)
smooth_pms_y = ms_fit.Tree(variables=['x','y'],
        parameters=['a%d' % i for i in range(8)],
        x=x,y=smooth_y,from_string=true_fits[1],prior_par=prior)


smooth_visited_dl.append(smooth_pms_x.E + smooth_pms_y.EP)
if (smooth_pms_x.E + smooth_pms_y.E) <smooth_mdl:
    smooth_mdl=copy(smooth_pms_x.E + smooth_pms_y.E)
    mdl_exp_smooth=f'{smooth_pms_x}____________{smooth_pms_y}'
    print('New MDL smooth',smooth_mdl,smooth_pms_x,smooth_pms_y,'####################')
    print(file)
    with open(f'./exh_smooth_mdl_{tail[:-4]}.pkl', 'wb') as file1:
        # A new file will be created
        pickle.dump({'x':smooth_pms_x,'y':smooth_pms_y}, file1)
count=0

for combo in itertools.product(models, repeat=2):
    #with open("test.txt", "a") as myfile:
    #    myfile.write(str(combo))
    #ODE BMS###############################################################
    pms_x = ms_ode.Tree(variables=['x','y'],
                parameters=['a%d' % i for i in range(8)],
                x=x,from_string=combo[0][0],prior_par=prior)
    pms_y = ms_ode.Tree(variables=['x','y'],
        parameters=['a%d' % i for i in range(8)],
        x=y,from_string=combo[1][0],prior_par=prior)
    pms_x.fy=pms_y
    pms_y.fy=pms_x
    pms_x.get_bic(reset=True, fit=True)
    pms_x.get_energy(bic=True, reset=True)
    visited_dl.append(pms_x.E + pms_y.EP)
    if (pms_x.E + pms_y.EP) <mdl:
        mdl=copy(pms_x.E + pms_y.EP)
        mdl_exp_ode=f'{pms_x}____________{pms_y}'
        print('New MDL ODE',mdl,pms_x,pms_y,'####################')
        with open(f'./exh_ode_mdl{tail[:-4]}.pkl', 'wb') as file1:
            # A new file will be created
            pickle.dump({'x':pms_x,'y':pms_y}, file1)
    #with open("test.txt", "a") as myfile:
    #    myfile.write("ode")
    #FIT BMS###############################################################
    fit_pms_x = ms_fit.Tree(variables=['x','y'],
                parameters=['a%d' % i for i in range(8)],
                x=x,y=fit_x,from_string=combo[0][0],prior_par=prior)
    fit_pms_y = ms_fit.Tree(variables=['x','y'],
        parameters=['a%d' % i for i in range(8)],
        x=x,y=fit_y,from_string=combo[1][0],prior_par=prior)

    fit_visited_dl.append(fit_pms_x.E + fit_pms_y.E)
    if (fit_pms_x.E + fit_pms_y.E) <fit_mdl:
        fit_mdl=copy(fit_pms_x.E + fit_pms_y.E)
        mdl_exp_fit=f'{fit_pms_x}____________{fit_pms_y}'
        print('New MDL fit',fit_mdl,fit_pms_x,fit_pms_y,'####################')
        print(tail)
        with open(f'./exh_fit_mdl_{tail[:-4]}.pkl', 'wb') as file1:
            # A new file will be created
            pickle.dump({'x':fit_pms_x,'y':fit_pms_y}, file1)
    #with open("test.txt", "a") as myfile:
    #    myfile.write("fit")
    #SMOOTH BMS#############################################################
    smooth_pms_x = ms_fit.Tree(variables=['x','y'],
                parameters=['a%d' % i for i in range(8)],
                x=x,y=smooth_x,from_string=combo[0][0],prior_par=prior)
    smooth_pms_y = ms_fit.Tree(variables=['x','y'],
        parameters=['a%d' % i for i in range(8)],
        x=x,y=smooth_y,from_string=combo[1][0],prior_par=prior)

    smooth_visited_dl.append(smooth_pms_x.E + smooth_pms_y.E)
    if (smooth_pms_x.E + smooth_pms_y.E) <smooth_mdl:
        smooth_mdl=copy(smooth_pms_x.E + smooth_pms_y.E)
        mdl_exp_smooth=f'{smooth_pms_x}____________{smooth_pms_y}'
        print('New MDL smooth',smooth_mdl,smooth_pms_x,smooth_pms_y,'####################')
        print(tail)
        with open(f'./exh_smooth_mdl_{tail[:-4]}.pkl', 'wb') as file1:
            # A new file will be created
            pickle.dump({'x':smooth_pms_x,'y':smooth_pms_y}, file1)
    #with open("test.txt", "a") as myfile:
    #    myfile.write("smooth")
    del pms_x, pms_y, fit_pms_x, fit_pms_y,smooth_pms_x, smooth_pms_y

    count+=1
print('End loop')
print(mdl,mdl_exp_ode)
print(fit_mdl,mdl_exp_fit)
print(smooth_mdl,mdl_exp_smooth)
with open(f'./exh_ode_dl_list{tail[:-4]}.pkl', 'wb') as file1:
    # A new file will be created
    pickle.dump(visited_dl, file1)
with open(f'./exh_fit_dl_list{tail[:-4]}.pkl', 'wb') as file1:
    # A new file will be created
    pickle.dump(fit_visited_dl, file1)
with open(f'./exh_smooth_dl_list{tail[:-4]}.pkl', 'wb') as file1:
    # A new file will be created
    pickle.dump(smooth_visited_dl, file1)
"""
with open(f'./end_file.txt', 'w') as file1:
    # A new file will be created
    file1.write("hello")
"""
print('End program')

