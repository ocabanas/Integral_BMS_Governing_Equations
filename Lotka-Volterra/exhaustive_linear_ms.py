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
from scipy.signal import savgol_filter
import pynumdiff

from sympy import zoo, oo, nan
from sympy.core.numbers import ComplexInfinity

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
        #all_exp += get_expressions_no_cte(the_vars, n=n)
    return all_exp



# Import Machine Scientist
#stderr_fileno = sys.stderr
#sys.stderr = open(os.devnull, 'w')

src = os.path.dirname('../I-BMS-2d/')
sys.path.append(src)

from importlib.machinery import SourceFileLoader

# Get the absolute path of the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
# Define the relative path to the module
relative_module_path = "../I-BMS-2d/mcmc_ode.py"
path = os.path.join(script_dir, relative_module_path)
ms_ode = SourceFileLoader("ms_ode", path).load_module()

# Import Machine Scientist FIT
from importlib.machinery import SourceFileLoader

# Get the absolute path of the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
# Define the relative path to the module
relative_module_path = "../rguimera-machine-scientist/mcmc.py"
path = os.path.join(script_dir, relative_module_path)
ms_fit = SourceFileLoader("ms_fit", path).load_module()

path = os.path.join(src, "Prior/")
sys.path.append(path)
from fit_prior import read_prior_par

priors = {
    "v2_p3": f"Prior/final_prior_param_sq.named_equations.nv2.np3.2016-09-09 18:49:42.927679.dat",
    "v2_p4": f"Prior/final_prior_param_sq.named_equations.nv2.np4.2016-09-09 18:49:43.056910.dat",
    "v2_p8": f"Prior/final_prior_param_sq.named_equations.nv2.np8.2016-09-09 18:49:42.800618.dat",
}

path = os.path.join(src, priors[f"v{2}_p{8}"])
prior = read_prior_par(path)

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
path, tail = os.path.split(file)
print("Directory:", path)
print("Filename:", tail)

data=pd.read_csv(file)
x_ode={'d0':deepcopy(data)}
y_ode={'d0':deepcopy(data)}
y_ode['d0'].x=deepcopy(x_ode['d0'].y)
y_ode['d0'].y=deepcopy(x_ode['d0'].x)
h = data.t.to_numpy()[1] - data.t.to_numpy()[0]

par = [2, 21, 21]
x_hat, dxdt_hat = deepcopy(pynumdiff.linear_model.polydiff(
        data.x, h, par, options=None))
y_hat, dydt_hat = deepcopy(pynumdiff.linear_model.polydiff(
        data.y, h, par, options=None))

dx_ode = {}
dx_ode['d0'] = [ pd.DataFrame(data={'x':x_hat,'y':y_hat}),
            pd.DataFrame(data={'x':dxdt_hat,'y':dydt_hat})]
dy_ode = {}
dy_ode['d0'] = [ pd.DataFrame(data={'x':y_hat,'y':x_hat}),
            pd.DataFrame(data={'x':dydt_hat,'y':dxdt_hat})]


fit_x={}
fit_y={}

B=deepcopy(data.x.values)
h=data.t.to_numpy()[1]-data.t.to_numpy()[0]
x_hat_finite, dxdt_hat_finite = pynumdiff.finite_difference._finite_difference.second_order(B, h)
fit_x['d0']=pd.Series(dxdt_hat_finite)
B = data.y.values
y_hat_finite, dydt_hat_finite = pynumdiff.finite_difference._finite_difference.second_order(B, h)
fit_y['d0']=pd.Series(dydt_hat_finite)

smooth_x = pd.DataFrame(data = {'x': x_hat, 'y': y_hat})
smooth_dx = pd.Series(dxdt_hat)
smooth_dy = pd.Series(dydt_hat)

terms=['x', '(pow2(x))',
         'y', '(pow2(y))',
         '(x * y)', '(x * (pow2(y)))',
         '((pow2(x)) * y)','(pow3(x))','(pow3(y))']

models = list(set(get_expressions(terms,4)))


mdl=np.inf
visited_dl=[]

fit_mdl=np.inf
fit_visited_dl=[]

smooth_mdl=np.inf
smooth_visited_dl=[]

#ODE BMS###############################################################
true=('((_a0_ * x) + (_a1_ * (y * x)))', '((_a0_ * x) + (_a1_ * (y * x)))')
print(dx_ode)
print(dy_ode)
pms_x = ms_ode.Tree(variables=['x','y'],
        parameters=['a%d' % i for i in range(8)],
        x=x_ode,dx=dx_ode,from_string=true[0],prior_par=prior)
pms_y = ms_ode.Tree(variables=['x','y'],
        parameters=['a%d' % i for i in range(8)],
        x=y_ode,dx=dy_ode,from_string=true[1],prior_par=prior)
pms_x.fy = pms_y
pms_y.fy = pms_x
# print('refit')
pms_x.get_bic(reset=True, fit=True,verbose=True)
pms_x.get_energy(bic=True, reset=True)
print('I-BMS true:',pms_x.E)
if pms_x.E <mdl:
    mdl=copy(pms_x.E)
    ibms_combo= (true[0],true[1] )
#FIT BMS###############################################################
true_fits=('((_a0_ * x) + (_a1_ * (y * x)))', '((_a0_ * y) + (_a1_ * (y * x)))')
fit_pms_x = ms_fit.Tree(variables=['x','y'],
        parameters=['a%d' % i for i in range(8)],
        x=x_ode,y=fit_x,from_string=true_fits[0],prior_par=prior)
fit_pms_y = ms_fit.Tree(variables=['x','y'],
        parameters=['a%d' % i for i in range(8)],
        x=x_ode,y=fit_y,from_string=true_fits[1],prior_par=prior)


if (fit_pms_x.E + fit_pms_y.E) <fit_mdl:
    fit_mdl=copy(fit_pms_x.E + fit_pms_y.E)
    fit_combo= (true_fits[0],true_fits[1] )
#SMOOTH BMS#############################################################
smooth_pms_x = ms_fit.Tree(variables=['x','y'],
        parameters=['a%d' % i for i in range(8)],
        x=smooth_x,y=smooth_dx,from_string=true_fits[0],prior_par=prior)
smooth_pms_y = ms_fit.Tree(variables=['x','y'],
        parameters=['a%d' % i for i in range(8)],
        x=smooth_x,y=smooth_dy,from_string=true_fits[1],prior_par=prior)


if (smooth_pms_x.E + smooth_pms_y.E) <smooth_mdl:
    smooth_mdl=copy(smooth_pms_x.E + smooth_pms_y.E)
    smooth_combo= (true_fits[0],true_fits[1] )


def evaluate_combo(combo):
    """Evaluate a single pair of models and return minimal results."""
    combo_x, combo_y = combo[0][0], combo[1][0]
    # ODE BMS
    pms_x = ms_ode.Tree(variables=['x','y'],
        parameters=['a%d' % i for i in range(8)],
        x=x_ode,dx=dx_ode,from_string=combo_x,prior_par=prior)
    pms_y = ms_ode.Tree(variables=['x','y'],
            parameters=['a%d' % i for i in range(8)],
            x=y_ode,dx=dy_ode,from_string=combo_y,prior_par=prior)
    pms_x.fy = pms_y
    pms_y.fy = pms_x
    # print('refit')
    pms_x.get_bic(reset=True, fit=True)
    pms_x.get_energy(bic=True, reset=True)
    mdl = pms_x.E

    # Fit BMS
    fit_pms_x = ms_fit.Tree(variables=['x','y'],
                            parameters=['a%d' % i for i in range(8)],
                            x=x_ode, y=fit_x, from_string=combo_x, prior_par=prior)
    fit_pms_y = ms_fit.Tree(variables=['x','y'],
                            parameters=['a%d' % i for i in range(8)],
                            x=x_ode, y=fit_y, from_string=combo_y, prior_par=prior)
    fit_mdl = fit_pms_x.E + fit_pms_y.E

    # Smooth BMS
    smooth_pms_x = ms_fit.Tree(variables=['x','y'],
                               parameters=['a%d' % i for i in range(8)],
                               x=smooth_x, y=smooth_dx, from_string=combo_x, prior_par=prior)
    smooth_pms_y = ms_fit.Tree(variables=['x','y'],
                               parameters=['a%d' % i for i in range(8)],
                               x=smooth_x, y=smooth_dy, from_string=combo_y, prior_par=prior)
    smooth_mdl = smooth_pms_x.E + smooth_pms_y.E

    # Cleanup to free memory
    del pms_x, pms_y, fit_pms_x, fit_pms_y, smooth_pms_x, smooth_pms_y
    gc.collect()

    return mdl, fit_mdl, smooth_mdl, combo_x, combo_y

from multiprocessing import Pool


combos = list(itertools.product(models, repeat=2))
def is_valid_real(x):
	return x.is_real and not x.has(zoo, oo, -oo, nan, ComplexInfinity)
# Sequential pool with automatic memory cleanup
with Pool(processes=5, maxtasksperchild=1) as pool:
	for result in pool.imap(evaluate_combo, combos):
		ibms_E, fit_E, smooth_E, combo_x, combo_y = result
		if ibms_E <mdl:
			mdl=copy(ibms_E)
			ibms_combo= (combo_x,combo_y )
			print('Better IBMS combo',ibms_combo,mdl)
		if is_valid_real(fit_E):
			if fit_E <fit_mdl:
				fit_mdl=copy(fit_E)
				fit_combo= (combo_x,combo_y )
		if is_valid_real(smooth_E):
			if smooth_E <smooth_mdl:
				smooth_mdl=copy(smooth_E)
				smooth_combo= (combo_x,combo_y )

# Create and save the best model.
pms_x = ms_ode.Tree(variables=['x','y'],
        parameters=['a%d' % i for i in range(8)],
        x=x_ode,dx=dx_ode,from_string=ibms_combo[0],prior_par=prior)
pms_y = ms_ode.Tree(variables=['x','y'],
        parameters=['a%d' % i for i in range(8)],
        x=y_ode,dx=dy_ode,from_string=ibms_combo[1],prior_par=prior)
pms_x.fy = pms_y
pms_y.fy = pms_x
# print('refit')
pms_x.get_bic(reset=True, fit=True)
pms_x.get_energy(bic=True, reset=True)
mdl_exp_ode=f'{pms_x}____________{pms_y}'
print('New MDL ODE',mdl,pms_x,pms_y,'####################')
with open(f'./results/exh_ode_mdl{tail[:-4]}.pkl', 'wb') as file1:
	# A new file will be created
	pickle.dump({'x':pms_x,'y':pms_y}, file1)

fit_pms_x = ms_fit.Tree(variables=['x','y'],
			parameters=['a%d' % i for i in range(8)],
			x=x_ode,y=fit_x,from_string=fit_combo[0],prior_par=prior)
fit_pms_y = ms_fit.Tree(variables=['x','y'],
	parameters=['a%d' % i for i in range(8)],
	x=x_ode,y=fit_y,from_string=fit_combo[1],prior_par=prior)
mdl_exp_fit=f'{fit_pms_x}____________{fit_pms_y}'
print('New MDL fit',fit_mdl,fit_pms_x,fit_pms_y,'####################')
print(tail)
with open(f'./results/exh_fit_mdl_{tail[:-4]}.pkl', 'wb') as file1:
	# A new file will be created
	pickle.dump({'x':fit_pms_x,'y':fit_pms_y}, file1)

smooth_pms_x = ms_fit.Tree(variables=['x','y'],
			parameters=['a%d' % i for i in range(8)],
			x=smooth_x,y=smooth_dx,from_string=smooth_combo[0],prior_par=prior)
smooth_pms_y = ms_fit.Tree(variables=['x','y'],
	parameters=['a%d' % i for i in range(8)],
	x=smooth_x,y=smooth_dy,from_string=smooth_combo[1],prior_par=prior)


mdl_exp_smooth=f'{smooth_pms_x}____________{smooth_pms_y}'
print('New MDL smooth',smooth_mdl,smooth_pms_x,smooth_pms_y,'####################')
print(tail)
with open(f'./results/exh_smooth_mdl_{tail[:-4]}.pkl', 'wb') as file1:
	# A new file will be created
	pickle.dump({'x':smooth_pms_x,'y':smooth_pms_y}, file1)
print('End program')

