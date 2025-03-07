import sys
import numpy as np 
import pandas as pd
import warnings
#import gc
import os
#from memory_profiler import profile
#warnings.filterwarnings('always')
#gc.disable()
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from copy import deepcopy, copy
#from ipywidgets import IntProgress
#from IPython.display import display,display_latex,Latex
import time
import pickle
import matplotlib.gridspec as gs

src='/export/home/oriolca/BMS_ODE/Lotka_Volterra/rguimera-machine-scientist/'
sys.path.append(src)
from mcmc_ode import *
from parallel_ode import *

sys.path.append(src+'Prior/')
from fit_prior import read_prior_par

priors={
    'v1_p4':f'{src}Prior/final_prior_param_sq.named_equations.nv1.np4.2017-10-18 18:07:35.214923.dat',
    'v1_p3':f'{src}Prior/final_prior_param_sq.named_equations.nv1.np3.2017-10-18 18:07:35.262530.dat',
    'v1_p2':f'{src}Prior/final_prior_param_sq.named_equations.nv1.np2.2017-10-18 18:07:35.163458.dat',
    'v1_p5':f'{src}Prior/final_prior_param_sq.named_equations.nv1.np5.2017-10-18 18:07:35.227360.dat',
    'v1_p6':f'{src}Prior/final_prior_param_sq.named_equations.nv1.np6.2017-10-18 18:07:35.224405.dat',
    'v1_p7':f'{src}Prior/final_prior_param_sq.named_equations.nv1.np7.2017-10-18 18:07:35.230340.dat',
    'v1_p8':f'{src}Prior/final_prior_param_sq.named_equations.nv1.np8.2017-10-18 18:07:35.261518.dat',
    'v1_p9':f'{src}Prior/final_prior_param_sq.named_equations.nv1.np9.2017-10-18 18:07:35.163833.dat',
    'v2_p3':f'{src}Prior/final_prior_param_sq.named_equations.nv2.np3.2016-09-09 18:49:42.927679.dat',
    'v2_p4':f'{src}Prior/final_prior_param_sq.named_equations.nv2.np4.2016-09-09 18:49:43.056910.dat',
    'v2_p8':f'{src}Prior/final_prior_param_sq.named_equations.nv2.np8.2016-09-09 18:49:42.800618.dat',
    'v3_p3':f'{src}Prior/final_prior_param_sq.named_equations.nv3.np3.2017-06-13 08:55:24.082204.dat',
    'v3_p6':f'{src}Prior/final_prior_param_sq.named_equations.nv3.np6.maxs50.2021-12-14 09:51:44.438445.dat',
    'v4_p8':f'{src}Prior/final_prior_param_sq.named_equations.nv4.np8.maxs200.2019-12-03 09:30:20.580307.dat',
    'v5_p10':f'{src}Prior/final_prior_param_sq.named_equations.nv5.np10.2016-07-11 17:12:38.129639.dat',
    'v5_p12':f'{src}Prior/final_prior_param_sq.named_equations.nv5.np12.2016-07-11 17:12:37.338812.dat',
    'v6_p12':f'{src}Prior/final_prior_param_sq.named_equations.nv6.np12.2016-07-11 17:20:51.957121.dat',
    'v7_p14':f'{src}Prior/final_prior_param_sq.named_equations.nv7.np14.2016-06-06 16:43:26.130179.dat'
}



def machinescientist(x,y,XLABS,n_params,resets=1,
                    steps_prod=1000,
                    Ts= [1] + [1.04**k for k in range(1, 20)],
                    log_scale_prediction=False,
                    ensemble_avg=None
                    ):
    """
    pms = Parallel(
        Ts,
        variables=XLABS,
        parameters=['a%d' % i for i in range(n_params)],
        x=x, y=y,
        prior_par=prior_par,
    )"""
    #print(list(priors.keys()))
    #print(Ts)
    if f'v{len(XLABS)}_p{str(n_params)}' not in list(priors.keys()):
        print(f'v{len(XLABS)}_p{str(n_params)}')
        raise ValueError
    if ensemble_avg != None:
        list_ens_mdls={}
    dict_mdls={}
    prior_par = read_prior_par(priors[f'v{len(XLABS)}_p{str(n_params)}'])
    del prior_par['Nopi_abs']
    del prior_par['Nopi2_abs']
    OPS = {
    'sin': 1,
    'cos': 1,
    'tan': 1,
    'exp': 1,
    'log': 1,
    'sinh' : 1,
    'cosh' : 1,
    'tanh' : 1,
    'pow2' : 1,
    'pow3' : 1,
    'sqrt' : 1,
    'fac' : 1,
    '-' : 1,
    '+' : 2,
    '*' : 2,
    '/' : 2,
    '**' : 2,
    }
    best_description_lengths,lowest_mdl, best_model = [],np.inf, None
    all_mdls={}
    #Start some MCMC
    runs=0
    while runs < resets:
        try: #Sometimes a NaN error appears. Therefore we forget the current MCMC and start again.
            # Initialize the parallel machine scientist
            pms = Parallel(
                Ts,ops=OPS,
                variables=XLABS,
                parameters=['a%d' % i for i in range(n_params)],
                x=x, y=y,
                prior_par=prior_par,
            )
            # MCMC 
            description_lengths, mdl, mdl_model = [], np.inf, None
            last_seen_by_can, last_seen_by_str = {}, {}
            for f in pms.trees.values():
                last_seen_by_can[f.canonical()] = 0
                last_seen_by_str[str(f)] = 0
            NCLEAN = 1000
            mc_start=time.time()
            for i in range(1,steps_prod+1):
                start = time.time()
                # MCMC update
                pms.mcmc_step() # MCMC step within each T
                pms.tree_swap() # Attempt to swap two randomly selected consecutive temps
                # Add the description length to the trace
                description_lengths.append(pms.t1.E)
                # Check if this is the MDL expression so far
                if pms.t1.E < mdl:
                    #if pms.t1.E==float('NaN'): print('NaN in best model mdl')
                    mdl, mdl_model = copy(pms.t1.E), deepcopy(pms.t1)
                    #gc.collect()
                # Save step of model
                for f in pms.trees.values():
                    last_seen_by_can[f.canonical()] = i
                    last_seen_by_str[str(f)] = i
                # Clean up old representatives and fitted parameters to speed up
                # sampling and save memory
                if (i % NCLEAN) == 0:
                    to_remove = []
                    for represent in pms.t1.representative:
                        try:
                            if last_seen_by_can[represent] < (i - NCLEAN):
                                to_remove.append(represent)
                        except KeyError: # This tree was tested but not visited anyway!
                            to_remove.append(represent)
                    for t in to_remove: 
                        del pms.t1.representative[t]
                        if t in last_seen_by_can:
                            del last_seen_by_can[t]
                    to_remove = []
                    for string in pms.t1.fit_par:
                        try:
                            if last_seen_by_str[string] < (i - NCLEAN):
                                to_remove.append(string)
                        except KeyError: # This tree was tested but not visited anyway!
                            to_remove.append(string)
                    for t in to_remove: 
                        del pms.t1.fit_par[t]
                        if t in last_seen_by_str:
                            del last_seen_by_str[t]
                if (i % 10) == 0:
                    end = time.time()
                    print(f'Progress: {int(float(i*100)/float(steps_prod))}%  | {round(1./(end-start),2)} MCs/s | Time left: {round(float(steps_prod-i)*float(end-mc_start)/(60.*(i)),2)}min.', end='\r')
                if ensemble_avg != None and i>= ensemble_avg[0] and (i%ensemble_avg[1])==0:
                    if pms.t1.E==float('NaN'): print('NaN in ensemble average')
                    list_ens_mdls+=[deepcopy(pms.t1)]
                    #gc.collect()
                if i>2500 and str(pms.t1) not in list(dict_mdls.keys()):
                    dict_mdls[str(pms.t1)]=deepcopy(pms.t1)
                    
            print()
            
            # End MCMC
            runs+=1
            if best_model==None:
                best_description_lengths,lowest_mdl,best_model=description_lengths,mdl, deepcopy(mdl_model)
            if mdl<lowest_mdl:
                best_description_lengths=deepcopy(description_lengths)
                lowest_mdl=deepcopy(mdl)
                best_model=deepcopy(mdl_model)
            all_mdls[mdl_model.latex()]=deepcopy(description_lengths)

            print('-'*20)
            print(f"Run {runs}")
            print('-'*20)
            print('Mdl for training data:',copy(mdl))
            print("Model",mdl_model)
            print(mdl_model.latex())
        except Exception as e:
            print('Error during MCMC evolution:')
            print(e)
            print('Current model',pms.t1)
            print('Current energy',pms.t1.E)
            print('Restarting MCMC')
    fig=plt.figure(figsize=(15, 5))
    
    g = gs.GridSpec(2,1)
    ax = fig.add_subplot(g[0])
    for i,j in all_mdls.items():
        if i!=best_model.latex():
            line='--'
        else:
            line='-'
        ax.plot(j,line,label=f'${i}$')
    ax.set_xlabel('MCMC step', fontsize=14)
    ax.set_ylabel('Description length', fontsize=14)
    ax.set_title('MDL model all MC runs')
    ax1=fig.add_subplot(g[1])
    h, l = ax.get_legend_handles_labels() 
    ax1.legend(h, l,fontsize='large')
    ax1.axis('off')
    print('#'*40)
    print('Lowest mdl for training data:',copy(lowest_mdl))
    print('Model:',copy(best_model))
    if ensemble_avg != None:
        return copy(best_model) , copy(list_ens_mdls),dict_mdls, fig
    else: 
        return copy(best_model), copy(lowest_mdl),dict_mdls, fig
def machinescientist_to_folder(x,y,XLABS,n_params,initial_guess_x=None,initial_guess_y=None,resets=1,
                    steps_prod=1000,
                    Ts= [1] + [1.04**k for k in range(1, 40,2)],
                    folder=''
                    ):
    #Ts= [1.08**-k for k in range(1, 10,2)[::-1]] + [1] + [1.04**k for k in range(1, 30,2)]
    Ts= [1] + [1.04**k for k in range(1, 20)]
    #Ts= [0.01,0.05,0.1,0.25,0.5,0.75] + [1] + [1.04**k for k in range(1, 10,2)]
    print('Temp series',Ts)
    """
    pms = Parallel(
        Ts,
        variables=XLABS,
        parameters=['a%d' % i for i in range(n_params)],
        x=x, y=y,
        prior_par=prior_par,
    )"""
    #print(list(priors.keys()))
    #print(Ts)
    if f'v{len(XLABS)}_p{str(n_params)}' not in list(priors.keys()):
        print(f'v{len(XLABS)}_p{str(n_params)}')
        raise ValueError
    prior_par = read_prior_par(priors[f'v{len(XLABS)}_p{str(n_params)}'])
    del prior_par['Nopi_abs']
    del prior_par['Nopi2_abs']
    del prior_par['Nopi_sin']
    del prior_par['Nopi2_sin']
    del prior_par['Nopi_cos']
    del prior_par['Nopi2_cos']
    del prior_par['Nopi_tan']
    del prior_par['Nopi2_tan']
    del prior_par['Nopi_sinh']
    del prior_par['Nopi2_sinh']
    del prior_par['Nopi_cosh']
    del prior_par['Nopi2_cosh']
    del prior_par['Nopi_tanh']
    del prior_par['Nopi2_tanh']
    OPS = {
    #'sin': 1,
    #'cos': 1,
    #'tan': 1,
    'exp': 1,
    #'log': 1,
    #'sinh' : 1,
    #'cosh' : 1,
    #'tanh' : 1,
    'pow2' : 1,
    'pow3' : 1,
    #'sqrt' : 1,
    #'fac' : 1,
    '-' : 1,
    '+' : 2,
    '*' : 2,
    '/' : 2,
    '**' : 2,
    }
    #best_description_lengths,lowest_mdl, best_model = [],np.inf, None
    #all_mdls={}
    description_lengths, mdl,mdl_x,mdl_y, mdl_model_x,mdl_model_y = [], np.inf,np.inf, np.inf,None,  None
    #Start some MCMC
    runs=0
    while runs < resets:
        try: #Sometimes a NaN error appears. Therefore we forget the current MCMC and start again.
            # Initialize the parallel machine scientist
            pms_x = Parallel(
                Ts,ops=OPS,
                variables=XLABS,
                parameters=['a%d' % i for i in range(n_params)],
                x=x,initial_guess=initial_guess_x, 
                prior_par=prior_par,
            )
            pms_y = Parallel(
                Ts,ops=OPS,
                variables=XLABS,
                parameters=['a%d' % i for i in range(n_params)],
                x=y,initial_guess=initial_guess_y, 
                prior_par=prior_par,
            )
            print('setting f-g links')
            for temp in pms_x.trees.keys():
                pms_x.trees[temp].fy=pms_y.trees[temp]
                pms_y.trees[temp].fy=pms_x.trees[temp]
                #print('refit')
                pms_x.trees[temp].get_bic(reset=True, fit=True)
                pms_x.trees[temp].get_energy(bic=True, reset=True)
            pms_x.t1 = pms_x.trees[str(min(Ts))]
            pms_y.t1 = pms_y.trees[str(min(Ts))]
            #print(pms_x.t1.E,str(pms_x.t1))
            #print(pms_y.t1.E,str(pms_y.t1))
            
            # MCMC 
            #exit(1)
            #description_lengths, mdl, mdl_model = [], np.inf, None
            """
            last_seen_by_can, last_seen_by_str = {}, {}
            for f in pms.trees.values():
                last_seen_by_can[f.canonical()] = 0
                last_seen_by_str[str(f)] = 0
            NCLEAN = 200
            """
            mc_start=time.time()
            description_lengths.append([])
            for i in range(1,steps_prod+1):
                start = time.time()
                # MCMC update
                pms_x.mcmc_step() # MCMC step within each T
                pms_y.mcmc_step()
                """
                if abs(pms.t1.E - pms.t1.get_energy(bic=True, reset=True)[0]) > 1.e-6:
                    print('Reset energy')
                    for tree in pms.trees.values():
                        tree.get_energy(bic=True,reset=True)"""
                ET1,ET2 = pms_x.tree_swap() # Attempt to swap two randomly selected consecutive temps
                #ET1,ET2 = pms_y.tree_swap() # Attempt to swap two randomly selected consecutive temps
            
                if ET1 != None:
                    t1 = pms_y.trees[ET1]
                    t2 = pms_y.trees[ET2]
                    BT1, BT2 = t1.BT, t2.BT
                    pms_y.trees[ET1] = t2
                    pms_y.trees[ET2] = t1
                    """
                    t1.BT = BT2
                    t2.BT = BT1
                    t1.BT = BT2
                    t2.BT = BT1
                    """
                    t1.BT = BT2
                    #t1.PT = BT2
                    t2.BT = BT1
                    #t2.PT = BT1


                    pms_x.trees[ET1].get_bic(reset=True,fit=True)
                    pms_x.trees[ET1].get_energy(bic=False,reset=True)
                    """
                    pms_x.trees[ET1].fy.get_energy(bic=False,reset=True)
                    """

                    #pms_x.trees[ET2].get_bic(reset=True,fit=True)
                    #pms_x.trees[ET2].get_energy(bic=False,reset=True)
                    """
                    pms_x.trees[ET2].fy.get_energy(bic=False,reset=True)
                    """

                    pms_y.t1 = pms_y.trees[str(min(Ts))]
                
                description_lengths[runs].append(copy(pms_x.t1.E+pms_y.t1.EP))
                # Add the description length to the trace
                #description_lengths.append(pms.t1.E)
                # Check if this is the MDL expression so far
                if pms_x.t1.E + pms_y.t1.EP < mdl:
                    #if pms.t1.E==float('NaN'): print('NaN in best model mdl')
                    mdl=copy(pms_x.t1.E + pms_y.t1.EP)
                    mdl_model_x=deepcopy(pms_x.t1)
                    mdl_model_y=deepcopy(pms_y.t1)
                    with open(f'./noise_data_ode/{folder}', 'wb') as f:
                        # A new file will be created
                        pickle.dump({'x':mdl_model_x,'y':mdl_model_y}, f)
                    #gc.collect()
                # Save step of model
                #print('main',runs,i,pms.t1.E, str(pms.t1),mdl,str(mdl_model))
                if (i % 10) == 0:
                    end = time.time()
                    print(f'Progress: {int(float(i*100)/float(steps_prod))}%  | Step {i} | {str(pms_x.t1)}{str(pms_x.t1.E)}_________{str(pms_y.t1)}{str(pms_y.t1.E)}', end='\r')
                """
                if (i % 50) == 0:
                    print('#'*20)
                    print(f'Progress: {int(float(i*100)/float(steps_prod))}%  | Step {i} | {str(pms_x.t1)}{str(pms_x.t1.E)}_________{str(pms_y.t1)}{str(pms_y.t1.E)}')#, end='\r')
                    print('_'*20)
                    for temp in pms_x.trees.keys():
                        print('-'*5,temp,'-'*5)   
                        print('x',pms_x.trees[temp],pms_x.trees[temp].E,pms_x.trees[temp].bic,pms_x.trees[temp].sse,{key:val for key,val in pms_x.trees[temp].par_values['A0'].items() if str(val)!=str(1.0)})
                        print('y',pms_y.trees[temp],pms_y.trees[temp].E,pms_y.trees[temp].bic,pms_y.trees[temp].sse,{key:val for key,val in pms_y.trees[temp].par_values['A0'].items() if str(val)!=str(1.0)})
                    print('#'*20)
                    with open(f'./{folder}/checkpoint_run_{runs}.pkl', 'wb') as file:
                        # A new file will be created
                        pickle.dump({'step':i,'x':pms_x,'y':pms_y}, file)
            
            with open(f'./{folder}/checkpoint_run_{runs}.pkl', 'wb') as file:
                # A new file will be created
                pickle.dump({'step':i,'x':pms_x,'y':pms_y}, file) 
            """
            
        except Exception as e:
            #q
            print('Error during MCMC evolution:')
            print(e)
            print(traceback.format_exc())
            exit(1)
            
        runs+=1
    print('End model',mdl_model_x,'......',mdl_model_y)
    print('End energy',mdl)
    return copy(mdl_model_x),copy(mdl_model_y),copy(description_lengths)
def from_string_model(x,initial_guess,string_model,n_vars,n_params,vars,silence=False):
    prior_par = read_prior_par(priors[f'v{str(n_vars)}_p{str(n_params)}'])
    model=Tree(prior_par=deepcopy(prior_par), from_string=string_model , x=x,initial_guess=initial_guess,variables=vars)
    if silence==False:
        print('Model summary')
        print('Par_values:',model.par_values)
        print(model.BT,model.PT)
        print('bic:',model.bic)
        print('E:',model.E)
        print('EB:',model.EB)
        print('EP:',model.EP)
        print('Representative:',model.representative)
        print('Variables:',model.variables)
        print('Parameters:',model.parameters)
    return model


if __name__ == "__main__":
    main()
    
