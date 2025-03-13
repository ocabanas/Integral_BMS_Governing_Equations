import sys
import numpy as np
import pandas as pd
import warnings
import os
import pickle
import time
warnings.filterwarnings("ignore")
from copy import deepcopy, copy

# Get the absolute path of the script's directory
src = os.path.dirname(os.path.abspath(__file__))
sys.path.append(src)
from mcmc_ode import *
from parallel_ode import *

path = os.path.join(src, "Prior/")
sys.path.append(path)
from fit_prior import read_prior_par

priors = {
    "v1_p4": f"Prior/final_prior_param_sq.named_equations.nv1.np4.2017-10-18 18:07:35.214923.dat",
    "v1_p3": f"Prior/final_prior_param_sq.named_equations.nv1.np3.2017-10-18 18:07:35.262530.dat",
    "v1_p2": f"Prior/final_prior_param_sq.named_equations.nv1.np2.2017-10-18 18:07:35.163458.dat",
    "v1_p5": f"Prior/final_prior_param_sq.named_equations.nv1.np5.2017-10-18 18:07:35.227360.dat",
    "v1_p6": f"Prior/final_prior_param_sq.named_equations.nv1.np6.2017-10-18 18:07:35.224405.dat",
    "v1_p7": f"Prior/final_prior_param_sq.named_equations.nv1.np7.2017-10-18 18:07:35.230340.dat",
    "v1_p8": f"Prior/final_prior_param_sq.named_equations.nv1.np8.2017-10-18 18:07:35.261518.dat",
    "v1_p9": f"Prior/final_prior_param_sq.named_equations.nv1.np9.2017-10-18 18:07:35.163833.dat",
    "v2_p3": f"Prior/final_prior_param_sq.named_equations.nv2.np3.2016-09-09 18:49:42.927679.dat",
    "v2_p4": f"Prior/final_prior_param_sq.named_equations.nv2.np4.2016-09-09 18:49:43.056910.dat",
    "v3_p3": f"Prior/final_prior_param_sq.named_equations.nv3.np3.2017-06-13 08:55:24.082204.dat",
    "v3_p6": f"Prior/final_prior_param_sq.named_equations.nv3.np6.maxs50.2021-12-14 09:51:44.438445.dat",
    "v4_p8": f"Prior/final_prior_param_sq.named_equations.nv4.np8.maxs200.2019-12-03 09:30:20.580307.dat",
    "v5_p10": f"Prior/final_prior_param_sq.named_equations.nv5.np10.2016-07-11 17:12:38.129639.dat",
    "v5_p12": f"Prior/final_prior_param_sq.named_equations.nv5.np12.2016-07-11 17:12:37.338812.dat",
    "v6_p12": f"Prior/final_prior_param_sq.named_equations.nv6.np12.2016-07-11 17:20:51.957121.dat",
    "v7_p14": f"Prior/final_prior_param_sq.named_equations.nv7.np14.2016-06-06 16:43:26.130179.dat",
}


def machinescientist_to_folder(
    x,
    y,
    XLABS,
    n_params,
    resets=1,
    steps_prod=1000,
    Ts=[1] + [1.04**k for k in range(1, 20)],
    folder="",
    constraint=None,
):
    if f"v{len(XLABS)}_p{str(n_params)}" not in list(priors.keys()):
        print(f"v{len(XLABS)}_p{str(n_params)}")
        raise ValueError
    path = os.path.join(src, priors[f"v{len(XLABS)}_p{str(n_params)}"])
    prior_par = read_prior_par(path)
    del prior_par["Nopi_abs"]
    del prior_par["Nopi2_abs"]
    del prior_par["Nopi_sin"]
    del prior_par["Nopi2_sin"]
    del prior_par["Nopi_cos"]
    del prior_par["Nopi2_cos"]
    del prior_par["Nopi_tan"]
    del prior_par["Nopi2_tan"]
    del prior_par["Nopi_sinh"]
    del prior_par["Nopi2_sinh"]
    del prior_par["Nopi_cosh"]
    del prior_par["Nopi2_cosh"]
    del prior_par["Nopi_tanh"]
    del prior_par["Nopi2_tanh"]
    OPS = {
        #'sin': 1,
        #'cos': 1,
        #'tan': 1,
        "exp": 1,
        #'log': 1,
        #'sinh' : 1,
        #'cosh' : 1,
        #'tanh' : 1,
        "pow2": 1,
        "pow3": 1,
        #'sqrt' : 1,
        #'fac' : 1,
        "-": 1,
        "+": 2,
        "*": 2,
        "/": 2,
        "**": 2,
    }
    description_lengths, mdl, mdl_model = [], np.inf, None
    # Start some MCMC
    runs = 0
    print(resets, steps_prod)
    while runs < resets:
        try:  # Sometimes a NaN error appears. Therefore we forget the current MCMC and start again.
            # Initialize the parallel machine scientist
            pms = Parallel(
                Ts,
                ops=OPS,
                variables=XLABS,
                parameters=["a%d" % i for i in range(n_params)],
                x=x,
                y=y,
                prior_par=prior_par,
                constraint=constraint,
            )
            # MCMC
            # description_lengths, mdl, mdl_model = [], np.inf, None
            last_seen_by_can, last_seen_by_str = {}, {}
            for f in pms.trees.values():
                last_seen_by_can[f.canonical()] = 0
                last_seen_by_str[str(f)] = 0
            NCLEAN = 200
            mc_start = time.time()
            for i in range(1, steps_prod + 1):
                start = time.time()
                # MCMC update
                pms.mcmc_step()  # MCMC step within each T

                if abs(pms.t1.E - pms.t1.get_energy(bic=True, reset=True)[0]) > 1.0e-6:
                    print("Reset energy")
                    for tree in pms.trees.values():
                        tree.get_energy(bic=True, reset=True)
                ET1, ET2 = (
                    pms.tree_swap()
                )  # Attempt to swap two randomly selected consecutive temps
                description_lengths.append(copy(pms.t1.E))
                # Add the description length to the trace
                # description_lengths.append(pms.t1.E)
                # Check if this is the MDL expression so far
                if pms.t1.E < mdl:
                    # if pms.t1.E==float('NaN'): print('NaN in best model mdl')
                    mdl, mdl_model = copy(pms.t1.E), deepcopy(pms.t1)
                    with open(f"./{folder}/mdl.pkl", "wb") as file:
                        # A new file will be created
                        pickle.dump(pms.t1, file)
                    # gc.collect()
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
                        except KeyError:  # This tree was tested but not visited anyway!
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
                        except KeyError:  # This tree was tested but not visited anyway!
                            to_remove.append(string)
                    for t in to_remove:
                        del pms.t1.fit_par[t]
                        del pms.t1.x0[t]
                        if t in last_seen_by_str:
                            del last_seen_by_str[t]
                if (i % 10) == 0:
                    end = time.time()
                    print(
                        f"Progress: {int(float(i*100)/float(steps_prod))}%  | {i}",
                        end="\r",
                    )
                if (i % 200) == 0:
                    print(str(pms.t1))
                    with open(f"./{folder}/checkpoint_run_{runs}.pkl", "wb") as file:
                        # A new file will be created
                        pickle.dump({"step": i, "object": pms}, file)
            with open(f"./{folder}/checkpoint_run_{runs}.pkl", "wb") as file:
                # A new file will be created
                pickle.dump({"step": i, "object": pms}, file)
            runs += 1
        except Exception as e:
            # q
            print("Error during MCMC evolution:")
            print(e)
            print("Current model", pms.t1)
            print("Current energy", pms.t1.E)
            print("Restarting MCMC")
    return copy(mdl_model), copy(description_lengths)


def from_string_model(
    x, y, string_model, n_vars, n_params, vars, constraint=None, silence=False
):
    prior_par = read_prior_par(priors[f"v{str(n_vars)}_p{str(n_params)}"])
    model = Tree(
        prior_par=deepcopy(prior_par),
        from_string=string_model,
        x=x,
        y=y,
        variables=vars,
        constraint=constraint,
    )
    if silence == False:
        print("Model summary")
        print("Par_values:", model.par_values)
        print(model.BT, model.PT)
        print("bic:", model.bic)
        print("E:", model.E)
        print("EB:", model.EB)
        print("EP:", model.EP)
        print("Representative:", model.representative)
        print("Variables:", model.variables)
        print("Parameters:", model.parameters)
    return model
