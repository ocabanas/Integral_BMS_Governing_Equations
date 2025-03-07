import sys
import numpy as np
import pandas as pd
import warnings
import os
from copy import deepcopy, copy
import time

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
    initial_guess_x=None,
    initial_guess_y=None,
    resets=1,
    steps_prod=1000,
    Ts=[1] + [1.04**k for k in range(1, 20)],
    folder="",
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
    description_lengths, mdl, mdl_x, mdl_y, mdl_model_x, mdl_model_y = (
        [],
        np.inf,
        np.inf,
        np.inf,
        None,
        None,
    )
    # Start some MCMC
    runs = 0
    while runs < resets:
        try:  # Sometimes a NaN error appears. Therefore we forget the current MCMC and start again.
            # Initialize the parallel machine scientist
            pms_x = Parallel(
                Ts,
                ops=OPS,
                variables=XLABS,
                parameters=["a%d" % i for i in range(n_params)],
                x=x,
                initial_guess=initial_guess_x,
                prior_par=prior_par,
            )
            pms_y = Parallel(
                Ts,
                ops=OPS,
                variables=XLABS,
                parameters=["a%d" % i for i in range(n_params)],
                x=y,
                initial_guess=initial_guess_y,
                prior_par=prior_par,
            )
            print("setting f-g links")
            for temp in pms_x.trees.keys():
                pms_x.trees[temp].fy = pms_y.trees[temp]
                pms_y.trees[temp].fy = pms_x.trees[temp]
                # print('refit')
                pms_x.trees[temp].get_bic(reset=True, fit=True)
                pms_x.trees[temp].get_energy(bic=True, reset=True)
            pms_x.t1 = pms_x.trees[str(min(Ts))]
            pms_y.t1 = pms_y.trees[str(min(Ts))]
            mc_start = time.time()
            description_lengths.append([])
            for i in range(1, steps_prod + 1):
                start = time.time()
                # MCMC update
                pms_x.mcmc_step()  # MCMC step within each T
                pms_y.mcmc_step()
                ET1, ET2 = (
                    pms_x.tree_swap()
                )  # Attempt to swap two randomly selected consecutive temps

                if ET1 != None:
                    t1 = pms_y.trees[ET1]
                    t2 = pms_y.trees[ET2]
                    BT1, BT2 = t1.BT, t2.BT
                    pms_y.trees[ET1] = t2
                    pms_y.trees[ET2] = t1
                    t1.BT = BT2
                    t2.BT = BT1
                    pms_x.trees[ET1].get_bic(reset=True, fit=True)
                    pms_x.trees[ET1].get_energy(bic=False, reset=True)
                    pms_y.t1 = pms_y.trees[str(min(Ts))]

                description_lengths[runs].append(copy(pms_x.t1.E + pms_y.t1.EP))
                # Add the description length to the trace
                # description_lengths.append(pms.t1.E)
                # Check if this is the MDL expression so far
                if pms_x.t1.E + pms_y.t1.EP < mdl:
                    # if pms.t1.E==float('NaN'): print('NaN in best model mdl')
                    mdl = copy(pms_x.t1.E + pms_y.t1.EP)
                    mdl_model_x = deepcopy(pms_x.t1)
                    mdl_model_y = deepcopy(pms_y.t1)
                    with open(f"./noise_data_ode/{folder}", "wb") as f:
                        # A new file will be created
                        pickle.dump({"x": mdl_model_x, "y": mdl_model_y}, f)
                # Save step of model
                if (i % 10) == 0:
                    end = time.time()
                    print(
                        f"Progress: {int(float(i*100)/float(steps_prod))}%  | Step {i} | {str(pms_x.t1)}{str(pms_x.t1.E)}_________{str(pms_y.t1)}{str(pms_y.t1.E)}",
                        end="\r",
                    )

        except Exception as e:
            # q
            print("Error during MCMC evolution:")
            print(e)
            print(traceback.format_exc())
            exit(1)

        runs += 1
    print("End model", mdl_model_x, "......", mdl_model_y)
    print("End energy", mdl)
    return copy(mdl_model_x), copy(mdl_model_y), copy(description_lengths)


def from_string_model(
    x, initial_guess, string_model, n_vars, n_params, vars, silence=False
):
    path = os.path.join(src, priors[f"v{n_vars}_p{str(n_params)}"])
    prior_par = read_prior_par(path)
    model = Tree(
        prior_par=deepcopy(prior_par),
        from_string=string_model,
        x=x,
        initial_guess=initial_guess,
        variables=vars,
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
