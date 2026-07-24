import numpy as np
import os
os.environ["OMP_NUM_THREADS"] = f"{10}" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = f"{1}" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = f"{1}" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = f"{2}" # export VECLIB_MAXIMUM_THREADS=1
#os.environ["NUMEXPR_NUM_THREADS"] = f"{2}" # export NUMEXPR_NUM_THREADS=1
#os.environ["PYTHON_JULIACALL_THREADS"] = "2"


#os.environ["OMP_NUM_THREADS"] = f"{2}" # export OMP_NUM_THREADS=1
#os.environ["OPENBLAS_NUM_THREADS"] = f"{2}" # export OPENBLAS_NUM_THREADS=1
#os.environ["MKL_NUM_THREADS"] = f"{2}" # export MKL_NUM_THREADS=1
#os.environ["VECLIB_MAXIMUM_THREADS"] = f"{2}" # export VECLIB_MAXIMUM_THREADS=1
#os.environ["NUMEXPR_NUM_THREADS"] = f"{2}" # export NUMEXPR_NUM_THREADS=1
#os.environ["PYTHON_JULIACALL_THREADS"] = "2"

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from copy import copy
import scipy
import pynumdiff
from pysr import PySRRegressor
from scipy.signal import savgol_filter
import pickle

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

file_name = os.path.basename(file)


data = pd.read_csv(file)

h=data.t.to_numpy()[1] - data.t.to_numpy()[0]
print(h)
print(data.x.to_numpy())
x_hat, dxdt_hat = pynumdiff.polydiff(data.x.to_numpy(), dt=h, params=[2,41,41])
print(dxdt_hat)
y_hat, dydt_hat = pynumdiff.polydiff(data.y.to_numpy(), dt=h, params=[2,41,41])

data_smoothed = pd.DataFrame(data={'x':x_hat,'y':y_hat})

# Fitting dx/dt
print('CPU', os.cpu_count())

model = PySRRegressor(
    populations=6, # Fewer populations (parallel groups)
    population_size=600,              # individuals per population
    ncycles_per_iteration=600,       # generations between migrations
    niterations=40000,                 # Run until wall-limit
    early_stop_condition="stop_if(loss, complexity) = loss < 1e-6 && complexity < 15",
    timeout_in_seconds=60 * 60 * 20,      # wall-time limit
    maxsize=30,
    maxdepth=10,
    procs = 8,
    model_selection="best",   # or 'accuracy'
    binary_operators=["*", "+", "-", "/"],
    unary_operators=["square", "cube", "exp","sin","cos"],
    constraints={
        "/": (-1, 9),
        "square": 9,
        "cube": 9,
        "exp": 9,
    },
    progress=True,
    # ^ Limit the complexity within each argument.
    # "inv": (-1, 9) states that the numerator has no constraint,
    # but the denominator has a max complexity of 9.
    # "exp": 9 simply states that `exp` can only have
    # an expression of complexity 9 as input.
    nested_constraints={
        "square": {"square": 1, "cube": 1, "exp": 0},
        "cube": {"square": 1, "cube": 1, "exp": 0},
        "exp": {"square": 1, "cube": 1, "exp": 0},
    }
    # ^ Define operator for SymPy as well
)

X = data_smoothed

model.fit(X, dxdt_hat)

p_x = model.predict(X)

print('Model X',model.get_best())
best = model.get_best()
with open(f"models/{file_name[:-4]}_PYSR_x.pkl", "wb") as f:
    pickle.dump(model, f)


# Fitting dy/dt

model = PySRRegressor(
    populations=6,                   # Fewer populations (parallel groups)
    population_size=600,              # individuals per population
    ncycles_per_iteration=600,       # generations between migrations
    niterations=40000,                 # Run until wall-limit
    early_stop_condition="stop_if(loss, complexity) = loss < 1e-6 && complexity < 15",
    timeout_in_seconds=60 * 60 * 20,      # wall-time limit
    maxsize=30,
    maxdepth=10,
    procs = 8,
    model_selection="best",   # or 'accuracy'
    binary_operators=["*", "+", "-", "/"],
    unary_operators=["square", "cube", "exp","sin","cos"],
    constraints={
        "/": (-1, 9),
        "square": 9,
        "cube": 9,
        "exp": 9,
    },
    progress=True,
    # ^ Limit the complexity within each argument.
    # "inv": (-1, 9) states that the numerator has no constraint,
    # but the denominator has a max complexity of 9.
    # "exp": 9 simply states that `exp` can only have
    # an expression of complexity 9 as input.
    nested_constraints={
        "square": {"square": 1, "cube": 1, "exp": 0},
        "cube": {"square": 1, "cube": 1, "exp": 0},
        "exp": {"square": 1, "cube": 1, "exp": 0},
    }
    # ^ Define operator for SymPy as well
)

X = data_smoothed

model.fit(X, dydt_hat)

p_x = model.predict(X)

print('Model Y',model.get_best())
best = model.get_best()
with open(f"models/{file_name[:-4]}_PYSR_y.pkl", "wb") as f:
    pickle.dump(model, f)
