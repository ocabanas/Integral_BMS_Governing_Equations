import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")
from copy import deepcopy, copy
from datetime import datetime
import pickle
from sklearn.model_selection import train_test_split
import sys, getopt
import matplotlib.pyplot as plt
import pynumdiff
import time
# Import Machine Scientist
from importlib.machinery import SourceFileLoader

# Get the absolute path of the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
# Define the relative path to the module
relative_module_path = "rguimera-machine-scientist-constrained/machinescientist_ode.py"
path = os.path.join(script_dir, relative_module_path)
ms = SourceFileLoader("ms", path).load_module()


import argparse

parser = argparse.ArgumentParser(description="Generate the train/test split")
parser.add_argument("--mode", action="store_true", help="Enable mode flag")
parser.add_argument("--free", action="store_true", help="Enable mode flag")
parser.add_argument("--prod", action="store_true", help="Enable mode flag")
parser.add_argument("--comb", action="store_true", help="Enable mode flag")

args = parser.parse_args()

if args.mode:
    # Generating train/test with 50% of columns

    folder_name = "Train_test_data_lin_term_com" + datetime.now().strftime(
        "%Y_%m_%d-%I_%M_%S"
    )
    dir = os.path.join(os.getcwd(), folder_name)
    if not os.path.exists(dir):
        os.mkdir(dir)
    x = {}
    y = {}

    # Dataset growth:
    data = pd.read_pickle("microbial_growth_full3.pkl")
    data["t"] = data["t"].shift(3)
    data.drop(index=data.index[:3], axis=0, inplace=True)
    data.reset_index(inplace=True)
    print(data.columns[2:])
    columns = [
        "A3",
        "A6",
        "A8",
        "A9",
        "B4",
        "B5",
        "B6",
        "B9",
        "B11",
        "B12",
        "C2",
        "C5",
        "C7",
        "C11",
        "D2",
        "D5",
        "D6",
        "D8",
        "D12",
        "E3",
        "E5",
        "E12",
        "F3",
        "F7",
        "G7",
        "G8",
        "G9",
        "H3",
        "H12",
    ]
    train_labels, test_labels = train_test_split(
        columns, test_size=0.5, random_state=42
    )
    print(len(columns), len(train_labels), len(test_labels))
    with open(f"./{folder_name}/train_test_0.pkl", "wb") as file:
        # A new file will be created
        pickle.dump({"train": train_labels, "test": test_labels}, file)
    for col in train_labels:
        No = data[col].to_numpy()[0]
        y[col] = pd.Series(deepcopy(data[col]))
        par = [2, 21, 21]
        h = data.t.to_numpy()[1] - data.t.to_numpy()[0]
        x_hat, dxdt_hat = pynumdiff.linear_model.polydiff(
            data[col].to_numpy(), h, par, options=None
        )
        x[col] = deepcopy(data[["t", col]].rename(columns={col: "B"}))
        x[col]["dx"] = dxdt_hat

    # Dataset growth 6:
    data = pd.read_csv("microbial_growth_full6.csv")
    csv_cols = data.columns.to_numpy()
    data = data.rename(columns={csv_cols[0]: "t"})
    data["t"] = data["t"].shift(4)
    data.drop(index=data.index[:4], axis=0, inplace=True)
    data.reset_index(inplace=True)
    columns = [
        "R.S.6 D-Glucose, 0.3%",
        "R.S.6 Glycerol, 0.3%",
        "R.S.6 Myo-Inositol, 0.3%",
        "R.S.6 D-Galactose, 0.3%",
        "R.S.6 Malate, 0.3%",
        "R.S.6 Alpha-D-Glucose, 0.3%",
        "R.S.6 D-Fructose, 0.3%",
        "R.S.6 Sodium Pyruvate, 0.3%",
        "R.S.6 Lactate, 0.3%",
        "R.S.6 Citrate, 0.3%",
        "R.S.6 D-Mannose, 0.3%",
        "R.S.6 D-Salicin, 0.3%",
        "R.S.6 D-Cellubiose, 0.3%",
        "R.S.6 Adonitol, 0.3%",
        "R.S.6 D-Glucose-6-phosphate, 0.3%",
        "R.S.6 D-Arabitol, 0.3%",
        "R.S.6 D-Trehalose, 0.3%",
        "R.S.6 L-Alanine, 0.3%",
        "R.S.6 L-Arabinose, 0.3%",
        "R.S.6 4 Hydroxy-Phenylacetate, 0.3%",
        "R.S.6 D-Ribose, 0.3%",
        "R.S.6 Alpha-D-Melebiose, 0.3%",
        "R.S.6 Inosine, 0.3%",
        "R.S.6 Thymidine, 0.3%",
        "R.S.6 Sucrose, 0.3%",
        "R.S.6 D-Xylose, 0.3%",
        "R.S.6 Alpha-D-Lactose, 0.3%",
        "R.S.6 D-Raffinose, 0.3%",
        "R.S.6 Adenosine, 0.2%",
        "R.S.6 L-Rhamnose, 0.3%",
        "R.S.6 D-Glucosamine, 0.3%",
        "R.S.6 D-Glutamate, 0.2%",
        "R.S.6 Cytidine, 0.3%",
        "R.S.6 Adenosine, 0.2%.1",
        "R.S.6 L-Arginine, 0.2%",
        "R.S.6 L-Histidine, 0.3%",
        "R.S.6 L-Glutathione, 0.2%",
        "R.S.6 Allantoin, 0.2%",
        "R.S.6 Adenine, 0.2%",
        "R.S.6 N-Acetyl-D-Glucosamine, 0.2%",
    ]
    train_labels, test_labels = train_test_split(
        columns, test_size=0.5, random_state=42
    )
    print(len(columns), len(train_labels), len(test_labels))
    with open(f"./{folder_name}/train_test_6.pkl", "wb") as file:
        # A new file will be created
        pickle.dump({"train": train_labels, "test": test_labels}, file)
    for col in train_labels:
        No = data[col].to_numpy()[0]
        y[col + "_6"] = pd.Series(deepcopy(data[col]))
        par = [2, 21, 21]
        h = data.t.to_numpy()[1] - data.t.to_numpy()[0]
        x_hat, dxdt_hat = pynumdiff.linear_model.polydiff(
            data[col].to_numpy(), h, par, options=None
        )
        x[col + "_6"] = deepcopy(data[["t", col]].rename(columns={col: "B"}))
        x[col + "_6"]["dx"] = dxdt_hat

    # Dataset growth 10:
    data = pd.read_csv("microbial_growth_full10.csv")
    csv_cols = data.columns.to_numpy()
    data = data.rename(columns={csv_cols[0]: "t"})
    data["t"] = data["t"].shift(4)

    data.drop(index=data.index[:4], axis=0, inplace=True)

    def convert_to_hours(time_str):
        parts = time_str.split(":")

        # If format is hh:mm:ss
        if len(parts) == 3:
            hours, minutes, seconds = map(int, parts)
        # If format is mm:ss (assume hours = 0)
        elif len(parts) == 2:
            hours = 0
            minutes, seconds = map(int, parts)
        else:
            return np.nan  # Handle unexpected cases
            print(parts)
        return hours + minutes / 60.0 + seconds / 3600.0  # Convert to hours

    data["t"] = data["t"].apply(lambda x: convert_to_hours(x))
    data.reset_index(inplace=True)
    columns = [
        "D Glucose",
        "Glycerol",
        "D Serine",
        "D Galactose",
        "Alpha D Glucose",
        "D Fructose",
        "Lactate",
        "D Mannose",
        "D Glucose 6 PO4",
        "D- Trehalose",
        "L Arabinose",
        "D Ribose",
        "Alpha D Melebiose",
        "Inosine",
        "D Xylose",
        "Alpha D Lactose",
        "Adenosine",
        "D Glucosamine",
        "Cytidine",
        "Adenosine.1",
        "L Arginine",
        "L Glutathione",
        "Adenine",
        "N- acetyl D Glucosamine",
    ]
    train_labels, test_labels = train_test_split(
        columns, test_size=0.5, random_state=42
    )
    print(len(columns), len(train_labels), len(test_labels))
    with open(f"./{folder_name}/train_test_10.pkl", "wb") as file:
        # A new file will be created
        pickle.dump({"train": train_labels, "test": test_labels}, file)
    for col in train_labels:
        No = data[col].to_numpy()[0]
        y[col + "_10"] = pd.Series(deepcopy(data[col]))
        par = [2, 21, 21]
        h = data.t.to_numpy()[1] - data.t.to_numpy()[0]
        x_hat, dxdt_hat = pynumdiff.linear_model.polydiff(
            data[col].to_numpy(), h, par, options=None
        )
        x[col + "_10"] = deepcopy(data[["t", col]].rename(columns={col: "B"}))
        x[col + "_10"]["dx"] = dxdt_hat

    # Dataset growth 12:
    data = pd.read_csv("microbial_growth_full12.csv")
    csv_cols = data.columns.to_numpy()
    data = data.rename(columns={csv_cols[0]: "t"})
    data["t"] = data["t"].shift(4)
    data.drop(index=data.index[:4], axis=0, inplace=True)
    data.reset_index(inplace=True)
    columns = [
        "R.S.12 Lactulose, 0.3%",
        "R.S.12 Glycerol, 0.3%",
        "R.S.12 Myo-Inositol, 0.3%",
        "R.S.12 D-Galactose, 0.3%",
        "R.S.12 Dulcitol, 0.3%",
        "R.S.12 Alpha-D-Glucose, 0.3%",
        "R.S.12 D-Fructose, 0.3%",
        "R.S.12 Citrate, 0.3%",
        "R.S.12 D-Mannose, 0.3%",
        "R.S.12 D-Salicin, 0.3%",
        "R.S.12 D-Cellubiose, 0.3%",
        "R.S.12 Adonitol, 0.3%",
        "R.S.12 D-Glucose-6-phosphate, 0.3%",
        "R.S.12 D-Arabitol, 0.3%",
        "R.S.12 D-Trehalose, 0.3%",
        "R.S.12 L-Arabinose, 0.3%",
        "R.S.12 D-Ribose, 0.3%",
        "R.S.12 Alpha-D-Melebiose, 0.3%",
        "R.S.12 L-Fucose, 0.3%",
        "R.S.12 Inosine, 0.3%",
        "R.S.12 Sucrose, 0.3%",
        "R.S.12 D-Xylose, 0.3%",
        "R.S.12 Alpha-D-Lactose, 0.3%",
        "R.S.12 D-Raffinose, 0.3%",
        "R.S.12 L-Glutamine, 0.3%",
        "R.S.12 L-Rhamnose, 0.3%",
        "R.S.12 D-Glucosamine, 0.3%",
        "R.S.12 D-Glutamate, 0.2%",
        "R.S.12 Cytidine, 0.3%",
        "R.S.12 Adenosine, 0.2%.1",
        "R.S.12 L-Arginine, 0.2%",
        "R.S.12 Guanidine, 0.3%",
        "R.S.12 L-Glutathione, 0.2%",
        "R.S.12 Allantoin, 0.2%",
        "R.S.12 Cytosine, 0.2%",
    ]
    train_labels, test_labels = [], columns
    print(len(columns), len(train_labels), len(test_labels))
    with open(f"./{folder_name}/train_test_12.pkl", "wb") as file:
        # A new file will be created
        pickle.dump({"train": [], "test": columns}, file)

    for col in train_labels:
        No = data[col].to_numpy()[0]
        y[col + "_12"] = pd.Series(deepcopy(data[col]))
        par = [2, 21, 21]
        h = data.t.to_numpy()[1] - data.t.to_numpy()[0]
        x_hat, dxdt_hat = pynumdiff.linear_model.polydiff(
            data[col].to_numpy(), h, par, options=None
        )
        x[col + "_12"] = deepcopy(data[["t", col]].rename(columns={col: "B"}))
        x[col + "_12"]["dx"] = dxdt_hat

    # Dataset growth 18:
    data = pd.read_csv("microbial_growth_full18.csv", header=1)

    csv_cols = data.columns.to_numpy()
    data = data.rename(columns={csv_cols[0]: "t"})
    data["t"] = data["t"].shift(4)
    data.drop(index=data.index[:4], axis=0, inplace=True)
    data.reset_index(inplace=True)
    columns = [
        "R.S.18 D-Glucose",
        "R.S.18 Glycerol",
        "R.S.18 D-Serine",
        "R.S.18 Sodium Succinate",
        "R.S.18 Alpha-D-Glucose",
        "R.S.18 L-Aspartate",
        "R.S.18 D-Fructose",
        "R.S.18 Sodium Pyruvate",
        "R.S.18 Citrate",
        "R.S.18 D-Mannose",
        "R.S.18 D-Salicin",
        "R.S.18 Inosine",
        "R.S.18 Thymidine",
        "R.S.18 Sucrose",
        "R.S.18 Adenosine",
        "R.S.18 Nitrogen Neg. Control",
        "R.S.18 Histamine",
        "R.S.18 L-Pyro-Glutamate.1",
        "R.S.18 Cytidine",
        "R.S.18 Adenosine.1",
        "R.S.18 L-Arginine",
        "R.S.18 Thiourea",
        "R.S.18 Biuret",
        "R.S.18 Guanidine",
        "R.S.18 L-Histidine",
        "R.S.18 Thymine",
        "R.S.18 L-Glutathione",
        "R.S.18 Cytosine",
    ]
    train_labels, test_labels = train_test_split(
        columns, test_size=0.5, random_state=42
    )
    print(len(columns), len(train_labels), len(test_labels))
    with open(f"./{folder_name}/train_test_18.pkl", "wb") as file:
        # A new file will be created
        pickle.dump({"train": train_labels, "test": test_labels}, file)
    for col in train_labels:
        No = data[col].to_numpy()[0]
        y[col + "_18"] = pd.Series(deepcopy(data[col]))
        par = [2, 21, 21]
        h = data.t.to_numpy()[1] - data.t.to_numpy()[0]
        x_hat, dxdt_hat = pynumdiff.linear_model.polydiff(
            data[col].to_numpy(), h, par, options=None
        )
        x[col + "_18"] = deepcopy(data[["t", col]].rename(columns={col: "B"}))
        x[col + "_18"]["dx"] = dxdt_hat

    # Dataset growth 19:
    data = pd.read_csv("microbial_growth_full19.csv")
    csv_cols = data.columns.to_numpy()
    data = data.rename(columns={csv_cols[0]: "t"})
    data["t"] = data["t"].shift(4)
    data.drop(index=data.index[:4], axis=0, inplace=True)
    data.reset_index(inplace=True)
    columns = [
        "R.S.19 D-Glucose",
        "R.S.19 Alpha-D-Glucose",
        "R.S.19 Nitrogen Neg. Control",
        "R.S.19 Histamine",
        "R.S.19 L-Pyro-Glutamate.1",
        "R.S.19 Cytidine",
        "R.S.19 L-Arginine",
        "R.S.19 Thiourea",
        "R.S.19 Guanidine",
        "R.S.19 Thymine",
        "R.S.19 L-Glutathione",
        "R.S.19 Allantoin",
        "R.S.19 Glycine.1",
    ]
    train_labels, test_labels = train_test_split(
        columns, test_size=0.5, random_state=42
    )
    print(len(columns), len(train_labels), len(test_labels))
    with open(f"./{folder_name}/train_test_19.pkl", "wb") as file:
        # A new file will be created
        pickle.dump({"train": train_labels, "test": test_labels}, file)
    for col in train_labels:
        No = data[col].to_numpy()[0]
        y[col + "_19"] = pd.Series(deepcopy(data[col]))
        par = [2, 21, 21]
        h = data.t.to_numpy()[1] - data.t.to_numpy()[0]
        x_hat, dxdt_hat = pynumdiff.linear_model.polydiff(
            data[col].to_numpy(), h, par, options=None
        )
        x[col + "_19"] = deepcopy(data[["t", col]].rename(columns={col: "B"}))
        x[col + "_19"]["dx"] = dxdt_hat

    with open(f"./{folder_name}/x.pkl", "wb") as file:
        # A new file will be created
        pickle.dump(x, file)

    with open(f"./{folder_name}/y.pkl", "wb") as file:
        # A new file will be created
        pickle.dump(y, file)
    exit()

if args.comb:
	#Sym A:
	# Physical contraint: lienar term combination
	folder_name='Full_data_lin_term_com_'+datetime.now().strftime("%Y_%m_%d-%I_%M_%S")
	dir=os.path.join(os.getcwd(),folder_name)
	if not os.path.exists(dir):
	    os.mkdir(dir)
	constraint=('((_a0_ * B) + ',')')

if args.prod:
	#Sym B:
	# Physical contraint: product term combination
	folder_name='Full_data_lin_term_prod_'+datetime.now().strftime("%Y_%m_%d-%I_%M_%S")
	dir=os.path.join(os.getcwd(),folder_name)
	if not os.path.exists(dir):
	    os.mkdir(dir)
	constraint=('((_a0_ * B) * ',')')
	
if args.free:
    # Sym C:
    # Physical contraint: No constraint
    folder_name = "Full_data_free_model" + datetime.now().strftime("%Y_%m_%d-%I_%M_%S")
    dir = os.path.join(os.getcwd(), folder_name)
    if not os.path.exists(dir):
        os.mkdir(dir)
    constraint = None

with open(f"./Train_test_data_lin_term_com2025_03_11-11_21_44/x.pkl", "rb") as file:
    # A new file will be created
    x = pickle.load(file)

with open(f"./Train_test_data_lin_term_com2025_03_11-11_21_44/y.pkl", "rb") as file:
    # A new file will be created
    y = pickle.load(file)

file1 = open(f"./{folder_name}/res.txt", "a")

mcmc_resets = 1
mcmc_steps = 5000
XLABS = ["B"]
params = 8

best_model, dls = ms.machinescientist_to_folder(
    x,
    y,
    XLABS=XLABS,
    n_params=params,
    resets=mcmc_resets,
    steps_prod=mcmc_steps,
    folder=folder_name,
    constraint=constraint,
)


with open(f"./{folder_name}/model_mdl.pkl", "wb") as file:
    # A new file will be created
    pickle.dump(best_model, file)


plt.plot(dls)
plt.savefig(f"./{folder_name}/model_mdl_dl.pdf", format="pdf")
plt.clf()
print()
print(file, best_model, best_model.E)
print(best_model.par_values)
print(best_model.x0)
print(len(best_model.x0), len(best_model.fit_par))


# fig_dl.savefig(f'./{folder_name}/1description_length_B.pdf',format='pdf')
file1.write(f"Best model: {best_model}\n")
file1.write(f"DL: {best_model.E}\n")
file1.write(f"Latex: {best_model.latex()}\n")
file1.write(f"Parameters: {best_model.par_values}\n")
file1.write("###################### \n")
