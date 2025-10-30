import warnings
import gc
from IPython.display import display
import os
import sys
import glob

args = glob.glob("datasets/*.csv")

print(args)

for arg in args:
    os.system(f'python3 model_data.py -f {arg}')