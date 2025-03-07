import warnings
import gc
from IPython.display import display
import os
import sys

sigmas=[0.1,0.5,1.,1.5,2.,2.5,3.,3.5,4.,4.5,5.,5.5,6.]

for i in range(0,40):#[34,38,39]:
    for sigma in sigmas:
        os.system(f'python3 ms_LV_v1_fit.py -f {sigma}_{i}.csv -s True')
        os.system(f'python3 ms_LV_v1_fit.py -f {sigma}_{i}.csv -s False')
        os.system(f'python3 ms_LV_v1_ode.py -f {sigma}_{i}.csv')
        os.system(f'python3 exhaustive_linear_MS.py -f {sigma}_{i}.csv')
