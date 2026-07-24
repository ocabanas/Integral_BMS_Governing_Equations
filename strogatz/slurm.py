#!/bin/python3

import subprocess
import sys
import numpy as np
import os
import glob
from pathlib import Path

BASE_PATH = '/export/home/shared/Projects/IntegralBMS/Integral_BMS_Governing_Equations/strogatz/'
NODES_PER_TASK = 1
PROC_PER_TASK = 5
USER_MAIL = 'oriol.cabanas@urv.cat'
JOB_NAME = 'IBMS' # strog
OUTPUT_PATH = BASE_PATH + 'logs_IBMS_shflow.txt'
COMMAND_PATH = BASE_PATH + '../venv/bin/python3'
SCRIPT_PATH = BASE_PATH + 'MCMC_sampling.py'

def generate_arguments():
    #args = glob.glob("noise_data/*.csv")
    folder = Path("datasets")
    files = [f.name for f in folder.iterdir() if f.is_file() and "20_2" in f.name]
    print(files)
    files = ['pred_prey_SNR_20_0.csv']
    return files

def build_command(arg):
    base_command = (
            f'srun --ntasks=1 --cpus-per-task={PROC_PER_TASK} --mem=10G --time=60-00:00:00 '
    f'--mail-user {USER_MAIL} -J {JOB_NAME} '
    f'--mail-type=ALL --error={OUTPUT_PATH} --output={OUTPUT_PATH} '
    )
    script_command = f'{COMMAND_PATH} {SCRIPT_PATH} -f datasets/{arg}'
    return base_command + script_command

def main():
	args = generate_arguments()
	#args=[]
	if len(args) == 0:
		print("no arguments passed")
		command = build_command('')
		process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
		#output, error = process.communicate()
	else:
		for arg in args:
			chunk,file=os.path.split(arg)
			print(arg,file[:-4])
			command = build_command(arg)
			print(command)
			proceses = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
if __name__ == "__main__":
	main()
