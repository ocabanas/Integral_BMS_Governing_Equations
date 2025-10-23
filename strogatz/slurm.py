#!/bin/python3

import subprocess
import sys
import numpy as np
import os
import glob

BASE_PATH = '/export/home/shared/Projects/IntegralBMS/Integral_BMS_Governing_Equations/strogatz/'
NODES_PER_TASK = 1
PROC_PER_TASK = 1
USER_MAIL = 'oriol.cabanas@urv.cat'
JOB_NAME = 'strogatz'
OUTPUT_PATH = BASE_PATH + 'logs_python.txt'
COMMAND_PATH = BASE_PATH + '../venv/bin/python3'
SCRIPT_PATH = BASE_PATH + 'MCMC_sampling.py'

# Genera una lista de strings que contiene los argumentos para el proceso.

def generate_arguments():
	args = glob.glob("datasets/*.csv")
	return args

# Construye el srun.
def build_command(arg):
	base_command = (
	f'srun --oversubscribe --cpus-per-task=1 --mem=2G '
	f'--mail-user {USER_MAIL} -J {JOB_NAME} '
	f'--mail-type=ALL --error={OUTPUT_PATH} --output={OUTPUT_PATH} '
	)
	script_command = f'{COMMAND_PATH} {SCRIPT_PATH} -f {arg}'
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
			if not os.path.isfile(f'models/{file[:-4]}_IBMS.pkl'):
				print(arg,file[:-4])
				command = build_command(arg)
				print(command)
				process = subprocess.Popen(command, shell=True)
				process.wait()  # wait for this job to finish before next

if __name__ == "__main__":
	main()