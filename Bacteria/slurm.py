#!/bin/python3

import subprocess
import sys
import numpy as np
import os

BASE_PATH = '/export/home/shared/Projects/IntegralBMS/Integral_BMS_Governing_Equations/Bacteria/'
NODES_PER_TASK = 1
PROC_PER_TASK = 1
USER_MAIL = 'oriol.cabanas@urv.cat'
JOB_NAME = 'ori'
OUTPUT_PATH = BASE_PATH + 'logs_python.txt'
COMMAND_PATH = BASE_PATH + 'venv/bin/python3'
#SCRIPT_PATH = BASE_PATH + 'ms_bacteris_v2_extended_datasets.py'
#SCRIPT_PATH = BASE_PATH + 'Extended_data_Refit_Params_Test.py'
SCRIPT_PATH = BASE_PATH + 'Extended_data_Refit_Params_Train.py'

# Genera una lista de strings que contiene los argumentos para el proceso.

def generate_arguments():
	
	"""
    sigmas=[0.1,0.5,1.,1.5,2.,2.5,3.,3.5,4.,4.5,5.,5.5,6.]
	args=[]
	#sigmas=[0.03]
	for i in range(0,10):#[34,38,39]:
		for sigma in sigmas:
			args.append(f'{BASE_PATH}MCMC-LV-PROGRAM-DATA/{sigma}_{i}.csv')
	"""
	args=[]
	for i in range(20,40):
		for n in range(800,1001,100):
			args.append(f'{BASE_PATH}MCMC-LV-PROGRAM-DATA/{n}_{i}.csv')
            
	#args=[f'{BASE_PATH}MCMC-LV-PROGRAM-DATA/0.1_9.csv']
	return args

# Construye el srun.
def build_command(arg):
	base_command = f'srun --oversubscribe --cpus-per-task=1 --mem=2G --mail-user {USER_MAIL} -J {JOB_NAME} --mail-type=ALL --error={OUTPUT_PATH} --output={OUTPUT_PATH} '
        #base_command = f"srun --oversubscribe --ntasks={NODES_PER_TASK} --cpus-per-task=1 --mem=3G --mail-user {USER_MAIL} -J {JOB_NAME}{arg[0]}_{arg[1]} --mail-type=ALL --error={OUTPUT_PATH} --output={OUTPUT_PATH} "
	script_command = f'{COMMAND_PATH} {SCRIPT_PATH} --prod'
	return base_command + script_command

def main():
	#args = generate_arguments()
	args=[]
	if len(args) == 0:
		print("no arguments passed")
		command = build_command('')
		process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
		#output, error = process.communicate()
	else:
		for arg in args:
			chunk,file=os.path.split(arg)
			if not os.path.isfile(f'./exh_ode_dl_list{file[:-4]}.pkl'):
				print(arg,file[:-4])
				command = build_command(arg)
				print(command)
				process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
				#output, error = process.communicate()

if __name__ == "__main__":
	main()
