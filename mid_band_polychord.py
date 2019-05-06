#!/usr/bin/python

'''

how to run:

$ python mid_band_polychord.py 0 5 test


'''


import sys
import numpy as np
import scipy as sp
import data_models as dm

import PyPolyChord
from PyPolyChord.settings import PolyChordSettings


N21 = int(sys.argv[1])
Nfg = int(sys.argv[2])
save_file_name = sys.argv[3]





# Constants
# -----------------------
Nparameters = N21+Nfg
Nderived    = 0



# Input parameters
# -----------------------
model_type_signal     = 'tanh'  #'tanh' #, 'tanh'
model_type_foreground = 'exp'  #'exp', 'linlog'

save_folder = '/home/raul/Desktop/test/'

data        = 'real'   # it could be 'real' or 'simulated'
case        =  1
FLOW        =  60
FHIGH       = 120 #159
v0          = 100




# Data
# -------------------------------------------------
# Choose to work either with simulated or real data
if data == 'simulated':
	v  = np.arange(61, 159, 0.39)
		
	#t, sigma, inv_sigma, det_sigma = dm.simulated_data([-0.5, 78, 19, 7, 1000, -2.5, -0.1, 1, 1], v, v0, 0.02, model_type_signal='exp', model_type_foreground='exp', N21par=4, NFGpar=5)
	t, sigma, inv_sigma, det_sigma = dm.simulated_data([1000, -2.5, -0.1, 1, 1], v, v0, 0.01, model_type_signal='exp', model_type_foreground='exp', N21par=0, NFGpar=5)


elif data == 'real':
	v, t, w, sigma, inv_sigma, det_sigma = dm.real_data(case, FLOW, FHIGH)
	









def prior_list(N21, Nfg, model_type_signal, model_type_foreground):
	

	pl      = np.zeros((Nparameters, 2))
	pl[:,0] = -1e4
	pl[:,1] = +1e4
	
	
	# Signal model
	if ((model_type_signal == 'exp') or (model_type_signal == 'tanh')) and (N21>=4):
		
		# Amplitude
		pl[0, 0] = -5
		pl[0, 1] =  0 #5

		# Center
		pl[1, 0] = 61
		pl[1, 1] = 100		
	
		# Width
		pl[2, 0] =  2
		pl[2, 1] = 40
		
		# Tau
		pl[3, 0] =  0.01
		pl[3, 1] =  20
		
		if (model_type_signal == 'exp') and (N21 == 5):
			
			# Tau
			pl[4, 0] =  -10
			pl[4, 1] =   10			
		
		
		if (model_type_signal == 'tanh') and (N21 == 5):
			
			# Tau
			pl[4, 0] =  0.01
			pl[4, 1] =  20
			
			#print('hola !!')
		
		
		
	# Foreground model
	if model_type_foreground == 'exp':
	
		# Temperature at reference frequency
		pl[N21,   0] =   100   # lower limit of first parameter, temperature at 100 MHz
		pl[N21,   1] = 10000   # upper limit of first parameter, temperature at 100 MHz
		
		# Spectral index
		pl[N21+1, 0] =  -2.0
		pl[N21+1, 1] =  -3.0
		

		
	elif model_type_foreground == 'linlog':
	
		# Temperature at reference frequency
		pl[N21,   0] =   100   # lower limit of first parameter, temperature at 100 MHz
		pl[N21,   1] = 10000   # upper limit of first parameter, temperature at 100 MHz
		

	return pl









def loglikelihood(theta):

	N = len(v)

	# Evaluating model
	m = dm.full_model(theta, v, v0, model_type_signal=model_type_signal, model_type_foreground=model_type_foreground, N21par=N21, NFGpar=Nfg)


	# Log-likelihood
	DELTA = t-m
	lnL2  = -(1/2)*np.dot(np.dot(DELTA, inv_sigma), DELTA)      -(N/2)*np.log(2*np.pi)       # -(1/2)*np.log(det_sigma)
	#lnL2 =  #-(1/2)*np.log(det_sigma)


	# This solves numerical errors
	if np.isnan(lnL2) == True:
		print('True')
		lnL2 = -np.infty


	#print(lnL)
	print(lnL2)
	#print('hola')

	return lnL2, 0











def prior(cube):

	"""

	A function defining the transform between the parameterisation in the unit hypercube to the true parameters.

	Args: cube (array, list): a list containing the parameters as drawn from a unit hypercube.

	Returns:
	list: the transformed parameters.

	"""


	theta = np.zeros(len(cube))

	pl = prior_list(N21, Nfg, model_type_signal, model_type_foreground) 

	for i in range(len(cube)):
		theta[i] = cube[i] * (pl[i,1] - pl[i,0]) + pl[i,0] 

	return theta










def dumper(live, dead, logweights, logZ, logZerr):
	print(dead[-1])







def run():

	# in python, or ipython >>  


	settings               = PolyChordSettings(Nparameters, Nderived)
	settings.base_dir      = save_folder
	settings.file_root     = save_file_name 
	settings.do_clustering = True
	settings.read_resume   = False
	output                 = PyPolyChord.run_polychord(loglikelihood, Nparameters, Nderived, settings, prior, dumper)

	return 0




run()







