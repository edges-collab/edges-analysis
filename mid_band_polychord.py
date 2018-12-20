#!/usr/bin/python

'''

how to run:

$ python mid_band_polychord 0 4 test


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
v  = np.arange(61, 159, 0.39)
v0 = 100

Nparameters = N21+Nfg
Nderived    = 0

model_type_signal     = 'exp'
model_type_foreground = 'exp'

data = 'simulated'   # it could be 'real' or 'simulated'

save_folder = '/home/raul/Desktop/'









def prior_list(N21, Nfg, model_type_signal, model_type_foreground):
	

	pl      = np.zeros((Npar, 2))
	pl[:,0] = -1e4
	pl[:,1] = +1e4
	
	
	# Signal model
	if (model_type_signal == 'exp') and (N21>=4):
		
		# Amplitude
		pl[0, 0] =  -5
		pl[0, 1] =   5

		# Center
		pl[1, 0] =  61
		pl[1, 1] = 159		
	
		# Width
		pl[2, 0] =   2
		pl[2, 1] =  60
		
		# Tau
		pl[3, 0] =   0.01
		pl[3, 1] =  30
		
		
		
	# Foreground model
	if model_type_foreground == 'exp':
	
		# Temperature at reference frequency
		pl[N21,   0] =   100   # lower limit of first parameter, temperature at 100 MHz
		pl[N21,   1] = 10000   # upper limit of first parameter, temperature at 100 MHz
		
		# Spectral index
		pl[N21+1, 0] =  -2.0
		pl[N21+1, 1] =  -3.0
		

	return pl









def loglikelihood(theta):

	N = len(v)

	# Evaluating model
	m = dm.full_model(theta,  model_type_signal='exp', model_type_foreground='exp', N21par=4, NFGpar=5)


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

	pl = prior_list() 

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



















# Choose to work either with simulated or real data
if data == 'simulated':	
	t, sigma, inv_sigma, det_sigma = simulated_data(4, [-0.5, 78, 19, 7, 1000, -2.5, 0.1, -4, -0.05], 0.02)  # power law

elif data == 'real':
	t, sigma, inv_sigma, det_sigma = real_data() 
	
	


run()












