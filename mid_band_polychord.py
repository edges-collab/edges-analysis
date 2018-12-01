#!/usr/bin/python

import sys
import numpy as np
import scipy as sp
import PyPolyChord
from PyPolyChord.settings import PolyChordSettings

#import matplotlib.pyplot as plt




N21 = int(sys.argv[1])
Nfg = int(sys.argv[2])



# Constants
# -----------------------
v0          = 100
Nparameters = N21+Nfg
Nderived    = 0

model_21 = 'flattened_gaussian'
model_fg = 'LINLOG'

simulated_noise_at_v0_std = 0.03 # K

# Prior limits
pl = np.zeros((Nparameters, 2))
pl[:,0] = pl[:,0] + -1e4  # lower limit
pl[:,1] = pl[:,1] +  1e4  # upper limit

pl[0,0] = 950   # lower limit of first parameter, temperature at 100 MHz
pl[0,1] = 1050  # upper limit of first parameter, temperature at 100 MHz  

print(pl)


v = np.arange(50,101)


save_folder = '/home/raul/Desktop/'
save_file_name = 'test'
















def full_model(theta):

	# 21-cm model
	model_21 = 0
	if N21 == 4:
		model_21 = model_eor_flattened_gaussian(model_type=1, T21=theta[0], vr=theta[1], dv=theta[2], tau0=theta[3], tilt=0)

	elif N21 == 5:
		model_21 = model_eor_flattened_gaussian(model_type=1, T21=theta[0], vr=theta[1], dv=theta[2], tau0=theta[3], tilt=theta[4])
	
	
	# Foreground model
	model_fg = 0
	for i in range(Nfg):
		j = N21 + i
		#print(i)
		model_fg = model_fg      +     theta[j] * ((v/v0)**(-2.5)) * ((np.log(v/v0))**i)

	model = model_21 + model_fg

	return model





def simulated_data(theta):

	#v          = np.arange(50, 101, 1)
	#T75        = 1500
	#beta       = -2.5
	#std_dev_vec  = 2*np.ones(len(v))   # frequency dependent or independent
	#std_dev = 1
	#noise   = np.random.normal(0, std_dev, size=len(v))
	
	
	std_dev_vec   = simulated_noise_at_v0_std*(v/v0)**(-2.5)
	#std_dev_vec   = 0.05*np.ones(len(v))
	
	
	sigma         = np.diag(std_dev_vec**2)     # uncertainty covariance matrix
	inv_sigma     = np.linalg.inv(sigma)
	det_sigma     = np.linalg.det(sigma)
	
	noise         = np.random.multivariate_normal(np.zeros(len(v)), sigma)
	
	d_no_noise    = full_model(theta)
	d             = d_no_noise + noise
	
	N             = len(v)


	return d, sigma, inv_sigma, det_sigma





def loglikelihood(theta):

	N = len(v)

	# Evaluating model
	m = full_model(theta)


	# Log-likelihood
	DELTA = t-m
	lnL2  = -(1/2)*np.dot(np.dot(DELTA, inv_sigma), DELTA)      -(N/2)*np.log(2*np.pi)      # -(1/2)*np.log(det_sigma)
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



















#
t, sigma, inv_sigma, det_sigma = simulated_data([1000, 1, 1, -1, 4])  # power law
#plt.plot(v, t)
#plt.show()


run()



#print(logL)












