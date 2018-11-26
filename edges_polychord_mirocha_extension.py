
import ares
import numpy as np
import matplotlib.pyplot as pl


#pars = ares.util.ParameterBundle('mirocha2016:base')
pars = ares.util.ParameterBundle('mirocha2016:dpl') 


# To speed things up
# Note that we can only get away with this in cases where we don't want to allow the HMF or stellar models to change in the fit!
hmf = ares.physics.HaloMassFunction()
src = ares.sources.SynthesisModel(source_Z=0.024, source_sed='eldridge2009')










def model_mirocha2016(theta):

	"""

	Usage:

	z, t = model_mirocha2016([np.log10(1e-2), np.log10(1e4), np.log10(1e-2), np.log10(1e40), 20, -0.5])

	"""





	# Assign new values to list
	theta_list = \
	        {
	                'pop_Z{0}':                 theta[0], # log10(1e-3, 0.04),
	                'pop_Tmin{0}':              theta[1], # log10(500, 5e5),
	                'pop_fesc{0}':              theta[2], # log10(1e-3, 0.5),
	                'pop_rad_yield{1}':         theta[3], # log10(1e38, 1e42),
	                'pop_logN{1}':              theta[4], # (17, 23),
	                'pop_rad_yield_Z_index{1}': theta[5], # (-1, 0),
	        }

	# List that defines if parameter is evaluated in log scale or not
	is_log = \
	        {
	                'pop_Z{0}':                 True,
	                'pop_Tmin{0}':              True,
	                'pop_fesc{0}':              True,
	                'pop_rad_yield{1}':         True,
	                'pop_logN{1}':              False,
	                'pop_rad_yield_Z_index{1}': False,
	        }





	# New list of parameter values
	updates = {}

	# Go parameter by parameter, and copy new value to list
	for parameter in theta_list.keys():

		# Copy new value in log or linear  scale
		if is_log[parameter]:

			# New value of parameter in log scale
			value_log = theta_list[parameter]			

			# Store parameter in linear scale
			updates[parameter] = 10**(value_log)


		else:

			# New value of parameter in linear scale
			value_linear = theta_list[parameter]			

			# Store parameter in linear scale
			updates[parameter] = value_linear





	# Create new parameter object
	p = pars.copy()


	# Assign star properties (loaded at the top of the file, IS THIS NECESSARY???????, or can we DO IT ONLY AT THE TOP)
	p['hmf_instance']     = hmf
	p['pop_psm_instance'] = src  # 'psm' is for "population synthesis model"


	# Assign input values to parameter object
	p.update(updates)


	# Run simulation
	sim = ares.simulations.Global21cm(**p)
	sim.run()


	# Extracting brightness temperature and redshift
	hist = sim.history
	t    = hist['dTb']
	z    = hist['z']

	return z, t














## This is for doing sweeps
#ranges = \
	#{
		#'pop_Z{0}': (1e-3, 0.04),
		#'pop_Tmin{0}': (500., 5e5),
		#'pop_fesc{0}': (1e-3, 0.5),
		#'pop_rad_yield{1}': (1e38, 1e42),
		#'pop_logN{1}': (17., 23.),
		#'pop_rad_yield_Z_index{1}': (-1., 0.),
	#}


#is_log = \
	#{
		#'pop_Z{0}': True,
		#'pop_Tmin{0}': True,
		#'pop_fesc{0}': True,
		#'pop_rad_yield{1}': True,
		#'pop_logN{1}': False,
		#'pop_rad_yield_Z_index{1}': False,
	#}



## Generate random models in these ranges
#for i in range(10):

	## Generate new parameters randomly in specified range
	#updates = {}
	#for par in ranges.keys():
		#lo, hi = ranges[par]
		#if is_log[par]:
			#lo = np.log10(lo)
			#hi = np.log10(hi)
			#updates[par] = 10**(np.random.random() * (hi - lo) + lo)
		#else:
			#updates[par] = np.random.random() * (hi - lo) + lo


	## Make new parameter dictionary
	#p = pars.copy()
	#p.update(updates)

	## Run & plot result
	#sim = ares.simulations.Global21cm(**p)
	#sim.run()




