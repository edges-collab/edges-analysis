


import sys
import numpy as np
import scipy as sp
import scipy.interpolate as spi
import matplotlib.pyplot as plt
import mpi4py

import emcee
import matlab.engine
from emcee.utils import MPIPool



# Run this program from the terminal as follows:
# 
# $ mpirun -np 12 python MPI_test.py
#
# It requires an MPI implementation. Best option is mpich. Install as: $ sudo apt-get install mpich



# Starting MATLAB engine
matlab_engine = matlab.engine.start_matlab()




# Output file name
filename = 'test_marginalized_fg_1.txt'


# Average 21cm fit parameters 
P1 = np.log10(0.001) + (np.log10(0.5)-np.log10(0.001))/2         # s.f.e
P2 = np.log10(16.5) + (np.log10(76.5)-np.log10(16.5))/2          # V_c  
P3 = np.log10(0.00001) + (np.log10(10)-np.log10(0.00001))/2      # f_X
P4 = 0.055 + (0.1-0.055)/2                                       # tau   
P5 = 1 + (1.5-1)/2                                               # alpha  
P6 = 0.1 + (3-0.1)/2                                             # nu_min       
P7 = 10 + (50-10)/2                                              # R_mfp



P40 = 0.07 #0.058 # 0.074
P50 = 1.3 #1.0   # 1.5
P60 = 2 ##0.1   # 3
P70 = 30 #10    # 50 
parameters_21cm = [-1, -1, -1, P40, P50, P60, P70]   # Parameters with -1 are fit parameters. Other parameters take the Px fixed value.


# MCMC chain parameters
MCMC_Nchain            = 40000 #0
MCMC_rejected_fraction = 0.3  # not used right now

# Dimension of accepted sampes is (MCMC_rejected_fraction*MCMC_Nchain*Nwalkers), (Ndim)    where Nwalkers is 2*(Nfg + N21) + 2






# One: P1, P3
# Two: P1-P2, P1-P6, P3-P6
# Three: P1-P2-P3












# Loading High-band data
data  = np.genfromtxt('move_along_nothing_to_see_here.txt')
vb    = data[:,0]
db    = data[:,1]
wb    = data[:,2]


# Cutting data within desired range
flow  = 90
fhigh = 190
v     = vb[(vb>=flow) & (vb<=fhigh)]
d     = db[(vb>=flow) & (vb<=fhigh)]
w     = wb[(vb>=flow) & (vb<=fhigh)]


# Normalized frequency
vn       = flow + (fhigh-flow)/2


# Uncertainty StdDev
sigma_noise = 0.035*np.ones(len(v))


# Foreground model
model_fg        = 'EDGES_polynomial'
Nfg             = 5











def fit_polynomial_fourier(model_type, xdata, ydata, nterms, Weights=1, external_model_in_K=0):

	"""

	2017-10-02
	This a cheap, short version of the same function in "edges.py"

	"""


	# Initializing "design" matrix
	AT  = np.zeros((nterms, len(xdata)))

	# Initializing auxiliary output array	
	aux = (0, 0)

	# Evaluating design matrix with foreground basis
	if (model_type == 'EDGES_polynomial') or (model_type == 'EDGES_polynomial_plus_external'):
		for i in range(nterms):
			AT[i,:] = xdata**(-2.5+i)

	# Adding external model to the design matrix
	if model_type == 'EDGES_polynomial_plus_external':
		AT = np.append(AT, external_model_in_K.reshape(1,-1), axis=0)





	# Applying General Least Squares Formalism, and Solving using QR decomposition
	# ----------------------------------------------------------------------------
	# ----------------------------------------------------------------------------
	# See: http://www2.imm.dtu.dk/pubdb/views/edoc_download.php/2804/pdf/imm2804.pdf

	# If no weights are given
	if np.isscalar(Weights):
		W = np.eye(len(xdata))

	# If a vector is given
	elif np.ndim(Weights) == 1:
		W = np.diag(Weights)

	# If a matrix is given
	elif np.ndim(Weights) == 2:
		W = Weights


	# Sqrt of weight matrix
	sqrtW = np.sqrt(W)


	# Transposing matrices so 'frequency' dimension is along columns
	A     = AT.T
	ydata = np.reshape(ydata, (-1,1))


	# A and ydata "tilde"
	WA     = np.dot(sqrtW, A)
	Wydata = np.dot(sqrtW, ydata)


	# Solving system using 'short' QR decomposition (see R. Butt, Num. Anal. Using MATLAB)
	Q1, R1      = sp.linalg.qr(WA, mode='economic') # returns

	param       = sp.linalg.solve(R1, np.dot(Q1.T, Wydata))

	model       = np.dot(A, param)
	error       = ydata - model
	DF          = len(xdata)-len(param)-1
	wMSE        = (1./DF) * np.dot(error.T, np.dot(W, error))  # This is correct because we also have the Weight matrix in the (AT * W * A)^-1.
	wRMS        = np.sqrt( np.dot(error.T, np.dot(W, error)) / np.sum(np.diag(W)))
	#inv_pre_cov = np.linalg.lstsq(np.dot(R1.T, R1), np.eye(nterms))   # using "lstsq" to compute the inverse: inv_pre_cov = (R1.T * R1) ^ -1
	#cov         = MSE * inv_pre_cov[0]
	inv_pre_cov = np.linalg.inv(np.dot(R1.T, R1))
	cov         = wMSE * inv_pre_cov


	# Back to input format
	ydata = ydata.flatten()
	model = model.flatten()
	param = param.flatten()


	return param, model, wRMS, cov, wMSE, aux     # wMSE = reduced chi square









def model_evaluate(model_type, par, xdata):

	"""

	2017-10-02
	This a cheap, short version of the same function in "edges.py"

	"""


	if (model_type == 'EDGES_polynomial'):
		summ = 0
		for i in range(len(par)):
			summ = summ + par[i] * xdata**(-2.5+i)


	else:
		summ = 0


	model = np.copy(summ)
	return model









def fialkov_model_21cm(f_new, values_21cm, interpolation_kind='linear'):

	"""


	2017-10-02
	Function that produces a 21cm model in MATLAB and sends it to Python

	Webpage: https://www.mathworks.com/help/matlab/matlab-engine-for-python.html


	"""


	#print f_new

	# Copying parameter list
	all_lin_values_21cm    = values_21cm[:]


	# Transforming first three parameters from Log10 to Linear
	all_lin_values_21cm[0] = 10**(values_21cm[0])
	all_lin_values_21cm[1] = 10**(values_21cm[1])
	all_lin_values_21cm[2] = 10**(values_21cm[2])


	# Calling MATLAB function
	# matlab_engine = matlab.engine.start_matlab()
	#print all_lin_values_21cm
	#future_model_md, future_Z, future_flags_md = matlab_engine.Global21cm.run(matlab.double(all_lin_values_21cm), nargout=3, async=True)
	#model_md = future_model_md.result()
	#Z        = future_Z.result()
	#flags_md = future_flags_md.result()



	# If one parameter is complex, by error, change it to one that is only the real part
	for i in range(len(all_lin_values_21cm)):
		if isinstance(all_lin_values_21cm[i], complex):
			all_lin_values_21cm[i] = np.real(all_lin_values_21cm[i])




	#print all_lin_values_21cm

	model_md, Z, flags_md = matlab_engine.Global21cm.run(matlab.double(all_lin_values_21cm), nargout=3)



	# Converting MATLAB outputs to numpy arrays
	model_raw = np.array(model_md._data.tolist())
	flags     = np.array(flags_md._data.tolist())


	# Redshift and frequency
	z_raw = np.arange(5,50.1,0.1)
	c     = 299792458	# wikipedia, m/s
	f21   = 1420.40575177e6  # wikipedia,    
	l21   = c / f21          # frequency to wavelength, as emitted
	l     = l21 * (1 + z_raw)
	f_raw = c / (l * 1e6)


	# print f_raw

	# Interpolation
	f_raw_increasing     = f_raw[::-1]
	model_raw_increasing = model_raw[::-1]
	func_model = spi.interp1d(f_raw_increasing, model_raw_increasing, kind=interpolation_kind)
	model_new  = func_model(f_new)	



	return model_new/1000, Z, flags






def fialkov_priors_21cm_part1(theta):

	P1, P2, P3, P4, P5, P6, P7 = theta

	if (np.log10(0.001) <= P1 <= np.log10(0.5)) \
	   and (np.log10(16.5) <= P2 <= np.log10(76.5)) \
	   and (np.log10(0.00001) <= P3 <= np.log10(10)) \
	   and (0.055 <= P4 <= 0.088) \
	   and (1 <= P5 <= 1.5) \
	   and (0.1 <= P6 <= 3) \
	   and (10 <= P7 <= 50):
		out = 0
		print 'good_21'
	else:
		out = -1e10
		print 'bad_21'

	return out




def fialkov_priors_21cm_part2(flags):

	if np.sum(flags) == 0:
		out = 0

	elif np.sum(flags) > 0:
		out = -1e10

	return out













def fialkov_log_likelihood(theta):


	# Evaluating model 21cm
	j = -1
	values_21cm = parameters_21cm[:]
	for i in range(len(parameters_21cm)):
		if parameters_21cm[i] == -1:
			j = j + 1
			values_21cm[i] = theta[j]

	# If one parameter is complex, by error, don't evaluate the model
	index_complex = 0
	for i in range(len(values_21cm)):
		if isinstance(values_21cm[i], complex):
			index_complex = index_complex + 1
			print 'complex'


	# Don't evaluate the model if tau falls outside its range
	par_index = []
	for i in range(len(parameters_21cm)):
		if parameters_21cm[i] == -1:
			par_index.append(i)
	#print par_index


	index_limits = 0
	for i in range(len(par_index)):
		if (par_index[i] == 0) and ((values_21cm[i] < np.log10(0.001)) or (values_21cm[i] > np.log10(0.5))):
			index_limits = index_limits + 1

		if (par_index[i] == 1) and ((values_21cm[i] < np.log10(16.5)) or (values_21cm[i] > np.log10(76.5))):
			index_limits = index_limits + 1

		if (par_index[i] == 2) and ((values_21cm[i] < np.log10(0.00001)) or (values_21cm[i] >np.log10(10))):
			index_limits = index_limits + 1

		if (par_index[i] == 3) and ((values_21cm[i] < 0.055) or (values_21cm[i] > 0.088)):
			index_limits = index_limits + 1

		if (par_index[i] == 4) and ((values_21cm[i] < 1.0) or (values_21cm[i] > 1.5)):
			index_limits = index_limits + 1

		if (par_index[i] == 5) and ((values_21cm[i] < 0.1) or (values_21cm[i] > 3)):
			index_limits = index_limits + 1

		if (par_index[i] == 6) and ((values_21cm[i] < 10) or (values_21cm[i] > 50)):
			index_limits = index_limits + 1






	# Applying priors
	if (index_complex > 0) or (index_limits > 0):
		log_priors_21 = -1e10
		T21 = 0
		#print 'hoha'       

	elif (index_complex == 0) and (index_limits == 0):
		log_priors_21_part1 = fialkov_priors_21cm_part1(values_21cm)
		print values_21cm
		print log_priors_21_part1
		if log_priors_21_part1 == 0:
			T21, Z, flags = fialkov_model_21cm(v, values_21cm, interpolation_kind='linear')
			log_priors_21_part2 = fialkov_priors_21cm_part2(flags)
			if log_priors_21_part2 < 0:
				T21 = 0

		else:
			log_priors_21_part2 = -1e10
			T21 = 0

		log_priors_21 = log_priors_21_part1 + log_priors_21_part2

	
	# Computing foreground model	
	if log_priors_21 == 0:
		pfit = fit_polynomial_fourier(model_fg, v/vn, d-T21, Nfg, Weights=w)
		Tfg  = pfit[1]
		
	else:
		Tfg = 0








	# Full model
	Tfull = Tfg + T21

	# Log-likelihood
	log_likelihood =  -(1/2)  *  np.sum(  ((d-Tfull)/sigma_noise)**2  )


	# Log-priors + Log-likelihood
	return log_priors_21 + log_likelihood

























































# Pool for MCMC
pool = MPIPool()
if not pool.is_master():
	pool.wait()
	sys.exit(0)




# Counting number of fit 21cm parameters
N21 = 0
for i in range(len(parameters_21cm)):
	if parameters_21cm[i] == -1:
		N21 = N21 + 1

# Number of parameters (dimensions) and walkers
Ndim     = N21
Nwalkers = 2*Ndim + 2




# Random starting point
# p0      = [np.random.rand(Ndim) for i in xrange(Nwalkers)]
p0 = (np.random.uniform(size=Ndim*Nwalkers).reshape((Nwalkers, Ndim)) - 0.5) / 10

# Appropriate offset to the cm parameters
P21_offset = [P1, P2, P3, P4, P5, P6, P7]
j=-1
for i in range(len(parameters_21cm)):
	if parameters_21cm[i] == -1:
		j = j+1
		p0[:,j] = p0[:,j] + P21_offset[i]





# MCMC processing
sampler = emcee.EnsembleSampler(Nwalkers, Ndim, fialkov_log_likelihood, pool=pool)
sampler.run_mcmc(p0, MCMC_Nchain)

# Dimension of accepted sampes is (MCMC_rejected_fraction*MCMC_Nchain*Nwalkers), (Ndim)
# samples_cut = sampler.chain[:, (np.int(MCMC_rejected_fraction*MCMC_Nchain)):, :].reshape((-1, Ndim))
samples_all = sampler.chain[:, :, :].reshape((-1, Ndim))

#print samples_cut.shape
pool.close()


np.savetxt('/home/ramo7131/Desktop/MCMC/results/' + filename, samples_all)

