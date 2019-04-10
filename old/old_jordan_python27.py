
import sys

from os.path import expanduser
home_folder = expanduser("~")

sys.path.insert(0, home_folder + '/emupy-master')
sys.path.insert(0, home_folder + '/ares')


#import emupy #ares
#import ares

import numpy as np
import scipy as sp
import scipy.interpolate as spi
import matplotlib.pyplot as plt
import emcee as ec
from ares.inference.ModelEmulator import ModelEmulator
#import ares















def frequency2redshift(fe):
	"""

	"""
	# Constants and definitions
	c    = 299792458	# wikipedia, m/s
	f21  = 1420.40575177e6  # wikipedia,    
	l21  = c / f21          # frequency to wavelength, as emitted 
	l    = c / (fe * 1e6)   # frequency to wavelength, observed. fe comes in MHz but it has to be converted to Hertz
	z    = (l - l21) / l21  # wavelegth to redshift	

	return z



def redshift2frequency(z):
	"""

	"""
	# Constants and definitions
	c    = 299792458	# wikipedia, m/s
	f21  = 1420.40575177e6  # wikipedia,    
	l21  = c / f21          # frequency to wavelength, as emitted
	l    = l21 * (1 + z)
	f    = c / (l * 1e6)
	return f






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












# Loading High-band data
# --------------------------------------------
d = np.genfromtxt(home_folder + '/DATA/EDGES/results/high_band/products/average_spectrum/high_band_2015_LST_0.26_6.26_dates_2015_250_299_nominal.txt')
v = d[:,0]
t = d[:,1]
w = d[:,2]



# Computing noise standard deviation
# --------------------------------------------
# We divide the antenna temperature by sqrt(weights), and scale this division to enclose 68% of data after fitting and removing a polynomal and differencing from channel to channel
Nk    = 16
par   = fit_polynomial_fourier('EDGES_polynomial', v/140, t, Nk, Weights=w)
model = par[1]
res   = t-model

diff    = np.zeros(len(v))
weights = np.zeros(len(v))
for i in range(len(v)-1):
	diff[i+1]    = res[i+1] - res[i]
	if np.abs(diff[i+1]) < 0.1:
		weights[i+1] = 1.0


profile_not_normalized = t/np.sqrt(w)
profile  = profile_not_normalized/np.max(profile_not_normalized[w>0])
max_diff = np.max(np.abs(diff[weights>0]))
resolution = 10*max_diff/1e4
scale = np.arange(0, 10*max_diff, resolution)
error = 1e6 + 0.0
for i in range(len(scale)):
	full = (np.abs(diff) - scale[i]*profile)[weights==1]
	len_full  = len(full) + 0.0
	len_below = len(full[full<0]) + 0.0
	error_new = np.abs((len_below/len_full)-0.68)
	if error_new < error:
		error = np.copy(error_new)
		correct_scale_diff = scale[i]


# Noise standard deviation
noise_std = correct_scale_diff*profile/np.sqrt(2)








# Produce 21cm interpolation object
# -------------------------------------------------
emu = ModelEmulator(home_folder + '/DATA/EDGES/global_21cm_models/mirocha/emupy/ares_fcoll_4d_mc')

# Whole training set has 10^5 models, just use 10^4	
#redshifts = np.flip(frequency2redshift(v),0)

redshifts = np.arange(6, 35, 1)
emu.train(ivars=redshifts, downsample=1e4, field='dTb', use_pca=False, method='poly', lognorm_pars=True)

# Low-res redshift to frequency
ff_flipped = redshift2frequency(redshifts)
ff = np.flip(ff_flipped, 0)

# Systematic uncertainty level
syst_std     = 0.035 # K








def likelihood_21cm(model_21, Nfg=5, v0=140):
	
	# HERE: THIS FUNCTION PRODUCES A LIKELIHOOD THAT IS INSENSITIVE TO INPUTS. FIX!!!

	vk = v[w>0]
	tk = t[w>0]
	wk = w[w>0]
	
	#print noise_std[0]


	# Covariance matrix of noise
	#if np.isscalar(noise_std):
		#cov_noise = (noise_std**2)*np.eye(len(wk))

	#elif (np.ndim(noise_std == 1) and len(noise_std) == len(w)):
	noise_std_k = noise_std[w>0]
	cov_noise = np.diag(noise_std_k**2)





	# Covariance matrix of systematics
	#if np.isscalar(syst_std):
	cov_syst = (syst_std**2)*np.eye(len(wk))

	#elif np.ndim(syst_std) == 1:
		#syst_std = syst_std[w>0]
		#cov_syst = np.diag(syst_std**2)




	# Foreground model and parameter covariance matrix
	if len(model_21) == len(w):
		model_21 = model_21[w>0]
	par          = fit_polynomial_fourier('EDGES_polynomial', vk/v0, tk-model_21, 5, Weights=wk)
	model_fg     = par[1]
	cov_theta_fg = par[3]

	# Producing "design" matrix with foreground functions
	AT  = np.zeros((Nfg, len(vk)))
	for i in range(Nfg):
		AT[i,:] = (vk/v0)**(-2.5+i)
	A = AT.T

	# Formal way of computing the fg uncertainties, i.e., adding the noise and systematics covariance matrices
	cov_theta_fg_2 = np.linalg.inv(np.dot(np.dot(A.T, np.linalg.inv(cov_noise + cov_syst)), A))




	# Foreground parameter covariance matrix projected onto frequency domain
	#cov_fg  = np.dot(np.dot(A, cov_theta_fg), (A.T))
	cov_fg  = np.dot(np.dot(A, cov_theta_fg_2), (A.T))



	# Total covariance matrix
	cov_total     = cov_noise + cov_syst + cov_fg

	# Inverse of total covariance matrix
	inv_cov_total = np.linalg.inv(cov_total)


	# Computing likelihood
	amplitude    = 1/(np.sqrt(((2*np.pi)**len(vk))*np.linalg.det(90*cov_total)))
	diff         = np.reshape(tk-model_21-model_fg, (1,-1))
	chi_sq       = np.dot(np.dot(diff, inv_cov_total), diff.T)
	
	L = amplitude * np.exp(-(1.0/2.0)*chi_sq) 	
	# This likelihood is not normalized, i.e., it is not absolute. To normalize it, it has to be multiplied by 1/np.sqrt(90**(len(v))). But this is a very small number. In practice it is zero. Therefore, I don't do it.


	return L[0,0]








def priors_21cm(theta):
	
	fx   = theta[0]	
	Nlw  = theta[1]
	Tmin = theta[2]
	Nion = theta[3]
		
	prior_21 = 0
		
	if (fx<1e-2) or (fx>1e2) or (Nlw<1e3) or (Nlw>1e5) or (Tmin<3*1e2) or (Tmin>3*1e5) or (Nion<1e3) or (Nion>1e5):
		prior_21 = -np.inf
			
	return prior_21









def model_21cm(theta):
	
	# Produce model with (linear) parameters entered
	X = emu.predict(theta)

	# This model is in redshift increasing
	model_21_flipped = X[0]/1000.0

	# Flipping model in redshift
	model_21 = np.flip(model_21_flipped, 0)

	# Interpolate to measured frequency vector
	func           = spi.interp1d(ff, model_21, kind='cubic')
	model_21_hires = func(v)

		
	return model_21_hires  # in Kelvin









def ln_likelihood_total(log10_theta):

	# Log10 to linear theta
	theta = 10**(log10_theta)
	
	# Prior probabilities
	priors_21 = priors_21cm(theta)
	
	# If parameter vector is good and within allowed range
	if priors_21 == 0:
		
		# Model 21 cm
		model_21_hires = model_21cm(theta)
		
		# Computing Likelihood of model
		L = likelihood_21cm(model_21_hires, Nfg=5, v0=140)
		
		#print L
		
		# Computing Ln-Likelihood of model
		ln_likelihood     = np.log(L)

	
	# If parameter vector is not good
	else:
		ln_likelihood     = -np.inf
					
					
	return ln_likelihood



















def main_function(Total_points, filename):

	
	#kwargs = {'fX': 3.9, 'Nion': 7300, 'Nlw': 11000, 'Tmin': 9.2e3} # random
	#kwarr  = np.array([kwargs[par] for par in emu.tset.parameters])
	#print kwarr
	#paramters_21cm = np.copy(kwarr)
	#signal = emu.predict(parameters_21cm)
	#signal = signal[0]


	
	Ndim     = 4                             # number of 21cm parameters
	Nwalkers = 2*Ndim + 2
	Nthreads = 1                             # number of CPUs used in parallel. Best value is just 1
	Nchain   = int(Total_points/Nwalkers)    # number of points in the MCMC chain per parameter
	
	
	# Random starting points
	
	log10_fx   = 4*np.random.uniform(size=Nwalkers) - 2               # in range [-2, 2]
	log10_Nlw  = 2*np.random.uniform(size=Nwalkers) + 3               # in range [3, 5]
	log10_Tmin = 3*np.random.uniform(size=Nwalkers) + np.log10(300)   # in range [log10(3x10^2), log10(3x10^5)]
	log10_Nion = 2*np.random.uniform(size=Nwalkers) + 3               # in range [3, 5]
	
	log10_p0_T = np.array([log10_fx, log10_Nlw, log10_Tmin, log10_Nion])
	log10_p0   = log10_p0_T.T
	
	
	
	
	# Setting up MCMC 
	sampler = ec.EnsembleSampler(Nwalkers, Ndim, ln_likelihood_total, threads=Nthreads)	

	# Running MCMC
	print('Computing MCMC ...')
	sampler.run_mcmc(log10_p0, Nchain)
	samples = sampler.chain.reshape((-1, 4))
	
	
	# corner.corner(samples, labels=['$log_{10}(f_{X})$', '$log_{10}(N_{lw})$', '$log_{10}(T_{min})$', '$log_{10}(N_{ion})$'], label_kwargs={'fontsize':17}, levels=(0.68,0.95), show_titles=True, title_kwargs={'fontsize':12})

	# Saving 
	np.savetxt('/media/ramo7131/SSD_4TB/EDGES/global_21cm_models/mirocha/emupy/output_files/' + filename, samples)

	return samples

