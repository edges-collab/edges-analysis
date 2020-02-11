

import numpy as np
import scipy as sp
import scipy.io as sio

import matplotlib.pyplot as plt
import datetime as dt

import astropy.time as apt
import ephem as eph

from src.edges_analysis import data_models as dm, rfi as rfi


def fit_polynomial_fourier(model_type, xdata, ydata, nterms, Weights=1, plot='no', fr=150, df=10, zr=8, dz=2, z_alpha=0, anastasia_model_number=0, jordan_model_number=0, xi_min=0.9, jordan_tau_e_min=0.02, jordan_tau_e_max=0.25, gaussian_flatness_tau=0, gaussian_flatness_tilt=0, external_model_in_K=0):
	
	"""

	This function computes a Least-Squares fit to data using the QR decomposition method.
	Two models are supported: 'polynomial', and 'fourier'.
	If P is the total number of parameters (P = nterms), the 'polynomial' model is: ydata = a0 + a1*xdata + a2*xdata**2 + ... + (aP-1)*xdata**(P-1).
	The 'fourier' model is: ydata = a0 + (a1*np.cos(1*xdata) + a2*np.sin(1*xdata)) + ... + ((aP-2)*np.cos(((P-1)/2)*xdata) + (aP-1)*np.sin(((P-1)/2)*xdata)).

	Definition:
	param, model, rms, cov = fit_polynomial_fourier(model_type, xdata, ydata, nterms, plot='no')

	Input parameters:
	model_type: 'polynomial', 'EDGES_polynomial', or 'fourier'
	xdata: 1D array of independent measurements, of length N, properly normalized to optimize the fit
	ydata: 1D array of dependent measurements, of length N
	nterms: total number of fit coefficients for baseline
	W: matrix of weights, expressed as the inverse of a covariance matrix. It doesn't have to be normalized to anything in particular. Relative weights are OK.
	plot: flag to plot measurements along with fit, and residuals. Use plot='yes' for plotting

	Output parameters:
	param: 1D array of fit parameters, in increasing order, i.e., [a0, a1, ... , aP-1]
	model: 1D array of length N, of model evaluated at fit parameters
	rms: RMS of residuals
	cov: covariance matrix of fit parameters, organized following 'param' array

	Usage:
	param, model, rms, cov = fit_polynomial_fourier('fourier', (f_MHz-150)/50, measured_spectrum, 11, plot='no')

	"""



	# initializing "design" matrix
	AT  = np.zeros((nterms, len(xdata)))

	# initializing auxiliary output array	
	aux = (0, 0)


	# assigning basis functions
	if model_type == 'polynomial':
		for i in range(nterms):
			AT[i,:] = xdata**i



	if model_type == 'fourier':
		AT[0,:] = np.ones(len(xdata))
		for i in range(int((nterms-1)/2)):
			AT[2*i+1,:] = np.cos((i+1)*xdata)
			AT[2*i+2,:] = np.sin((i+1)*xdata)



	if (model_type == 'EDGES_polynomial') or (model_type == 'EDGES_polynomial_plus_gaussian_frequency') or (model_type == 'EDGES_polynomial_plus_gaussian_redshift') or (model_type == 'EDGES_polynomial_plus_tanh') or (model_type == 'EDGES_polynomial_plus_anastasia')  or (model_type == 'EDGES_polynomial_plus_jordan') or (model_type == 'EDGES_polynomial_plus_external'):
		for i in range(nterms):
			AT[i,:] = xdata**(-2.505+i)



	if (model_type == 'LINLOG'):		
		for i in range(nterms):
			AT[i,:] = (xdata**(-2.5))  *  ((np.log(xdata))**i)



	# Physical model from Memo 172
	if (model_type == 'Physical_model') or (model_type == 'Physical_model_plus_gaussian_frequency') or (model_type == 'Physical_model_plus_gaussian_redshift') or (model_type == 'Physical_model_plus_tanh') or (model_type == 'Physical_model_plus_anastasia') or (model_type == 'Physical_model_plus_jordan') or (model_type == 'Physical_model_plus_external'):
		if nterms >= 3:
		#if (nterms == 4) or (nterms == 5):
			AT = np.zeros((nterms,len(xdata)))
			AT[0,:] = xdata**(-2.5)
			AT[1,:] = np.log(xdata) * xdata**(-2.5)
			AT[2,:] = (np.log(xdata))**2 * xdata**(-2.5)

			if nterms >= 4:
				AT[3,:] = xdata**(-4.5)
				if nterms == 5:
					AT[4,:] = xdata**(-2)


		else:
			print('ERROR: For the Physical model it has to be 4 or 5 terms.')
			AT = 0



	# nterms ONLY includes the number of parameters for the baseline.

	# Gaussian in frequency
	if (model_type == 'EDGES_polynomial_plus_gaussian_frequency') or (model_type == 'Physical_model_plus_gaussian_frequency'):

		# Regular Gaussian in frequency
		if gaussian_flatness_tau == 0:
			gaussian_function, xHI, z = model_eor(xdata, T21=1, model_type='gaussian_frequency', fr=fr, df=df)
			aux                       = (z, xHI)
			AT                        = np.append(AT, gaussian_function.reshape(1,-1), axis=0)


		# Flattened Gaussian in frequency
		elif gaussian_flatness_tau > 0:
			#print('HOLE')
			gaussian_function, xHI, z = model_eor(xdata, T21=1, model_type='gaussian_flattened_1', fr=fr, df=df, tau0=gaussian_flatness_tau, tilt=gaussian_flatness_tilt)
			aux                       = (z, xHI)
			AT                        = np.append(AT, gaussian_function.reshape(1,-1), axis=0)






	# Gaussian in redshift
	if (model_type == 'EDGES_polynomial_plus_gaussian_redshift') or (model_type == 'Physical_model_plus_gaussian_redshift'):
		gaussian_function, xHI, z = model_eor(xdata, T21=1, model_type='gaussian_redshift', zr=zr, dz=dz, z_alpha=z_alpha, dz_accuracy_skewed_gaussian=0.0025)
		aux                       = (z, xHI)
		AT                        = np.append(AT, gaussian_function.reshape(1,-1), axis=0)



	# Tanh
	if (model_type == 'EDGES_polynomial_plus_tanh') or (model_type == 'Physical_model_plus_tanh'):
		tanh_function, xHI, z = model_eor(xdata, T21=1, zr=zr, dz=dz)
		aux                   = (z, xHI)
		AT                    = np.append(AT, tanh_function.reshape(1,-1), axis=0)




	if (model_type == 'EDGES_polynomial_plus_anastasia') or (model_type == 'Physical_model_plus_anastasia'):
		model_in_K, ao = model_eor_anastasia(anastasia_model_number, xdata)   # xdata: frequency in MHz, model_in_K: it is in K
		aux            = ao
		AT             = np.append(AT, model_in_K.reshape(1,-1), axis=0)




	if (model_type == 'EDGES_polynomial_plus_jordan') or (model_type == 'Physical_model_plus_jordan'):
		model_in_K, ao = model_eor_jordan(jordan_model_number, xdata, xi_min=xi_min, tau_e_min=jordan_tau_e_min, tau_e_max=jordan_tau_e_max)   # xdata: frequency in MHz, model_in_K: it is in K
		aux            = ao
		print('---------------------------------------------')
		AT             = np.append(AT, model_in_K.reshape(1,-1), axis=0)



	if (model_type == 'EDGES_polynomial_plus_external') or (model_type == 'Physical_model_plus_external'):
		aux            = 0
		AT             = np.append(AT, external_model_in_K.reshape(1,-1), axis=0)






	# Applying General Least Squares Formalism, and Solving using QR decomposition
	# ----------------------------------------------------------------------------
	# ----------------------------------------------------------------------------
	# See: http://www2.imm.dtu.dk/pubdb/views/edoc_download.php/2804/pdf/imm2804.pdf

	# if no weights are given
	if np.isscalar(Weights):
		W = np.eye(len(xdata))

	# if a vector is given
	elif np.ndim(Weights) == 1:
		W = np.diag(Weights)

	# if a matrix is given
	elif np.ndim(Weights) == 2:
		W = Weights


	# sqrt of weight matrix
	sqrtW = np.sqrt(W)


	# transposing matrices so 'frequency' dimension is along columns
	A     = AT.T
	ydata = np.reshape(ydata, (-1,1))


	# A and ydata "tilde"
	WA     = np.dot(sqrtW, A)
	Wydata = np.dot(sqrtW, ydata)


	# solving system using 'short' QR decomposition (see R. Butt, Num. Anal. Using MATLAB)
	Q1, R1      = sp.linalg.qr(WA, mode='economic') # returns

	#print(R1)
	#print(np.dot(Q1.T, Wydata))
	param       = sp.linalg.solve(R1, np.dot(Q1.T, Wydata))

	model       = np.dot(A, param)
	error       = ydata - model
	DF          = len(xdata)-len(param)-1
	wMSE        = (1/DF) * np.dot(error.T, np.dot(W, error))  # This is correct because we also have the Weight matrix in the (AT * W * A)^-1.
	wRMS        = np.sqrt( np.dot(error.T, np.dot(W, error)) / np.sum(np.diag(W)))
	#inv_pre_cov = np.linalg.lstsq(np.dot(R1.T, R1), np.eye(nterms))   # using "lstsq" to compute the inverse: inv_pre_cov = (R1.T * R1) ^ -1
	#cov         = MSE * inv_pre_cov[0]
	inv_pre_cov = np.linalg.inv(np.dot(R1.T, R1))
	cov         = wMSE * inv_pre_cov



	# back to input format
	ydata = ydata.flatten()
	model = model.flatten()
	param = param.flatten()





	# plotting ?
	if plot == 'yes':
		plt.close()

		plt.subplot(3,1,1)
		plt.plot(xdata, ydata, 'b')
		plt.plot(xdata, model, 'r.')

		plt.subplot(3,1,2)
		plt.plot(xdata, ydata-model, 'b')

		plt.subplot(3,1,3)
		plt.errorbar(np.arange(len(param)), param, np.sqrt(np.diag(cov)), marker='o')
		plt.xlim([-1, len(param)])

		plt.show()

	return param, model, wRMS, cov, wMSE, aux     # wMSE = reduced chi square










def model_evaluate(model_type, par, xdata, fr=150, df=10, zr=8, dz=2, z_alpha=0, anastasia_model_number=0, jordan_model_number=0, gaussian_flatness_tau=0, gaussian_flatness_tilt=0):
	"""
	Last modification: May 24, 2015.

	This function evaluates 'polynomial' or 'fourier' models at array 'xdata', using parameters 'par'.
	It is a direct complement to the function 'fit_polynomial_fourier'.
	If P is the total number of parameters, the 'polynomial' model is: model = a0 + a1*xdata + a2*xdata**2 + ... + (aP-1)*xdata**(P-1).
	The 'fourier' model is: model = a0 + (a1*np.cos(1*xdata) + a2*np.sin(1*xdata)) + ... + ((aP-2)*np.cos(((P-1)/2)*xdata) + (aP-1)*np.sin(((P-1)/2)*xdata)).

	Definition:
	model = model_evaluate(model_type, par, xdata)

	Input parameters:
	model_type: 'polynomial', 'EDGES_polynomial', or 'fourier'
	par: 1D array of parameters, in increasing order, i.e., [a0, a1, ... , aP-1]
	xdata: 1D array of the independent variable

	Output parameters:
	model: 1D array with model

	Usage:
	model = model_evaluate('fourier', par_array, fn)
	"""


	if model_type == 'polynomial':
		summ = 0
		for i in range(len(par)):
			summ = summ + par[i] * xdata**i




	elif model_type == 'fourier':
		summ = par[0]

		n_cos_sin = int((len(par)-1)/2)
		for i in range(n_cos_sin):
			icos = 2*i + 1
			isin = 2*i + 2
			summ = summ + par[icos] * np.cos((i+1)*xdata) + par[isin] * np.sin((i+1)*xdata)




	elif (model_type == 'EDGES_polynomial'):
		summ = 0
		for i in range(len(par)):
			summ = summ + par[i] * xdata**(-2.5+i)





	elif (model_type == 'LINLOG'):
		summ = 0
		for i in range(len(par)):
			summ = summ      +      par[i] * (xdata**(-2.5)) * ((np.log(xdata))**i)






	elif (model_type == 'EDGES_polynomial_plus_gaussian_frequency') or (model_type == 'EDGES_polynomial_plus_gaussian_redshift') or (model_type == 'EDGES_polynomial_plus_tanh') or (model_type == 'EDGES_polynomial_plus_anastasia') or (model_type == 'EDGES_polynomial_plus_jordan'):
		summ = 0
		for i in range(len(par)-1):  # Here is the difference with the case above. The last parameters is the amplitude of the Gaussian/Tanh.
			summ = summ + par[i] * xdata**(-2.5+i)



	# Physical model from Memo 172
	elif (model_type == 'Physical_model'):
		summ = 0
		basis = np.zeros((5,len(xdata)))
		basis[0,:] = xdata**(-2.5)
		basis[1,:] = np.log(xdata) * xdata**(-2.5)
		basis[2,:] = (np.log(xdata))**2 * xdata**(-2.5)
		basis[3,:] = xdata**(-4.5)
		basis[4,:] = xdata**(-2)		

		for i in range(len(par)):
			summ = summ + par[i] * basis[i,:]



	# Physical model from Memo 172
	elif (model_type == 'Physical_model_plus_gaussian_frequency') or (model_type == 'Physical_model_plus_gaussian_redshift') or (model_type == 'Physical_model_plus_tanh') or (model_type == 'Physical_model_plus_anastasia') or (model_type == 'Physical_model_plus_jordan'):
		summ = 0
		basis = np.zeros((5,len(xdata)))
		basis[0,:] = xdata**(-2.5)
		basis[1,:] = np.log(xdata) * xdata**(-2.5)
		basis[2,:] = (np.log(xdata))**2 * xdata**(-2.5)
		basis[3,:] = xdata**(-4.5)
		basis[4,:] = xdata**(-2)		

		for i in range(len(par)-1):  # Here is the difference with the case above. The last parameters is the amplitude of the Gaussian/Tanh.
			summ = summ + par[i] * basis[i,:]                




	else:
		summ = 0




	if (model_type == 'EDGES_polynomial_plus_gaussian_frequency') or (model_type == 'Physical_model_plus_gaussian_frequency'):
		if gaussian_flatness_tau == 0:
			gaussian_function, xHI, z = model_eor(xdata, T21=1, model_type='gaussian_frequency', fr=fr, df=df)

		elif gaussian_flatness_tau > 0:
			gaussian_function, xHI, z = model_eor(xdata, T21=1, model_type='gaussian_flattened_1', fr=fr, df=df, tau0=gaussian_flatness_tau, tilt=gaussian_flatness_tilt)

		summ = summ + par[-1] * gaussian_function




	if (model_type == 'EDGES_polynomial_plus_gaussian_redshift') or (model_type == 'Physical_model_plus_gaussian_redshift'):
		gaussian_function, xHI, z = model_eor(xdata, T21=1, model_type='gaussian_redshift', zr=zr, dz=dz, z_alpha=z_alpha)
		summ = summ + par[-1] * gaussian_function




	if (model_type == 'EDGES_polynomial_plus_tanh') or (model_type == 'Physical_model_plus_tanh'):
		tanh_function, xHI, z = model_eor(xdata, T21=1, zr=zr, dz=dz)
		summ = summ + par[-1] * tanh_function




	if (model_type == 'EDGES_polynomial_plus_anastasia') or (model_type == 'Physical_model_plus_anastasia'):
		model_in_K = model_eor_anastasia(anastasia_model_number, xdata)   # xdata: frequency in MHz, model_in_K: it is in K
		summ = summ + par[-1] * model_in_K




	if (model_type == 'EDGES_polynomial_plus_jordan') or (model_type == 'Physical_model_plus_jordan'):
		model_in_K = model_eor_jordan(jordan_model_number, xdata)   # xdata: frequency in MHz, model_in_K: it is in K
		summ = summ + par[-1] * model_in_K




	model = summ
	return model



































def frequency_edges(flow, fhigh):
	"""
	Last modification: May 24, 2015.

	This function returns the raw EDGES frequency array, in MHz.

	Definition:
	freqs, index_flow, index_fhigh = frequency_edges(flow, fhigh)

	Input parameters:
	flow: low-end limit of frequency range, in MHz
	fhigh: high-end limit of frequency range, in MHz

	Output parameters:
	freqs: full frequency array from 0 to 200 MHz, at raw resolution
	index_flow: index of flow
	index_fhigh: index of fhigh

	Usage:
	freqs, index_flow, index_fhigh = frequency_edges(90, 190)
	"""

	# Full frequency vector
	nchannels = 16384*2
	max_freq = 200
	fstep = max_freq/nchannels
	freqs = np.arange(0, max_freq, fstep)

	# Indices of frequency limits
	if (flow < 0) or (flow >= max(freqs)) or (fhigh < 0) or (fhigh >= max(freqs)):
		print('ERROR. Limits are 0 MHz and ' + str(max(freqs)) + ' MHz')
	else:
		for i in range(len(freqs)-1):
			if (freqs[i] <= flow) and (freqs[i+1] >= flow):
				index_flow = i
			if (freqs[i] <= fhigh) and (freqs[i+1] >= fhigh):
				index_fhigh = i

		return freqs, index_flow, index_fhigh









def level1_MAT(file_name, plot='no'):
	"""
	Last modification: May 24, 2015.

	This function loads the antenna temperature and date/time from MAT files produced by the MATLAB function acq2level1.m

	Definition:
	ds, dd = level1_MAT(file_name, plot='no')

	Input parameters:
	file_name: path and name of MAT file
	plot: flag for plotting spectrum data. Use plot='yes' for plotting

	Output parameters:
	ds: 2D spectra array
	dd: Nx6 date/time array

	Usage:
	ds, dd = level1_MAT('/file.MAT', plot='yes')
	"""


	# loading data and extracting main array
	d = sio.loadmat(file_name)
	darray = d['array']

	# extracting spectra and date/time
	ds = darray[0,0]
	dd = darray[0,1]

	# plotting ?
	if plot == 'yes':
		plt.imshow(ds, aspect = 'auto', vmin = 0, vmax = 2000)
		plt.xlabel('frequency channels')
		plt.ylabel('trace')
		plt.colorbar()
		plt.show()

	return ds, dd








def spectral_binning_number_of_samples(freq_in, spectrum_in, weights_in, nsamples=64):

	flag_start = 0
	i = 0
	for j in range(len(freq_in)):
		#print(j)

		if i == 0:
			sum_fre = 0
			sum_num = 0
			sum_den = 0
			samples_spectrum_in = []
			samples_weights_in  = []
			

		if (i >= 0) and (i < nsamples):

			sum_fre = sum_fre + freq_in[j]
			#if spectrum_in[j]  0:
			sum_num = sum_num + spectrum_in[j]*weights_in[j]
			sum_den = sum_den + weights_in[j]

			av_fr_temp = sum_fre / nsamples
			if sum_den > 0:
				av_sp_temp = sum_num / sum_den
			else:
				av_sp_temp = 0
				
			if weights_in[j]>0:
				samples_spectrum_in = np.append(samples_spectrum_in, spectrum_in[j])
				samples_weights_in  = np.append(samples_weights_in, weights_in[j])
			
			
		if i < (nsamples-1):
			i = i+1

		elif i == (nsamples-1):
			
			if len(samples_spectrum_in) <= 1:
				std_of_the_mean = 1e6			
			
			if len(samples_spectrum_in) > 1:
				sample_variance = np.sum(((samples_spectrum_in - av_sp_temp)**2)*samples_weights_in)/np.sum(samples_weights_in)
				#sample_variance = (np.std(samples_spectrum_in))**2
				std_of_the_mean = np.sqrt(sample_variance/len(samples_spectrum_in))
			
			
			if flag_start == 0:
				av_fr = av_fr_temp
				av_sp = av_sp_temp
				av_we = sum_den
				av_std = std_of_the_mean
				flag_start = 1

			elif flag_start > 0:
				av_fr = np.append(av_fr, av_fr_temp)
				av_sp = np.append(av_sp, av_sp_temp)
				av_we = np.append(av_we, sum_den)
				av_std = np.append(av_std, std_of_the_mean)

			i = 0


	return av_fr, av_sp, av_we, av_std













def weighted_mean(data_array, weights_array):

	# Number of frequency channels
	lf = len(data_array[0,:])

	# Number of spectra
	ls = len(data_array[:,0])

	# Initializing arrays
	av = np.zeros(lf)
	w  = np.zeros(lf)

	# Cycle over frequency channels
	for k in range(lf):
		num = 0
		den = 0
		wei = 0

		# Cycle over number of spectra
		for j in range(ls):
			if (weights_array[j,k] > 0): # (data_array[j,k] > 0) and 
				num = num + data_array[j,k] * weights_array[j,k]
				den = den + weights_array[j,k]


		# Computing averages
		if (num != 0) and (den != 0): 
			av[k] = num/den
			w[k]  = den


	return av, w









def weighted_standard_deviation(av, data_array, std_array):

	ls = len(data_array[0,:])
	la = len(data_array[:,0])

	std_sq = np.zeros(ls)

	for k in range(ls):
		num = 0
		den = 0		
		for j in range(la):
			num = num + ((data_array[j,k]-av[k])/std_array[j,k])**2
			den = den + 1/(std_array[j,k]**2)

		if (num != 0) and (den != 0): 
			std_sq[k] = num/den


	std = np.sqrt(std_sq)	

	return std









def spectral_averaging(data_array, weights_array):

	"""
	array: 2D format ls x lf
	ls: number of spectra (number of 1D arrays)
	lf: number of frequency points per spectra
	"""

	av, w  = weighted_mean(data_array, weights_array)	

	return av, w







def utc2lst(utc_time_array, LONG_float):
	"""
	Last modification: May 29, 2015.

	This function converts an Nx6 array of floats or integers representing UTC date/time, to a 1D array of LST date/time.

	Definition:
	LST = utc2lst(utc_time_array, LONG_float)

	Input parameters:
	utc_time_array: Nx6 array of floats or integers, where each row is of the form [yyyy, mm, dd, HH, MM, SS]. It can also be a 6-element 1D array.
	LONG_float: terrestrial longitude of observatory (float) in degrees.

	Output parameters:
	LST: 1D array of LST dates/times

	Usage:
	LST = utc2lst(utc_time_array, -27.5)
	"""

	# converting input array to "int"
	uta = utc_time_array.astype(int)



	# compatibility for 1D inputs (only 1 array of date/time), and 2D (N arrays of date/time)
	if uta.ndim == 1:
		len_array = 1
	if uta.ndim == 2:
		len_array = len(uta)



	# converting UTC to LST
	LST = np.zeros(len_array)

	for i in range(len_array):
		if uta.ndim == 1:
			yyyy = uta[0]
			mm   = uta[1]
			dd   = uta[2]
			HH   = uta[3]
			MM   = uta[4]
			SS   = uta[5]

		elif uta.ndim == 2:
			yyyy = uta[i,0]
			mm   = uta[i,1]
			dd   = uta[i,2]
			HH   = uta[i,3]
			MM   = uta[i,4]
			SS   = uta[i,5]


		# time stamp in python "datetime" format
		udt = dt.datetime(yyyy, mm, dd, HH, MM, SS)

		# python "datetime" to astropy "Time" format
		t = apt.Time(udt, format='datetime', scale='utc')

		# necessary approximation to compute sidereal time
		t.delta_ut1_utc = 0

		# LST at longitude LONG_float, in degrees
		LST_object = t.sidereal_time('apparent', str(LONG_float)+'d', model='IAU2006A')
		LST[i] = LST_object.value


	return LST










def SUN_MOON_azel(LAT, LON, UTC_array):
	# 
	# Local coordinates of the Sun using Astropy
	#
	# EDGES_lat_deg = -26.7
	# EDGES_lon_deg = 116.6
	#	


	# Observation coordinates
	OBS_lat_deg = str(LAT) 
	OBS_lon_deg = str(LON) 
	#print(' ')
	#print('Observation Coordinates: ' + 'LAT: ' + OBS_lat_deg + ' LON: ' + OBS_lon_deg)
	#print('------------------------')

	OBS_location     = eph.Observer()
	OBS_location.lat = OBS_lat_deg 
	OBS_location.lon = OBS_lon_deg


	# Compute local coordinates of Sun and Moon
	SH = UTC_array.shape

	if len(SH) == 1:
		coord    = np.zeros(4)

		OBS_location.date = dt.datetime(UTC_array[0], UTC_array[1], UTC_array[2], UTC_array[3], UTC_array[4], UTC_array[5])
		Sun  = eph.Sun(OBS_location)
		Moon = eph.Moon(OBS_location)

		coord[0] = (180/np.pi)*eph.degrees(Sun.az)
		coord[1] = (180/np.pi)*eph.degrees(Sun.alt)		
		coord[2] = (180/np.pi)*eph.degrees(Moon.az)
		coord[3] = (180/np.pi)*eph.degrees(Moon.alt)


		#print('Sun AZ:  ' + str(round(coord[0],2))  + '      Sun EL:  ' + str(round(coord[1],2)))
		#print('Moon AZ: ' + str(round(coord[2],2))  + '      Moon EL: ' + str(round(coord[3],2)))

	elif len(SH) == 2:
		coord    = np.zeros((SH[0],4))
		for i in range(SH[0]):
			OBS_location.date = dt.datetime(UTC_array[i,0], UTC_array[i,1], UTC_array[i,2], UTC_array[i,3], UTC_array[i,4], UTC_array[i,5])
			Sun  = eph.Sun(OBS_location)
			Moon = eph.Moon(OBS_location)

			coord[i,0] = (180/np.pi)*eph.degrees(Sun.az)
			coord[i,1] = (180/np.pi)*eph.degrees(Sun.alt)		
			coord[i,2] = (180/np.pi)*eph.degrees(Moon.az)
			coord[i,3] = (180/np.pi)*eph.degrees(Moon.alt)


			#print('Sun AZ:  ' + str(round(coord[i,0],2))  + '      Sun EL:  ' + str(round(coord[i,1],2)))
			#print('Moon AZ: ' + str(round(coord[i,2],2))  + '      Moon EL: ' + str(round(coord[i,3],2)))
			#print('-------------------------------------')

	return coord





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







def temperature_thermistor_oven_industries_TR136_170(R, unit):

	"""
	
	This function is good for two thermistors:
	- Oven Industries TR-136-170
	- Tetech MP-3193
	
	The match between the two is better than 0.001 degC
	
	"""



	# Steinhart-Hart coefficients
	a1 = 1.03514e-3
	a2 = 2.33825e-4
	a3 = 7.92467e-8


	# TK in Kelvin
	TK = 1/(a1 + a2*np.log(R) + a3*(np.log(R))**3)

	# Kelvin or Celsius
	if unit == 'K':
		T = TK
	if unit == 'C':
		T = TK - 273.15

	return T







def temperature_thermistor_omega_ON_930_44006(R, unit):

	# Steinhart-Hart coefficients
	a1 = 0.001296751267466723
	a2 = 0.00019737361897609893
	a3 = 3.0403175473012516e-7


	# TK in Kelvin
	TK = 1/(a1 + a2*np.log(R) + a3*(np.log(R))**3)

	# Kelvin or Celsius
	if unit == 'K':
		T = TK
	if unit == 'C':
		T = TK - 273.15

	return T







def average_calibration_spectrum(spectrum_files, RFI_cleaning, FLOW, FHIGH, thermistor_model, resistance_file, start_percent=0, plot='no'):
	"""
	Last modification: May 24, 2015.

	This function loads and averages (in time) calibration data (ambient, hot, open, shorted, simulators, etc.) in MAT format produced by the "acq2level1.m" MATLAB program. It also returns the average physical temperature of the corresponding calibrator, measured with an Oven Industries TR136-170 thermistor.

	Definition:
	av_ta, av_temp = average_calibration_spectrum(spectrum_files, resistance_file, start_percentage=0, plot='no')

	Input parameters:
	spectrum_files: string, or list of strings, with the paths and names of spectrum files to process
	resistance_file: string, or list, with the path and name of resistance file to process
	start_percent: percentage of initial data to dismiss, for both, spectra and resistance
	plot: flag for plotting representative data cuts. Use plot='yes' for plotting

	Output parameters:
	av_ta: average spectrum at raw frequency resolution, starting at 0 Hz
	av_temp: average physical temperature

	Usage:
	spec_file1 = '/file1.mat'
	spec_file2 = '/file2.mat'
	spec_files = [spec_file1, spec_file2]
	res_file = 'res_file.txt'
	av_ta, av_temp = average_calibration_spectrum(spec_files, res_file, start_percentage=10, plot='yes')
	"""



	# spectra
	for i in range(len(spectrum_files)):
		tai, xxx = level1_MAT(spectrum_files[i], plot='no')
		if i == 0:
			ta = tai
		elif i > 0:
			ta = np.concatenate((ta, tai), axis=0)

	index_start_spectra = int((start_percent/100)*len(ta[:,0]))
	ta_sel0 = ta[index_start_spectra::,:]
	   
	
	
	# Selecting data in desired frequency range
	# -----------------------------------------
	fx, x1, x2 = ba.frequency_edges(1,2)
	f          = fx[(fx>=FLOW) & (fx<=FHIGH)] # 45 MHz -- 195 MHz
	ta_sel1    = ta_sel0[:,(fx>=FLOW) & (fx<=FHIGH)]
	
	
	
	# total power filter
	# ------------------
	tp = np.sum(ta_sel1, axis=1)
	W  = np.ones(len(tp))
	IN = np.arange(len(tp))
	
	Nsigma_tp = 3
	bad_old   = -1
	bad       =  0		
	iteration =  0
	
	

	while (bad > bad_old):

		iteration = iteration + 1
		print('Iteration: ' + str(iteration))
			
		std = np.std(tp[W>0])
		ave = np.mean(tp[W>0])
		res = tp-ave	

		IN_bad    = IN[(np.abs(res) > Nsigma_tp*std)]
		W[IN_bad] = 0
		
		bad_old = np.copy(bad)
		bad     = len(IN_bad)

		print('STD: ' + str(np.round(std,3)) + ' K')
		print('Number of bad points excised: ' + str(bad))
		print(IN_bad)



	# indices of good data points
	# ---------------------------
	IN_good = np.setdiff1d(IN, IN_bad)



	# selecting data with normal total power
	# --------------------------------------
	ta_sel2 = ta_sel1[IN_good,:]



	
	
	
	# RFI filter
	# ----------

	if RFI_cleaning == 'yes':
		ta_sel3 = np.zeros((len(IN_good), len(f)))
		ww      = np.zeros((len(IN_good), len(f)))
		
		Nsigma_RFI = 2.5
		
		print('RFI cleaning')
		print('---------------------------------')
		for i in range(len(IN_good)):  # range(2): #
			
			print('RFI trace: ' + str(i+1) + ' of ' + str(len(IN_good)))
			ti1, wi1  = rfi.cleaning_sweep(f, ta_sel2[i, :], np.ones(len(f)), window_width_MHz=4, Npolyterms_block=4, N_choice=20, N_sigma=Nsigma_RFI)
	
			flip_ti2, flip_wi2  = rfi.cleaning_sweep(f, np.flip(ta_sel2[i, :]), np.ones(len(f)), window_width_MHz=4, Npolyterms_block=4, N_choice=20, N_sigma=Nsigma_RFI)
			ti2 = np.flip(flip_ti2)
			wi2 = np.flip(flip_wi2)
	
			ind      = np.arange(len(f))
			ind_bad  = np.union1d(ind[wi1==0], ind[wi2==0])
	
			ti3          = ta_sel2[i,:]
			ti3[ind_bad] = 0
	
			wi3          = np.ones(len(f))
			wi3[ind_bad] = 0
	
	
			ta_sel3[i,:] = ti3
			ww[i,:]      = wi3

	
	elif RFI_cleaning == 'no':
		ta_sel3 = np.copy(ta_sel2)
		ww      = np.ones((len(IN_good), len(f)))



	
	
	#av_ta = np.mean(ta_sel2, axis=0)
	avt, avw = spectral_averaging(ta_sel3, ww)











	# temperature
	if isinstance(resistance_file, list):
		for i in range(len(resistance_file)):
			if i == 0:
				R = np.genfromtxt(resistance_file[i])
			else:	
				R = np.concatenate((R, np.genfromtxt(resistance_file[i])), axis=0)
	else:
		R = np.genfromtxt(resistance_file)

	# compute physical temperature depending on thermistor model
	if (thermistor_model == 'tr136') or (thermistor_model == 'mp3139'):
		temp = temperature_thermistor_oven_industries_TR136_170(R, 'K')
	
	elif thermistor_model == 'on930':
		temp = temperature_thermistor_omega_ON_930_44006(R, 'K')


	
	index_start_temp = int((start_percent/100)*len(temp))
	temp_sel = temp[index_start_temp::]
	av_temp = np.mean(temp_sel)




	# plot
	if plot == 'yes':
		plt.close()
		plt.subplot(2,2,1)
		plt.plot(ta[:,30000],'r')
		plt.plot([index_start_spectra, index_start_spectra],[min(ta[:,30000])-5, max(ta[:,30000])+5], 'k--')
		plt.ylabel('spectral temperature')
		plt.ylim([min(ta[:,30000])-5, max(ta[:,30000])+5])

		plt.subplot(2,2,2)
		plt.plot(ta_sel0[:,30000],'r')
		plt.ylim([min(ta[:,30000])-5, max(ta[:,30000])+5])

		plt.subplot(2,2,3)
		plt.plot(temp,'r')
		plt.plot([index_start_temp, index_start_temp],[min(temp)-5, max(temp)+5], 'k--')
		plt.xlabel('sample')
		plt.ylabel('physical temperature')
		plt.ylim([min(temp)-5, max(temp)+5])

		plt.subplot(2,2,4)
		plt.plot(temp_sel,'r')
		plt.xlabel('sample')
		plt.ylim([min(temp)-5, max(temp)+5])

	return f, avt, avw, av_temp    #, ta_sel2





def signal_edges2018_uncertainties(v):

	# Nominal model reported in Bowman et al. (2018)
	# ----------------------------------------------
	model_nominal = dm.signal_model('exp', [-0.5, 78, 19, 7], v)
	
	
	
	# Computing distribution of models and limits
	# -------------------------------------------
	N_MC = 10000
	dx   = 0.003/2   # probability tails for 99.7% (3 sigma) probabilities
	
	# Centers and Widths of Gaussian Parameter Distributions
	uA = (-1-(-0.2))/2 -0.2
	sA = (0.2+0.5)/6
	
	uv0 = 78
	sv0 = 1/3
	
	uW = (23-17)/2 + 17
	sW = (23-17)/6
	
	ut = (12-4)/2 + 4
	st = (12-4)/6
	
	# Random models
	model_perturbed = np.zeros((N_MC, len(v)))
	
	for i in range(N_MC):	
		x   = np.random.multivariate_normal([0,0,0,0], np.diag(np.ones(4)))
		dA  = sA  * x[0]
		dv0 = sv0 * x[1]
		dW  = sW  * x[2]
		dt  = st  * x[3]
			
		model_perturbed[i,:] = dm.signal_model('exp', [uA+dA, uv0+dv0, uW+dW, ut+dt], v)
		
	# Limits
	limits = np.zeros((len(v),2))
	for i in range(len(v)):
		x_low  = -1
		x_high = -1
		x            = model_perturbed[:,i]
		x_increasing = np.sort(x)
		cx           = np.abs(np.cumsum(x_increasing))
		norm_cx      = cx/np.max(cx)
		for j in range(len(norm_cx)-1):
			#print(norm_cx[j])
			if (norm_cx[j]<dx) and (norm_cx[j+1]>=dx):
				x_low = x_increasing[j+1]
				
			if (norm_cx[j]<(1-dx)) and (norm_cx[j+1]>(1-dx)):
				x_high = x_increasing[j]
		
		if x_low == -1:
			x_low = x_increasing[0]
		if x_high == -1:
			x_high = x_increasing[-1]
			
			
		limits[i,0] = x_low
		limits[i,1] = x_high
			
		
		
		
			
			
	return model_nominal, model_perturbed, limits









def HFSS_beam_read(path_to_file, dB_or_linear, theta_min=0, theta_max=180, theta_resolution=1, phi_min=0, phi_max=359, phi_resolution=1):
	
	'''
	
	'''
	
	d = np.genfromtxt(path_to_file, skip_header=1, delimiter=',')
	
	theta        = np.arange(theta_min, theta_max+theta_resolution, theta_resolution)
	
	phi          = np.arange(phi_min, phi_max+phi_resolution, phi_resolution)
	#phiV[phiV<0] = phiV[phiV<0] + 360
	#iV           = np.argsort(phiV)
	#phiV         = phiV[iV]	
	#phi          = np.unique(phiV, axis=0)
	
	
	beam_map = np.zeros((len(theta), len(phi)))
	for i in range(len(theta)):
		
		phiX   = d[d[:,1] == theta[i], 0]
		powerX = d[d[:,1] == theta[i], 2]
		
		phiX[phiX<0]  = phiX[phiX<0] + 360
		phiX, iX      = np.unique(phiX, axis=0, return_index=True)
		powerX        = powerX[iX]
		
		
		iY     = np.argsort(phiX)
		powerY = powerX[iY]
		
		if dB_or_linear == 'dB':
			linear_power  = 10**(powerY/10)
			
		elif dB_or_linear == 'linear':
			linear_power = np.copy(powerY)
	
		linear_power[np.isnan(linear_power) == True] = 0
		beam_map[i,:] = linear_power	
		
		
	
	
	return theta, phi, beam_map







def WIPLD_beam_read(filename, AZ_antenna_axis=0):
	
	# '/EDGES_vol2/others/beam_simulations/wipl-d/20191012/blade_dipole_infinite_PEC.ra1'
	with open(filename) as fn:
		
		file_length = 0
		number_of_frequencies = 0
		
		flag_columns = 0
		frequencies_list = []
		for line in fn:
			file_length = file_length + 1
			if line[2] == '>':
				number_of_frequencies = number_of_frequencies + 1
				frequencies_list.append(float(line[19:32]))
				print(line)
				
			if (line[2] != '>') and (flag_columns == 0):
				line_splitted = line.split()
				number_of_columns = len(line_splitted)
				flag_columns = 1
		
		rows_per_frequency = (file_length - number_of_frequencies) / number_of_frequencies
		
		print(file_length)
		print(int(number_of_frequencies))
		print(int(rows_per_frequency))
		print(int(number_of_columns))
		
		output = np.zeros((int(number_of_frequencies), int(rows_per_frequency), int(number_of_columns)))
		
		
	frequencies = np.array(frequencies_list)



	with open(filename) as fn:
		i = -1
		for line in fn:
			
			if line[2] == '>':
				i = i + 1
				j = -1
				
				print(line)
			else:
				j = j + 1
				line_splitted = line.split()
				line_array    = np.array(line_splitted, dtype=float)
				
				output[i,j,:] = line_array
				#print(line_array)
				
				

	# Rearranging data
	# ----------------
	phi_u   = np.unique(output[0,:,0])
	theta_u = np.unique(output[0,:,1])
	
	beam = np.zeros((len(frequencies), len(theta_u), len(phi_u))) 
	
	for i in range(len(frequencies)):
		out_2D = output[i,:,:]
		
		phi   = out_2D[:,0]
		theta = 90 - out_2D[:,1]  # theta is zero at the zenith, and goes to 180 deg
		gain  = out_2D[:,6]
		
		theta_u = np.unique(theta)
		it = np.argsort(theta_u)
		theta_a = theta_u[it]
		
		for j in range(len(theta_a)):
			#print(theta_a[j])
			
			phi_j  = phi[theta == theta_a[j]]
			gain_j = gain[theta == theta_a[j]]
			
			ip = np.argsort(phi_j)
			gp = gain_j[ip]
			
			beam[i, j, :] = gp
	
	
	
	# Flip beam from theta to elevation
	# ---------------------------------
	beam_maps = beam[:,::-1,:]
	
				
	# Change coordinates from theta/phi, to AZ/EL
	# -------------------------------------------
	EL = np.arange(0,91) #, dtype='uint32')
	AZ = np.arange(0,360) #, dtype='uint32')
				
	
	
	# Shifting beam relative to true AZ (referenced at due North)
	# Due to angle of orientation of excited antenna panels relative to due North
	# ---------------------------------------------------------
	print('AZ_antenna_axis = ' + str(AZ_antenna_axis) + ' deg')
	if AZ_antenna_axis < 0:
		AZ_index          = -AZ_antenna_axis
		bm1               = beam_maps[:,:,AZ_index::]
		bm2               = beam_maps[:,:,0:AZ_index]
		beam_maps_shifted = np.append(bm1, bm2, axis=2)

	elif AZ_antenna_axis > 0:
		AZ_index          = AZ_antenna_axis
		bm1               = beam_maps[:,:,0:(-AZ_index)]
		bm2               = beam_maps[:,:,(360-AZ_index)::]
		beam_maps_shifted = np.append(bm2, bm1, axis=2)

	elif AZ_antenna_axis == 0:
		beam_maps_shifted = np.copy(beam_maps)
	
	return frequencies, AZ, EL, beam_maps_shifted


