

import numpy as np
import scipy as sp
import scipy.io as sio

import matplotlib.pyplot as plt
import datetime as dt

import astropy.time as apt
import ephem as eph








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

		if i == 0:
			sum_fre = 0
			sum_num = 0
			sum_den = 0

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


		if i < (nsamples-1):
			i = i+1

		elif i == (nsamples-1):
			if flag_start == 0:
				av_fr = av_fr_temp
				av_sp = av_sp_temp
				av_we = sum_den
				flag_start = 1

			elif flag_start > 0:
				av_fr = np.append(av_fr, av_fr_temp)
				av_sp = np.append(av_sp, av_sp_temp)
				av_we = np.append(av_we, sum_den)

			i = 0


	return av_fr, av_sp, av_we












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





def calibrated_antenna_temperature(Tde, rd, rl, sca, off, TU, TC, TS, Tamb_internal=300):

	# S11 quantities
	Fd = np.sqrt( 1 - np.abs(rl) ** 2 ) / ( 1 - rd*rl )	
	PHId = np.angle( rd*Fd )
	G = 1 - np.abs(rl) ** 2
	K1d = (1 - np.abs(rd) **2) * np.abs(Fd) ** 2 / G
	K2d = (np.abs(rd) ** 2) * (np.abs(Fd) ** 2) / G
	K3d = (np.abs(rd) * np.abs(Fd) / G) * np.cos(PHId)
	K4d = (np.abs(rd) * np.abs(Fd) / G) * np.sin(PHId)


	# Applying scale and offset to raw spectrum
	Tde_corrected = (Tde - Tamb_internal)*sca + Tamb_internal - off

	# Noise wave contribution
	NWPd = TU*K2d + TC*K3d + TS*K4d

	# Antenna temperature
	Td = (Tde_corrected - NWPd) / K1d

	return Td





def uncalibrated_antenna_temperature(Td, rd, rl, sca, off, TU, TC, TS, Tamb_internal=300):

	# S11 quantities
	Fd = np.sqrt( 1 - np.abs(rl) ** 2 ) / ( 1 - rd*rl )
	PHId = np.angle( rd*Fd )
	G = 1 - np.abs(rl) ** 2
	K1d = (1 - np.abs(rd) **2) * np.abs(Fd) ** 2 / G
	K2d = (np.abs(rd) ** 2) * (np.abs(Fd) ** 2) / G
	K3d = (np.abs(rd) * np.abs(Fd) / G) * np.cos(PHId)
	K4d = (np.abs(rd) * np.abs(Fd) / G) * np.sin(PHId)	


	# Noise wave contribution
	NWPd = TU*K2d + TC*K3d + TS*K4d	

	# Scaled and offset spectrum 
	Tde_corrected = Td*K1d + NWPd

	# Removing scale and offset
	Tde = Tamb_internal + (Tde_corrected - Tamb_internal + off) / sca

	return Tde








