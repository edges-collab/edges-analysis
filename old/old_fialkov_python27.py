


# This file is for ipython for python 2.7
# $ /usr/bin/ipython


import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.interpolate as spi
import emcee as ec
import corner
import h5py

import matlab.engine

from os.path import expanduser









# Determining home folder
home_folder = expanduser("~")









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












def fialkov_MCMC(vb, db, nb, wb, save='no', save_filename='x', flow=90, fhigh=190, model_fg='EDGES_polynomial', Nfg=5, parameters_21cm=[-1,-1,-1,-1,-1,-1,-1], Nchain=10000, Nthreads=6, rejected_fraction=0.3):


	# 
	# EXAMPLE: 
	#
	# import numpy as np
	# d = np.genfromtxt('/home/ramo7131/DATA/EDGES/results/high_band/products/average_spectrum/high_band_2015_LST_0.26_6.26_dates_2015_250_299_nominal.txt')
	#
	#
	# P1_center  = np.log10(0.001) + (np.log10(0.5)-np.log10(0.001))/2        # star formation efficiency (fraction, e.g. 0.05)
	# P2_center  = np.log10(16.5) + (np.log10(76.5)-np.log10(16.5))/2         # minimum virial circular velocity (e.g. 16.5 km/s for atomic cooling) 
	# P3_center  = np.log10(0.00001) + (np.log10(10)-np.log10(0.00001))/2     # X-ray efficiency (fraction, e.g. 1) 
	# P4_center  = 0.055 + (0.1-0.055)/2                                      # CMB optical depth (e.g. 0.06) 
	# P5_center  = 1 + (1.5-1)/2                                              # slope of the X-ray spectrum (e.g. 1 for sources with spectrum that goes as nu^(-1) )
	# P6_center  = 0.1 + (3-0.1)/2                                            # low cut-off of the X-ray spectrum (e.g. 0.2 KeV)
	# P7_center  = 10 + (50-10)/2                                             # mean free path of ionizing photons (e.g. 30 Mpc)
	#
	# sc = fk.fialkov_MCMC(d[:,0], d[:,1], 0.03*np.ones(len(d[:,0])), d[:,2], save='no', save_filename='x', flow=90, fhigh=190, model_fg='EDGES_polynomial', Nfg=5, parameters_21cm=[-1, P2_center, -1, P4_center, P5_center, P6_center, P7_center], Nchain=5000, Nthreads=1, rejected_fraction=0.3)

	# Setting Up MCMC
	# -----------------------------------------------------
	# Cutting data within desired range	
	v = vb[(vb>=flow) & (vb<=fhigh)]
	d = db[(vb>=flow) & (vb<=fhigh)]
	n = nb[(vb>=flow) & (vb<=fhigh)]
	w = wb[(vb>=flow) & (vb<=fhigh)]
	
	# Counting number of 21cm parameters
	N21 = 0
	for i in range(len(parameters_21cm)):
		if parameters_21cm[i] == -1:
			N21 = N21 + 1
	
	# Number of parameters (dimensions) and walkers
	Ndim     = Nfg + N21
	Nwalkers = 2*Ndim + 2
	
	# Normalized frequency
	vn       = flow + (fhigh-flow)/2
	


	# Random start point of the chain
	p0 = (np.random.uniform(size=Ndim*Nwalkers).reshape((Nwalkers, Ndim)) - 0.5) / 10         # noise STD: ([0,1] - 0.5) / 10  =  [-0.05, 0.05]
	
	
	
	# Appropriate offset to the random starting point of some parameters
	# First are foreground model coefficients
	x        = fit_polynomial_fourier(model_fg, v/vn, d, Nfg, Weights=w)
	poly_par = x[0]	
	for i in range(Nfg):
		p0[:,i] = p0[:,i] + poly_par[i]
	
	# Then, are the 21cm parameters
	P1_center  = np.log10(0.001) + (np.log10(0.5)-np.log10(0.001))/2
	P2_center  = np.log10(16.5) + (np.log10(76.5)-np.log10(16.5))/2
	P3_center  = np.log10(0.00001) + (np.log10(10)-np.log10(0.00001))/2
	P4_center  = 0.055 + (0.1-0.055)/2
	P5_center  = 1 + (1.5-1)/2
	P6_center  = 0.1 + (3-0.1)/2
	P7_center  = 10 + (50-10)/2
	p21_center = [P1_center, P2_center, P3_center, P4_center, P5_center, P6_center, P7_center]
	
	j=-1
	for i in range(len(parameters_21cm)):
		if parameters_21cm[i] == -1:
			j = j+1
			p0[:,Nfg+j] = p0[:,Nfg+j] + p21_center[i]
			print p0[:,Nfg+j]
	
	
	# Starting MATLAB engine
	print '\nStarting 4 MATLAB engines ...'
	eng1 = matlab.engine.start_matlab()
	eng2 = matlab.engine.start_matlab()
	eng3 = matlab.engine.start_matlab()
	eng4 = matlab.engine.start_matlab()
		
	matlab_engine_list = [eng1, eng2, eng3, eng4]
	engine_flag_array  = np.zeros(len(matlab_engine_list))
		
		
	# Computing results
	# ------------------------------------------------------
	print '\nComputing MCMC ...'
	sampler = ec.EnsembleSampler(Nwalkers, Ndim, fialkov_log_likelihood, threads=Nthreads, args=(v[w>0], d[w>0], n[w>0], matlab_engine_list, engine_flag_array), kwargs={'model_fg':model_fg, 'Nfg':Nfg, 'vn':vn, 'parameters_21cm':parameters_21cm})  # 
	sampler.run_mcmc(p0, Nchain)
	
	
	# Only accepted samples
	samples_cut = sampler.chain[:, (np.int(rejected_fraction*Nchain)):, :].reshape((-1, Ndim))	
		
	return samples_cut









def fialkov_log_likelihood(theta, v, d, sigma_noise, matlab_engine_list, engine_flag_array, model_fg='EDGES_polynomial', Nfg=5, vn=140, parameters_21cm=[-1, -1, -1, -1, -1, -1, -1]):


	# Evaluating model foregrounds
	if Nfg == 0:
		Tfg = 0
		log_priors_fg = 0

	elif Nfg > 0:
		Tfg           = model_evaluate(model_fg, theta[0:Nfg], v/vn)
		log_priors_fg = fialkov_priors_foreground(theta[0:Nfg])



	# Evaluating model 21cm
	j = -1
	values_21cm = parameters_21cm[:]
	for i in range(len(parameters_21cm)):
		if parameters_21cm[i] == -1:
			j = j + 1
			values_21cm[i] = theta[Nfg + j]
	
							
	T21, Z, flags = fialkov_model_21cm(v, values_21cm, matlab_engine_list, engine_flag_array, interpolation_kind='linear')
	log_priors_21 = fialkov_priors_21cm(values_21cm, flags)



	# Full model
	Tfull = Tfg + T21

	# Log-likelihood
	log_likelihood =  -(1/2)  *  np.sum(  ((d-Tfull)/sigma_noise)**2  )


	# Log-priors + Log-likelihood
	return log_priors_fg + log_priors_21 + log_likelihood






def fialkov_priors_foreground(theta):
	
	# a0, a1, a2, a3 = theta

	# Counting the parameters with allowable values
	flag = 0
	for i in range(len(theta)):
		if (-1e5 <= theta[i] <= 1e5):
			flag = flag + 1
		
	# Assign likelihood
	if flag == len(theta):
		out = 0
	else:
		out = -1e10
			
	return out









def fialkov_priors_21cm(theta, flags):
	
	P1, P2, P3, P4, P5, P6, P7 = theta
	
	if (np.log10(0.001) <= P1 <= np.log10(0.5)) \
	and (np.log10(16.5) <= P2 <= np.log10(76.5)) \
	and (np.log10(0.00001) <= P3 <= np.log10(10)) \
	and (0.055 <= P4 <= 0.088) \
	and (1 <= P5 <= 1.5) \
	and (0.1 <= P6 <= 3) \
	and (10 <= P7 <= 50) \
	and (np.sum(flags) == 0):
		out = 0
		print 'Y'
	else:
		out = -1e10
		print 'X'
		
	
	#print theta
	#print flags
	
	return out
	
	
	
	



	
def fialkov_model_21cm(f_new, values_21cm, matlab_engine_list, engine_flag_array, interpolation_kind='linear'):
	
	"""
	2017-10-02
	Function that produces a 21cm model in MATLAB and sends it to Python
	
	Webpage: https://www.mathworks.com/help/matlab/matlab-engine-for-python.html
	"""
	
	# Copying parameter list
	all_lin_values_21cm    = values_21cm[:]
	
	# Transforming first three parameters from Log10 to Linear
	all_lin_values_21cm[0] = 10**(values_21cm[0])
	all_lin_values_21cm[1] = 10**(values_21cm[1])
	all_lin_values_21cm[2] = 10**(values_21cm[2])
	
	# Calling MATLAB function
	# matlab_engine = matlab.engine.start_matlab()
	#print all_lin_values_21cm
	
	
	index_flag_list = np.arange(len(engine_flag_array))
	index_engine = int(np.min(index_flag_list[engine_flag_array == 0]))	
	engine_flag_array[index_engine] = 1
	print 'MATLAB engine number: ' + str(index_engine)
	model_md, Z, flags_md = matlab_engine_list[index_engine].Global21cm.run(matlab.double(all_lin_values_21cm), nargout=3)
	engine_flag_array[index_engine] = 0
	
	
	
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
	
	
	# Interpolation
	func_model = spi.interp1d(f_raw, model_raw, kind=interpolation_kind)
	model_new  = func_model(f_new)	
	

	
	return model_new/1000, Z, flags



























def grid_sweep_amplitude_fit(N, save_filename):
	
	# Loading High-band data
	d = np.genfromtxt('/home/ramo7131/DATA/EDGES/results/high_band/products/average_spectrum/high_band_2015_LST_0.26_6.26_dates_2015_250_299_nominal.txt')
	
	# Starting MATLAB engine
	print '\nStarting MATLAB engine ...'
	matlab_engine = matlab.engine.start_matlab()
	
	# Evaluating and fitting 21cm models
	# N  = 3
	rnd_offset = 0 #0.000001
	P1 = np.linspace(np.log10(0.001+rnd_offset), np.log10(0.5-rnd_offset), N)
	P2 = np.linspace(np.log10(16.5+rnd_offset), np.log10(76.5-rnd_offset), N)
	P3 = np.linspace(np.log10(0.00001+rnd_offset), np.log10(10-rnd_offset), N)
	P4 = np.linspace(0.055+rnd_offset, 0.088-rnd_offset, N)
	P5 = np.linspace(1+rnd_offset, 1.5-rnd_offset, N)
	P6 = np.linspace(0.1+rnd_offset, 3-rnd_offset, N)
	P7 = np.linspace(10+rnd_offset, 50-rnd_offset, N)
	
	



	
	p21_array  = np.zeros((N,N,N,N,N,N,N))
	dp21_array = np.zeros((N,N,N,N,N,N,N))
	
	P1_array = np.zeros((N,N,N,N,N,N,N))
	P2_array = np.zeros((N,N,N,N,N,N,N))
	P3_array = np.zeros((N,N,N,N,N,N,N))
	P4_array = np.zeros((N,N,N,N,N,N,N))
	P5_array = np.zeros((N,N,N,N,N,N,N))
	P6_array = np.zeros((N,N,N,N,N,N,N))
	P7_array = np.zeros((N,N,N,N,N,N,N))
	
	flag1_array = np.ones((N,N,N,N,N,N,N))
	flag2_array = np.ones((N,N,N,N,N,N,N))
	flag3_array = np.ones((N,N,N,N,N,N,N))
	flag4_array = np.ones((N,N,N,N,N,N,N))
	
	
	model_number = 0
	for i1 in range(len(P1)):
		for i2 in range(len(P2)):
			for i3 in range(len(P3)):
				for i4 in range(len(P4)):
					for i5 in range(len(P5)):
						for i6 in range(len(P6)):
							for i7 in range(len(P7)):
																
								P1_array[i1, i2, i3, i4, i5, i6, i7]    = 10**P1[i1]
								P2_array[i1, i2, i3, i4, i5, i6, i7]    = 10**P2[i2]
								P3_array[i1, i2, i3, i4, i5, i6, i7]    = 10**P3[i3]
								P4_array[i1, i2, i3, i4, i5, i6, i7]    = P4[i4]
								P5_array[i1, i2, i3, i4, i5, i6, i7]    = P5[i5]
								P6_array[i1, i2, i3, i4, i5, i6, i7]    = P6[i6]
								P7_array[i1, i2, i3, i4, i5, i6, i7]    = P7[i7]
								
								values_21cm     = [P1[i1], P2[i2], P3[i3], P4[i4], P5[i5], P6[i6], P7[i7]]
								model, Z, flags = fialkov_model_21cm(d[:,0], values_21cm, matlab_engine, interpolation_kind='linear')
								
								flag1_array[i1, i2, i3, i4, i5, i6, i7] = flags[0]
								flag2_array[i1, i2, i3, i4, i5, i6, i7] = flags[1]
								flag3_array[i1, i2, i3, i4, i5, i6, i7] = flags[2]
								flag4_array[i1, i2, i3, i4, i5, i6, i7] = flags[3]							
								
								if flags[0] == 0:
																	
									Nfg  = 5
									par  = fit_polynomial_fourier('EDGES_polynomial_plus_external', d[:,0], d[:,1], Nfg, Weights=d[:,2], external_model_in_K=model)
									p21  = par[0][-1]
									dp21 = np.sqrt(np.diag(par[3])[-1])
									
									# Storing 21cm results
									p21_array[i1, i2, i3, i4, i5, i6, i7]   = p21
									dp21_array[i1, i2, i3, i4, i5, i6, i7]  = dp21									
									
									
																	
								model_number = model_number + 1
								print '-------------------------------\nModel ' + str(model_number) +  ' of ' + str(N**7)
								print str(values_21cm) + ' ' + str(flags)
								
								#+ ' ' + str( np.round( (1-p21)/((35./17.)*dp21), 2) ) + ' ' + str( np.round(p21/((35./17.)*dp21),2) ) + ' ' + str(np.round(1000*par[2][0,0],0)) + ' mK\n\n'
								
								

	# Saving data
	#if save == 'yes':		
	save_file = home_folder + '/DATA/EDGES/results/high_band/products/model_rejection/global_21cm_anastasia/lowres_rejection_significance/' + save_filename + '.hdf5'
	with h5py.File(save_file, 'w') as hf:

		hf.create_dataset('P1',    data = P1_array)
		hf.create_dataset('P2',    data = P2_array)
		hf.create_dataset('P3',    data = P3_array)
		hf.create_dataset('P4',    data = P4_array)
		hf.create_dataset('P5',    data = P5_array)
		hf.create_dataset('P6',    data = P6_array)
		hf.create_dataset('P7',    data = P7_array)
	
		hf.create_dataset('flag1', data = flag1_array)
		hf.create_dataset('flag2', data = flag2_array)
		hf.create_dataset('flag3', data = flag3_array)
		hf.create_dataset('flag4', data = flag4_array)
		
		hf.create_dataset('p21',   data = p21_array)
		hf.create_dataset('dp21',  data = dp21_array)


								
									
	return 0 #P1_array, P2_array, P3_array, P4_array, P5_array, P6_array, P7_array, flag1_array, flag2_array, flag3_array, flag4_array, p21_array, dp21_array







def read_rejection_significance(path_file):
	
	with h5py.File(path_file,'r') as hf:
		
		hfx    = hf.get('P1')
		P1     = np.array(hfx)
	
		hfx    = hf.get('P2')
		P2     = np.array(hfx)
		
		hfx    = hf.get('P3')
		P3     = np.array(hfx)

		hfx    = hf.get('P4')
		P4     = np.array(hfx)

		hfx    = hf.get('P5')
		P5     = np.array(hfx)

		hfx    = hf.get('P6')
		P6     = np.array(hfx)

		hfx    = hf.get('P7')
		P7     = np.array(hfx)



		hfx    = hf.get('flag1')
		flag1  = np.array(hfx)

		hfx    = hf.get('flag2')
		flag2  = np.array(hfx)
		
		hfx    = hf.get('flag3')
		flag3  = np.array(hfx)
		
		hfx    = hf.get('flag4')
		flag4  = np.array(hfx)

	
		hfx    = hf.get('p21')
		p21    = np.array(hfx)			

		hfx    = hf.get('dp21')
		dp21   = np.array(hfx)			

		
	return P1, P2, P3, P4, P5, P6, P7, flag1, flag2, flag3, flag4, p21, dp21









def plot_lowres_rejection_significance():
	
	P1, P2, P3, P4, P5, P6, P7, flag1, flag2, flag3, flag4, p21, dp21 = read_rejection_significance(home_folder + '/DATA/EDGES/results/high_band/products/model_rejection/global_21cm_anastasia/lowres_rejection_significance/lowres_rej_sig_N5.hdf5')

	dp21_new = (35/17)*dp21
	rs       = (1-p21)/dp21_new
	ds       = np.abs(p21/dp21_new)
	ic       = 2 # Index of cut


	cmap_name = 'inferno'
	min_rej_sig = 0
	max_rej_sig = 5	

	N    = 5
	PA1  = np.linspace(np.log10(0.001), np.log10(0.5), N)
	PA2  = np.linspace(np.log10(16.5),  np.log10(76.5), N)
	PA3  = np.linspace(np.log10(0.00001),  np.log10(10), N)
	PA4  = np.linspace(0.055, 0.088, N)
	PA5  = np.linspace(1, 1.5, N)
	PA6  = np.linspace(0.1, 3, N)
	PA7  = np.linspace(10, 50, N)	
	
	
	
	
	
	
	
	
	
	

	# Figure
	plt.close()
	plt.close()
	plt.close()
	plt.close()
	plt.close()
	plt.close()
	plt.close()
	plt.close()
	

	size_x = 12
	size_y = 12
	
	dx     = 0.15
	dy     = 0.15	
	x0     = 0.052
	y0     = 0.059
	x00    = 0.007
	y00    = 0.007




	# Rejection significance
	# ----------------------------------------------------------------------------------


	f1     = plt.figure(num=1, figsize=(size_x, size_y))
	
	ax17   = f1.add_axes([1*x0 + 0*(dx+x00), 1*y0 + 0*(dy+y00), dx, dy])
	ax16   = f1.add_axes([1*x0 + 0*(dx+x00), 1*y0 + 1*(dy+y00), dx, dy])
	ax15   = f1.add_axes([1*x0 + 0*(dx+x00), 1*y0 + 2*(dy+y00), dx, dy])
	ax14   = f1.add_axes([1*x0 + 0*(dx+x00), 1*y0 + 3*(dy+y00), dx, dy])
	ax13   = f1.add_axes([1*x0 + 0*(dx+x00), 1*y0 + 4*(dy+y00), dx, dy])
	ax12   = f1.add_axes([1*x0 + 0*(dx+x00), 1*y0 + 5*(dy+y00), dx, dy])
	
	ax27   = f1.add_axes([1*x0 + 1*(dx+x00), 1*y0 + 0*(dy+y00), dx, dy])
	ax26   = f1.add_axes([1*x0 + 1*(dx+x00), 1*y0 + 1*(dy+y00), dx, dy])
	ax25   = f1.add_axes([1*x0 + 1*(dx+x00), 1*y0 + 2*(dy+y00), dx, dy])
	ax24   = f1.add_axes([1*x0 + 1*(dx+x00), 1*y0 + 3*(dy+y00), dx, dy])
	ax23   = f1.add_axes([1*x0 + 1*(dx+x00), 1*y0 + 4*(dy+y00), dx, dy])

	ax37   = f1.add_axes([1*x0 + 2*(dx+x00), 1*y0 + 0*(dy+y00), dx, dy])
	ax36   = f1.add_axes([1*x0 + 2*(dx+x00), 1*y0 + 1*(dy+y00), dx, dy])
	ax35   = f1.add_axes([1*x0 + 2*(dx+x00), 1*y0 + 2*(dy+y00), dx, dy])
	ax34   = f1.add_axes([1*x0 + 2*(dx+x00), 1*y0 + 3*(dy+y00), dx, dy])
	
	ax47   = f1.add_axes([1*x0 + 3*(dx+x00), 1*y0 + 0*(dy+y00), dx, dy])
	ax46   = f1.add_axes([1*x0 + 3*(dx+x00), 1*y0 + 1*(dy+y00), dx, dy])
	ax45   = f1.add_axes([1*x0 + 3*(dx+x00), 1*y0 + 2*(dy+y00), dx, dy])	

	ax57   = f1.add_axes([1*x0 + 4*(dx+x00), 1*y0 + 0*(dy+y00), dx, dy])
	ax56   = f1.add_axes([1*x0 + 4*(dx+x00), 1*y0 + 1*(dy+y00), dx, dy])
	
	ax67   = f1.add_axes([1*x0 + 5*(dx+x00), 1*y0 + 0*(dy+y00), dx, dy])
	
	
	
	E12 = [PA1[0]-(PA1[1]-PA1[0])/2, PA1[4]+(PA1[1]-PA1[0])/2, PA2[0]-(PA2[1]-PA2[0])/2, PA2[4]+(PA2[1]-PA2[0])/2]
	E13 = [PA1[0]-(PA1[1]-PA1[0])/2, PA1[4]+(PA1[1]-PA1[0])/2, PA3[0]-(PA3[1]-PA3[0])/2, PA3[4]+(PA3[1]-PA3[0])/2]
	E14 = [PA1[0]-(PA1[1]-PA1[0])/2, PA1[4]+(PA1[1]-PA1[0])/2, PA4[0]-(PA4[1]-PA4[0])/2, PA4[4]+(PA4[1]-PA4[0])/2]
	E15 = [PA1[0]-(PA1[1]-PA1[0])/2, PA1[4]+(PA1[1]-PA1[0])/2, PA5[0]-(PA5[1]-PA5[0])/2, PA5[4]+(PA5[1]-PA5[0])/2]
	E16 = [PA1[0]-(PA1[1]-PA1[0])/2, PA1[4]+(PA1[1]-PA1[0])/2, PA6[0]-(PA6[1]-PA6[0])/2, PA6[4]+(PA6[1]-PA6[0])/2]
	E17 = [PA1[0]-(PA1[1]-PA1[0])/2, PA1[4]+(PA1[1]-PA1[0])/2, PA7[0]-(PA7[1]-PA7[0])/2, PA7[4]+(PA7[1]-PA7[0])/2]
	
	E23 = [PA2[0]-(PA2[1]-PA2[0])/2, PA2[4]+(PA2[1]-PA2[0])/2, PA3[0]-(PA3[1]-PA3[0])/2, PA3[4]+(PA3[1]-PA3[0])/2]
	E24 = [PA2[0]-(PA2[1]-PA2[0])/2, PA2[4]+(PA2[1]-PA2[0])/2, PA4[0]-(PA4[1]-PA4[0])/2, PA4[4]+(PA4[1]-PA4[0])/2]
	E25 = [PA2[0]-(PA2[1]-PA2[0])/2, PA2[4]+(PA2[1]-PA2[0])/2, PA5[0]-(PA5[1]-PA5[0])/2, PA5[4]+(PA5[1]-PA5[0])/2]
	E26 = [PA2[0]-(PA2[1]-PA2[0])/2, PA2[4]+(PA2[1]-PA2[0])/2, PA6[0]-(PA6[1]-PA6[0])/2, PA6[4]+(PA6[1]-PA6[0])/2]
	E27 = [PA2[0]-(PA2[1]-PA2[0])/2, PA2[4]+(PA2[1]-PA2[0])/2, PA7[0]-(PA7[1]-PA7[0])/2, PA7[4]+(PA7[1]-PA7[0])/2]

	E34 = [PA3[0]-(PA3[1]-PA3[0])/2, PA3[4]+(PA3[1]-PA3[0])/2, PA4[0]-(PA4[1]-PA4[0])/2, PA4[4]+(PA4[1]-PA4[0])/2]
	E35 = [PA3[0]-(PA3[1]-PA3[0])/2, PA3[4]+(PA3[1]-PA3[0])/2, PA5[0]-(PA5[1]-PA5[0])/2, PA5[4]+(PA5[1]-PA5[0])/2]
	E36 = [PA3[0]-(PA3[1]-PA3[0])/2, PA3[4]+(PA3[1]-PA3[0])/2, PA6[0]-(PA6[1]-PA6[0])/2, PA6[4]+(PA6[1]-PA6[0])/2]
	E37 = [PA3[0]-(PA3[1]-PA3[0])/2, PA3[4]+(PA3[1]-PA3[0])/2, PA7[0]-(PA7[1]-PA7[0])/2, PA7[4]+(PA7[1]-PA7[0])/2]
	
	E45 = [PA4[0]-(PA4[1]-PA4[0])/2, PA4[4]+(PA4[1]-PA4[0])/2, PA5[0]-(PA5[1]-PA5[0])/2, PA5[4]+(PA5[1]-PA5[0])/2]
	E46 = [PA4[0]-(PA4[1]-PA4[0])/2, PA4[4]+(PA4[1]-PA4[0])/2, PA6[0]-(PA6[1]-PA6[0])/2, PA6[4]+(PA6[1]-PA6[0])/2]
	E47 = [PA4[0]-(PA4[1]-PA4[0])/2, PA4[4]+(PA4[1]-PA4[0])/2, PA7[0]-(PA7[1]-PA7[0])/2, PA7[4]+(PA7[1]-PA7[0])/2]	
	
	E56 = [PA5[0]-(PA5[1]-PA5[0])/2, PA5[4]+(PA5[1]-PA5[0])/2, PA6[0]-(PA6[1]-PA6[0])/2, PA6[4]+(PA6[1]-PA6[0])/2]
	E57 = [PA5[0]-(PA5[1]-PA5[0])/2, PA5[4]+(PA5[1]-PA5[0])/2, PA7[0]-(PA7[1]-PA7[0])/2, PA7[4]+(PA7[1]-PA7[0])/2]	

	E67 = [PA6[0]-(PA6[1]-PA6[0])/2, PA6[4]+(PA6[1]-PA6[0])/2, PA7[0]-(PA7[1]-PA7[0])/2, PA7[4]+(PA7[1]-PA7[0])/2]
	
	

	# Column 1
	ax12.imshow(np.flipud(rs[:,:,ic,ic,ic,ic,ic]), interpolation='none', aspect=(PA1[4]-PA1[0])/(PA2[4]-PA2[0]), extent=E12, vmin=min_rej_sig, vmax=max_rej_sig, cmap=plt.get_cmap(cmap_name))
	ax12.set_xticks(PA1)
	ax12.set_xticklabels('')
	ax12.set_yticks(PA2)
	ax12.set_yticklabels(np.round(PA2,2), rotation=45)
	ax12.set_ylabel('log10(m.v.c.v. [km/s])', fontsize=9)	
		
	ax13.imshow(np.flipud(rs[:,ic,:,ic,ic,ic,ic]), interpolation='none', aspect=(PA1[4]-PA1[0])/(PA3[4]-PA3[0]), extent=E13, vmin=min_rej_sig, vmax=max_rej_sig, cmap=plt.get_cmap(cmap_name))
	ax13.set_xticks(PA1)
	ax13.set_xticklabels('')
	ax13.set_yticks(PA3)
	ax13.set_yticklabels(np.round(PA3,2), rotation=45)
	ax13.set_ylabel('log10(x.r.e)', fontsize=9)	
		
	ax14.imshow(np.flipud(rs[:,ic,ic,:,ic,ic,ic]), interpolation='none', aspect=(PA1[4]-PA1[0])/(PA4[4]-PA4[0]), extent=E14, vmin=min_rej_sig, vmax=max_rej_sig, cmap=plt.get_cmap(cmap_name))
	ax14.set_xticks(PA1)
	ax14.set_xticklabels('')
	ax14.set_yticks(PA4)
	ax14.set_yticklabels(np.round(PA4,2), rotation=45)
	ax14.set_ylabel('tau', fontsize=9)		
		
	ax15.imshow(np.flipud(rs[:,ic,ic,ic,:,ic,ic]), interpolation='none', aspect=(PA1[4]-PA1[0])/(PA5[4]-PA5[0]), extent=E15, vmin=min_rej_sig, vmax=max_rej_sig, cmap=plt.get_cmap(cmap_name))
	ax15.set_xticks(PA1)
	ax15.set_xticklabels('')
	ax15.set_yticks(PA5)
	ax15.set_yticklabels(np.round(PA5,2), rotation=45)
	ax15.set_ylabel('slope x-ray', fontsize=9)
		
	ax16.imshow(np.flipud(rs[:,ic,ic,ic,ic,:,ic]), interpolation='none', aspect=(PA1[4]-PA1[0])/(PA6[4]-PA6[0]), extent=E16, vmin=min_rej_sig, vmax=max_rej_sig, cmap=plt.get_cmap(cmap_name))
	ax16.set_xticks(PA1)
	ax16.set_xticklabels('')
	ax16.set_yticks(PA6)
	ax16.set_yticklabels(np.round(PA6,2), rotation=45)
	ax16.set_ylabel('low cut-off x-ray [KeV]', fontsize=9)
		
	ax17.imshow(np.flipud(rs[:,ic,ic,ic,ic,ic,:]), interpolation='none', aspect=(PA1[4]-PA1[0])/(PA7[4]-PA7[0]), extent=E17, vmin=min_rej_sig, vmax=max_rej_sig, cmap=plt.get_cmap(cmap_name))
	ax17.set_xticks(PA1)
	ax17.set_xticklabels(np.round(PA1,2), rotation=45)
	ax17.set_yticks(PA7)
	ax17.set_yticklabels(np.round(PA7,2), rotation=45)
	ax17.set_xlabel('log10(s.f.e)', fontsize=9)
	ax17.set_ylabel('m.f.p. [Mpc]', fontsize=9)
	






	# Column 2		
	ax23.imshow(np.flipud(rs[ic,:,:,ic,ic,ic,ic]), interpolation='none', aspect=(PA2[4]-PA2[0])/(PA3[4]-PA3[0]), extent=E23, vmin=min_rej_sig, vmax=max_rej_sig, cmap=plt.get_cmap(cmap_name))
	ax23.set_xticks(PA2)
	ax23.set_xticklabels('')
	ax23.set_yticks(PA3)
	ax23.set_yticklabels('')
		
	ax24.imshow(np.flipud(rs[ic,:,ic,:,ic,ic,ic]), interpolation='none', aspect=(PA2[4]-PA2[0])/(PA4[4]-PA4[0]), extent=E24, vmin=min_rej_sig, vmax=max_rej_sig, cmap=plt.get_cmap(cmap_name))
	ax24.set_xticks(PA2)
	ax24.set_xticklabels('')
	ax24.set_yticks(PA4)
	ax24.set_yticklabels('')	
		
	ax25.imshow(np.flipud(rs[ic,:,ic,ic,:,ic,ic]), interpolation='none', aspect=(PA2[4]-PA2[0])/(PA5[4]-PA5[0]), extent=E25, vmin=min_rej_sig, vmax=max_rej_sig, cmap=plt.get_cmap(cmap_name))
	ax25.set_xticks(PA2)
	ax25.set_xticklabels('')
	ax25.set_yticks(PA5)
	ax25.set_yticklabels('')
		
	ax26.imshow(np.flipud(rs[ic,:,ic,ic,ic,:,ic]), interpolation='none', aspect=(PA2[4]-PA2[0])/(PA6[4]-PA6[0]), extent=E26, vmin=min_rej_sig, vmax=max_rej_sig, cmap=plt.get_cmap(cmap_name))
	ax26.set_xticks(PA2)
	ax26.set_xticklabels('')
	ax26.set_yticks(PA6)
	ax26.set_yticklabels('')
		
	ax27.imshow(np.flipud(rs[ic,:,ic,ic,ic,ic,:]), interpolation='none', aspect=(PA2[4]-PA2[0])/(PA7[4]-PA7[0]), extent=E27, vmin=min_rej_sig, vmax=max_rej_sig, cmap=plt.get_cmap(cmap_name))
	ax27.set_xticks(PA2)
	ax27.set_xticklabels(np.round(PA2,2), rotation=45)
	ax27.set_yticks(PA7)
	ax27.set_yticklabels('')
	ax27.set_xlabel('log10(m.v.c.v. [km/s])', fontsize=9)
	




	# Column 3		
	ax34.imshow(np.flipud(rs[ic,ic,:,:,ic,ic,ic]), interpolation='none', aspect=(PA3[4]-PA3[0])/(PA4[4]-PA4[0]), extent=E34, vmin=min_rej_sig, vmax=max_rej_sig, cmap=plt.get_cmap(cmap_name))
	ax34.set_xticks(PA3)
	ax34.set_xticklabels('')
	ax34.set_yticks(PA4)
	ax34.set_yticklabels('') 		
		
	ax35.imshow(np.flipud(rs[ic,ic,:,ic,:,ic,ic]), interpolation='none', aspect=(PA3[4]-PA3[0])/(PA5[4]-PA5[0]), extent=E35, vmin=min_rej_sig, vmax=max_rej_sig, cmap=plt.get_cmap(cmap_name))
	ax35.set_xticks(PA3)
	ax35.set_xticklabels('')
	ax35.set_yticks(PA5)
	ax35.set_yticklabels('')
		
	ax36.imshow(np.flipud(rs[ic,ic,:,ic,ic,:,ic]), interpolation='none', aspect=(PA3[4]-PA3[0])/(PA6[4]-PA6[0]), extent=E36, vmin=min_rej_sig, vmax=max_rej_sig, cmap=plt.get_cmap(cmap_name))
	ax36.set_xticks(PA3)
	ax36.set_xticklabels('')
	ax36.set_yticks(PA6)
	ax36.set_yticklabels('')
		
	ax37.imshow(np.flipud(rs[ic,ic,:,ic,ic,ic,:]), interpolation='none', aspect=(PA3[4]-PA3[0])/(PA7[4]-PA7[0]), extent=E37, vmin=min_rej_sig, vmax=max_rej_sig, cmap=plt.get_cmap(cmap_name))
	ax37.set_xticks(PA3)
	ax37.set_xticklabels(np.round(PA3,2), rotation=45)
	ax37.set_yticks(PA7)
	ax37.set_yticklabels('') 
	ax37.set_xlabel('log10(x.r.e)', fontsize=9)
	
	
	
	
	
	# Column 4
	ax45.imshow(np.flipud(rs[ic,ic,ic,:,:,ic,ic]), interpolation='none', aspect=(PA4[4]-PA4[0])/(PA5[4]-PA5[0]), extent=E45, vmin=min_rej_sig, vmax=max_rej_sig, cmap=plt.get_cmap(cmap_name))
	ax45.set_xticks(PA4)
	ax45.set_xticklabels('')
	ax45.set_yticks(PA5)
	ax45.set_yticklabels('')
		
	ax46.imshow(np.flipud(rs[ic,ic,ic,:,ic,:,ic]), interpolation='none', aspect=(PA4[4]-PA4[0])/(PA6[4]-PA6[0]), extent=E46, vmin=min_rej_sig, vmax=max_rej_sig, cmap=plt.get_cmap(cmap_name))
	ax46.set_xticks(PA4)
	ax46.set_xticklabels('')
	ax46.set_yticks(PA6)
	ax46.set_yticklabels('')
		
	ax47.imshow(np.flipud(rs[ic,ic,ic,:,ic,ic,:]), interpolation='none', aspect=(PA4[4]-PA4[0])/(PA7[4]-PA7[0]), extent=E47, vmin=min_rej_sig, vmax=max_rej_sig, cmap=plt.get_cmap(cmap_name))
	ax47.set_xticks(PA4)
	ax47.set_xticklabels(np.round(PA4,2), rotation=45)
	ax47.set_yticks(PA7)
	ax47.set_yticklabels('') 
	ax47.set_xlabel('tau', fontsize=9)
		
	
	
	
	
	# Column 5
	ax56.imshow(np.flipud(rs[ic,ic,ic,ic,:,:,ic]), interpolation='none', aspect=(PA5[4]-PA5[0])/(PA6[4]-PA6[0]), extent=E56, vmin=min_rej_sig, vmax=max_rej_sig, cmap=plt.get_cmap(cmap_name))
	ax56.set_xticks(PA5)
	ax56.set_xticklabels('')
	ax56.set_yticks(PA6)
	ax56.set_yticklabels('')
		
	ax57.imshow(np.flipud(rs[ic,ic,ic,ic,:,ic,:]), interpolation='none', aspect=(PA5[4]-PA5[0])/(PA7[4]-PA7[0]), extent=E57, vmin=min_rej_sig, vmax=max_rej_sig, cmap=plt.get_cmap(cmap_name))
	ax57.set_xticks(PA5)
	ax57.set_xticklabels(np.round(PA5,2), rotation=45)
	ax57.set_yticks(PA7)
	ax57.set_yticklabels('') 
	ax57.set_xlabel('slope x-ray', fontsize=9)	
	
	
	# Column 6
	IMAGE_MAP = ax67.imshow(np.flipud(rs[ic,ic,ic,ic,ic,:,:]), interpolation='none', aspect=(PA6[4]-PA6[0])/(PA7[4]-PA7[0]), extent=E67, vmin=min_rej_sig, vmax=max_rej_sig, cmap=plt.get_cmap(cmap_name))
	ax67.set_xticks(PA6)
	ax67.set_xticklabels(np.round(PA6,2), rotation=45)
	ax67.set_yticks(PA7)
	ax67.set_yticklabels('') 
	ax67.set_xlabel('low cut-off x-ray [KeV]', fontsize=9)
	
	
	
	cb_x0 = 0.6
	cb_y0 = 0.8
	cb_dx = 0.2
	cb_dy = 0.01
	cax1 = f1.add_axes([cb_x0, cb_y0, cb_dx, cb_dy])
	cbar = plt.colorbar(IMAGE_MAP, cax=cax1, ticks=np.arange(min_rej_sig, max_rej_sig+0.1, 1), orientation="horizontal")
	#cbar.ax.set_xticklabels([0,1,2,3,4], fontsize=10)
	cbar.ax.set_xlabel(r'rejection significance [$\hat{\sigma}_{21}$]', fontsize=12)
	cbar.ax.tick_params(labelsize=12)

	
	path_plot_save = home_folder + '/DATA/EDGES/results/plots/20171005/'
	plt.savefig(path_plot_save + 'lowres_cut_rejection_significance.png', bbox_inches='tight')
	plt.close()
	plt.close()






	# Consistency with zero
	# ----------------------------------------------------------------------------------


	f2     = plt.figure(num=2, figsize=(size_x, size_y))
	
	ax17   = f2.add_axes([1*x0 + 0*(dx+x00), 1*y0 + 0*(dy+y00), dx, dy])
	ax16   = f2.add_axes([1*x0 + 0*(dx+x00), 1*y0 + 1*(dy+y00), dx, dy])
	ax15   = f2.add_axes([1*x0 + 0*(dx+x00), 1*y0 + 2*(dy+y00), dx, dy])
	ax14   = f2.add_axes([1*x0 + 0*(dx+x00), 1*y0 + 3*(dy+y00), dx, dy])
	ax13   = f2.add_axes([1*x0 + 0*(dx+x00), 1*y0 + 4*(dy+y00), dx, dy])
	ax12   = f2.add_axes([1*x0 + 0*(dx+x00), 1*y0 + 5*(dy+y00), dx, dy])
	
	ax27   = f2.add_axes([1*x0 + 1*(dx+x00), 1*y0 + 0*(dy+y00), dx, dy])
	ax26   = f2.add_axes([1*x0 + 1*(dx+x00), 1*y0 + 1*(dy+y00), dx, dy])
	ax25   = f2.add_axes([1*x0 + 1*(dx+x00), 1*y0 + 2*(dy+y00), dx, dy])
	ax24   = f2.add_axes([1*x0 + 1*(dx+x00), 1*y0 + 3*(dy+y00), dx, dy])
	ax23   = f2.add_axes([1*x0 + 1*(dx+x00), 1*y0 + 4*(dy+y00), dx, dy])

	ax37   = f2.add_axes([1*x0 + 2*(dx+x00), 1*y0 + 0*(dy+y00), dx, dy])
	ax36   = f2.add_axes([1*x0 + 2*(dx+x00), 1*y0 + 1*(dy+y00), dx, dy])
	ax35   = f2.add_axes([1*x0 + 2*(dx+x00), 1*y0 + 2*(dy+y00), dx, dy])
	ax34   = f2.add_axes([1*x0 + 2*(dx+x00), 1*y0 + 3*(dy+y00), dx, dy])
	
	ax47   = f2.add_axes([1*x0 + 3*(dx+x00), 1*y0 + 0*(dy+y00), dx, dy])
	ax46   = f2.add_axes([1*x0 + 3*(dx+x00), 1*y0 + 1*(dy+y00), dx, dy])
	ax45   = f2.add_axes([1*x0 + 3*(dx+x00), 1*y0 + 2*(dy+y00), dx, dy])	

	ax57   = f2.add_axes([1*x0 + 4*(dx+x00), 1*y0 + 0*(dy+y00), dx, dy])
	ax56   = f2.add_axes([1*x0 + 4*(dx+x00), 1*y0 + 1*(dy+y00), dx, dy])
	
	ax67   = f2.add_axes([1*x0 + 5*(dx+x00), 1*y0 + 0*(dy+y00), dx, dy])
	
	
	
	E12 = [PA1[0]-(PA1[1]-PA1[0])/2, PA1[4]+(PA1[1]-PA1[0])/2, PA2[0]-(PA2[1]-PA2[0])/2, PA2[4]+(PA2[1]-PA2[0])/2]
	E13 = [PA1[0]-(PA1[1]-PA1[0])/2, PA1[4]+(PA1[1]-PA1[0])/2, PA3[0]-(PA3[1]-PA3[0])/2, PA3[4]+(PA3[1]-PA3[0])/2]
	E14 = [PA1[0]-(PA1[1]-PA1[0])/2, PA1[4]+(PA1[1]-PA1[0])/2, PA4[0]-(PA4[1]-PA4[0])/2, PA4[4]+(PA4[1]-PA4[0])/2]
	E15 = [PA1[0]-(PA1[1]-PA1[0])/2, PA1[4]+(PA1[1]-PA1[0])/2, PA5[0]-(PA5[1]-PA5[0])/2, PA5[4]+(PA5[1]-PA5[0])/2]
	E16 = [PA1[0]-(PA1[1]-PA1[0])/2, PA1[4]+(PA1[1]-PA1[0])/2, PA6[0]-(PA6[1]-PA6[0])/2, PA6[4]+(PA6[1]-PA6[0])/2]
	E17 = [PA1[0]-(PA1[1]-PA1[0])/2, PA1[4]+(PA1[1]-PA1[0])/2, PA7[0]-(PA7[1]-PA7[0])/2, PA7[4]+(PA7[1]-PA7[0])/2]
	
	E23 = [PA2[0]-(PA2[1]-PA2[0])/2, PA2[4]+(PA2[1]-PA2[0])/2, PA3[0]-(PA3[1]-PA3[0])/2, PA3[4]+(PA3[1]-PA3[0])/2]
	E24 = [PA2[0]-(PA2[1]-PA2[0])/2, PA2[4]+(PA2[1]-PA2[0])/2, PA4[0]-(PA4[1]-PA4[0])/2, PA4[4]+(PA4[1]-PA4[0])/2]
	E25 = [PA2[0]-(PA2[1]-PA2[0])/2, PA2[4]+(PA2[1]-PA2[0])/2, PA5[0]-(PA5[1]-PA5[0])/2, PA5[4]+(PA5[1]-PA5[0])/2]
	E26 = [PA2[0]-(PA2[1]-PA2[0])/2, PA2[4]+(PA2[1]-PA2[0])/2, PA6[0]-(PA6[1]-PA6[0])/2, PA6[4]+(PA6[1]-PA6[0])/2]
	E27 = [PA2[0]-(PA2[1]-PA2[0])/2, PA2[4]+(PA2[1]-PA2[0])/2, PA7[0]-(PA7[1]-PA7[0])/2, PA7[4]+(PA7[1]-PA7[0])/2]

	E34 = [PA3[0]-(PA3[1]-PA3[0])/2, PA3[4]+(PA3[1]-PA3[0])/2, PA4[0]-(PA4[1]-PA4[0])/2, PA4[4]+(PA4[1]-PA4[0])/2]
	E35 = [PA3[0]-(PA3[1]-PA3[0])/2, PA3[4]+(PA3[1]-PA3[0])/2, PA5[0]-(PA5[1]-PA5[0])/2, PA5[4]+(PA5[1]-PA5[0])/2]
	E36 = [PA3[0]-(PA3[1]-PA3[0])/2, PA3[4]+(PA3[1]-PA3[0])/2, PA6[0]-(PA6[1]-PA6[0])/2, PA6[4]+(PA6[1]-PA6[0])/2]
	E37 = [PA3[0]-(PA3[1]-PA3[0])/2, PA3[4]+(PA3[1]-PA3[0])/2, PA7[0]-(PA7[1]-PA7[0])/2, PA7[4]+(PA7[1]-PA7[0])/2]
	
	E45 = [PA4[0]-(PA4[1]-PA4[0])/2, PA4[4]+(PA4[1]-PA4[0])/2, PA5[0]-(PA5[1]-PA5[0])/2, PA5[4]+(PA5[1]-PA5[0])/2]
	E46 = [PA4[0]-(PA4[1]-PA4[0])/2, PA4[4]+(PA4[1]-PA4[0])/2, PA6[0]-(PA6[1]-PA6[0])/2, PA6[4]+(PA6[1]-PA6[0])/2]
	E47 = [PA4[0]-(PA4[1]-PA4[0])/2, PA4[4]+(PA4[1]-PA4[0])/2, PA7[0]-(PA7[1]-PA7[0])/2, PA7[4]+(PA7[1]-PA7[0])/2]	
	
	E56 = [PA5[0]-(PA5[1]-PA5[0])/2, PA5[4]+(PA5[1]-PA5[0])/2, PA6[0]-(PA6[1]-PA6[0])/2, PA6[4]+(PA6[1]-PA6[0])/2]
	E57 = [PA5[0]-(PA5[1]-PA5[0])/2, PA5[4]+(PA5[1]-PA5[0])/2, PA7[0]-(PA7[1]-PA7[0])/2, PA7[4]+(PA7[1]-PA7[0])/2]	

	E67 = [PA6[0]-(PA6[1]-PA6[0])/2, PA6[4]+(PA6[1]-PA6[0])/2, PA7[0]-(PA7[1]-PA7[0])/2, PA7[4]+(PA7[1]-PA7[0])/2]
	
	

	# Column 1
	ax12.imshow(np.flipud(ds[:,:,ic,ic,ic,ic,ic]), interpolation='none', aspect=(PA1[4]-PA1[0])/(PA2[4]-PA2[0]), extent=E12, vmin=min_rej_sig, vmax=max_rej_sig, cmap=plt.get_cmap(cmap_name))
	ax12.set_xticks(PA1)
	ax12.set_xticklabels('')
	ax12.set_yticks(PA2)
	ax12.set_yticklabels(np.round(PA2,2), rotation=45)
	ax12.set_ylabel('log10(m.v.c.v. [km/s])', fontsize=9)
		
	ax13.imshow(np.flipud(ds[:,ic,:,ic,ic,ic,ic]), interpolation='none', aspect=(PA1[4]-PA1[0])/(PA3[4]-PA3[0]), extent=E13, vmin=min_rej_sig, vmax=max_rej_sig, cmap=plt.get_cmap(cmap_name))
	ax13.set_xticks(PA1)
	ax13.set_xticklabels('')
	ax13.set_yticks(PA3)
	ax13.set_yticklabels(np.round(PA3,2), rotation=45)
	ax13.set_ylabel('log10(x.r.e)', fontsize=9)	
		
	ax14.imshow(np.flipud(ds[:,ic,ic,:,ic,ic,ic]), interpolation='none', aspect=(PA1[4]-PA1[0])/(PA4[4]-PA4[0]), extent=E14, vmin=min_rej_sig, vmax=max_rej_sig, cmap=plt.get_cmap(cmap_name))
	ax14.set_xticks(PA1)
	ax14.set_xticklabels('')
	ax14.set_yticks(PA4)
	ax14.set_yticklabels(np.round(PA4,2), rotation=45)
	ax14.set_ylabel('tau', fontsize=9)		
		
	ax15.imshow(np.flipud(ds[:,ic,ic,ic,:,ic,ic]), interpolation='none', aspect=(PA1[4]-PA1[0])/(PA5[4]-PA5[0]), extent=E15, vmin=min_rej_sig, vmax=max_rej_sig, cmap=plt.get_cmap(cmap_name))
	ax15.set_xticks(PA1)
	ax15.set_xticklabels('')
	ax15.set_yticks(PA5)
	ax15.set_yticklabels(np.round(PA5,2), rotation=45)
	ax15.set_ylabel('slope x-ray', fontsize=9)
		
	ax16.imshow(np.flipud(ds[:,ic,ic,ic,ic,:,ic]), interpolation='none', aspect=(PA1[4]-PA1[0])/(PA6[4]-PA6[0]), extent=E16, vmin=min_rej_sig, vmax=max_rej_sig, cmap=plt.get_cmap(cmap_name))
	ax16.set_xticks(PA1)
	ax16.set_xticklabels('')
	ax16.set_yticks(PA6)
	ax16.set_yticklabels(np.round(PA6,2), rotation=45)
	ax16.set_ylabel('low cut-off x-ray [KeV]', fontsize=9)
		
	ax17.imshow(np.flipud(ds[:,ic,ic,ic,ic,ic,:]), interpolation='none', aspect=(PA1[4]-PA1[0])/(PA7[4]-PA7[0]), extent=E17, vmin=min_rej_sig, vmax=max_rej_sig, cmap=plt.get_cmap(cmap_name))
	ax17.set_xticks(PA1)
	ax17.set_xticklabels(np.round(PA1,2), rotation=45)
	ax17.set_yticks(PA7)
	ax17.set_yticklabels(np.round(PA7,2), rotation=45)
	ax17.set_xlabel('log10(s.f.e)', fontsize=9)
	ax17.set_ylabel('m.f.p. [Mpc]', fontsize=9)
	






	# Column 2		
	ax23.imshow(np.flipud(ds[ic,:,:,ic,ic,ic,ic]), interpolation='none', aspect=(PA2[4]-PA2[0])/(PA3[4]-PA3[0]), extent=E23, vmin=min_rej_sig, vmax=max_rej_sig, cmap=plt.get_cmap(cmap_name))
	ax23.set_xticks(PA2)
	ax23.set_xticklabels('')
	ax23.set_yticks(PA3)
	ax23.set_yticklabels('')
		
	ax24.imshow(np.flipud(ds[ic,:,ic,:,ic,ic,ic]), interpolation='none', aspect=(PA2[4]-PA2[0])/(PA4[4]-PA4[0]), extent=E24, vmin=min_rej_sig, vmax=max_rej_sig, cmap=plt.get_cmap(cmap_name))
	ax24.set_xticks(PA2)
	ax24.set_xticklabels('')
	ax24.set_yticks(PA4)
	ax24.set_yticklabels('')	
		
	ax25.imshow(np.flipud(ds[ic,:,ic,ic,:,ic,ic]), interpolation='none', aspect=(PA2[4]-PA2[0])/(PA5[4]-PA5[0]), extent=E25, vmin=min_rej_sig, vmax=max_rej_sig, cmap=plt.get_cmap(cmap_name))
	ax25.set_xticks(PA2)
	ax25.set_xticklabels('')
	ax25.set_yticks(PA5)
	ax25.set_yticklabels('')
		
	ax26.imshow(np.flipud(ds[ic,:,ic,ic,ic,:,ic]), interpolation='none', aspect=(PA2[4]-PA2[0])/(PA6[4]-PA6[0]), extent=E26, vmin=min_rej_sig, vmax=max_rej_sig, cmap=plt.get_cmap(cmap_name))
	ax26.set_xticks(PA2)
	ax26.set_xticklabels('')
	ax26.set_yticks(PA6)
	ax26.set_yticklabels('')
		
	ax27.imshow(np.flipud(ds[ic,:,ic,ic,ic,ic,:]), interpolation='none', aspect=(PA2[4]-PA2[0])/(PA7[4]-PA7[0]), extent=E27, vmin=min_rej_sig, vmax=max_rej_sig, cmap=plt.get_cmap(cmap_name))
	ax27.set_xticks(PA2)
	ax27.set_xticklabels(np.round(PA2,2), rotation=45)
	ax27.set_yticks(PA7)
	ax27.set_yticklabels('')
	ax27.set_xlabel('log10(m.v.c.v. [km/s])', fontsize=9)
	




	# Column 3		
	ax34.imshow(np.flipud(ds[ic,ic,:,:,ic,ic,ic]), interpolation='none', aspect=(PA3[4]-PA3[0])/(PA4[4]-PA4[0]), extent=E34, vmin=min_rej_sig, vmax=max_rej_sig, cmap=plt.get_cmap(cmap_name))
	ax34.set_xticks(PA3)
	ax34.set_xticklabels('')
	ax34.set_yticks(PA4)
	ax34.set_yticklabels('') 		
		
	ax35.imshow(np.flipud(ds[ic,ic,:,ic,:,ic,ic]), interpolation='none', aspect=(PA3[4]-PA3[0])/(PA5[4]-PA5[0]), extent=E35, vmin=min_rej_sig, vmax=max_rej_sig, cmap=plt.get_cmap(cmap_name))
	ax35.set_xticks(PA3)
	ax35.set_xticklabels('')
	ax35.set_yticks(PA5)
	ax35.set_yticklabels('')
		
	ax36.imshow(np.flipud(ds[ic,ic,:,ic,ic,:,ic]), interpolation='none', aspect=(PA3[4]-PA3[0])/(PA6[4]-PA6[0]), extent=E36, vmin=min_rej_sig, vmax=max_rej_sig, cmap=plt.get_cmap(cmap_name))
	ax36.set_xticks(PA3)
	ax36.set_xticklabels('')
	ax36.set_yticks(PA6)
	ax36.set_yticklabels('')
		
	ax37.imshow(np.flipud(ds[ic,ic,:,ic,ic,ic,:]), interpolation='none', aspect=(PA3[4]-PA3[0])/(PA7[4]-PA7[0]), extent=E37, vmin=min_rej_sig, vmax=max_rej_sig, cmap=plt.get_cmap(cmap_name))
	ax37.set_xticks(PA3)
	ax37.set_xticklabels(np.round(PA3,2), rotation=45)
	ax37.set_yticks(PA7)
	ax37.set_yticklabels('') 
	ax37.set_xlabel('log10(x.r.e)', fontsize=9)
	
	
	
	
	
	# Column 4
	ax45.imshow(np.flipud(ds[ic,ic,ic,:,:,ic,ic]), interpolation='none', aspect=(PA4[4]-PA4[0])/(PA5[4]-PA5[0]), extent=E45, vmin=min_rej_sig, vmax=max_rej_sig, cmap=plt.get_cmap(cmap_name))
	ax45.set_xticks(PA4)
	ax45.set_xticklabels('')
	ax45.set_yticks(PA5)
	ax45.set_yticklabels('')
		
	ax46.imshow(np.flipud(ds[ic,ic,ic,:,ic,:,ic]), interpolation='none', aspect=(PA4[4]-PA4[0])/(PA6[4]-PA6[0]), extent=E46, vmin=min_rej_sig, vmax=max_rej_sig, cmap=plt.get_cmap(cmap_name))
	ax46.set_xticks(PA4)
	ax46.set_xticklabels('')
	ax46.set_yticks(PA6)
	ax46.set_yticklabels('')
		
	ax47.imshow(np.flipud(ds[ic,ic,ic,:,ic,ic,:]), interpolation='none', aspect=(PA4[4]-PA4[0])/(PA7[4]-PA7[0]), extent=E47, vmin=min_rej_sig, vmax=max_rej_sig, cmap=plt.get_cmap(cmap_name))
	ax47.set_xticks(PA4)
	ax47.set_xticklabels(np.round(PA4,2), rotation=45)
	ax47.set_yticks(PA7)
	ax47.set_yticklabels('') 
	ax47.set_xlabel('tau', fontsize=9)
		
	
	
	
	
	# Column 5
	ax56.imshow(np.flipud(ds[ic,ic,ic,ic,:,:,ic]), interpolation='none', aspect=(PA5[4]-PA5[0])/(PA6[4]-PA6[0]), extent=E56, vmin=min_rej_sig, vmax=max_rej_sig, cmap=plt.get_cmap(cmap_name))
	ax56.set_xticks(PA5)
	ax56.set_xticklabels('')
	ax56.set_yticks(PA6)
	ax56.set_yticklabels('')
		
	ax57.imshow(np.flipud(ds[ic,ic,ic,ic,:,ic,:]), interpolation='none', aspect=(PA5[4]-PA5[0])/(PA7[4]-PA7[0]), extent=E57, vmin=min_rej_sig, vmax=max_rej_sig, cmap=plt.get_cmap(cmap_name))
	ax57.set_xticks(PA5)
	ax57.set_xticklabels(np.round(PA5,2), rotation=45)
	ax57.set_yticks(PA7)
	ax57.set_yticklabels('') 
	ax57.set_xlabel('slope x-ray', fontsize=9)	
	
	
	
	
	
	# Column 6
	IMAGE_MAP = ax67.imshow(np.flipud(ds[ic,ic,ic,ic,ic,:,:]), interpolation='none', aspect=(PA6[4]-PA6[0])/(PA7[4]-PA7[0]), extent=E67, vmin=min_rej_sig, vmax=max_rej_sig, cmap=plt.get_cmap(cmap_name))
	ax67.set_xticks(PA6)
	ax67.set_xticklabels(np.round(PA6,2), rotation=45)
	ax67.set_yticks(PA7)
	ax67.set_yticklabels('') 
	ax67.set_xlabel('low cut-off x-ray [KeV]', fontsize=9)
	
	
	
	
	
	cb_x0 = 0.6
	cb_y0 = 0.8
	cb_dx = 0.2
	cb_dy = 0.01
	cax1 = f2.add_axes([cb_x0, cb_y0, cb_dx, cb_dy])
	cbar = plt.colorbar(IMAGE_MAP, cax=cax1, ticks=np.arange(min_rej_sig, max_rej_sig+0.1, 1), orientation="horizontal")
	#cbar.ax.set_xticklabels([0,1,2,3,4], fontsize=10)
	cbar.ax.set_xlabel(r'consistency with zero [$\hat{\sigma}_{21}$]', fontsize=12)
	cbar.ax.tick_params(labelsize=12)
	
	
	
	path_plot_save = home_folder + '/DATA/EDGES/results/plots/20171005/'
	plt.savefig(path_plot_save + 'lowres_cut_consistency_with_zero.png', bbox_inches='tight')
	plt.close()
	plt.close()	
	
	
	
	return 














def plot_lowres_rejection_significance_2():
	
	P1, P2, P3, P4, P5, P6, P7, flag1, flag2, flag3, flag4, p21, dp21 = read_rejection_significance(home_folder + '/DATA/EDGES/results/high_band/products/model_rejection/global_21cm_anastasia/lowres_rejection_significance/lowres_rej_sig_N5.hdf5')


	dp21_new = (35/17)*dp21
	rs       = (1-p21)/dp21_new
	ds       = np.abs(p21/dp21_new)
	ic       = 2 # Index of cut


	cmap_name   = 'inferno'
	min_rej_sig = 0
	max_rej_sig = 5	


	N    = 5
	PA1  = np.linspace(np.log10(0.001),    np.log10(0.5), N)
	PA2  = np.linspace(np.log10(16.5),     np.log10(76.5), N)
	PA3  = np.linspace(np.log10(0.00001),  np.log10(10), N)
	PA4  = np.linspace(0.055, 0.088, N)
	PA5  = np.linspace(1, 1.5, N)
	PA6  = np.linspace(0.1, 3, N)
	PA7  = np.linspace(10, 50, N)
	
	
	E21 = [PA2[0]-(PA2[1]-PA2[0])/2, PA2[4]+(PA2[1]-PA2[0])/2, PA1[0]-(PA1[1]-PA1[0])/2, PA1[4]+(PA1[1]-PA1[0])/2]
	E23 = [PA2[0]-(PA2[1]-PA2[0])/2, PA2[4]+(PA2[1]-PA2[0])/2, PA3[0]-(PA3[1]-PA3[0])/2, PA3[4]+(PA3[1]-PA3[0])/2]
	E24 = [PA2[0]-(PA2[1]-PA2[0])/2, PA2[4]+(PA2[1]-PA2[0])/2, PA4[0]-(PA4[1]-PA4[0])/2, PA4[4]+(PA4[1]-PA4[0])/2]
	E25 = [PA2[0]-(PA2[1]-PA2[0])/2, PA2[4]+(PA2[1]-PA2[0])/2, PA5[0]-(PA5[1]-PA5[0])/2, PA5[4]+(PA5[1]-PA5[0])/2]
	E26 = [PA2[0]-(PA2[1]-PA2[0])/2, PA2[4]+(PA2[1]-PA2[0])/2, PA6[0]-(PA6[1]-PA6[0])/2, PA6[4]+(PA6[1]-PA6[0])/2]
	E27 = [PA2[0]-(PA2[1]-PA2[0])/2, PA2[4]+(PA2[1]-PA2[0])/2, PA7[0]-(PA7[1]-PA7[0])/2, PA7[4]+(PA7[1]-PA7[0])/2]
	
	
	# Figure
	plt.close()
	plt.close()
	plt.close()
	plt.close()
	plt.close()
	plt.close()
	plt.close()
	plt.close()
	
	
	
	
	f1 = plt.figure(figsize=(14,12))
	
	
	
	
	
	
	
	
	plt.subplot(6,5,1)
	ic = 0
	plt.imshow(np.flipud(rs[:,:,ic,ic,ic,ic,ic].T), interpolation='none', aspect=(PA2[4]-PA2[0])/(PA1[4]-PA1[0]), extent=E21, vmin=min_rej_sig, vmax=max_rej_sig, cmap=plt.get_cmap(cmap_name))
	plt.xticks(PA2, '') #np.round(PA2,2), rotation=45)
	plt.yticks(PA1, np.round(PA1,2), rotation=0)
	plt.ylabel('log10(s.f.e)', fontsize=9)
	plt.title('CUT 1', fontsize=20)
		
	plt.subplot(6,5,2)
	ic = 1
	plt.imshow(np.flipud(rs[:,:,ic,ic,ic,ic,ic].T), interpolation='none', aspect=(PA2[4]-PA2[0])/(PA1[4]-PA1[0]), extent=E21, vmin=min_rej_sig, vmax=max_rej_sig, cmap=plt.get_cmap(cmap_name))
	plt.xticks(PA2, '') #np.round(PA2,2), rotation=45)
	plt.yticks(PA1, np.round(PA1,2), rotation=0)
	#plt.xlabel('log10(m.v.c.v. [km/s])', fontsize=9)
	plt.title('CUT 2', fontsize=20)

	plt.subplot(6,5,3)
	ic = 2
	plt.imshow(np.flipud(rs[:,:,ic,ic,ic,ic,ic].T), interpolation='none', aspect=(PA2[4]-PA2[0])/(PA1[4]-PA1[0]), extent=E21, vmin=min_rej_sig, vmax=max_rej_sig, cmap=plt.get_cmap(cmap_name))
	plt.xticks(PA2, '') #np.round(PA2,2), rotation=45)
	plt.yticks(PA1, np.round(PA1,2), rotation=0)
	#plt.xlabel('log10(m.v.c.v. [km/s])', fontsize=9)
	plt.title('CUT 3', fontsize=20)
	
	plt.subplot(6,5,4)
	ic = 3
	plt.imshow(np.flipud(rs[:,:,ic,ic,ic,ic,ic].T), interpolation='none', aspect=(PA2[4]-PA2[0])/(PA1[4]-PA1[0]), extent=E21, vmin=min_rej_sig, vmax=max_rej_sig, cmap=plt.get_cmap(cmap_name))
	plt.xticks(PA2, '') #np.round(PA2,2), rotation=45)
	plt.yticks(PA1, np.round(PA1,2), rotation=0)
	#plt.xlabel('log10(m.v.c.v. [km/s])', fontsize=9)
	plt.title('CUT 4', fontsize=20)
		
	plt.subplot(6,5,5)
	ic = 4
	plt.imshow(np.flipud(rs[:,:,ic,ic,ic,ic,ic].T), interpolation='none', aspect=(PA2[4]-PA2[0])/(PA1[4]-PA1[0]), extent=E21, vmin=min_rej_sig, vmax=max_rej_sig, cmap=plt.get_cmap(cmap_name))
	plt.xticks(PA2, '') #np.round(PA2,2), rotation=45)
	plt.yticks(PA1, np.round(PA1,2), rotation=0)
	#plt.xlabel('log10(m.v.c.v. [km/s])', fontsize=9)
	plt.title('CUT 5', fontsize=20)
	
	
	
	
	
	
	
	plt.subplot(6,5,6)
	ic = 0
	plt.imshow(np.flipud(rs[ic,:,:,ic,ic,ic,ic]), interpolation='none', aspect=(PA2[4]-PA2[0])/(PA3[4]-PA3[0]), extent=E23, vmin=min_rej_sig, vmax=max_rej_sig, cmap=plt.get_cmap(cmap_name))
	plt.xticks(PA2, '') #np.round(PA2,2), rotation=45)
	plt.yticks(PA3, np.round(PA3,2), rotation=0)
	plt.ylabel('log10(x.r.e)', fontsize=9)		
		
	plt.subplot(6,5,7)
	ic = 1
	plt.imshow(np.flipud(rs[ic,:,:,ic,ic,ic,ic]), interpolation='none', aspect=(PA2[4]-PA2[0])/(PA3[4]-PA3[0]), extent=E23, vmin=min_rej_sig, vmax=max_rej_sig, cmap=plt.get_cmap(cmap_name))
	plt.xticks(PA2, '') #np.round(PA2,2), rotation=45)
	plt.yticks(PA3, np.round(PA3,2), rotation=0)
	#plt.xlabel('log10(m.v.c.v. [km/s])', fontsize=9)	

	plt.subplot(6,5,8)
	ic = 2
	plt.imshow(np.flipud(rs[ic,:,:,ic,ic,ic,ic]), interpolation='none', aspect=(PA2[4]-PA2[0])/(PA3[4]-PA3[0]), extent=E23, vmin=min_rej_sig, vmax=max_rej_sig, cmap=plt.get_cmap(cmap_name))
	plt.xticks(PA2, '') #np.round(PA2,2), rotation=45)
	plt.yticks(PA3, np.round(PA3,2), rotation=0)
	#plt.xlabel('log10(m.v.c.v. [km/s])', fontsize=9)
	
	plt.subplot(6,5,9)
	ic = 3
	plt.imshow(np.flipud(rs[ic,:,:,ic,ic,ic,ic]), interpolation='none', aspect=(PA2[4]-PA2[0])/(PA3[4]-PA3[0]), extent=E23, vmin=min_rej_sig, vmax=max_rej_sig, cmap=plt.get_cmap(cmap_name))
	plt.xticks(PA2, '') #np.round(PA2,2), rotation=45)
	plt.yticks(PA3, np.round(PA3,2), rotation=0)
	#plt.xlabel('log10(m.v.c.v. [km/s])', fontsize=9)
		
	plt.subplot(6,5,10)
	ic = 4
	plt.imshow(np.flipud(rs[ic,:,:,ic,ic,ic,ic]), interpolation='none', aspect=(PA2[4]-PA2[0])/(PA3[4]-PA3[0]), extent=E23, vmin=min_rej_sig, vmax=max_rej_sig, cmap=plt.get_cmap(cmap_name))
	plt.xticks(PA2, '') #np.round(PA2,2), rotation=45)
	plt.yticks(PA3, np.round(PA3,2), rotation=0)
	#plt.xlabel('log10(m.v.c.v. [km/s])', fontsize=9)	
	
	
	
	
	
	
	plt.subplot(6,5,11)
	ic = 0
	plt.imshow(np.flipud(rs[ic,:,ic,:,ic,ic,ic]), interpolation='none', aspect=(PA2[4]-PA2[0])/(PA4[4]-PA4[0]), extent=E24, vmin=min_rej_sig, vmax=max_rej_sig, cmap=plt.get_cmap(cmap_name))
	plt.xticks(PA2, '') #np.round(PA2,2), rotation=45)
	plt.yticks(PA4, np.round(PA4,2), rotation=0)
	plt.ylabel('tau', fontsize=9)		
		
	plt.subplot(6,5,12)
	ic = 1
	plt.imshow(np.flipud(rs[ic,:,ic,:,ic,ic,ic]), interpolation='none', aspect=(PA2[4]-PA2[0])/(PA4[4]-PA4[0]), extent=E24, vmin=min_rej_sig, vmax=max_rej_sig, cmap=plt.get_cmap(cmap_name))
	plt.xticks(PA2, '') #np.round(PA2,2), rotation=45)
	plt.yticks(PA4, np.round(PA4,2), rotation=0)
	#plt.xlabel('log10(m.v.c.v. [km/s])', fontsize=9)	

	plt.subplot(6,5,13)
	ic = 2
	plt.imshow(np.flipud(rs[ic,:,ic,:,ic,ic,ic]), interpolation='none', aspect=(PA2[4]-PA2[0])/(PA4[4]-PA4[0]), extent=E24, vmin=min_rej_sig, vmax=max_rej_sig, cmap=plt.get_cmap(cmap_name))
	plt.xticks(PA2, '') #np.round(PA2,2), rotation=45)
	plt.yticks(PA4, np.round(PA4,2), rotation=0)
	#plt.xlabel('log10(m.v.c.v. [km/s])', fontsize=9)
	
	plt.subplot(6,5,14)
	ic = 3
	plt.imshow(np.flipud(rs[ic,:,ic,:,ic,ic,ic]), interpolation='none', aspect=(PA2[4]-PA2[0])/(PA4[4]-PA4[0]), extent=E24, vmin=min_rej_sig, vmax=max_rej_sig, cmap=plt.get_cmap(cmap_name))
	plt.xticks(PA2, '') #np.round(PA2,2), rotation=45)
	plt.yticks(PA4, np.round(PA4,2), rotation=0)
	#plt.xlabel('log10(m.v.c.v. [km/s])', fontsize=9)
		
	plt.subplot(6,5,15)
	ic = 4
	plt.imshow(np.flipud(rs[ic,:,ic,:,ic,ic,ic]), interpolation='none', aspect=(PA2[4]-PA2[0])/(PA4[4]-PA4[0]), extent=E24, vmin=min_rej_sig, vmax=max_rej_sig, cmap=plt.get_cmap(cmap_name))
	plt.xticks(PA2, '') #np.round(PA2,2), rotation=45)
	plt.yticks(PA4, np.round(PA4,2), rotation=0)
	#plt.xlabel('log10(m.v.c.v. [km/s])', fontsize=9)		
	
	



	plt.subplot(6,5,16)
	ic = 0
	plt.imshow(np.flipud(rs[ic,:,ic,ic,:,ic,ic]), interpolation='none', aspect=(PA2[4]-PA2[0])/(PA5[4]-PA5[0]), extent=E25, vmin=min_rej_sig, vmax=max_rej_sig, cmap=plt.get_cmap(cmap_name))
	plt.xticks(PA2, '') #np.round(PA2,2), rotation=45)
	plt.yticks(PA5, np.round(PA5,2), rotation=0)
	plt.ylabel('slope x-ray', fontsize=9)		
		
	plt.subplot(6,5,17)
	ic = 1
	plt.imshow(np.flipud(rs[ic,:,ic,ic,:,ic,ic]), interpolation='none', aspect=(PA2[4]-PA2[0])/(PA5[4]-PA5[0]), extent=E25, vmin=min_rej_sig, vmax=max_rej_sig, cmap=plt.get_cmap(cmap_name))
	plt.xticks(PA2, '') #np.round(PA2,2), rotation=45)
	plt.yticks(PA5, np.round(PA5,2), rotation=0)
	#plt.xlabel('log10(m.v.c.v. [km/s])', fontsize=9)	

	plt.subplot(6,5,18)
	ic = 2
	plt.imshow(np.flipud(rs[ic,:,ic,ic,:,ic,ic]), interpolation='none', aspect=(PA2[4]-PA2[0])/(PA5[4]-PA5[0]), extent=E25, vmin=min_rej_sig, vmax=max_rej_sig, cmap=plt.get_cmap(cmap_name))
	plt.xticks(PA2, '') #np.round(PA2,2), rotation=45)
	plt.yticks(PA5, np.round(PA5,2), rotation=0)
	#plt.xlabel('log10(m.v.c.v. [km/s])', fontsize=9)
	
	plt.subplot(6,5,19)
	ic = 3
	plt.imshow(np.flipud(rs[ic,:,ic,ic,:,ic,ic]), interpolation='none', aspect=(PA2[4]-PA2[0])/(PA5[4]-PA5[0]), extent=E25, vmin=min_rej_sig, vmax=max_rej_sig, cmap=plt.get_cmap(cmap_name))
	plt.xticks(PA2, '') #np.round(PA2,2), rotation=45)
	plt.yticks(PA5, np.round(PA5,2), rotation=0)
	#plt.xlabel('log10(m.v.c.v. [km/s])', fontsize=9)
		
	plt.subplot(6,5,20)
	ic = 4
	plt.imshow(np.flipud(rs[ic,:,ic,ic,:,ic,ic]), interpolation='none', aspect=(PA2[4]-PA2[0])/(PA5[4]-PA5[0]), extent=E25, vmin=min_rej_sig, vmax=max_rej_sig, cmap=plt.get_cmap(cmap_name))
	plt.xticks(PA2, '') #np.round(PA2,2), rotation=45)
	plt.yticks(PA5, np.round(PA5,2), rotation=0)
	#plt.xlabel('log10(m.v.c.v. [km/s])', fontsize=9)
	
	
	



	plt.subplot(6,5,21)
	ic = 0
	plt.imshow(np.flipud(rs[ic,:,ic,ic,ic,:,ic]), interpolation='none', aspect=(PA2[4]-PA2[0])/(PA6[4]-PA6[0]), extent=E26, vmin=min_rej_sig, vmax=max_rej_sig, cmap=plt.get_cmap(cmap_name))
	plt.xticks(PA2, '') #np.round(PA2,2), rotation=45)
	plt.yticks(PA6, np.round(PA6,2), rotation=0)
	plt.ylabel('low cut-off x-ray [KeV]', fontsize=9)		
		
	plt.subplot(6,5,22)
	ic = 1
	plt.imshow(np.flipud(rs[ic,:,ic,ic,ic,:,ic]), interpolation='none', aspect=(PA2[4]-PA2[0])/(PA6[4]-PA6[0]), extent=E26, vmin=min_rej_sig, vmax=max_rej_sig, cmap=plt.get_cmap(cmap_name))
	plt.xticks(PA2, '') #np.round(PA2,2), rotation=45)
	plt.yticks(PA6, np.round(PA6,2), rotation=0)
	#plt.xlabel('log10(m.v.c.v. [km/s])', fontsize=9)	

	plt.subplot(6,5,23)
	ic = 2
	plt.imshow(np.flipud(rs[ic,:,ic,ic,ic,:,ic]), interpolation='none', aspect=(PA2[4]-PA2[0])/(PA6[4]-PA6[0]), extent=E26, vmin=min_rej_sig, vmax=max_rej_sig, cmap=plt.get_cmap(cmap_name))
	plt.xticks(PA2, '') #np.round(PA2,2), rotation=45)
	plt.yticks(PA6, np.round(PA6,2), rotation=0)
	#plt.xlabel('log10(m.v.c.v. [km/s])', fontsize=9)
	
	plt.subplot(6,5,24)
	ic = 3
	plt.imshow(np.flipud(rs[ic,:,ic,ic,ic,:,ic]), interpolation='none', aspect=(PA2[4]-PA2[0])/(PA6[4]-PA6[0]), extent=E26, vmin=min_rej_sig, vmax=max_rej_sig, cmap=plt.get_cmap(cmap_name))
	plt.xticks(PA2, '') #np.round(PA2,2), rotation=45)
	plt.yticks(PA6, np.round(PA6,2), rotation=0)
	#plt.xlabel('log10(m.v.c.v. [km/s])', fontsize=9)
		
	plt.subplot(6,5,25)
	ic = 4
	plt.imshow(np.flipud(rs[ic,:,ic,ic,ic,:,ic]), interpolation='none', aspect=(PA2[4]-PA2[0])/(PA6[4]-PA6[0]), extent=E26, vmin=min_rej_sig, vmax=max_rej_sig, cmap=plt.get_cmap(cmap_name))
	plt.xticks(PA2, '') #np.round(PA2,2), rotation=45)
	plt.yticks(PA6, np.round(PA6,2), rotation=0)
	#plt.xlabel('log10(m.v.c.v. [km/s])', fontsize=9)
	






	plt.subplot(6,5,26)
	ic = 0
	plt.imshow(np.flipud(rs[ic,:,ic,ic,ic,ic,:]), interpolation='none', aspect=(PA2[4]-PA2[0])/(PA7[4]-PA7[0]), extent=E27, vmin=min_rej_sig, vmax=max_rej_sig, cmap=plt.get_cmap(cmap_name))
	plt.xticks(PA2, np.round(PA2,2), rotation=45)
	plt.yticks(PA7, np.round(PA7,0), rotation=0)
	plt.ylabel('m.f.p. [Mpc]', fontsize=9)
	plt.xlabel('log10(m.v.c.v. [km/s])', fontsize=9)
		
	plt.subplot(6,5,27)
	ic = 1
	plt.imshow(np.flipud(rs[ic,:,ic,ic,ic,ic,:]), interpolation='none', aspect=(PA2[4]-PA2[0])/(PA7[4]-PA7[0]), extent=E27, vmin=min_rej_sig, vmax=max_rej_sig, cmap=plt.get_cmap(cmap_name))
	plt.xticks(PA2, np.round(PA2,2), rotation=45)
	plt.yticks(PA7, np.round(PA7,0), rotation=0)
	plt.xlabel('log10(m.v.c.v. [km/s])', fontsize=9)	

	plt.subplot(6,5,28)
	ic = 2
	plt.imshow(np.flipud(rs[ic,:,ic,ic,ic,ic,:]), interpolation='none', aspect=(PA2[4]-PA2[0])/(PA7[4]-PA7[0]), extent=E27, vmin=min_rej_sig, vmax=max_rej_sig, cmap=plt.get_cmap(cmap_name))
	plt.xticks(PA2, np.round(PA2,2), rotation=45)
	plt.yticks(PA7, np.round(PA7,0), rotation=0)
	plt.xlabel('log10(m.v.c.v. [km/s])', fontsize=9)
	
	plt.subplot(6,5,29)
	ic = 3
	plt.imshow(np.flipud(rs[ic,:,ic,ic,ic,ic,:]), interpolation='none', aspect=(PA2[4]-PA2[0])/(PA7[4]-PA7[0]), extent=E27, vmin=min_rej_sig, vmax=max_rej_sig, cmap=plt.get_cmap(cmap_name))
	plt.xticks(PA2, np.round(PA2,2), rotation=45)
	plt.yticks(PA7, np.round(PA7,0), rotation=0)
	plt.xlabel('log10(m.v.c.v. [km/s])', fontsize=9)
		
	plt.subplot(6,5,30)
	ic = 4
	plt.imshow(np.flipud(rs[ic,:,ic,ic,ic,ic,:]), interpolation='none', aspect=(PA2[4]-PA2[0])/(PA7[4]-PA7[0]), extent=E27, vmin=min_rej_sig, vmax=max_rej_sig, cmap=plt.get_cmap(cmap_name))
	plt.xticks(PA2, np.round(PA2,2), rotation=45)
	plt.yticks(PA7, np.round(PA7,0), rotation=0)
	plt.xlabel('log10(m.v.c.v. [km/s])', fontsize=9)	
	
	
	
	
	
	path_plot_save = home_folder + '/DATA/EDGES/results/plots/20171005/'
	plt.savefig(path_plot_save + 'lowres_rs_mvcv.png', bbox_inches='tight')
	plt.close()
	plt.close()		
	
	
	
	
	return rs

	