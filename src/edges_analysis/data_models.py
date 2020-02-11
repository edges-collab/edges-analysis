from src.edges_analysis import basic as ba
import numpy as np
import os

from os import listdir

edges_folder       = os.environ['EDGES_vol2']
print('EDGES Folder: ' + edges_folder)










def simulated_data(theta, v, vr, noise_std_at_vr, model_type_signal='exp', model_type_foreground='exp', N21par=4, NFGpar=5):

	#v          = np.arange(50, 101, 1)
	#T75        = 1500
	#beta       = -2.5
	#std_dev_vec  = 2*np.ones(len(v))   # frequency dependent or independent
	#std_dev = 1
	#noise   = np.random.normal(0, std_dev, size=len(v))


	std_dev_vec   = noise_std_at_vr * (v/vr)**(-2.5)
	#std_dev_vec   = noise_std_at_vr*np.ones(len(v))


	sigma         = np.diag(std_dev_vec**2)     # uncertainty covariance matrix
	inv_sigma     = np.linalg.inv(sigma)
	det_sigma     = np.linalg.det(sigma)

	noise         = np.random.multivariate_normal(np.zeros(len(v)), sigma)



	d_no_noise    = full_model(theta, v, vr, model_type_signal=model_type_signal, model_type_foreground=model_type_foreground, N21par=N21par, NFGpar=NFGpar)
	d             = d_no_noise + noise

	N             = len(v)


	return d, sigma, inv_sigma, det_sigma










def real_data(case, FLOW, FHIGH, gap_FLOW=0, gap_FHIGH=0):
	
	#if case == 0:
		##dd = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case_nominal/integrated_spectrum_case_nominal_days_186_219_60-120MHz.txt')
		##dd = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case_nominal/integrated_spectrum_case_nominal_days_186_219_58-120MHz_v2.txt')
		#dd = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case_nominal/integrated_spectrum_case_nominal_days_186_219_58-120MHz.txt')
	
	
	if case == 1:
		#dd = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case_nominal/integrated_spectrum_case_nominal_days_186_219_60-120MHz.txt')
		#dd = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case_nominal/integrated_spectrum_case_nominal_days_186_219_58-120MHz_v2.txt')
		#dd = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/rcv18_ant19_nominal/integrated_spectrum_rcv18_ant19_nominal_days_147_182.txt')
		dd = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/one_day_tests/another_one.txt')
		
		
	if case == 101:
		#dd = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/rcv18_sw18_nominal_GHA_every_1hr/integrated_spectrum_rcv18_sw18_every_1hr_GHA_15-17hr.txt')
		dd = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/rcv18_sw18_nominal_GHA_every_1hr/integrated_spectrum_rcv18_sw18_every_1hr_GHA_6-18hr.txt')
		#dd = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/rcv18_sw18_nominal_GHA_every_1hr/integrated_spectrum_rcv18_sw18_every_1hr_GHA_18-6hr.txt')
		
		
		
	
	

	#if case == 11:
		#dd = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case1/integrated_spectrum_case1_days_146_219_GHA_6-18.txt')
		
	#if case == 12:
		#dd = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case1/integrated_spectrum_case1_days_146_219_GHA_18-6.txt')


	
		
	#if case == 2:
		##dd = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case2/integrated_spectrum_case2_days_164_219.txt')
		##dd = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case2/integrated_spectrum_case2_days_147_219_65-105MHz.txt')
		#dd = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case2/integrated_spectrum_case2_days_186_219_60-120MHz.txt')


	#if case == 3:
		#dd = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/one_day_tests/integrated_spectrum_2018_188_gha6_18_case87.txt')






		
	#if case == 26:
		#dd = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case26/integrated_spectrum_case26.txt')
		
		
	vv = dd[:,0]
	tt = dd[:,1]
	ww = dd[:,2]
	ss = dd[:,3]
	
	#v0 = 100
	#A  = 12
	#ss = A*(1/np.sqrt(ww))*(vv/v0)**(-2.5)
		
			
	vp = vv[(vv>=FLOW) & (vv<=FHIGH)]
	tp = tt[(vv>=FLOW) & (vv<=FHIGH)]
	wp = ww[(vv>=FLOW) & (vv<=FHIGH)]
	sp = ss[(vv>=FLOW) & (vv<=FHIGH)]


	# Possibility of removing from analysis the data range between FLOW_gap and FHIGH_gap
	if (gap_FLOW > 0) and (gap_FHIGH > 0):
		vx = np.copy(vp)
		tx = np.copy(tp)
		wx = np.copy(wp)
		sx = np.copy(sp)
		
		vp = vx[(vx<=gap_FLOW) | (vx>=gap_FHIGH)]
		tp = tx[(vx<=gap_FLOW) | (vx>=gap_FHIGH)]
		wp = wx[(vx<=gap_FLOW) | (vx>=gap_FHIGH)]
		sp = sx[(vx<=gap_FLOW) | (vx>=gap_FHIGH)]




	v = vp[wp>0]
	t = tp[wp>0]
	w = wp[wp>0]
	std_dev_vec = sp[wp>0]


	sigma     = np.diag(std_dev_vec**2)     # uncertainty covariance matrix
	inv_sigma = np.linalg.inv(sigma)
	det_sigma = np.linalg.det(sigma)	

	
	return v, t, w, sigma, inv_sigma, det_sigma 





























def foreground_model(model_type, theta_fg, v, vr, ion_abs_coeff='free', ion_emi_coeff='free'):
	
	#print(theta_fg)


	number_of_parameters = len(theta_fg)

	model_fg = 0

	# ########################
	if model_type == 'powerlog':
		
		
		if (ion_abs_coeff == 'free') and (ion_emi_coeff == 'free'):
			number_astro_parameters = number_of_parameters - 2
			
		elif (ion_abs_coeff == 'free') or (ion_emi_coeff == 'free'):
			number_astro_parameters = number_of_parameters - 1
			
		else:
			number_astro_parameters = number_of_parameters
	
			
		astro_exponent = 0
		for i in range(number_astro_parameters-1):
			astro_exponent = astro_exponent + theta_fg[i+1] * ((np.log(v/vr))**i)

		astro_fg = theta_fg[0] * ((v/vr) ** astro_exponent)

		
		
		# Ionospheric absorption
		if ion_abs_coeff == 'free':
			IAC = theta_fg[-2]
		else:
			IAC = ion_abs_coeff			
		ionos_abs = np.exp(IAC*((v/vr)**(-2)))
		
		
		# Ionospheric emission
		if ion_emi_coeff == 'free':
			IEC = theta_fg[-1]
		else:
			IEC = ion_emi_coeff		
		ionos_emi = IEC*(1-ionos_abs)




		model_fg = (astro_fg * ionos_abs) + ionos_emi





	# ########################
	if model_type == 'linlog':
		for i in range(number_of_parameters):
			model_fg = model_fg      +     theta_fg[i] * ((v/vr)**(-2.5)) * ((np.log(v/vr))**i)

	return model_fg














def signal_model(model_type, theta, v):
	
	# Parameter assignment
	T21   = theta[0]
	vr    = theta[1]
	dv    = theta[2]
	tau0  = theta[3]
	
	


	# Memo 220 and 226
	if model_type == 'exp':
		
		if len(theta) == 4:
			tilt = 0
			
		elif len(theta) == 5:
			tilt  = theta[4]		
		
		b  = -np.log(-np.log( (1 + np.exp(-tau0))/2 )/tau0)
		K1 = T21 * (1 - np.exp( -tau0 * np.exp( (-b*(v-vr)**2) / ((dv**2)/4))))
		K2 = 1 + (tilt * (v - vr) / dv)
		K3 = 1 - np.exp(-tau0)
		T  = K1 * K2 / K3


	# Memo 226
	if model_type == 'tanh':
		if len(theta) == 4:
			tau1 = np.copy(tau0)
		elif len(theta) == 5:
			tau1 = theta[4]
		
		K1 = np.tanh( (1/(v + dv/2) - 1/vr) / (dv/(tau0*(vr**2))) )
		K2 = np.tanh( (1/(v - dv/2) - 1/vr) / (dv/(tau1*(vr**2))) )
		T  = -(T21/2) * (K1 - K2) 
		
		#print('tanh !!!')

	return T   # The amplitude is equal to T21, not to -T21













def full_model(theta, v, vr, model_type_signal='exp', model_type_foreground='exp', N21par=4, NFGpar=5):


	# Signal model
	if N21par == 0:
		model_21 = 0

	elif N21par > 0:
		model_21 = signal_model(model_type_signal, theta[0:N21par], v)



	# Foreground model
	model_fg = foreground_model(model_type_foreground, theta[N21par::], v, vr, ion_abs_coeff=0, ion_emi_coeff=0)



	# Full model
	model = model_21 + model_fg


	return model





















def spectrum_channel_to_channel_difference(f, t, w, FLOW, FHIGH, noise_of_residuals='yes', Nfg=5):
	
	fc  = f[(f>=FLOW) & (f<=FHIGH)]; tc = t[(f>=FLOW) & (f<=FHIGH)]; wc = w[(f>=FLOW) & (f<=FHIGH)]
	
	par = ba.fit_polynomial_fourier('LINLOG', fc, tc, Nfg, Weights=wc)
	
	rc  = (tc-par[1])
	
	x1  = np.arange(0,len(rc),2)
	x2  = np.arange(1,len(rc),2)
	print(x1)
	print(x2)
	print(len(rc))
	
	r1 = rc[x1]
	r2 = rc[x2]
	
	print(len(r1))
	print(len(r2))
	if len(r1) > len(r2):
		
		r1 = r1[0:len(r1)]
	
	diff = rc[x2] - rc[x1]
	
	
	
	
	
	
	
	return diff














def svd_functions(folder_with_spectra, FLOW, FHIGH, number_of_functions, method='average_removed'):
	
	'''
	
	Computing SVD functions
	
	'''
	
	
	# Listing files to be processed
	# -----------------------------
	folder = edges_folder + 'mid_band/spectra/level5/' + folder_with_spectra + '/'
	list_of_spectra = listdir(folder)
	list_of_spectra.sort()
	
	
	
	
	# Generate the original matrix of spectra
	# ---------------------------------------
	for i in range(len(list_of_spectra)):
		
		filename = list_of_spectra[i]
		#print(filename)
		
		d = np.genfromtxt(folder + filename)
		
		if i == 0:
			A  = d[:,1]
			fx = d[:,0] 
		
		if i > 0:
			A = np.vstack((A, d[:,1]))
			
			
	# Cut to desired frequency range
	f = fx[(fx>=FLOW) & (fx<=FHIGH)]
	A = A[:,(fx>=FLOW) & (fx<=FHIGH)]
	
	
	# Remove channels with no data
	P = np.prod(A, axis=0)
	f = f[P>0]
	A = A[:,P>0]
	
	
	
	
	# Generating one of the two possible data matrices for SVD
	# --------------------------------------------------------
	
	# Remove the average spectrum
	if method == 'average_removed':
		
		avA = np.mean(A, axis=0)
		C   = A - avA
		
		
	# Generate matrix of differences. The number of rows is N*(N-1)/2. If there are 24 spectra per day (1 per hour), N=24.
	elif method == 'delta_all':
		
		flag = 0
		for j in range(len(list_of_spectra)-1):
			#print(j)
			for i in range(len(list_of_spectra)-1-j):
	
				k = i + (j+1)
				B = A[k] - A[j]				
						
						
				if flag == 0:
					C = np.copy(B)
					
				elif flag > 0:
					C = np.vstack((C, B))
					
				print(str(flag) + ': ' + str(k) + '-' + str(j))	
				
				flag = flag + 1
	
	
	
	
	# SVD
	# ---------------------------------------
	u, EValues, EFunctions = np.linalg.svd(C)
		
		
		
	return f, EValues, EFunctions











