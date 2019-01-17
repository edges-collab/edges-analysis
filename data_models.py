
import numpy as np
import os, sys

edges_folder       = os.environ['EDGES']
print('EDGES Folder: ' + edges_folder)





def foreground_model(model_type, theta_fg, v, vr, ion_abs_coeff='free', ion_emi_coeff='free'):
	
	#print(theta_fg)


	number_of_parameters = len(theta_fg)

	model_fg = 0

	# ########################
	if model_type == 'exp':
		
		
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










def real_data(case, FLOW, FHIGH, index=1):
	
	
	if case == 'a':
		vv = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case2/case2_frequency.txt')
		tt = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case2/case2_12hr_temperature.txt')
		ww = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case2/case2_12hr_weights.txt')
		
	if case == 'b':
		vv = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case2/case2_frequency.txt')
		tt = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case2/case2_8hr_temperature.txt')
		ww = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case2/case2_8hr_weights.txt')		
		
	
	
	
	vp  = vv[(vv>=FLOW) & (vv<=FHIGH)]
	tpp = tt[:,(vv>=FLOW) & (vv<=FHIGH)]
	wpp = ww[:,(vv>=FLOW) & (vv<=FHIGH)]
	
	tp  = tpp[index,:]
	wp  = wpp[index,:]
	
	
	
	
	#vk = vp[wp>0]
	#tk = tp[wp>0]
	#wk = wp[wp>0]
	
	#v = vk[(vk<65) | ((vk>94.7) & (vk<104)) | (vk>112)]
	#t = tk[(vk<65) | ((vk>94.7) & (vk<104)) | (vk>112)]
	#w = wk[(vk<65) | ((vk>94.7) & (vk<104)) | (vk>112)]
	
	
	
	
	v = vp[wp>0]
	t = tp[wp>0]
	w = wp[wp>0]	
	
	
	
	
	#std_dev_vec   = noise_std_at_vr * (v/vr)**(-2.5)
	std_dev_vec = np.ones(len(v))
	std_dev_vec[v <= 100] = 0.015 * std_dev_vec[v <= 100]
	std_dev_vec[v  > 100] = 0.010 * std_dev_vec[v  > 100]

	sigma         = np.diag(std_dev_vec**2)     # uncertainty covariance matrix
	inv_sigma     = np.linalg.inv(sigma)
	det_sigma     = np.linalg.det(sigma)	

	
		
	return v, t, w, sigma, inv_sigma, det_sigma 

