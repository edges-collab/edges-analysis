
import numpy as np



def foreground_model(model_type, theta_fg, f, fr, ion_abs_coeff='free', ion_emi_coeff='free'):


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
			astro_exponent = astro_exponent + theta_fg[i+1] * ((np.log(f/fr))**i)

		astro_fg = theta_fg[0] * ((f/fr) ** astro_exponent)

		
		
		# Ionospheric absorption
		if ion_abs_coeff == 'free':
			IAC = theta_fg[-2]
		else:
			IAC = ion_abs_coeff			
		ionos_abs = np.exp(IAC*((f/fr)**(-2)))
		
		
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
			model_fg = model_fg      +     theta_fg[i] * ((f/fr)**(-2.5)) * ((np.log(f/fr))**i)

	return model_fg












