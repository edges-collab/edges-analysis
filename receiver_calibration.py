

import numpy as np
import basic as ba
import reflection_coefficient as rc



# Determining home folder
from os.path import expanduser
home_folder = expanduser("~")

import os
edges_folder       = os.environ['EDGES']
print('EDGES Folder: ' + edges_folder)






def models_calibration_spectra(band, folder, f, MC_spectra_noise=np.zeros(4)):

	if band == 'mid_band_2018':
		
		path_par_spec       = edges_folder + '/mid_band/calibration/receiver_calibration/receiver1/2018_01_25C/results/' + folder + '/spectra/'
		
		# Loading parameters
		par_spec_amb        = np.genfromtxt(path_par_spec + 'par_spec_amb.txt')
		par_spec_hot        = np.genfromtxt(path_par_spec + 'par_spec_hot.txt')
		par_spec_open       = np.genfromtxt(path_par_spec + 'par_spec_open.txt')
		par_spec_shorted    = np.genfromtxt(path_par_spec + 'par_spec_shorted.txt')
		RMS_spec	    = np.genfromtxt(path_par_spec + 'RMS_spec.txt')


		# Normalized frequency
		fn = (f - 120)/60


		# Evaluating models
		Tae  = ba.model_evaluate('fourier',  par_spec_amb,     fn)
		The  = ba.model_evaluate('fourier',  par_spec_hot,     fn)
		Toe  = ba.model_evaluate('fourier',  par_spec_open,    fn)
		Tse  = ba.model_evaluate('fourier',  par_spec_shorted, fn)
		
		
	
	

	# RMS noise
	RMS_Tae  = RMS_spec[0]
	RMS_The  = RMS_spec[1]
	RMS_Toe  = RMS_spec[2]
	RMS_Tse  = RMS_spec[3]



	# Adding noise to models
	if MC_spectra_noise[0] > 0:
		Tae  = Tae  + MC_spectra_noise[0] * RMS_Tae * np.random.normal(0, np.ones(len(fn)))

	if MC_spectra_noise[1] > 0:
		The  = The  + MC_spectra_noise[1] * RMS_The * np.random.normal(0, np.ones(len(fn)))	

	if MC_spectra_noise[2] > 0:
		Toe  = Toe  + MC_spectra_noise[2] * RMS_Toe * np.random.normal(0, np.ones(len(fn)))

	if MC_spectra_noise[3] > 0:
		Tse  = Tse  + MC_spectra_noise[3] * RMS_Tse * np.random.normal(0, np.ones(len(fn)))	




	# Output
	output = np.array([Tae, The, Toe, Tse])



	return output















def models_calibration_physical_temperature(band, folder, f, s_parameters=np.zeros(1), MC_temp=np.zeros(4)):


	if band == 'mid_band_2018':
		
		# Paths	
		path         = edges_folder + '/mid_band/calibration/receiver_calibration/receiver1/2018_01_25C/results/' + folder + '/temp/'

		

	# Physical temperatures
	phys_temp = np.genfromtxt(path + 'physical_temperatures.txt')
	Ta  = phys_temp[0] * np.ones(len(f))
	Th  = phys_temp[1] * np.ones(len(f))
	To  = phys_temp[2] * np.ones(len(f))
	Ts  = phys_temp[3] * np.ones(len(f))



	# MC realizations of physical temperatures
	STD_temp = 0.1
	if MC_temp[0] > 0:
		Ta  = Ta  + MC_temp[0] * STD_temp * np.random.normal(0,1)

	if MC_temp[1] > 0:	
		Th  = Th  + MC_temp[1] * STD_temp * np.random.normal(0,1)

	if MC_temp[2] > 0:	
		To  = To  + MC_temp[2] * STD_temp * np.random.normal(0,1)

	if MC_temp[3] > 0:
		Ts  = Ts  + MC_temp[3] * STD_temp * np.random.normal(0,1)



	# S-parameters of hot load device
	if len(s_parameters) == 1:
		out       = models_calibration_s11(band, folder, f)
		rh        = out[2]
		s11_sr    = out[5]
		s12s21_sr = out[6]
		s22_sr    = out[7]

	if len(s_parameters) == 4:
		rh        = s_parameters[0]
		s11_sr    = s_parameters[1]
		s12s21_sr = s_parameters[2]
		s22_sr    = s_parameters[3]



	# Temperature of hot device

	# reflection coefficient of termination
	rht = rc.gamma_de_embed(s11_sr, s12s21_sr, s22_sr, rh)

	# inverting the direction of the s-parameters,
	# since the port labels have to be inverted to match those of Pozar eqn 10.25
	s11_sr_rev = s22_sr
	s22_sr_rev = s11_sr

	# absolute value of S_21
	abs_s21 = np.sqrt(np.abs(s12s21_sr))

	# available power gain
	G = ( abs_s21**2 ) * ( 1-np.abs(rht)**2 ) / ( (np.abs(1-s11_sr_rev*rht))**2 * (1-(np.abs(rh))**2) )

	# temperature
	Thd  = G*Th + (1-G)*Ta



	# Output
	output = np.array([Ta, Thd, To, Ts])

	return output




















def models_calibration_s11(band, folder, calibration_temperature, f):
	
	
	if band == 'mid_band_2018':
		
		path_par_s11 = edges_folder + '/mid_band/calibration/receiver_calibration/receiver1/2018_01_25C/results/' + folder + '/s11/'


			

	# Loading S11 parameters
	par_s11_LNA_mag     = np.genfromtxt(path_par_s11 + 'par_s11_LNA_mag.txt')
	par_s11_LNA_ang     = np.genfromtxt(path_par_s11 + 'par_s11_LNA_ang.txt')
	
	par_s11_amb_mag     = np.genfromtxt(path_par_s11 + 'par_s11_amb_mag.txt')
	par_s11_amb_ang     = np.genfromtxt(path_par_s11 + 'par_s11_amb_ang.txt')
	
	par_s11_hot_mag     = np.genfromtxt(path_par_s11 + 'par_s11_hot_mag.txt')
	par_s11_hot_ang     = np.genfromtxt(path_par_s11 + 'par_s11_hot_ang.txt')
	
	par_s11_open_mag    = np.genfromtxt(path_par_s11 + 'par_s11_open_mag.txt')
	par_s11_open_ang    = np.genfromtxt(path_par_s11 + 'par_s11_open_ang.txt')
	
	par_s11_shorted_mag = np.genfromtxt(path_par_s11 + 'par_s11_shorted_mag.txt')
	par_s11_shorted_ang = np.genfromtxt(path_par_s11 + 'par_s11_shorted_ang.txt')

	par_s11_sr_mag      = np.genfromtxt(path_par_s11 + 'par_s11_sr_mag.txt')
	par_s11_sr_ang      = np.genfromtxt(path_par_s11 + 'par_s11_sr_ang.txt')
	
	par_s12s21_sr_mag   = np.genfromtxt(path_par_s11 + 'par_s12s21_sr_mag.txt')
	par_s12s21_sr_ang   = np.genfromtxt(path_par_s11 + 'par_s12s21_sr_ang.txt')
	
	par_s22_sr_mag      = np.genfromtxt(path_par_s11 + 'par_s22_sr_mag.txt')
	par_s22_sr_ang      = np.genfromtxt(path_par_s11 + 'par_s22_sr_ang.txt')

	
	
	fen = (f-120)/60
	
	
	# Evaluating S11 models at EDGES frequency
	s11_LNA_mag     = ba.model_evaluate('polynomial', par_s11_LNA_mag,     fen)
	s11_LNA_ang     = ba.model_evaluate('polynomial', par_s11_LNA_ang,     fen)

	s11_amb_mag     = ba.model_evaluate('fourier',    par_s11_amb_mag,     fen)
	s11_amb_ang     = ba.model_evaluate('fourier',    par_s11_amb_ang,     fen)

	s11_hot_mag     = ba.model_evaluate('fourier',    par_s11_hot_mag,     fen)
	s11_hot_ang     = ba.model_evaluate('fourier',    par_s11_hot_ang,     fen)

	s11_open_mag    = ba.model_evaluate('fourier',    par_s11_open_mag,    fen)
	s11_open_ang    = ba.model_evaluate('fourier',    par_s11_open_ang,    fen)

	s11_shorted_mag = ba.model_evaluate('fourier',    par_s11_shorted_mag, fen)
	s11_shorted_ang = ba.model_evaluate('fourier',    par_s11_shorted_ang, fen)

	s11_sr_mag      = ba.model_evaluate('polynomial', par_s11_sr_mag,      fen)
	s11_sr_ang      = ba.model_evaluate('polynomial', par_s11_sr_ang,      fen)
	
	s12s21_sr_mag   = ba.model_evaluate('polynomial', par_s12s21_sr_mag,   fen)
	s12s21_sr_ang   = ba.model_evaluate('polynomial', par_s12s21_sr_ang,   fen)
	
	s22_sr_mag      = ba.model_evaluate('polynomial', par_s22_sr_mag,      fen)
	s22_sr_ang      = ba.model_evaluate('polynomial', par_s22_sr_ang,      fen)	






	# ----- Make these input parameters!!!
	RMS_expec_mag = 0.0001
	RMS_expec_ang = 0.1*(np.pi/180)
	Npar_max  = 15
	# ---------------------------------
	
	
	# ---------------------- LNA --------------------------------
	if MC_s11_syst[0] > 0:		
		pert_mag = random_signal_perturbation(f, RMS_expec_mag, Npar_max)
		s11_LNA_mag       = s11_LNA_mag       +   MC_s11_syst[0] * pert_mag

	if MC_s11_syst[1] > 0:
		pert_ang = random_signal_perturbation(f, RMS_expec_ang, Npar_max)
		s11_LNA_ang       = s11_LNA_ang       +   MC_s11_syst[1] * pert_ang
	# -----------------------------------------------------------
		
		
		

	# ---------------------- Amb ---------------------------------
	if MC_s11_syst[2] > 0:
		pert_mag = random_signal_perturbation(f, RMS_expec_mag, Npar_max)
		s11_amb_mag       = s11_amb_mag       +   MC_s11_syst[2] * pert_mag		

	if MC_s11_syst[3] > 0:
		pert_ang = random_signal_perturbation(f, RMS_expec_ang, Npar_max)
		s11_amb_ang       = s11_amb_ang       +   MC_s11_syst[3] * pert_ang						
	# -----------------------------------------------------------




	# ---------------------- Hot ---------------------------------
	if MC_s11_syst[4] > 0:
		pert_mag = random_signal_perturbation(f, RMS_expec_mag, Npar_max)
		s11_hot_mag       = s11_hot_mag       +   MC_s11_syst[4] * pert_mag

	if MC_s11_syst[5] > 0:
		pert_ang = random_signal_perturbation(f, RMS_expec_ang, Npar_max)
		s11_hot_ang       = s11_hot_ang       +   MC_s11_syst[5] * pert_ang
	# -------------------------------------------------------------




	# ---------------------- Open --------------------------------
	if MC_s11_syst[6] > 0:
		pert_mag = random_signal_perturbation(f, RMS_expec_mag, Npar_max)
		s11_open_mag      = s11_open_mag      +   MC_s11_syst[6] * pert_mag

	if MC_s11_syst[7] > 0:
		pert_ang = random_signal_perturbation(f, RMS_expec_ang, Npar_max)
		s11_open_ang      = s11_open_ang      +   MC_s11_syst[7] * pert_ang
	# ------------------------------------------------------------
	



	# ---------------------- Shorted -----------------------------
	if MC_s11_syst[8] > 0:
		pert_mag = random_signal_perturbation(f, RMS_expec_mag, Npar_max)
		s11_shorted_mag   = s11_shorted_mag   +   MC_s11_syst[8] * pert_mag

	if MC_s11_syst[9] > 0:
		pert_ang = random_signal_perturbation(f, RMS_expec_ang, Npar_max)
		s11_shorted_ang   = s11_shorted_ang   +   MC_s11_syst[9] * pert_ang
	# ------------------------------------------------------------




	# ---------------------- S11 short cable -----------------------------------
	if MC_s11_syst[10] > 0:
		pert_mag = random_signal_perturbation(f, RMS_expec_mag, Npar_max)
		s11_sr_mag        = s11_sr_mag        +   MC_s11_syst[10] * pert_mag

	if MC_s11_syst[11] > 0:
		pert_ang = random_signal_perturbation(f, RMS_expec_ang, Npar_max)
		s11_sr_ang        = s11_sr_ang        +   MC_s11_syst[11] * pert_ang
	# --------------------------------------------------------------------------




	if MC_s11_syst[12] > 0:
		s12s21_sr_mag     = s12s21_sr_mag     +   MC_s11_syst[12] * np.random.normal(0, 1) * 2*sigma_mag_s21        # not correlated. This uncertainty value is technically (2 * A * dA), where A is |S21|. But also |S21| ~ 1

	if MC_s11_syst[13] > 0:
		s12s21_sr_ang     = s12s21_sr_ang     +   MC_s11_syst[13] * np.random.normal(0, 1) * (np.pi/180) * 2 * sigma_phase_1mag    # not correlated, This uncertainty value is technically (2 * dP) where dP is the uncertainty in the phase of S21




	# ---------------------- S22 short cable -----------------------------------
	if MC_s11_syst[14] > 0:
		pert_mag = random_signal_perturbation(f, RMS_expec_mag, Npar_max)
		s22_sr_mag        = s22_sr_mag        +   MC_s11_syst[14] * pert_mag

	if MC_s11_syst[15] > 0:
		pert_ang = random_signal_perturbation(f, RMS_expec_ang, Npar_max)
		s22_sr_ang        = s22_sr_ang        +   MC_s11_syst[15] * pert_ang
	# --------------------------------------------------------------------------



	
	return 0









def random_signal_perturbation(f, RMS_expectation, Npar_max):

	# Choose randomly, from a Uniform distribution, an integer corresponding to the number of polynomial terms 
	Npar = np.random.choice(Npar_max+1)
	
	# Choose randomly, from a Gaussian distribution, a number corresponding to the RMS of the perturbation across frequency
	RMS              = RMS_expectation * np.random.normal(0,1)

	# Generate random noise across frequency, with MEAN=0 and STD=1
	noise            = np.random.normal(0,1, size=len(f))
	
	# Fit a generic polynomial to the noise, using a polynomial with Npar terms
	par              = np.polyfit(f, noise, Npar+1)
	model            = np.polyval(par, f)
	
	# Compute the current RMS of the polynomial
	RMS_model        = np.std(model)
	
	# Scale the polynomial to have the new desired RMS
	model_normalized = (RMS/RMS_model) * model
	
	return model_normalized
