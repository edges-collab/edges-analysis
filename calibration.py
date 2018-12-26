

import numpy as np
import scipy as sp
import basic as ba
import reflection_coefficient as rc



# Determining home folder
from os.path import expanduser
home_folder = expanduser("~")

import os
edges_folder       = os.environ['EDGES']
print('EDGES Folder: ' + edges_folder)






def models_calibration_spectra(case, f, MC_spectra_noise=np.zeros(4)):

	if case == 1:
		
		path_par_spec       = edges_folder + '/mid_band/calibration/receiver_calibration/receiver1/2018_01_25C/results/nominal/spectra/'
		
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







def models_calibration_s11(case, f, MC_s11_syst=np.zeros(16), Npar_max=15):
	
	
	if case == 1:
		
		path_par_s11 = edges_folder + '/mid_band/calibration/receiver_calibration/receiver1/2018_01_25C/results/nominal/s11/'


			

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






	# ----- Make these input parameters ??
	RMS_expec_mag = 0.0001
	RMS_expec_ang = 0.1*(np.pi/180)
	
	
	# The following were obtained using the function "two_port_network_uncertainties()" also contained in this file
	RMS_expec_mag_2port  = 0.0002
	RMS_expec_ang_2port  = 2*(np.pi/180)
	
	RMS_expec_mag_s12s21 = 0.0002
	RMS_expec_ang_s12s21 = 0.1*(np.pi/180)	
	# ------------------------------------
	
	
	
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
		pert_mag = random_signal_perturbation(f, RMS_expec_mag_2port, Npar_max)
		s11_sr_mag        = s11_sr_mag        +   MC_s11_syst[10] * pert_mag

	if MC_s11_syst[11] > 0:
		pert_ang = random_signal_perturbation(f, RMS_expec_ang_2port, Npar_max)
		s11_sr_ang        = s11_sr_ang        +   MC_s11_syst[11] * pert_ang
	# --------------------------------------------------------------------------



	# ---------------------- S12S21 short cable -----------------------------------
	if MC_s11_syst[12] > 0:
		pert_mag = random_signal_perturbation(f, RMS_expec_mag_s12s21, Npar_max)
		s12s21_sr_mag     = s12s21_sr_mag     +   MC_s11_syst[12] * pert_mag 

	if MC_s11_syst[13] > 0:
		pert_ang = random_signal_perturbation(f, RMS_expec_ang_s12s21, Npar_max)
		s12s21_sr_ang     = s12s21_sr_ang     +   MC_s11_syst[13] * pert_ang
	# -----------------------------------------------------------------------------



	# ---------------------- S22 short cable -----------------------------------
	if MC_s11_syst[14] > 0:
		pert_mag = random_signal_perturbation(f, RMS_expec_mag_2port, Npar_max)
		s22_sr_mag        = s22_sr_mag        +   MC_s11_syst[14] * pert_mag

	if MC_s11_syst[15] > 0:
		pert_ang = random_signal_perturbation(f, RMS_expec_ang_2port, Npar_max)
		s22_sr_ang        = s22_sr_ang        +   MC_s11_syst[15] * pert_ang
	# --------------------------------------------------------------------------





	# Output
	# ------------------------------------------------------------------------------------
	s11_LNA     = s11_LNA_mag      *  (np.cos(s11_LNA_ang)     + 1j*np.sin(s11_LNA_ang))
	s11_amb     = s11_amb_mag      *  (np.cos(s11_amb_ang)     + 1j*np.sin(s11_amb_ang))
	s11_hot     = s11_hot_mag      *  (np.cos(s11_hot_ang)     + 1j*np.sin(s11_hot_ang))
	s11_open    = s11_open_mag     *  (np.cos(s11_open_ang)    + 1j*np.sin(s11_open_ang))
	s11_shorted = s11_shorted_mag  *  (np.cos(s11_shorted_ang) + 1j*np.sin(s11_shorted_ang))
	s11_sr      = s11_sr_mag       *  (np.cos(s11_sr_ang)      + 1j*np.sin(s11_sr_ang))
	s12s21_sr   = s12s21_sr_mag    *  (np.cos(s12s21_sr_ang)   + 1j*np.sin(s12s21_sr_ang))
	s22_sr      = s22_sr_mag       *  (np.cos(s22_sr_ang)      + 1j*np.sin(s22_sr_ang))

	
	out = (s11_LNA, s11_amb, s11_hot, s11_open, s11_shorted, s11_sr, s12s21_sr, s22_sr)
	# ------------------------------------------------------------------------------------
	
	return out












def models_calibration_physical_temperature(case, f, s_parameters=np.zeros(1), MC_temp=np.zeros(4)):


	if case == 1:
		
		# Paths	
		path         = edges_folder + '/mid_band/calibration/receiver_calibration/receiver1/2018_01_25C/results/nominal/temp/'

		

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
		out       = models_calibration_s11(case, f)
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








def two_port_network_uncertainties():
	
	"""
	This function propagates the uncertainty in 1-port measurements 
	to uncertainties in the 2-port S-parameters of the short cable between the hot load and the receiver
	"""
	
	
	# Simulated measurements at the VNA input
	# ---------------------------------------
	
	# Reflection standard models
	f = np.arange(50, 181)  # In MHz
	resistance_of_match = 50.1
	
	Y = rc.agilent_85033E((10**6)*f, resistance_of_match, m = 1, md_value_ps = 38)
	oa = Y[0]
	sa = Y[1]
	la = Y[2]
	
	

	# Simulated measurements at the VNA input
	o1m =  1*np.ones(len(f))
	s1m = -1*np.ones(len(f))
	l1m =  0.001*np.ones(len(f))
	
	# S-parameters of VNA calibration network
	X, s11V, s12s21V, s22V = rc.de_embed(oa, sa, la, o1m, s1m, l1m, o1m)	





	# Simulated measurements at the end of the 2-port network
	# -------------------------------------------------------
	
	# Load 2-port parameters and models
		
	path_par_s11 = edges_folder + '/mid_band/calibration/receiver_calibration/receiver1/2018_01_25C/results/nominal/s11/'

	
	par_s11_sr_mag      = np.genfromtxt(path_par_s11 + 'par_s11_sr_mag.txt')
	par_s11_sr_ang      = np.genfromtxt(path_par_s11 + 'par_s11_sr_ang.txt')
	
	par_s12s21_sr_mag   = np.genfromtxt(path_par_s11 + 'par_s12s21_sr_mag.txt')
	par_s12s21_sr_ang   = np.genfromtxt(path_par_s11 + 'par_s12s21_sr_ang.txt')
	
	par_s22_sr_mag      = np.genfromtxt(path_par_s11 + 'par_s22_sr_mag.txt')
	par_s22_sr_ang      = np.genfromtxt(path_par_s11 + 'par_s22_sr_ang.txt')
	
	fen = (f-120)/60

	s11_mag      = ba.model_evaluate('polynomial', par_s11_sr_mag,      fen)
	s11_ang      = ba.model_evaluate('polynomial', par_s11_sr_ang,      fen)
	
	s12s21_mag   = ba.model_evaluate('polynomial', par_s12s21_sr_mag,   fen)
	s12s21_ang   = ba.model_evaluate('polynomial', par_s12s21_sr_ang,   fen)
	
	s22_mag      = ba.model_evaluate('polynomial', par_s22_sr_mag,      fen)
	s22_ang      = ba.model_evaluate('polynomial', par_s22_sr_ang,      fen)
	
	s11    = s11_mag *    (np.cos(s11_ang)    + 1j*np.sin(s11_ang))
	s12s21 = s12s21_mag * (np.cos(s12s21_ang) + 1j*np.sin(s12s21_ang))
	s22    = s22_mag *    (np.cos(s22_ang)    + 1j*np.sin(s22_ang))


	
	# Measurements as seen at the input of 2-port network
	oX = rc.gamma_shifted(s11, s12s21, s22, oa)
	sX = rc.gamma_shifted(s11, s12s21, s22, sa)
	lX = rc.gamma_shifted(s11, s12s21, s22, la)	

	# Measurements at the uncalibrated VNA port
	o2m = rc.gamma_shifted(s11V, s12s21V, s22V, oX)
	s2m = rc.gamma_shifted(s11V, s12s21V, s22V, sX)
	l2m = rc.gamma_shifted(s11V, s12s21V, s22V, lX)
	
	

	

	
	# Simulating and propagating uncertainties
	# ----------------------------------------
	N        = 10000   # MC repetitions
	
	RMS_mag  = 0.0001  # magnitude uncertainty STD
	RMS_ang  = 0.1     # phase uncertainty STD
	Npar_max = 15      # maximum number of polynomial terms
	
	s11_N    = np.zeros((N, len(f))) + 1j*0
	s12s21_N = np.zeros((N, len(f))) + 1j*0
	s22_N    = np.zeros((N, len(f))) + 1j*0
	
	
	
	for i in range(N):
		
		print(i)
		
		# Add perturbations to measurements
		
		# Open at VNA port
		pert_mag = random_signal_perturbation(f, RMS_mag, Npar_max)
		pert_ang = random_signal_perturbation(f, RMS_ang, Npar_max)
		
		o1mp_mag = np.abs(o1m)              + pert_mag
		o1mp_ang = np.unwrap(np.angle(o1m)) + (np.pi/180)*pert_ang
		
		o1mp     = o1mp_mag * (np.cos(o1mp_ang) + 1j*np.sin(o1mp_ang))
		
		
		
		# Short at VNA port
		pert_mag = random_signal_perturbation(f, RMS_mag, Npar_max)
		pert_ang = random_signal_perturbation(f, RMS_ang, Npar_max)
		
		s1mp_mag = np.abs(s1m)              + pert_mag
		s1mp_ang = np.unwrap(np.angle(s1m)) + (np.pi/180)*pert_ang
		
		s1mp     = s1mp_mag * (np.cos(s1mp_ang) + 1j*np.sin(s1mp_ang))
		
		
			
		# Load at VNA port
		pert_mag = random_signal_perturbation(f, RMS_mag, Npar_max)
		pert_ang = random_signal_perturbation(f, RMS_ang, Npar_max)
		
		l1mp_mag = np.abs(l1m)              + pert_mag
		l1mp_ang = np.unwrap(np.angle(l1m)) + (np.pi/180)*pert_ang
		
		l1mp     = l1mp_mag * (np.cos(l1mp_ang) + 1j*np.sin(l1mp_ang))
		
		
		
		
		
		# Open at the end of network
		pert_mag = random_signal_perturbation(f, RMS_mag, Npar_max)
		pert_ang = random_signal_perturbation(f, RMS_ang, Npar_max)
		
		o2mp_mag = np.abs(o2m)              + pert_mag
		o2mp_ang = np.unwrap(np.angle(o2m)) + (np.pi/180)*pert_ang
		
		o2mp     = o2mp_mag * (np.cos(o2mp_ang) + 1j*np.sin(o2mp_ang))
		
		
		
		# Short at the end of network
		pert_mag = random_signal_perturbation(f, RMS_mag, Npar_max)
		pert_ang = random_signal_perturbation(f, RMS_ang, Npar_max)
		
		s2mp_mag = np.abs(s2m)              + pert_mag
		s2mp_ang = np.unwrap(np.angle(s2m)) + (np.pi/180)*pert_ang
		
		s2mp     = s2mp_mag * (np.cos(s2mp_ang) + 1j*np.sin(s2mp_ang))
		
		
			
		# Load at the end of network
		pert_mag = random_signal_perturbation(f, RMS_mag, Npar_max)
		pert_ang = random_signal_perturbation(f, RMS_ang, Npar_max)
		
		l2mp_mag = np.abs(l2m)              + pert_mag
		l2mp_ang = np.unwrap(np.angle(l2m)) + (np.pi/180)*pert_ang
	
		l2mp     = l2mp_mag * (np.cos(l2mp_ang) + 1j*np.sin(l2mp_ang))	
		
		
		

		
		
		# Calibrate measurements at the end of 2-port network, to VNA port
		o2mx, xa, xb, xc = rc.de_embed(oa, sa, la, o1mp, s1mp, l1mp, o2mp)
		s2mx, xa, xb, xc = rc.de_embed(oa, sa, la, o1mp, s1mp, l1mp, s2mp)
		l2mx, xa, xb, xc = rc.de_embed(oa, sa, la, o1mp, s1mp, l1mp, l2mp)
		
		
		# Compute S-parameters of DUT
		o, s11new, s12s21new, s22new = rc.de_embed(oa, sa, la, o2mx, s2mx, l2mx, o2mx)
		s, s11new, s12s21new, s22new = rc.de_embed(oa, sa, la, o2mx, s2mx, l2mx, s2mx)
		l, s11new, s12s21new, s22new = rc.de_embed(oa, sa, la, o2mx, s2mx, l2mx, l2mx)

		
		# Store S-parameters of DUT
		s11_N[i,:]    = s11new
		s12s21_N[i,:] = s12s21new
		s22_N[i,:]    = s22new
		
		
		
	# Estimated STD values:
	# |s11|    = 0.0002
	# |s22|    = 0.0015   # higher because this network was measured only in one direction
	# |s12s21| = 0.0002   # this means that the uncertainty on |s12|=|s21|=0.0002/2
	#
	# ang(s11)    =   2 degrees
	# ang(s22)    =  20 degrees  # higher because this network was measured only in one direction
	# ang(s12s21) = 0.1 degree
	
		
		
		
	# The first three S-parameters are the input, nominal values. The last three are the MC arrays from which the uncertainty STD could be estimated
	return f, s11, s12s21, s22, s11_N, s12s21_N, s22_N

















def NWP_fit(fn, rl, ro, rs, Toe, Tse, To, Ts, wterms):

	"""
	It is preferable to compute externally, and use a normalized frequency. For instance, fn = (f-120)/60
	
	"""


	# S11 quantities
	Fo = np.sqrt( 1 - np.abs(rl) ** 2 ) / ( 1 - ro*rl ) 
	Fs = np.sqrt( 1 - np.abs(rl) ** 2 ) / ( 1 - rs*rl )

	PHIo = np.angle( ro*Fo )
	PHIs = np.angle( rs*Fs )

	G = 1 - np.abs(rl) ** 2

	K1o = (1 - np.abs(ro) ** 2) * (np.abs(Fo) ** 2) / G
	K1s = (1 - np.abs(rs) ** 2) * (np.abs(Fs) ** 2) / G

	K2o = (np.abs(ro) ** 2) * (np.abs(Fo) ** 2) / G
	K2s = (np.abs(rs) ** 2) * (np.abs(Fs) ** 2) / G

	K3o = (np.abs(ro) * np.abs(Fo) / G) * np.cos(PHIo)
	K3s = (np.abs(rs) * np.abs(Fs) / G) * np.cos(PHIs)

	K4o = (np.abs(ro) * np.abs(Fo) / G) * np.sin(PHIo)
	K4s = (np.abs(rs) * np.abs(Fs) / G) * np.sin(PHIs)



	# Matrices A and b
	A = np.zeros((3 * wterms, 2*len(fn)))
	for i in range(wterms):
		A[i, :] = np.append(K2o * fn ** i, K2s * fn ** i) 
		A[i + 1 * wterms, :] = np.append(K3o * fn ** i, K3s * fn ** i)
		A[i + 2 * wterms, :] = np.append(K4o * fn ** i, K4s * fn ** i)
	b = np.append( (Toe - To*K1o), (Tse - Ts*K1s) )

	# Transposing matrices so 'frequency' dimension is along columns
	M = A.T
	ydata = np.reshape(b, (-1,1))



	# Solving system using 'short' QR decomposition (see R. Butt, Num. Anal. Using MATLAB)
	Q1, R1 = sp.linalg.qr(M, mode='economic')
	param  = sp.linalg.solve(R1, np.dot(Q1.T, ydata))	



	# Evaluating TU, TC, and TS
	TU = np.zeros(len(fn))
	TC = np.zeros(len(fn))
	TS = np.zeros(len(fn))

	for i in range(wterms):
		TU = TU + param[i, 0] * fn ** i
		TC = TC + param[i+1*wterms, 0] * fn ** i
		TS = TS + param[i+2*wterms, 0] * fn ** i



	# Parameters
	pU = param[0:int(len(param) / 3), 0].T
	pC = param[int((len(param)/3)):int((2 * len(param) / 3)), 0].T
	pS = param[int((2 * len(param)/3)):int(len(param)), 0].T



	return TU, TC, TS





















def calibration_quantities(fn, Tae, The, Toe, Tse, rl, ra, rh, ro, rs, Ta, Th, To, Ts, Tamb_internal, cterms, wterms):

	"""
	It is preferable to compute externally, and use a normalized frequency. For instance, fn = (f-120)/60
	
	"""


	# S11 quantities 
	Fa = np.sqrt( 1 - np.abs(rl) ** 2 ) / ( 1 - ra*rl ) 
	Fh = np.sqrt( 1 - np.abs(rl) ** 2 ) / ( 1 - rh*rl )

	PHIa = np.angle( ra*Fa )
	PHIh = np.angle( rh*Fh )

	G = 1 - np.abs(rl) ** 2

	K1a = (1 - np.abs(ra) **2) * np.abs(Fa) ** 2 / G
	K1h = (1 - np.abs(rh) **2) * np.abs(Fh) ** 2 / G

	K2a = (np.abs(ra) ** 2) * (np.abs(Fa) ** 2) / G
	K2h = (np.abs(rh) ** 2) * (np.abs(Fh) ** 2) / G

	K3a = (np.abs(ra) * np.abs(Fa) / G) * np.cos(PHIa)
	K3h = (np.abs(rh) * np.abs(Fh) / G) * np.cos(PHIh)

	K4a = (np.abs(ra) * np.abs(Fa) / G) * np.sin(PHIa)
	K4h = (np.abs(rh) * np.abs(Fh) / G) * np.sin(PHIh)



	# Initializing arrays
	niter = 4
	Ta_iter = np.zeros((niter, len(fn)))
	Th_iter = np.zeros((niter, len(fn)))

	sca = np.zeros((niter, len(fn)))
	off = np.zeros((niter, len(fn)))

	Tae_iter = np.zeros((niter, len(fn)))
	The_iter = np.zeros((niter, len(fn)))
	Toe_iter = np.zeros((niter, len(fn)))
	Tse_iter = np.zeros((niter, len(fn)))

	TU = np.zeros((niter, len(fn)))
	TC = np.zeros((niter, len(fn)))
	TS = np.zeros((niter, len(fn)))




	# Calibration loop
	for i in range(niter):

		print(i)

		# Step 1: approximate physical temperature
		if i == 0:
			Ta_iter[i,:] = Tae / K1a
			Th_iter[i,:] = The / K1h

		if i > 0:		
			NWPa = TU[i-1,:]*K2a + TC[i-1,:]*K3a + TS[i-1,:]*K4a
			NWPh = TU[i-1,:]*K2h + TC[i-1,:]*K3h + TS[i-1,:]*K4h			

			Ta_iter[i,:] = (Tae_iter[i-1,:] - NWPa) / K1a
			Th_iter[i,:] = (The_iter[i-1,:] - NWPh) / K1h	


		# Step 2: scale and offset

		# Updating scale and offset
		sca_new  = (Th - Ta) / (Th_iter[i,:] - Ta_iter[i,:])
		off_new  = Ta_iter[i,:] - Ta

		if i == 0:
			sca_raw = sca_new
			off_raw = off_new
		if i > 0:
			sca_raw = sca[i-1,:] * sca_new
			off_raw = off[i-1,:] + off_new

		# Modeling scale
		p_sca    = np.polyfit(fn, sca_raw, cterms-1)
		m_sca    = np.polyval(p_sca, fn)
		sca[i,:] = m_sca

		# Modeling offset
		p_off    = np.polyfit(fn, off_raw, cterms-1)
		m_off    = np.polyval(p_off, fn)		
		off[i,:] = m_off




		# Step 3: corrected "uncalibrated spectrum" of cable
		#Tamb_internal = 300  # same as used for 3-pos switch computation. BUT RESULTS DON'T CHANGE IF ANOTHER VALUE IS USED

		Tae_iter[i,:] = (Tae - Tamb_internal) * sca[i,:] + Tamb_internal - off[i,:]
		The_iter[i,:] = (The - Tamb_internal) * sca[i,:] + Tamb_internal - off[i,:]
		Toe_iter[i,:] = (Toe - Tamb_internal) * sca[i,:] + Tamb_internal - off[i,:]
		Tse_iter[i,:] = (Tse - Tamb_internal) * sca[i,:] + Tamb_internal - off[i,:]



		# Step 4: computing NWP
		TU[i,:], TC[i,:], TS[i,:] = NWP_fit(fn, rl, ro, rs, Toe_iter[i,:], Tse_iter[i,:], To, Ts, wterms)

	return sca[-1,:], off[-1,:], TU[-1,:], TC[-1,:], TS[-1,:]














	
def models_antenna_s11_remove_delay(band, f_MHz, year=2018, day=145, delay_0=0.17, model_type='polynomial', Nfit=10, plot_fit_residuals='no', MC_mag='no', MC_ang='no', sigma_mag=0.0001, sigma_ang_deg=0.1):	


	


	# Paths
	path_data = edges_folder + band + '/calibration/antenna_s11/corrected/' + str(year) + '_' + str(day) + '/' 

	
	# Mid-Band
	if band == 'mid_band':	
		if (year == 2018) and (day == 145):
			d = np.genfromtxt(path_data + 'antenna_s11_mid_band_2018_145.txt'); print('Antenna S11: 2018-145')
	
		if (year == 2018) and (day == 147):
			d = np.genfromtxt(path_data + 'antenna_s11_mid_band_2018_147.txt'); print('Antenna S11: 2018-147')
	
		if (year == 2018) and (day == 222):
			d = np.genfromtxt(path_data + 'antenna_s11_mid_band_2018_222.txt'); print('Antenna S11: 2018-222')


	# Low-Band 3
	if band == 'low_band3':	
		if (year == 2018) and (day == 225):
			d = np.genfromtxt(path_data + 'antenna_s11_low_band3_2018_225.txt'); print('Antenna S11: 2018-225')
	
		if (year == 2018) and (day == 227):
			d = np.genfromtxt(path_data + 'antenna_s11_low_band3_2018_227.txt'); print('Antenna S11: 2018-227')







	flow  = np.min(f_MHz)
	fhigh = np.max(f_MHz)

	d_cut = d[(d[:,0]>= flow) & (d[:,0]<= fhigh), :]


	f_orig_MHz  =  d_cut[:,0]
	s11         =  d_cut[:,1] + 1j*d_cut[:,2]


	# Removing delay from S11
	delay = delay_0 * f_orig_MHz
	re_wd = np.abs(s11) * np.cos(delay + np.unwrap(np.angle(s11)))
	im_wd = np.abs(s11) * np.sin(delay + np.unwrap(np.angle(s11)))


	# Fitting data with delay applied
	if model_type == 'polynomial':
		par_re_wd = np.polyfit(f_orig_MHz, re_wd, Nfit-1)
		par_im_wd = np.polyfit(f_orig_MHz, im_wd, Nfit-1)
		
		# Evaluating models at original frequency for evaluation
		rX = np.polyval(par_re_wd, f_orig_MHz)
		iX = np.polyval(par_im_wd, f_orig_MHz)
		
	#elif model_type == 'fourier':
		#KK1 = ba.fit_polynomial_fourier('fourier', f_orig_MHz, re_wd, Nfit)
		#KK2 = ba.fit_polynomial_fourier('fourier', f_orig_MHz, im_wd, Nfit)
		
		#rX = KK1[1]
		#iX = KK2[1]


	mX = rX + 1j*iX
	rX = mX * np.exp(-1j*delay_0 * f_orig_MHz)
	
	delta_mag_dB  = 20*np.log10(np.abs(s11)) - 20*np.log10(np.abs(rX))
	delta_ang_deg = (180/np.pi)*np.unwrap(np.angle(s11)) - (180/np.pi)*np.unwrap(np.angle(rX))
	
	print('RMS mag [dB]: ' + np.str(np.std(delta_mag_dB)))
	print('RMS ang [deg]: ' + np.str(np.std(delta_ang_deg)))
	
	
	
	# Plotting residuals
	if plot_fit_residuals == 'yes':
		plt.figure()
		
		plt.subplot(2,1,1)
		plt.plot(f_orig_MHz, 20*np.log10(np.abs(s11)) - 20*np.log10(np.abs(rX)))
		plt.ylim([-0.01, 0.01])
		plt.ylabel(r'$\Delta$ mag [dB]')
	
		plt.subplot(2,1,2)
		plt.plot(f_orig_MHz, (180/np.pi)*np.unwrap(np.angle(s11)) - (180/np.pi)*np.unwrap(np.angle(rX)))
		plt.ylim([-0.1, 0.1])
		plt.ylabel(r'$\Delta$ ang [deg]')
		
		plt.xlabel('frequency [MHz]')
	
	
	
	# Evaluating models at new input frequency
	if model_type == 'polynomial':
		model_re_wd = np.polyval(par_re_wd, f_MHz)
		model_im_wd = np.polyval(par_im_wd, f_MHz)
		
	#elif model_type == 'fourier':
		#model_re_wd = ba.model_evaluate('fourier', KK1[0], f_MHz)
		#model_im_wd = ba.model_evaluate('fourier', KK2[0], f_MHz)
		
		
		

	model_s11_wd = model_re_wd + 1j*model_im_wd
	ra    = model_s11_wd * np.exp(-1j*delay_0 * f_MHz)

	
	return ra













def balun_and_connector_loss(band, f, ra, MC=[0,0,0,0,0,0,0,0]):

	"""
	
	f:    frequency in MHz
	ra: reflection coefficient of antenna at the reference plane, the LNA input

	MC Switches:
	-------------------------
	MC[0] = tube_inner_radius
	MC[1] = tube_outer_radius
	MC[2] = tube_length
	MC[3] = connector_inner_radius
	MC[4] = connector_outer_radius
	MC[5] = connector_length
	MC[6] = metal_conductivity   
	MC[7] = teflon_permittivity

	"""


	if band == 'mid_band':

		# Angular frequency
		w = 2 * np.pi * f * 1e6   
	
		# Inch-to-meters conversion
		inch2m = 1/39.370
	
		# Conductivity of copper
		sigma_copper0 = 5.96 * 10**7     # Pozar 3rd edition. Alan uses a different number. What is his source??  5.813
	
	
		# Information from memo 273 and email from Alan Oct 18, 2018
		# -----------------------------------------------------------------
	
		# These are valid for the mid-band antenna
		balun_length     = 35 # inches
		connector_length = 0.03/inch2m # (3 cm <-> 1.18 inch) # Fairview SC3792
	
		
		# Balun dimensions 
		ric_b  = ((16/32)*inch2m)/2  # radius of outer wall of inner conductor
		if MC[0] == 1:
			ric_b = ric_b + 0.03*ric_b*np.random.normal()	# 1-sigma of 3%, about 0.04 mm
	
		roc_b  = ((1.25)*inch2m)/2  # radius of inner wall of outer conductor
		if MC[1] == 1:
			roc_b = roc_b + 0.03*roc_b*np.random.normal()	# 1-sigma of 3%, about 0.08 mm 
	
		l_b    = balun_length*inch2m    # length in meters 
		if MC[2] == 1:
			l_b = l_b + 0.001*np.random.normal()	# 1-sigma of 1 mm
	
	
		# Connector dimensions (Fairview SC3792)
		ric_c  = (0.05*inch2m)/2  # radius of outer wall of inner conductor
		if MC[3] == 1:
			ric_c = ric_c + 0.03*ric_c*np.random.normal()	# 1-sigma of 3%, about < 0.04 mm
	
		roc_c  = (0.161*inch2m)/2  # radius of inner wall of outer conductor
		if MC[4] == 1:
			roc_c = roc_c + 0.03*roc_c*np.random.normal()	# 1-sigma of 3%, about 0.04 mm	
	
		l_c    = connector_length*inch2m  # length    in meters
		if MC[5] == 1:
			l_c = l_c + 0.0001*np.random.normal()	# 1-sigma of 0.1 mm
	
	
		# Metal conductivity
		sigma_copper   = 1     * sigma_copper0   
		sigma_brass    = 0.24  * sigma_copper0
	
		sigma_xx_inner = 0.24  * sigma_copper0
		sigma_xx_outer = 0.024 * sigma_copper0
	
		if MC[6] == 1:
			sigma_copper   = sigma_copper   + 0.01*sigma_copper   * np.random.normal()   # 1-sigma of 1% of value
			sigma_brass    = sigma_brass    + 0.01*sigma_brass    * np.random.normal()   # 1-sigma of 1% of value
			sigma_xx_inner = sigma_xx_inner + 0.01*sigma_xx_inner * np.random.normal()   # 1-sigma of 1% of value
			sigma_xx_outer = sigma_xx_outer + 0.01*sigma_xx_outer * np.random.normal()   # 1-sigma of 1% of value
	
		
		# Permeability
		u0        = 4*np.pi*10**(-7)  # permeability of free space (same for copper, brass, etc., all nonmagnetic materials)
	
		ur_air    = 1 # relative permeability of air
		u_air     = u0 * ur_air
	
		ur_teflon = 1 # relative permeability of teflon
		u_teflon  = u0 * ur_teflon
	
		
		# Permittivity
		c          = 299792458       # speed of light
		e0         = 1/(u0 * c**2)   # permittivity of free space
	
		er_air           = 1.2 # Question for Alan. Why this value ???? shouldn't it be closer to 1 ?
		ep_air           = e0 * er_air
		tan_delta_air    = 0 
		epp_air          = ep_air    * tan_delta_air
	
		er_teflon        = 2.05 # why Alan????   	
		ep_teflon        = e0 * er_teflon
		tan_delta_teflon = 0.0002 # http://www.kayelaby.npl.co.uk/general_physics/2_6/2_6_5.html
		epp_teflon       = ep_teflon * tan_delta_teflon
	
		if MC[7] == 1:
			epp_teflon = epp_teflon + 0.01*epp_teflon * np.random.normal()	# 1-sigma of 1%
	















	
	# Skin Depth 
	skin_depth_copper = np.sqrt(2 / (w * u0 * sigma_copper))
	skin_depth_brass  = np.sqrt(2 / (w * u0 * sigma_brass))
	
	skin_depth_xx_inner = np.sqrt(2 / (w * u0 * sigma_xx_inner))
	skin_depth_xx_outer = np.sqrt(2 / (w * u0 * sigma_xx_outer))
	
	
	
	
	# Surface resistance
	Rs_copper = 1 / (sigma_copper * skin_depth_copper)
	Rs_brass  = 1 / (sigma_brass  * skin_depth_brass)
	
	Rs_xx_inner = 1 / (sigma_xx_inner * skin_depth_xx_inner)
	Rs_xx_outer = 1 / (sigma_xx_outer * skin_depth_xx_outer)
	
	
	
	
	# Transmission Line Parameters
	# ----------------------------
	
	# Balun
	# -----
	
	# Inductance per unit length
	Lb_inner  = u0 * skin_depth_copper / (4 * np.pi * ric_b)
	Lb_dielec = (u_air / (2 * np.pi)) * np.log(roc_b/ric_b) 
	Lb_outer  = u0 * skin_depth_brass / (4 * np.pi * roc_b)
	Lb        = Lb_inner + Lb_dielec + Lb_outer
	
	# Capacitance per unit length	
	Cb = 2 * np.pi * ep_air / np.log(roc_b/ric_b)
	
	# Resistance per unit length
	Rb = (Rs_copper / (2 * np.pi * ric_b))   +   (Rs_brass / (2 * np.pi * roc_b))
	
	# Conductance per unit length
	Gb = 2 * np.pi * w * epp_air / np.log(roc_b/ric_b)
	
	
	
	
	
	# Connector
	# ---------
	
	# Inductance per unit length
	Lc_inner  = u0 * skin_depth_xx_inner / (4 * np.pi * ric_c)
	Lc_dielec = (u_teflon / (2 * np.pi)) * np.log(roc_c/ric_c)
	Lc_outer  = u0 * skin_depth_xx_outer / (4 * np.pi * roc_c)
	Lc = Lc_inner + Lc_dielec + Lc_outer
	
	# Capacitance per unit length	
	Cc = 2 * np.pi * ep_teflon / np.log(roc_c/ric_c)
	
	# Resistance per unit length
	Rc = (Rs_xx_inner / (2 * np.pi * ric_c))   +   (Rs_xx_outer / (2 * np.pi * roc_c))
	
	# Conductance per unit length
	Gc = 2 * np.pi * w * epp_teflon / np.log(roc_c/ric_c)
	
	
	
	
	
	# Propagation constant
	gamma_b = np.sqrt( (Rb + 1j*w*Lb) * (Gb + 1j*w*Cb) )
	gamma_c = np.sqrt( (Rc + 1j*w*Lc) * (Gc + 1j*w*Cc) )
	
	
	# Complex Cable Impedance
	Zchar_b = np.sqrt( (Rb + 1j*w*Lb) / (Gb + 1j*w*Cb) )
	Zchar_c = np.sqrt( (Rc + 1j*w*Lc) / (Gc + 1j*w*Cc) )
	
	
	
	
	# ----------------------------------------------------------------------------------
	# The following loss calculations employ the expressions in Memo 126, and end of 125 
	# ----------------------------------------------------------------------------------
	
	# Impedance of Agilent terminations
	Zref = 50
	Ropen, Rshort, Rmatch = rc.agilent_85033E(f*1e6, Zref, 1)
	Zopen  = rc.gamma2impedance(Ropen,  Zref)
	Zshort = rc.gamma2impedance(Rshort, Zref)
	Zmatch = rc.gamma2impedance(Rmatch, Zref)
	
	
	
	# Impedance of terminated transmission lines
	Zin_b_open  = rc.input_impedance_transmission_line(Zchar_b, gamma_b, l_b, Zopen)
	Zin_b_short = rc.input_impedance_transmission_line(Zchar_b, gamma_b, l_b, Zshort)
	Zin_b_match = rc.input_impedance_transmission_line(Zchar_b, gamma_b, l_b, Zmatch)
	
	Zin_c_open  = rc.input_impedance_transmission_line(Zchar_c, gamma_c, l_c, Zopen)
	Zin_c_short = rc.input_impedance_transmission_line(Zchar_c, gamma_c, l_c, Zshort)
	Zin_c_match = rc.input_impedance_transmission_line(Zchar_c, gamma_c, l_c, Zmatch)
	
	
	
	# Reflection of terminated transmission lines
	Rin_b_open  = rc.impedance2gamma(Zin_b_open,  Zref)
	Rin_b_short = rc.impedance2gamma(Zin_b_short, Zref)
	Rin_b_match = rc.impedance2gamma(Zin_b_match, Zref)
	
	Rin_c_open  = rc.impedance2gamma(Zin_c_open,  Zref)
	Rin_c_short = rc.impedance2gamma(Zin_c_short, Zref)
	Rin_c_match = rc.impedance2gamma(Zin_c_match, Zref)
	
	
	
	
	
	# S-parameters (it has to be done in this order, first the Connector+Bend, then the Balun)
	ra_c, S11c, S12S21c, S22c = rc.de_embed(Ropen, Rshort, Rmatch, Rin_c_open, Rin_c_short, Rin_c_match, ra) # Reflection of antenna + balun, at the input of bend+connector	
	ra_b, S11b, S12S21b, S22b = rc.de_embed(Ropen, Rshort, Rmatch, Rin_b_open, Rin_b_short, Rin_b_match, ra_c) # Reflection of antenna only, at the input of balun
	
	
	# Inverting S11 and S22
	S11b_rev = S22b
	S22b_rev = S11b
	
	S11c_rev = S22c
	S22c_rev = S11c
	
	
	
	# Absolute value of S_21
	abs_S21b = np.sqrt(np.abs(S12S21b))
	abs_S21c = np.sqrt(np.abs(S12S21c))
	
	
	
	# Available Power Gain (Gain Factor, also known as Loss Factor)
	Gb = ( abs_S21b**2 ) * ( 1-np.abs(ra_b)**2 ) / ( (np.abs(1-S11b_rev*ra_b))**2 * (1-(np.abs(ra_c))**2) )
	Gc = ( abs_S21c**2 ) * ( 1-np.abs(ra_c)**2 ) / ( (np.abs(1-S11c_rev*ra_c))**2 * (1-(np.abs(ra))**2) )
	
	
	
	
	## ---------------------------------------------------------------
	## Alternative way, Memo 126. Gives the Same results as above !!!!
	## ---------------------------------------------------------------
	
	## Impedance of Antenna at Reference Plane
	#Zant = rc.gamma2impedance(ra, 50)
	
	## Impedance of Antenna at the input of Connector + Bend
	#Zant_before_c = rc.gamma2impedance(ra_c, 50)
	
	
	
	## Factor R (Gamma)
	#Rb = ((Zant_before_c - Zchar_b)/(Zant_before_c + Zchar_b)) * np.exp(2 * gamma_b * l_b)
	#Rc = ((Zant - Zchar_c)/(Zant + Zchar_c)) * np.exp(2 * gamma_c * l_c)
	
	
	## Voltage and Currents
	#Vin_b  =   np.exp(gamma_b * l_b) + Rb * np.exp(-gamma_b * l_b)
	#Iin_b  = ( np.exp(gamma_b * l_b) - Rb * np.exp(-gamma_b * l_b) ) / Zchar_b
	#Vout_b =  1 + Rb
	#Iout_b = (1 - Rb) / Zchar_b
	
	#Vin_c  =   np.exp(gamma_c * l_c) + Rc * np.exp(-gamma_c * l_c)
	#Iin_c  = ( np.exp(gamma_c * l_c) - Rc * np.exp(-gamma_c * l_c) ) / Zchar_c
	#Vout_c =  1 + Rc
	#Iout_c = (1 - Rc) / Zchar_c
	
	
	## Loss parameter
	#Gb = np.real(Vout_b * np.conj(Iout_b) ) / np.real(Vin_b * np.conj(Iin_b))
	#Gc = np.real(Vout_c * np.conj(Iout_c) ) / np.real(Vin_c * np.conj(Iin_c))
	
	
	
	return Gb, Gc





























def error_propagation(case):
	
	# Settings
	if case == 1:
		case_models   = 1  # mid band 2018
		s11_Npar_max  = 14
		
		band          = 'mid_band'
		
		fx, il, ih    = ba.frequency_edges(60,160)
		f             = fx[il:(ih+1)]
		fn            = (f-120)/60

		Tamb_internal = 300
		
		G_Npar_max    = 10

		
		# MC flags
		MC_spectra_noise = np.ones(4)
		MC_s11_syst      = np.ones(16)
		MC_temp          = np.ones(4)
		
		cterms = 14
		wterms = 14
		
	
	

	

	# MC Calibration quantities	
	N_MC = 10000
	
	rl_all   = np.random.normal(0, 1, size=(N_MC, len(f))) + 0*1j
	rant_all = np.random.normal(0, 1, size=(N_MC, len(f))) + 0*1j
	
	C1_all = np.random.normal(0, 1, size=(N_MC, len(f)))
	C2_all = np.random.normal(0, 1, size=(N_MC, len(f)))
	TU_all = np.random.normal(0, 1, size=(N_MC, len(f)))
	TC_all = np.random.normal(0, 1, size=(N_MC, len(f)))
	TS_all = np.random.normal(0, 1, size=(N_MC, len(f)))
	
	G_all  = np.random.normal(0, 1, size=(N_MC, len(f)))
	
	
	
	for i in range(30):
		
		# Computing "Perturbed" receiver spectra, reflection coefficients, and physical temperatures
		ms = models_calibration_spectra(case_models, f, MC_spectra_noise=MC_spectra_noise)
		mr = models_calibration_s11(case_models, f, MC_s11_syst=MC_s11_syst, Npar_max=s11_Npar_max)
		mt = models_calibration_physical_temperature(case_models, f, s_parameters=[mr[2], mr[5], mr[6], mr[7]], MC_temp=MC_temp)
		
		Tae = ms[0]
		The = ms[1]
		Toe = ms[2]
		Tse = ms[3]
		
		rl  = mr[0]
		ra  = mr[1]
		rh  = mr[2]
		ro  = mr[3]
		rs  = mr[4]
		
		Ta  = mt[0]
		Thd = mt[1]
		To  = mt[2]
		Ts  = mt[3]
		
		

		# Computing receiver calibration quantities
		print(' ')
		print('----- Calibration quantities: ' + str(i+1) + ' -----')
		C1, C2, TU, TC, TS = calibration_quantities(fn, Tae, The, Toe, Tse, rl, ra, rh, ro, rs, Ta, Thd, To, Ts, Tamb_internal, cterms, wterms)
		
		rl_all[i, :] = rl
		C1_all[i, :] = C1
		C2_all[i, :] = C2
		TU_all[i, :] = TU
		TC_all[i, :] = TC
		TS_all[i, :] = TS
		
		print('-------------------------------------')
		
		
		# Producing perturbed antenna reflection coefficient
		rant = models_antenna_s11_remove_delay(band, f, year=2018, day=147, delay_0=0.17, model_type='polynomial', Nfit=14, plot_fit_residuals='no')

		RMS_mag  = 0.0001
		RMS_ang  = 0.1*(np.pi/180) 
		pert_mag = random_signal_perturbation(f, RMS_mag, s11_Npar_max)
		pert_ang = random_signal_perturbation(f, RMS_ang, s11_Npar_max)
		rant_mag_MC = np.abs(rant) + pert_mag
		rant_ang_MC = np.unwrap(np.angle(rant)) + pert_ang
		
		rant_MC     = rant_mag_MC * (np.cos(rant_ang_MC) + 1j*np.sin(rant_ang_MC))
		
		rant_all[i, :] = rant_MC
		
		
		# Producing perturbed balun and connector loss
		Gb, Gc = balun_and_connector_loss(band, f, rant_MC)
		G = Gb*Gc
		RMS_G = 0.00025  # 5% of typical (i.e. 0.5%)
	
		flag=1
		while flag==1:
			pert_G = random_signal_perturbation(f, RMS_G, G_Npar_max)
			if np.max(np.abs(pert_G)) <= 6*RMS_G:     # 6 sigma = 0.0015, forcing the loss to stay within reason
				flag=0
		
		G_MC = G + pert_G
		G_all[i, :] = G_MC
		
		
	
	# Save to HDF5
	# -----------------
	
	
	
	return f, rl_all, C1_all, C2_all, TU_all, TC_all, TS_all, rant_all, G_all


