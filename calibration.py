

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import basic as ba
import reflection_coefficient as rc
import calibration_receiver1 as cr1

import astropy.units as apu
import astropy.time as apt
import astropy.coordinates as apc

import h5py
import healpy as hp
import datetime as dt
import scipy.interpolate as spi

from os.path import exists
from os import makedirs
from astropy.io import fits



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
	print('hola')
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



























	
def models_antenna_s11_remove_delay(band, f_MHz, year=2018, day=145, case=5, delay_0=0.17, model_type='polynomial', Nfit=10, plot_fit_residuals='no'):	


	


	# Paths
	path_data = edges_folder + band + '/calibration/antenna_s11/corrected/' + str(year) + '_' + str(day) + '/' 

	
	# Mid-Band
	if band == 'mid_band':	
		if (year == 2018) and (day == 145):
			d = np.genfromtxt(path_data + 'antenna_s11_mid_band_2018_145.txt'); print('Antenna S11: 2018-145')
	
		if (year == 2018) and (day == 147):
			if case == 1:
				d = np.genfromtxt(path_data + 'antenna_s11_2018_147_16_50_13.txt'); print('Antenna S11: 2018-147: 147_16_50_13')
			if case == 2:
				d = np.genfromtxt(path_data + 'antenna_s11_2018_147_16_51_33.txt'); print('Antenna S11: 2018-147: 147_16_51_33')
			if case == 3:
				d = np.genfromtxt(path_data + 'antenna_s11_2018_147_16_52_34.txt'); print('Antenna S11: 2018-147: 147_16_52_34')			
			if case == 4:
				d = np.genfromtxt(path_data + 'antenna_s11_2018_147_17_03_25.txt'); print('Antenna S11: 2018-147: 147_17_03_25')			
			if case == 5:
				d = np.genfromtxt(path_data + 'antenna_s11_2018_147_17_04_33.txt'); print('Antenna S11: 2018-147: 147_17_04_33')			
			
			
			
			
			
	
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
	









	# This loss is the same as for the Low-Band 1 system of 2015. It is the same balun/connector
	if band == 'low_band3':
		
		print('Balun loss model: ' + band)

		# Angular frequency
		w = 2 * np.pi * f * 1e6   
	
		# Inch-to-meters conversion
		inch2m = 1/39.370
	
		# Conductivity of copper
		sigma_copper0 = 5.96 * 10**7     # Pozar 3rd edition. Alan uses a different number. What is his source??  5.813




		# These are valid for the low-band 1 antenna
		balun_length     = 43.6 # inches
		connector_length = 0.8 # inch
		
		

		# Information from memos 166 and 181
		# -------------------------------------------------------------------


		# Balun dimensions 
		ric_b  = ((5/16)*inch2m)/2  # radius of outer wall of inner conductor
		if MC[0] == 1:
			ric_b = ric_b + 0.03*ric_b*np.random.normal()	# 1-sigma of 3%, about 0.04 mm

		roc_b  = ((3/4)*inch2m)/2  # radius of inner wall of outer conductor
		if MC[1] == 1:
			roc_b = roc_b + 0.03*roc_b*np.random.normal()	# 1-sigma of 3%, about 0.08 mm 

		l_b    = balun_length*inch2m    # length in meters
		if MC[2] == 1:
			l_b = l_b + 0.001*np.random.normal()	# 1-sigma of 1 mm


		# Connector dimensions
		ric_c  = (0.05*inch2m)/2  # radius of outer wall of inner conductor
		if MC[3] == 1:
			ric_c = ric_c + 0.03*ric_c*np.random.normal()	# 1-sigma of 3%, about < 0.04 mm

		roc_c  = (0.16*inch2m)/2  # radius of inner wall of outer conductor
		if MC[4] == 1:
			roc_c = roc_c + 0.03*roc_c*np.random.normal()	# 1-sigma of 3%, about 0.04 mm	

		l_c    = connector_length*inch2m  # length
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

		er_air           = 1.07 # why Alan???? shouldn't it be closer to 1 ?
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














def MC_receiver(band, MC_spectra_noise = np.ones(4), MC_s11_syst = np.ones(16), MC_temp = np.ones(4)):
	
	# Settings
	case_models   = 1  # mid band 2018
	s11_Npar_max  = 14
	
	#band          = 'mid_band'
	
	fx, il, ih    = ba.frequency_edges(60,160)
	f             = fx[il:(ih+1)]
	fn            = (f-120)/60

	Tamb_internal = 300
	
	# MC flags
	
	
	
	cterms = 6
	wterms = 14
	

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
	C1, C2, TU, TC, TS = calibration_quantities(fn, Tae, The, Toe, Tse, rl, ra, rh, ro, rs, Ta, Thd, To, Ts, Tamb_internal, cterms, wterms)
	
	
	return f, rl, C1, C2, TU, TC, TS
	
	
	
	
	


def MC_antenna_s11(f, rant, s11_Npar_max=14):	

	# rant = models_antenna_s11_remove_delay(band, f, year=2018, day=147, delay_0=0.17, model_type='polynomial', Nfit=14, plot_fit_residuals='no')

	# Producing perturbed antenna reflection coefficient

	RMS_mag  = 0.0001
	RMS_ang  = 0.1*(np.pi/180) 
	
	pert_mag = random_signal_perturbation(f, RMS_mag, s11_Npar_max)
	pert_ang = random_signal_perturbation(f, RMS_ang, s11_Npar_max)
	
	rant_mag_MC = np.abs(rant)              + pert_mag
	rant_ang_MC = np.unwrap(np.angle(rant)) + pert_ang
	
	rant_MC     = rant_mag_MC * (np.cos(rant_ang_MC) + 1j*np.sin(rant_ang_MC))
	
	return rant_MC
	
	





def MC_antenna_loss(f, G_Npar_max=10):

	"""
		
	Output:  dG
	Perturbed MC loss: G_nominal + dG
	
	where:
	Gb, Gc = balun_and_connector_loss(band, f, rant_MC)
	G_nominal = Gb*Gc  # balun loss x connector loss
	
	"""


	
	RMS_G       = 0.00025  # 5% of typical (i.e., of 0.5%)

	flag=1
	while flag==1:
		dG = random_signal_perturbation(f, RMS_G, G_Npar_max)
		if np.max(np.abs(dG)) <= 6*RMS_G:     # 6 sigma = 0.0015, forcing the loss to stay within reason
			flag=0
	
	return dG











def MC_error_propagation():
	
	band  = 'mid_band'
	FLOW  =  61
	FHIGH = 121
	
	
	
	
	#rcv_file = edges_folder + 'mid_band/calibration/receiver_calibration/receiver1/2018_01_25C/results/nominal/calibration_files/calibration_file_mid_band_cfit6_wfit14.txt'
	#rcv      = np.genfromtxt(rcv_file)
	#fX       = rcv[:,0]
	#rcv2     = rcv[(fX>=FLOW) & (fX<=FHIGH),:]
	
	#f   = rcv2[:,0]
	#rl  = rcv2[:,1] + 1j*rcv2[:,2]
	#C1  = rcv2[:,3]
	#C2  = rcv2[:,4]
	#TU  = rcv2[:,5]
	#TC  = rcv2[:,6]
	#TS  = rcv2[:,7]
	

	ff, rl_X, C1_X, C2_X, TU_X, TC_X, TS_X = MC_receiver(band, MC_spectra_noise = np.zeros(4), MC_s11_syst = np.zeros(16), MC_temp = np.zeros(4))
	
	rant1_X = models_antenna_s11_remove_delay('mid_band', ff, year=2018, day=147, delay_0=0.17, model_type='polynomial', Nfit=14, plot_fit_residuals='no')
	rant2_X = models_antenna_s11_remove_delay('low_band3', ff, year=2018, day=227, delay_0=0.17, model_type='polynomial', Nfit=14, plot_fit_residuals='no')
	
	Gb, Gc = balun_and_connector_loss('mid_band', ff, rant1_X)
	G1_X   = Gb*Gc
	
	Gb, Gc = balun_and_connector_loss('low_band3', ff, rant1_X)
	G2_X   = Gb*Gc	
	
	
	
	

	f     = ff[(ff>=FLOW) & (ff<=FHIGH)]
	rl    = rl_X[(ff>=FLOW) & (ff<=FHIGH)]
	C1    = C1_X[(ff>=FLOW) & (ff<=FHIGH)]
	C2    = C2_X[(ff>=FLOW) & (ff<=FHIGH)]
	TU    = TU_X[(ff>=FLOW) & (ff<=FHIGH)]
	TC    = TC_X[(ff>=FLOW) & (ff<=FHIGH)]
	TS    = TS_X[(ff>=FLOW) & (ff<=FHIGH)]

	rant1 = rant1_X[(ff>=FLOW) & (ff<=FHIGH)]
	rant2 = rant2_X[(ff>=FLOW) & (ff<=FHIGH)]
	
	G1    = G1_X[(ff>=FLOW) & (ff<=FHIGH)]
	G2    = G2_X[(ff>=FLOW) & (ff<=FHIGH)]
	
	
	
	
	
	#rant = models_antenna_s11_remove_delay(band, f, year=2018, day=147, delay_0=0.17, model_type='polynomial', Nfit=14, plot_fit_residuals='no')
	
	tsky1 = 1000*(f/100)**(-2.5)
	tsky2 = 3000*(f/100)**(-2.5)
	
	Tamb  = 273+25
	
	tsky1L1 = tsky1*G1 + Tamb*(1-G1)
	tsky2L1 = tsky2*G1 + Tamb*(1-G1)
	
	tsky1L2 = tsky1*G2 + Tamb*(1-G2)
	tsky2L2 = tsky2*G2 + Tamb*(1-G2)
	
	
	
	tunc1 = uncalibrated_antenna_temperature(tsky1L1, rant1, rl, C1, C2, TU, TC, TS, Tamb_internal=300)
	tunc2 = uncalibrated_antenna_temperature(tsky1L2, rant2, rl, C1, C2, TU, TC, TS, Tamb_internal=300)
	
	tunc3 = uncalibrated_antenna_temperature(tsky2L1, rant1, rl, C1, C2, TU, TC, TS, Tamb_internal=300)
	tunc4 = uncalibrated_antenna_temperature(tsky2L2, rant2, rl, C1, C2, TU, TC, TS, Tamb_internal=300)

	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	ff, rl_Y, C1_Y, C2_Y, TU_Y, TC_Y, TS_Y = MC_receiver(band, MC_spectra_noise = np.zeros(4), MC_s11_syst = np.zeros(16), MC_temp = np.zeros(4))

	rant1_Y  = MC_antenna_s11(ff, rant1_X, s11_Npar_max=14)
	rant2_Y  = MC_antenna_s11(ff, rant2_X, s11_Npar_max=14)

	dG1_Y = MC_antenna_loss(ff, G_Npar_max=10)
	dG2_Y = MC_antenna_loss(ff, G_Npar_max=10)
	
	G1_Y = G1_X + dG1_Y
	G2_Y = G2_X + dG2_Y
	
	

	
	rl_MC = rl_Y[(ff>=FLOW) & (ff<=FHIGH)]
	C1_MC = C1_Y[(ff>=FLOW) & (ff<=FHIGH)]
	C2_MC = C2_Y[(ff>=FLOW) & (ff<=FHIGH)]
	TU_MC = TU_Y[(ff>=FLOW) & (ff<=FHIGH)]
	TC_MC = TC_Y[(ff>=FLOW) & (ff<=FHIGH)]
	TS_MC = TS_Y[(ff>=FLOW) & (ff<=FHIGH)]
	
	rant1_MC = rant1_Y[(ff>=FLOW) & (ff<=FHIGH)]
	rant2_MC = rant2_Y[(ff>=FLOW) & (ff<=FHIGH)]	
	
	G1_MC = G1_Y[(ff>=FLOW) & (ff<=FHIGH)]
	G2_MC = G2_Y[(ff>=FLOW) & (ff<=FHIGH)]		
	
	
	
	
	#rant_MC = MC_antenna_s11(f, band)
	
	tcal1L1 = calibrated_antenna_temperature(tunc1, rant1_MC, rl_MC, C1_MC, C2_MC, TU_MC, TC_MC, TS_MC, Tamb_internal=300)
	tcal1L2 = calibrated_antenna_temperature(tunc2, rant2_MC, rl_MC, C1_MC, C2_MC, TU_MC, TC_MC, TS_MC, Tamb_internal=300)
	
	tcal2L1 = calibrated_antenna_temperature(tunc3, rant1_MC, rl_MC, C1_MC, C2_MC, TU_MC, TC_MC, TS_MC, Tamb_internal=300)
	tcal2L2 = calibrated_antenna_temperature(tunc4, rant2_MC, rl_MC, C1_MC, C2_MC, TU_MC, TC_MC, TS_MC, Tamb_internal=300)


	
	



	tcal1  = (tcal1L1 - Tamb * (1-G1_MC)) / G1_MC
	tcal2  = (tcal1L2 - Tamb * (1-G2_MC)) / G2_MC
	
	tcal3  = (tcal2L1 - Tamb * (1-G1_MC)) / G1_MC
	tcal4  = (tcal2L2 - Tamb * (1-G2_MC)) / G2_MC
		



	
	#tcal1 = calibrated_antenna_temperature(tunc1, rant1, rl, C1, C2, TU, TC, TS, Tamb_internal=300)
	
	#tcal2 = calibrated_antenna_temperature(tunc2, rant2, rl, C1, C2, TU, TC, TS, Tamb_internal=300)
	
	
	#par0 = ba.fit_polynomial_fourier('LINLOG', f, tsky, 5)
	
	Nfg = 1
	par1 = ba.fit_polynomial_fourier('LINLOG', f, tcal1, Nfg)
	par2 = ba.fit_polynomial_fourier('LINLOG', f, tcal2, Nfg)

	par3 = ba.fit_polynomial_fourier('LINLOG', f, tcal3, Nfg)
	par4 = ba.fit_polynomial_fourier('LINLOG', f, tcal4, Nfg)
	
	
	
	# plt.plot(f, tsky-par0[1]); plt.ylim([-0.5, 0.5])
	# plt.plot(f, tcal1-par1[1]); plt.ylim([-0.5, 0.5])
	# plt.plot(f, tcal2-par2[1]); plt.ylim([-0.5, 0.5])
	
	
	

	# f, tsky, tunc, tcal = cal.MC_error_propagation()
	
	return f, rant1, rant2, tcal1, tcal2, tcal3, tcal4, par1[1], par2[1], par3[1], par4[1]








































def FEKO_blade_beam(band, beam_file, frequency_interpolation='no', frequency=np.array([0]), AZ_antenna_axis=0):

	"""

	AZ_antenna_axis = 0    #  Angle of orientation (in integer degrees) of excited antenna panels relative to due North. Best value is around X ???

	"""


	if band == 'mid_band':

		data_folder = edges_folder + '/mid_band/calibration/beam/alan/'
	
	
		# Loading beam
	
		if beam_file == 1:		
			# FROM ALAN, 50-200 MHz
			print('BEAM MODEL #1 FROM ALAN')
			ff         = data_folder + 'azelq_blade9mid0.78.txt'
			f_original = np.arange(50,201,2)   #between 50 and 200 MHz in steps of 2 MHz
	
		if beam_file == 2:
			# FROM ALAN, 50-200 MHz
			print('BEAM MODEL #2 FROM ALAN')
			ff         = data_folder + 'azelq_blade9perf7mid.txt'
			f_original = np.arange(50,201,2)   #between 50 and 200 MHz in steps of 2 MHz
	
		if beam_file == 3:
			# FROM NIVEDITA, 60-200 MHz
			print('BEAM MODEL FROM NIVEDITA')
			ff         = data_folder + 'FEKO_midband_realgnd_Simple-blade_niv.txt'
			f_original = np.arange(60,201,2)   #between 60 and 200 MHz in steps of 2 MHz
			
			
			
			
			
			
	if band == 'low_band3':

		data_folder = edges_folder + '/low_band3/calibration/beam/alan/'
	
	
		# Loading beam
	
		if beam_file == 1:		
			# FROM ALAN, 50-120 MHz
			print('BEAM MODEL #1 FROM ALAN')
			ff         = data_folder + 'azelq_low3.txt'
			f_original = np.arange(50,121,2)   #between 50 and 120 MHz in steps of 2 MHz			
			
			
			
			



	data = np.genfromtxt(ff)




	# Loading data and convert to linear representation
	beam_maps = np.zeros((len(f_original),91,360))
	for i in range(len(f_original)):
		beam_maps[i,:,:] = (10**(data[(i*360):((i+1)*360),2::]/10)).T



	# Frequency interpolation
	if frequency_interpolation == 'yes':

		interp_beam = np.zeros((len(frequency), len(beam_maps[0,:,0]), len(beam_maps[0,0,:])))
		for j in range(len(beam_maps[0,:,0])):
			for i in range(len(beam_maps[0,0,:])):
				#print('Elevation: ' + str(j) + ', Azimuth: ' + str(i))
				par   = np.polyfit(f_original/200, beam_maps[:,j,i], 13)
				model = np.polyval(par, frequency/200)
				interp_beam[:,j,i] = model

		beam_maps = np.copy(interp_beam)




	# Shifting beam relative to true AZ (referenced at due North)
	# Due to angle of orientation of excited antenna panels relative to due North
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



	return beam_maps_shifted









def haslam_408MHz_map():

	'''
	
	This function will return the Haslam map in NESTED Galactic Coordinates
	
	'''


	# Loading NESTED galactic coordinates
	# -----------------------------------
	coord              = fits.open(edges_folder + '/sky_models/coordinate_maps/pixel_coords_map_nested_galactic_res9.fits')
	coord_array        = coord[1].data
	lon                = coord_array['LONGITUDE']
	lat                = coord_array['LATITUDE']
	GALAC_COORD_object = apc.SkyCoord(lon, lat, frame='galactic', unit='deg')  # defaults to ICRS frame
	
	
	
	# Loading Haslam map
	# ------------------
	haslam_map = fits.open(edges_folder + '/sky_models/haslam_map/lambda_haslam408_dsds.fits')
	haslam408  = (haslam_map[1].data)['temperature']


	print('HASLAM 408MHz MAP')

	return haslam408, lon, lat, GALAC_COORD_object








def LW_150MHz_map():
	
	'''
	
	This function will return the Haslam map in NESTED Galactic Coordinates
	
	'''
	
	# Loading NESTED galactic coordinates
	# -----------------------------------
	coord              = fits.open(edges_folder + '/sky_models/coordinate_maps/pixel_coords_map_nested_galactic_res8.fits')
	coord_array        = coord[1].data
	lon                = coord_array['LONGITUDE']
	lat                = coord_array['LATITUDE']
	GALAC_COORD_object = apc.SkyCoord(lon, lat, frame='galactic', unit='deg')  # defaults to ICRS frame
	
	
	
	# Loading LW map
	# --------------
	LW150 = np.genfromtxt(edges_folder + '/sky_models/LW_150MHz_map/150_healpix_gal_nested_R8.txt')   # Here we use the raw map, without destriping


	print('LW 150MHz MAP')

	return LW150, lon, lat, GALAC_COORD_object








def guzman_45MHz_map():

	'''
	
	This function will return the Guzman map in NESTED Galactic Coordinates
	
	'''
	
	# Loading NESTED galactic coordinates
	# -----------------------------------
	coord              = fits.open(edges_folder + '/sky_models/coordinate_maps/pixel_coords_map_nested_galactic_res9.fits')
	coord_array        = coord[1].data
	lon_raw            = coord_array['LONGITUDE']
	lat_raw            = coord_array['LATITUDE']
	GALAC_COORD_object = apc.SkyCoord(lon_raw, lat_raw, frame='galactic', unit='deg')  # defaults to ICRS frame	



	
	# 45-MHz map. The map is in Plate Caree projection (unprojected, see https://en.wikipedia.org/wiki/Equirectangular_projection)
	# ----------------------------------------------------------------------------------------------------------------------------
	map45_fit = fits.open(edges_folder + '/sky_models/map_45MHz/wlb45.fits')	
	map45     = map45_fit[0].data
	map45_1D  = map45.flatten() 

	# Coordinate grid
	lat    = np.arange(-90,90.25,0.25)
	lon    = np.arange(180,-180.25,-0.25)
	lat_2D = np.tile(lat.reshape(-1,1), len(map45[0,:]))
	lon_2D = np.tile(lon.reshape(-1,1), len(map45[:,0]))
	lon_2D = lon_2D.T

	# Converting to Healpix Nside=128
	hp_pix_2D = hp.ang2pix(128, (np.pi/180)*(90-lat_2D), (np.pi/180)*lon_2D)
	hp_pix    = hp_pix_2D.flatten()

	# Index for pixels with data, in the format Healpix Nside=128
	unique_hp_pix, unique_hp_pix_index = np.unique(hp_pix, return_index=True)

	# Map in format Healpix Nside=128
	map45_hp = np.zeros(12*128**2)
	map45_hp[unique_hp_pix]   = map45_1D[unique_hp_pix_index]
	map45_hp[map45_hp < 3300] = hp.UNSEEN   # assigning a characteristic value to the hole at high declinations

	# Converting map to format Healpix Nside=512 (same as Haslam map)
	map45_512  = hp.pixelfunc.ud_grade(map45_hp, nside_out=512)
	
	# Converting map from RING to NESTED
	m = hp.reorder(map45_512, r2n=True)
	


	# Loading celestial coordinates to fill in the temperature hole around the north pole
	coord              = fits.open(edges_folder + '/sky_models/coordinate_maps/pixel_coords_map_nested_celestial_res9.fits')
	coord_array        = coord[1].data
	RA                 = coord_array['LONGITUDE']
	DEC                = coord_array['LATITUDE']

	# Filling the hole
	m[DEC>68] = np.mean(m[(DEC>60) & (DEC<68)])
	
	
	print('GUZMAN 45MHz MAP')
	
	return m, lon_raw, lat_raw, GALAC_COORD_object

















def antenna_beam_factor(band, name_save, beam_file=1, sky_model='haslam', rotation_from_north=90, band_deg=10, index_inband=2.5, index_outband=2.62, reference_frequency=100):



	#band      = 'mid_band'
	#sky_model = 'haslam', 'LW', 'guzman'
	


	# Data paths
	path_data = edges_folder + 'sky_models/'
	path_save = edges_folder + band + '/calibration/beam_factors/raw/'






	# Antenna beam
	# ---------------------------------------------------------------------------
	AZ_beam  = np.arange(0, 360)
	EL_beam  = np.arange(0, 91)


	# FEKO blade beam	
	# Fixing rotation angle due to diferent rotation (by 90deg) in Nivedita's map
	if (band == 'mid_band') and (beam_file == 3):
		rotation_from_north = rotation_from_north - 90
		
	beam_all = FEKO_blade_beam(band, beam_file, AZ_antenna_axis=rotation_from_north)


	# Frequency array
	if band == 'mid_band':
		if beam_file == 1:
			# ALAN #1
			freq_array = np.arange(50, 201, 2, dtype='uint32')  
	
		elif beam_file == 2:
			# ALAN #2
			freq_array = np.arange(50, 201, 2, dtype='uint32')  
	
		elif beam_file == 3:
			# NIVEDITA
			freq_array = np.arange(60, 201, 2, dtype='uint32')
			
	elif band == 'low_band3':
		if beam_file == 1:
			freq_array = np.arange(50, 121, 2, dtype='uint32')		
			
					
	# Index of reference frequency
	index_freq_array = np.arange(len(freq_array))
	irf = index_freq_array[freq_array == reference_frequency]
	print('Reference frequency: ' + str(freq_array[irf][0]) + ' MHz')







	# Sky model
	# ------------------------------------------------------------------
	if sky_model == 'haslam':

		# Loading Haslam map
		map_orig, lon, lat, GALAC_COORD_object = haslam_408MHz_map()
		v0 = 408
		
		
	elif sky_model == 'LW': 
	
		# Loading LW map
		map_orig, lon, lat, GALAC_COORD_object = LW_150MHz_map()
		v0 = 150
		
		
	elif sky_model == 'guzman':
		
		# Loading Guzman map
		map_orig, lon, lat, GALAC_COORD_object = guzman_45MHz_map()
		v0 = 45
		

	print(v0)


	# Scaling Haslam map (the map contains the CMB, which has to be removed at 408 MHz, and then added back)
	Tcmb    = 2.725
	

	
	
	sky_map = np.zeros((len(map_orig), len(freq_array)))
	
	print(map_orig.shape)
	print(sky_map.shape)
	
	
	for i in range(len(freq_array)):

		# Band of the Galactic center, using spectral index
		sky_map[(lat >= -band_deg) & (lat <= band_deg), i] = (map_orig - Tcmb)[(lat >= -band_deg) & (lat <= band_deg)] * (freq_array[i]/v0)**(-index_inband) + Tcmb

		# Range outside the Galactic center, using second spectral index
		sky_map[(lat < -band_deg) | (lat > band_deg), i]   = (map_orig - Tcmb)[(lat < -band_deg) | (lat > band_deg)] * (freq_array[i]/v0)**(-index_outband) + Tcmb
		
			




	# Calculation 
	# --------------------------------------------------------------------------------------

	# EDGES location	
	EDGES_lat_deg  = -26.714778
	EDGES_lon_deg  = 116.605528 
	EDGES_location = apc.EarthLocation(lat=EDGES_lat_deg*apu.deg, lon=EDGES_lon_deg*apu.deg)


	# Reference UTC observation time. At this time, the LST is 0.1666 (00:10 Hrs LST) at the EDGES location (it was wrong before, now it is correct)
	Time_iter    = np.array([2014, 1, 1, 9, 39, 42])     
	Time_iter_dt = dt.datetime(Time_iter[0], Time_iter[1], Time_iter[2], Time_iter[3], Time_iter[4], Time_iter[5]) 


	# Looping over LST
	LST             = np.zeros(72)
	convolution_ref = np.zeros((len(LST), len(beam_all[:,0,0])))
	convolution     = np.zeros((len(LST), len(beam_all[:,0,0])))	
	numerator       = np.zeros((len(LST), len(beam_all[:,0,0])))
	denominator     = np.zeros((len(LST), len(beam_all[:,0,0])))


	#for i in range(len(LST)):
	for i in range(len(LST)):


		print(name_save + '. LST: ' + str(i+1) + ' out of 72')


		# Advancing time ( 19:57 minutes UTC correspond to 20 minutes LST )
		minutes_offset = 19
		seconds_offset = 57
		if i > 0:
			Time_iter_dt = Time_iter_dt + dt.timedelta(minutes = minutes_offset, seconds = seconds_offset)
			Time_iter    = np.array([Time_iter_dt.year, Time_iter_dt.month, Time_iter_dt.day, Time_iter_dt.hour, Time_iter_dt.minute, Time_iter_dt.second]) 



		# LST 
		LST[i] = ba.utc2lst(Time_iter, EDGES_lon_deg)



		# Transforming Galactic coordinates of Sky to Local coordinates		
		altaz          = GALAC_COORD_object.transform_to(apc.AltAz(location=EDGES_location, obstime=apt.Time(Time_iter_dt, format='datetime')))
		AZ             = np.asarray(altaz.az)
		EL             = np.asarray(altaz.alt)



		# Selecting coordinates above the horizon
		AZ_above_horizon         = AZ[EL>=0]
		EL_above_horizon         = EL[EL>=0]



		# Selecting sky data above the horizon
		sky_above_horizon     = sky_map[EL>=0,:]	
		sky_ref_above_horizon = sky_above_horizon[:, irf].flatten()



		# Arranging AZ and EL arrays corresponding to beam model
		az_array   = np.tile(AZ_beam,91)
		el_array   = np.repeat(EL_beam,360)
		az_el_original      = np.array([az_array, el_array]).T
		az_el_above_horizon = np.array([AZ_above_horizon, EL_above_horizon]).T



		# Loop over frequency
		for j in range(len(freq_array)):

			print(name_save + ', Freq: ' + str(j) + ' out of ' + str(len(beam_all[:,0,0])))

			beam_array         = beam_all[j,:,:].reshape(1,-1)[0]
			beam_above_horizon = spi.griddata(az_el_original, beam_array, az_el_above_horizon, method='cubic')  # interpolated beam


			no_nan_array = np.ones(len(AZ_above_horizon)) - np.isnan(beam_above_horizon)
			index_no_nan = np.nonzero(no_nan_array)[0]

	
			# Convolution between (beam at all frequencies) and (sky at reference frequency)
			convolution_ref[i, j]   = np.sum(beam_above_horizon[index_no_nan]*sky_ref_above_horizon[index_no_nan])/np.sum(beam_above_horizon[index_no_nan])


			# Antenna temperature, i.e., Convolution between (beam at all frequencies) and (sky at all frequencies)
			sky_above_horizon_ff    = sky_above_horizon[:, j].flatten()
			convolution[i, j]       = np.sum(beam_above_horizon[index_no_nan]*sky_above_horizon_ff[index_no_nan])/np.sum(beam_above_horizon[index_no_nan])



	
	beam_factor_T = convolution_ref.T/convolution_ref[:,irf].T
	beam_factor   = beam_factor_T.T





	# Saving
	# ---------------------------------------------------------
	np.savetxt(path_save + name_save + '_tant.txt', convolution)
	np.savetxt(path_save + name_save + '_data.txt', beam_factor)
	np.savetxt(path_save + name_save + '_LST.txt',  LST)
	np.savetxt(path_save + name_save + '_freq.txt', freq_array)



	return freq_array, LST, convolution, beam_factor
























def antenna_beam_factor_interpolation(band, case, lst_hires, fnew):

	"""


	"""


	if band == 'mid_band':
		file_path = edges_folder + 'mid_band/calibration/beam_factors/raw/'
		
		if case == 1:
			bf_old  = np.genfromtxt(file_path + 'mid_band_50-200MHz_90deg_alan1_haslam_2.5_2.62_reffreq_100MHz_data.txt')
			freq    = np.genfromtxt(file_path + 'mid_band_50-200MHz_90deg_alan1_haslam_2.5_2.62_reffreq_100MHz_freq.txt')
			lst_old = np.genfromtxt(file_path + 'mid_band_50-200MHz_90deg_alan1_haslam_2.5_2.62_reffreq_100MHz_LST.txt')
			
		elif case == 2:
			bf_old  = np.genfromtxt(file_path + 'mid_band_50-200MHz_90deg_alan1_haslam_2.55_reffreq_100MHz_data.txt')
			freq    = np.genfromtxt(file_path + 'mid_band_50-200MHz_90deg_alan1_haslam_2.55_reffreq_100MHz_freq.txt')
			lst_old = np.genfromtxt(file_path + 'mid_band_50-200MHz_90deg_alan1_haslam_2.55_reffreq_100MHz_LST.txt')
			
		elif case == 3:
			bf_old  = np.genfromtxt(file_path + 'mid_band_50-200MHz_90deg_alan1_LW_2.55_reffreq_100MHz_data.txt')
			freq    = np.genfromtxt(file_path + 'mid_band_50-200MHz_90deg_alan1_LW_2.55_reffreq_100MHz_freq.txt')
			lst_old = np.genfromtxt(file_path + 'mid_band_50-200MHz_90deg_alan1_LW_2.55_reffreq_100MHz_LST.txt')
			
		elif case == 4:
			bf_old  = np.genfromtxt(file_path + 'mid_band_50-200MHz_90deg_alan1_guzman_2.55_reffreq_100MHz_data.txt')
			freq    = np.genfromtxt(file_path + 'mid_band_50-200MHz_90deg_alan1_guzman_2.55_reffreq_100MHz_freq.txt')
			lst_old = np.genfromtxt(file_path + 'mid_band_50-200MHz_90deg_alan1_guzman_2.55_reffreq_100MHz_LST.txt')	


		
		
		
		
		
		
		
		
		
	elif band == 'low_band3':
		file_path = edges_folder + 'low_band3/calibration/beam_factors/raw/'
		
		bf_old  = np.genfromtxt(file_path + 'low_band3_50-120MHz_85deg_alan_haslam_2.5_2.62_reffreq_76MHz_data.txt')
		freq    = np.genfromtxt(file_path + 'low_band3_50-120MHz_85deg_alan_haslam_2.5_2.62_reffreq_76MHz_freq.txt')
		lst_old = np.genfromtxt(file_path + 'low_band3_50-120MHz_85deg_alan_haslam_2.5_2.62_reffreq_76MHz_LST.txt')		
		

	


	# Wrap beam factor and LST for 24-hr interpolation 
	bf   = np.vstack((bf_old[-1,:], bf_old, bf_old[0,:]))
	lst0 = np.append(lst_old[-1]-24, lst_old)
	lst  = np.append(lst0, lst_old[0]+24)

	# Arranging original arrays in preparation for interpolation
	freq_array        = np.tile(freq, len(lst))
	lst_array         = np.repeat(lst, len(freq))
	bf_array          = bf.reshape(1,-1)[0]
	freq_lst_original = np.array([freq_array, lst_array]).T

	# Producing high-resolution array of LSTs (frequencies are the same as the original)
	freq_hires       = np.copy(freq)
	freq_hires_array = np.tile(freq_hires, len(lst_hires))
	lst_hires_array  = np.repeat(lst_hires, len(freq_hires)) 
	freq_lst_hires   = np.array([freq_hires_array, lst_hires_array]).T

	# Interpolating beam factor to high LST resolution
	bf_hires_array = spi.griddata(freq_lst_original, bf_array, freq_lst_hires, method='cubic')	
	bf_2D          = bf_hires_array.reshape(len(lst_hires), len(freq_hires))

	# Interpolating beam factor to high frequency resolution
	for i in range(len(bf_2D[:,0])):
		par       = np.polyfit(freq_hires, bf_2D[i,:], 11)
		bf_single = np.polyval(par, fnew)

		if i == 0:
			bf_2D_hires = np.copy(bf_single)
		elif i > 0:
			bf_2D_hires = np.vstack((bf_2D_hires, bf_single))


	return bf_2D_hires   #beam_factor_model  #, freq_hires, bf_lst_average














def beam_factor_table_computation(band, case, f, N_lst, file_name_hdf5):


	# Produce the beam factor at high resolution
	# ------------------------------------------
	#N_lst = 6000   # number of LST points within 24 hours
	
	
	lst_hires = np.arange(0,24,24/N_lst)
	bf        = antenna_beam_factor_interpolation(band, case, lst_hires, f)
	
	
	
	# Save
	# ----
	file_path = edges_folder + band + '/calibration/beam_factors/table/'
	with h5py.File(file_path + file_name_hdf5, 'w') as hf:
		hf.create_dataset('frequency',           data = f)
		hf.create_dataset('lst',                 data = lst_hires)
		hf.create_dataset('beam_factor',         data = bf)		
		
	return 0









def beam_factor_table_read(path_file):

	# path_file = home_folder + '/EDGES/calibration/beam_factors/mid_band/file.hdf5'

	# Show keys (array names inside HDF5 file)
	with h5py.File(path_file,'r') as hf:

		hf_freq  = hf.get('frequency')
		f        = np.array(hf_freq)

		hf_lst   = hf.get('lst')
		lst      = np.array(hf_lst)

		hf_bf    = hf.get('beam_factor')
		bf       = np.array(hf_bf)

	return f, lst, bf	









def beam_factor_table_evaluate(f_table, lst_table, bf_table, lst_in):
	
	# f_table, lst_table, bf_table = eg.beam_factor_table_read('/data5/raul/EDGES/calibration/beam_factors/mid_band/beam_factor_table_hires.hdf5')
	
	beam_factor = np.zeros((len(lst_in), len(f_table)))
	
	for i in range(len(lst_in)):
		d = np.abs(lst_table - lst_in[i])
		
		index = np.argsort(d)
		IX = index[0]
		
		beam_factor[i,:] = bf_table[IX,:]
				
	return beam_factor
























def corrected_antenna_s11(day, case, save_name_flag):
	
	# Creating folder if necessary
	path_save = home_folder + '/DATA/EDGES/mid_band/calibration/antenna_s11/corrected/2018_' + str(day) + '/'
	if not exists(path_save):
		makedirs(path_save)	
	





	
	# Mid-Band
	if day == 145:
		path_antenna_s11 = home_folder + '/DATA/EDGES/mid_band/calibration/antenna_s11/raw/mid-recv1-20180525/'	
		date_all = ['145_08_28_27', '145_08_28_27']
	
	
	# Mid-Band
	if day == 147:
		path_antenna_s11 = home_folder + '/DATA/EDGES/mid_band/calibration/antenna_s11/raw/mid-rcv1-20180527/'

		if case == 1:
			date_all = ['147_16_50_13', '147_16_50_13']

		if case == 2:
			date_all = ['147_16_51_33', '147_16_51_33']
			
		if case == 3:
			date_all = ['147_16_52_34', '147_16_52_34']
			
		if case == 4:
			date_all = ['147_17_03_25', '147_17_03_25']
		
		if case == 5:
			date_all = ['147_17_04_33', '147_17_04_33']
			
			
		
		
	# Mid-Band
	if day == 222:
		path_antenna_s11 = home_folder + '/DATA/EDGES/mid_band/calibration/antenna_s11/raw/mid_20180809/'
		date_all = ['222_05_41_09', '222_05_43_53', '222_05_45_58', '222_05_48_06']
	
			
	# Alan's noise source + 6dB attenuator
	if day == 223:
		path_antenna_s11 = home_folder + '/DATA/EDGES/mid_band/calibration/antenna_s11/raw/mid_20180812_ans1_6db/'
		date_all = ['223_22_40_15', '223_22_41_22', '223_22_42_44']  # Leaving the first one out: '223_22_38_08'


	# Low-Band 3 (Receiver 1, Low-Band 1 antenna, High-Band ground plane with wings)
	if day == 225:  
		path_antenna_s11 = home_folder + '/DATA/EDGES/mid_band/calibration/antenna_s11/raw/low3_20180813_rcv1/'
		date_all = ['225_12_17_56', '225_12_19_22', '225_12_20_39', '225_12_21_47']
		









	# Load and correct individual measurements
	for i in range(len(date_all)):

		# Four measurements
		o_m, f_m  = rc.s1p_read(path_antenna_s11 + '2018_' + date_all[i] + '_input1.s1p')
		s_m, f_m  = rc.s1p_read(path_antenna_s11 + '2018_' + date_all[i] + '_input2.s1p')
		l_m, f_m  = rc.s1p_read(path_antenna_s11 + '2018_' + date_all[i] + '_input3.s1p')
		a_m, f_m  = rc.s1p_read(path_antenna_s11 + '2018_' + date_all[i] + '_input4.s1p')

	
		# Standards assumed at the switch
		o_sw =  1 * np.ones(len(f_m))
		s_sw = -1 * np.ones(len(f_m))
		l_sw =  0 * np.ones(len(f_m))
		
		
		# Correction at switch
		a_sw_c, x1, x2, x3  = rc.de_embed(o_sw, s_sw, l_sw, o_m, s_m, l_m, a_m)
		
		
		# Correction at receiver input
		a = cr1.switch_correction_receiver1_2018_01_25C(a_sw_c, f_in = f_m)
		
		if i == 0:
			a_all = np.copy(a[0])
		
		elif i > 0:
			a_all = np.vstack((a_all, a[0]))
			
			
			
			
			
	# Average measurement	
	f  = f_m/1e6
	av = np.mean(a_all, axis=0)			
		
		
	# Output array
	ra_T = np.array([f, np.real(av), np.imag(av)])
	ra   = ra_T.T



		
	
	# Save
	np.savetxt(path_save + 'antenna_s11_2018_' + str(day) + save_name_flag + '.txt', ra, header='freq [MHz]   real(s11)   imag(s11)')
	
	return ra, a_all




























