
import numpy as np
import scipy as sp
import reflection_coefficient as rc
import time  as tt
import edges as eg
import datetime as dt
import receiver_calibration as rcv


import basic as ba


import scipy.io as sio
import scipy.interpolate as spi

import matplotlib.pyplot as plt

import astropy.units as apu
import astropy.time as apt
import astropy.coordinates as apc

import h5py




from os.path import expanduser
from os.path import exists
from os import listdir, makedirs, system

from astropy.io import fits






# Determining home folder
home_folder = expanduser("~")

import os
edges_folder       = os.environ['EDGES']
print('EDGES Folder: ' + edges_folder)
















def switch_correction_mid_band_2018_01_25C(ant_s11, f_in = np.zeros([0,1]), case = 1):  


	"""
	
	Aug 13, 2018
	
	The characterization of the 4-position switch was done using the male standard of Phil's kit
	
	"""

	
	
	
	

	# Loading measurements
	if case == 1:
		path_folder     = home_folder + '/DATA/EDGES/calibration/receiver_calibration/mid_band/2018_01_25C/data/s11/raw/InternalSwitch/'
		
		resistance_of_match = 50.027 # male
		
		o_in, f = rc.s1p_read(path_folder + 'Open01.s1p')
		s_in, f = rc.s1p_read(path_folder + 'Short01.s1p')
		l_in, f = rc.s1p_read(path_folder + 'Match01.s1p')
	
		o_ex, f = rc.s1p_read(path_folder + 'ExternalOpen01.s1p')
		s_ex, f = rc.s1p_read(path_folder + 'ExternalShort01.s1p')
		l_ex, f = rc.s1p_read(path_folder + 'ExternalMatch01.s1p')




	if case == 2:
		path_folder     = home_folder + '/DATA/EDGES/calibration/receiver_calibration/mid_band/2018_01_25C/data/s11/raw/InternalSwitch/'
		
		resistance_of_match = 50.027 # male
		
		o_in, f = rc.s1p_read(path_folder + 'Open02.s1p')
		s_in, f = rc.s1p_read(path_folder + 'Short02.s1p')
		l_in, f = rc.s1p_read(path_folder + 'Match02.s1p')
	
		o_ex, f = rc.s1p_read(path_folder + 'ExternalOpen02.s1p')
		s_ex, f = rc.s1p_read(path_folder + 'ExternalShort02.s1p')
		l_ex, f = rc.s1p_read(path_folder + 'ExternalMatch02.s1p')




	if case == 3:
		path_folder     = home_folder + '/DATA/EDGES/calibration/receiver_calibration/mid_band/2018_01_15C/data/s11/raw/InternalSwitch/'
		
		resistance_of_match = 50.099 # male
		
		o_in, f = rc.s1p_read(path_folder + 'Open01.s1p')
		s_in, f = rc.s1p_read(path_folder + 'Short01.s1p')
		l_in, f = rc.s1p_read(path_folder + 'Match01.s1p')
	
		o_ex, f = rc.s1p_read(path_folder + 'ExternalOpen01.s1p')
		s_ex, f = rc.s1p_read(path_folder + 'ExternalShort01.s1p')
		l_ex, f = rc.s1p_read(path_folder + 'ExternalMatch01.s1p')



	if case == 4:
		path_folder     = home_folder + '/DATA/EDGES/calibration/receiver_calibration/mid_band/2018_01_15C/data/s11/raw/InternalSwitch/'
		
		resistance_of_match = 50.099 # male
		
		o_in, f = rc.s1p_read(path_folder + 'Open02.s1p')
		s_in, f = rc.s1p_read(path_folder + 'Short02.s1p')
		l_in, f = rc.s1p_read(path_folder + 'Match02.s1p')
	
		o_ex, f = rc.s1p_read(path_folder + 'ExternalOpen02.s1p')
		s_ex, f = rc.s1p_read(path_folder + 'ExternalShort02.s1p')
		l_ex, f = rc.s1p_read(path_folder + 'ExternalMatch02.s1p')




	if case == 5:
		path_folder     = home_folder + '/DATA/EDGES/calibration/receiver_calibration/mid_band/2018_01_35C/data/s11/raw/InternalSwitch/'
		
		resistance_of_match = 50.002 # male
		
		o_in, f = rc.s1p_read(path_folder + 'Open01.s1p')
		s_in, f = rc.s1p_read(path_folder + 'Short01.s1p')
		l_in, f = rc.s1p_read(path_folder + 'Match01.s1p')
	
		o_ex, f = rc.s1p_read(path_folder + 'ExternalOpen01.s1p')
		s_ex, f = rc.s1p_read(path_folder + 'ExternalShort01.s1p')
		l_ex, f = rc.s1p_read(path_folder + 'ExternalMatch01.s1p')


	
	# THESE MEASUREMENTS ARE BAD. DO NOT USE CASE 6
	# if case == 6:
		# path_folder     = home_folder + '/DATA/EDGES/calibration/receiver_calibration/mid_band/2018_01_35C/data/s11/raw/InternalSwitch/'
		
		# resistance_of_match = 50.10 # male
		
		# o_in, f = rc.s1p_read(path_folder + 'Open02.s1p')
		# s_in, f = rc.s1p_read(path_folder + 'Short02.s1p')
		# l_in, f = rc.s1p_read(path_folder + 'Match02.s1p')
	
		# o_ex, f = rc.s1p_read(path_folder + 'ExternalOpen02.s1p')
		# s_ex, f = rc.s1p_read(path_folder + 'ExternalShort02.s1p')
		# l_ex, f = rc.s1p_read(path_folder + 'ExternalMatch02.s1p')


















	# Standards assumed at the switch
	o_sw =  1 * np.ones(len(f))
	s_sw = -1 * np.ones(len(f))
	l_sw =  0 * np.ones(len(f))	




	# Correction at the switch
	o_ex_c, xx1, xx2, xx3 = rc.de_embed(o_sw, s_sw, l_sw, o_in, s_in, l_in, o_ex)
	s_ex_c, xx1, xx2, xx3 = rc.de_embed(o_sw, s_sw, l_sw, o_in, s_in, l_in, s_ex)
	l_ex_c, xx1, xx2, xx3 = rc.de_embed(o_sw, s_sw, l_sw, o_in, s_in, l_in, l_ex)




	# Computation of S-parameters to the receiver input
	#md = 1
	oa, sa, la = rc.agilent_85033E(f, resistance_of_match)  #, md)
	xx, s11, s12s21, s22 = rc.de_embed(oa, sa, la, o_ex_c, s_ex_c, l_ex_c, o_ex_c)





	# Polynomial fit of S-parameters from "f" to input frequency vector "f_in"
	# ------------------------------------------------------------------------


	# Frequency normalization
	fn = f/75e6

	if len(f_in) > 10:
		if f_in[0] > 1e5:
			fn_in = f_in/75e6
		elif f_in[-1] < 300:
			fn_in = f_in/75
	else:
		fn_in = fn	



	# Real-Imaginary parts
	real_s11    = np.real(s11)
	imag_s11    = np.imag(s11)
	real_s12s21 = np.real(s12s21)
	imag_s12s21 = np.imag(s12s21)
	real_s22    = np.real(s22)
	imag_s22    = np.imag(s22)



	# Polynomial fits
	Nterms_poly = 14
	
	p = np.polyfit(fn, real_s11, Nterms_poly-1)
	fit_real_s11 = np.polyval(p, fn_in)

	p = np.polyfit(fn, imag_s11, Nterms_poly-1)
	fit_imag_s11 = np.polyval(p, fn_in)


	p = np.polyfit(fn, real_s12s21, Nterms_poly-1)
	fit_real_s12s21 = np.polyval(p, fn_in)

	p = np.polyfit(fn, imag_s12s21, Nterms_poly-1)
	fit_imag_s12s21 = np.polyval(p, fn_in)


	p = np.polyfit(fn, real_s22, Nterms_poly-1)
	fit_real_s22 = np.polyval(p, fn_in)

	p = np.polyfit(fn, imag_s22, Nterms_poly-1)
	fit_imag_s22 = np.polyval(p, fn_in)


	fit_s11    = fit_real_s11    + 1j*fit_imag_s11
	fit_s12s21 = fit_real_s12s21 + 1j*fit_imag_s12s21
	fit_s22    = fit_real_s22    + 1j*fit_imag_s22



	# Corrected antenna S11
	corr_ant_s11 = rc.gamma_de_embed(fit_s11, fit_s12s21, fit_s22, ant_s11)






	return (corr_ant_s11, fit_s11, fit_s12s21, fit_s22)





















def s11_calibration_measurements_mid_band_2018_01_25C(flow=40, fhigh=200, save='no', flag=''):




	# Data paths
	main_path    = home_folder + '/DATA/EDGES/calibration/receiver_calibration/mid_band/2018_01_25C/data/s11/raw/'


	path_LNA     = main_path + 'ReceiverReading03/'    # seems slightly better than the others
	path_ambient = main_path + 'Ambient/'
	path_hot     = main_path + 'HotLoad/'
	path_open    = main_path + 'LongCableOpen/'
	path_shorted = main_path + 'LongCableShorted/'
	path_sim     = main_path + 'AntSim3/'
	



	# Receiver reflection coefficient
	# -------------------------------

	# Reading measurements
	o,   fr  = rc.s1p_read(path_LNA + 'Open01.s1p')
	s,   fr  = rc.s1p_read(path_LNA + 'Short01.s1p')
	l,   fr  = rc.s1p_read(path_LNA + 'Match01.s1p')
	LNA0, fr = rc.s1p_read(path_LNA + 'ReceiverReading01.s1p')


	# Models of standards
	resistance_of_match = 50 # 49.98 # female
	md = 1
	oa, sa, la = rc.agilent_85033E(fr, resistance_of_match, md)


	# Correction of measurements
	LNAc, x1, x2, x3   = rc.de_embed(oa, sa, la, o, s, l, LNA0)






	# Calibration loads
	# -----------------



	# -----------------------------------------------------------------------------------------
	# Ambient load before
	# -------------------
	o_m,  f_a = rc.s1p_read(path_ambient + 'Open02.s1p')
	s_m,  f_a = rc.s1p_read(path_ambient + 'Short02.s1p')
	l_m,  f_a = rc.s1p_read(path_ambient + 'Match02.s1p')
	a_m,  f_a = rc.s1p_read(path_ambient + 'External02.s1p')


	# Standards assumed at the switch
	o_sw =  1 * np.ones(len(f_a))
	s_sw = -1 * np.ones(len(f_a))
	l_sw =  0 * np.ones(len(f_a))


	# Correction at switch
	a_sw_c, x1, x2, x3  = rc.de_embed(o_sw, s_sw, l_sw, o_m, s_m, l_m, a_m)


	# Correction at receiver input
	out = switch_correction_mid_band_2018_01_25C(a_sw_c, f_in = f_a)
	a_c = out[0]










	# -----------------------------------------------------------------------------------------
	# Hot load before
	# -------------------
	o_m, f_h = rc.s1p_read(path_hot + 'Open02.s1p')
	s_m, f_h = rc.s1p_read(path_hot + 'Short02.s1p')
	l_m, f_h = rc.s1p_read(path_hot + 'Match02.s1p')
	h_m, f_h = rc.s1p_read(path_hot + 'External02.s1p')


	# Standards assumed at the switch
	o_sw =  1 * np.ones(len(f_h))
	s_sw = -1 * np.ones(len(f_h))
	l_sw =  0 * np.ones(len(f_h))


	# Correction at switch
	h_sw_c, x1, x2, x3  = rc.de_embed(o_sw, s_sw, l_sw, o_m, s_m, l_m, h_m)


	# Correction at receiver input
	out = switch_correction_mid_band_2018_01_25C(h_sw_c, f_in = f_h)
	h_c = out[0]












	# -----------------------------------------------------------------------------------------
	# Open Cable before
	# -------------------
	o_m, f_o = rc.s1p_read(path_open  + 'Open02.s1p')
	s_m, f_o = rc.s1p_read(path_open  + 'Short02.s1p')
	l_m, f_o = rc.s1p_read(path_open  + 'Match02.s1p')
	oc_m, f_o = rc.s1p_read(path_open + 'External02.s1p')


	# Standards assumed at the switch
	o_sw =  1 * np.ones(len(f_o))
	s_sw = -1 * np.ones(len(f_o))
	l_sw =  0 * np.ones(len(f_o))


	# Correction at switch
	oc_sw_c, x1, x2, x3  = rc.de_embed(o_sw, s_sw, l_sw, o_m, s_m, l_m, oc_m)


	# Correction at receiver input
	out = switch_correction_mid_band_2018_01_25C(oc_sw_c, f_in = f_o)
	o_c = out[0]










	# -----------------------------------------------------------------------------------------
	# Short Cable before
	# -------------------
	o_m,  f_s = rc.s1p_read(path_shorted + 'Open02.s1p')
	s_m,  f_s = rc.s1p_read(path_shorted + 'Short02.s1p')
	l_m,  f_s = rc.s1p_read(path_shorted + 'Match02.s1p')
	sc_m,  f_s = rc.s1p_read(path_shorted + 'External02.s1p')


	# Standards assumed at the switch
	o_sw =  1 * np.ones(len(f_s))
	s_sw = -1 * np.ones(len(f_s))
	l_sw =  0 * np.ones(len(f_s))


	# Correction at switch
	sc_sw_c, x1, x2, x3  = rc.de_embed(o_sw, s_sw, l_sw, o_m, s_m, l_m, sc_m)


	# Correction at receiver input
	out = switch_correction_mid_band_2018_01_25C(sc_sw_c, f_in = f_s)
	s_c = out[0]	







	# -----------------------------------------------------------------------------------------
	# Antenna Simulator 3
	# --------------------------
	o_m, f_q = rc.s1p_read(path_sim + 'Open02.s1p')
	s_m, f_q = rc.s1p_read(path_sim + 'Short02.s1p')
	l_m, f_q = rc.s1p_read(path_sim + 'Match02.s1p')
	q_m, f_q = rc.s1p_read(path_sim + 'External02.s1p')


	# Standards assumed at the switch
	o_sw =  1 * np.ones(len(f_q))
	s_sw = -1 * np.ones(len(f_q))
	l_sw =  0 * np.ones(len(f_q))


	# Correction at switch
	q_sw_c, x1, x2, x3  = rc.de_embed(o_sw, s_sw, l_sw, o_m, s_m, l_m, q_m)


	# Correction at receiver input
	out  = switch_correction_mid_band_2018_01_25C(q_sw_c, f_in = f_q)
	q_c  = out[0]	
	









	# S-parameters of semi-rigid cable 
	# ---------------------------------
	d = np.genfromtxt(home_folder + '/DATA/EDGES/calibration/receiver_calibration/high_band1/2015_03_25C/data/S11/corrected_original/semi_rigid_s_parameters.txt')
			
	Nterms = 17	
	
	column      = 1
	p           = np.polyfit(d[:,0], d[:, column], Nterms-1)
	sr_s11r     = np.polyval(p, fr/1e6)
	
	column      = 2
	p           = np.polyfit(d[:,0], d[:, column], Nterms-1)
	sr_s11i     = np.polyval(p, fr/1e6)
	
	column      = 3
	p           = np.polyfit(d[:,0], d[:, column], Nterms-1)
	sr_s12s21r  = np.polyval(p, fr/1e6)	
	
	column      = 4
	p           = np.polyfit(d[:,0], d[:, column], Nterms-1)
	sr_s12s21i  = np.polyval(p, fr/1e6)

	column      = 5
	p           = np.polyfit(d[:,0], d[:, column], Nterms-1)
	sr_s22r     = np.polyval(p, fr/1e6)

	column      = 6
	p           = np.polyfit(d[:,0], d[:, column], Nterms-1)
	sr_s22i     = np.polyval(p, fr/1e6)

			
			


	# Output array
	# ----------------------
	tempT = np.array([ fr/1e6, 
	np.real(LNAc),   np.imag(LNAc),
	np.real(a_c),    np.imag(a_c),
	np.real(h_c),    np.imag(h_c), 
	np.real(o_c),    np.imag(o_c),
	np.real(s_c),    np.imag(s_c),
	sr_s11r,         sr_s11i,
	sr_s12s21r,      sr_s12s21i,
	sr_s22r,         sr_s22i,
	np.real(q_c),    np.imag(q_c)])
	
	temp = tempT.T	
	kk = temp[(fr/1e6 >= flow) & (fr/1e6 <= fhigh), :]











	# -----------------------------------------------------------------------------------------
	# Saving
	if save == 'yes':

		save_path       = home_folder + '/DATA/EDGES/calibration/receiver_calibration/mid_band/2018_01_25C/data/s11/corrected/'
		temperature_LNA = '25degC'
		output_file_str = save_path + 's11_calibration_mid_band_LNA' + temperature_LNA + '_' + tt.strftime('%Y-%m-%d-%H-%M-%S') + flag + '.txt'
		np.savetxt(output_file_str, kk)

		print('File saved to: ' + output_file_str)


	return fr, LNAc, a_c, h_c, o_c, s_c, q_c, sr_s11r, sr_s11i, sr_s12s21r, sr_s12s21i, sr_s22r, sr_s22i, q_c


















def calibration_processing_mid_band_2018_01_25C(flow=50, fhigh=180, save='no', save_folder=0):


	"""
	
	Modification: Aug 3, 2018.

	"""	




	# Main folder
	main_folder     = home_folder + '/DATA/EDGES/calibration/receiver_calibration/mid_band/2018_01_25C/'


	# Paths for source data
	path_spectra    = main_folder + 'data/spectra/'
	path_resistance = main_folder + 'data/resistance/corrected/'
	path_s11        = main_folder + 'data/s11/corrected/'


	# Creating output folders
	if save == 'yes':

		if not exists(main_folder + 'results/' + save_folder + '/temp/'):
			makedirs(main_folder + 'results/' + save_folder + '/temp/')

		if not exists(main_folder + 'results/' + save_folder + '/spectra/'):
			makedirs(main_folder + 'results/' + save_folder + '/spectra/')

		if not exists(main_folder + 'results/' + save_folder + '/s11/'):
			makedirs(main_folder + 'results/' + save_folder + '/s11/')

		if not exists(main_folder + 'results/' + save_folder + '/data/'):
			makedirs(main_folder + 'results/' + save_folder + '/data/')

		if not exists(main_folder + 'results/' + save_folder + '/calibration_files/'):
			makedirs(main_folder + 'results/' + save_folder + '/calibration_files/')			

		if not exists(main_folder + 'results/' + save_folder + '/plots/'):
			makedirs(main_folder + 'results/' + save_folder + '/plots/')


		# Output folders
		path_par_temp    = main_folder + 'results/' + save_folder + '/temp/'
		path_par_spectra = main_folder + 'results/' + save_folder + '/spectra/'
		path_par_s11     = main_folder + 'results/' + save_folder + '/s11/'
		path_data        = main_folder + 'results/' + save_folder + '/data/'







	# Ambient
	file_ambient1 = path_spectra + 'level1_AmbientLoad_2018_060_02_300_350.mat'
	file_ambient2 = path_spectra + 'level1_AmbientLoad_2018_061_00_300_350.mat'
	file_ambient3 = path_spectra + 'level1_AmbientLoad_2018_062_00_300_350.mat'	
	spec_ambient  = [file_ambient1, file_ambient2, file_ambient3]
	res_ambient   = path_resistance + 'AmbientLoad.txt'



	# Hot
	file_hot1 = path_spectra + 'level1_HotLoad_2018_062_02_300_350.mat'	
	file_hot2 = path_spectra + 'level1_HotLoad_2018_063_00_300_350.mat'
	file_hot3 = path_spectra + 'level1_HotLoad_2018_064_00_300_350.mat'
	spec_hot  = [file_hot1, file_hot2, file_hot3]
	res_hot   = path_resistance + 'HotLoad.txt'



	# Open Cable
	file_open1 = path_spectra + 'level1_LongCableOpen_2018_057_23_300_350.mat' 
	file_open2 = path_spectra + 'level1_LongCableOpen_2018_058_00_300_350.mat'
	spec_open  = [file_open1, file_open2]
	res_open   = path_resistance + 'OpenCable.txt'



	# Shorted Cable
	file_shorted1 = path_spectra + 'level1_LongCableShorted_2018_059_01_300_350.mat'
	file_shorted2 = path_spectra + 'level1_LongCableShorted_2018_060_00_300_350.mat'
	spec_shorted  = [file_shorted1, file_shorted2]
	res_shorted   = path_resistance + 'ShortedCable.txt'



	# Antenna Simulator 3
	file_sim1  = path_spectra + 'level1_AntSim3_2018_055_05_300_350.mat'	
	file_sim2  = path_spectra + 'level1_AntSim3_2018_056_00_300_350.mat'
	file_sim3  = path_spectra + 'level1_AntSim3_2018_057_00_300_350.mat'	
	spec_sim   = [file_sim1, file_sim2, file_sim3]
	res_sim    = path_resistance + 'SimAnt.txt'









	# Average calibration spectra / physical temperature
	# Percentage of initial data to leave out
	percent = 5 # 5%
	ssa,    phys_temp_ambient  = average_calibration_spectrum(spec_ambient, res_ambient, 1*percent, plot='no')
	ssh,    phys_temp_hot      = average_calibration_spectrum(spec_hot,     res_hot,     2*percent, plot='no')
	sso,    phys_temp_open     = average_calibration_spectrum(spec_open,    res_open,    1*percent, plot='no')
	sss,    phys_temp_shorted  = average_calibration_spectrum(spec_shorted, res_shorted, 1*percent, plot='no')
	sss1,   phys_temp_sim      = average_calibration_spectrum(spec_sim,     res_sim,     1*percent, plot='no')













	# Select frequency range
	ff, ilow, ihigh = frequency_edges(flow, fhigh)
	fe    = ff[ilow:ihigh+1]
	sa    = ssa[ilow:ihigh+1]
	sh    = ssh[ilow:ihigh+1]
	so    = sso[ilow:ihigh+1]
	ss    = sss[ilow:ihigh+1]
	ss1   = sss1[ilow:ihigh+1]








	# Spectra modeling
	fen = (fe-120)/60

	fit_spec_ambient    = fit_polynomial_fourier('fourier',    fen, sa,     17,  plot='no')
	fit_spec_hot        = fit_polynomial_fourier('fourier',    fen, sh,     17,  plot='no')
	fit_spec_open       = fit_polynomial_fourier('fourier',    fen, so,    121,  plot='no')
	fit_spec_shorted    = fit_polynomial_fourier('fourier',    fen, ss,    121,  plot='no')
	fit_spec_sim        = fit_polynomial_fourier('fourier',    fen, ss1,    37,  plot='no')

	model_spec_ambient  = model_evaluate('fourier', fit_spec_ambient[0],    fen)
	model_spec_hot      = model_evaluate('fourier', fit_spec_hot[0],        fen)
	model_spec_open     = model_evaluate('fourier', fit_spec_open[0],       fen)
	model_spec_shorted  = model_evaluate('fourier', fit_spec_shorted[0],    fen)
	model_spec_sim      = model_evaluate('fourier', fit_spec_sim[0],        fen)



	# Loading S11 data
	s11_all = np.genfromtxt(path_s11 + 's11_calibration_mid_band_LNA25degC_2018-08-13-22-05-34.txt')
	s11     = s11_all[(s11_all[:,0]>=flow) & (s11_all[:,0]<=fhigh), :]
	
	



	# Frequency / complex data
	f_s11       = s11[:, 0]

	s11_LNA     = s11[:, 1]  + 1j*s11[:, 2]

	s11_amb     = s11[:, 3]  + 1j*s11[:, 4]
	s11_hot     = s11[:, 5]  + 1j*s11[:, 6]

	s11_open    = s11[:, 7]  + 1j*s11[:, 8]
	s11_shorted = s11[:, 9]  + 1j*s11[:, 10]

	s11_sr      = s11[:, 11] + 1j*s11[:, 12]
	s12s21_sr   = s11[:, 13] + 1j*s11[:, 14]
	s22_sr      = s11[:, 15] + 1j*s11[:, 16]

	s11_simu    = s11[:, 17] + 1j*s11[:, 18]





	# Magnitude / Angle

	# LNA
	s11_LNA_mag     = np.abs(s11_LNA)
	s11_LNA_ang     = np.unwrap(np.angle(s11_LNA))

	# Ambient
	s11_amb_mag     = np.abs(s11_amb)
	s11_amb_ang     = np.unwrap(np.angle(s11_amb))

	# Hot
	s11_hot_mag     = np.abs(s11_hot)
	s11_hot_ang     = np.unwrap(np.angle(s11_hot))

	# Open
	s11_open_mag    = np.abs(s11_open)
	s11_open_ang    = np.unwrap(np.angle(s11_open))

	# Shorted
	s11_shorted_mag = np.abs(s11_shorted)
	s11_shorted_ang = np.unwrap(np.angle(s11_shorted))

	# sr-s11
	s11_sr_mag      = np.abs(s11_sr)
	s11_sr_ang      = np.unwrap(np.angle(s11_sr))

	# sr-s12s21
	s12s21_sr_mag   = np.abs(s12s21_sr)
	s12s21_sr_ang   = np.unwrap(np.angle(s12s21_sr))

	# sr-s22
	s22_sr_mag      = np.abs(s22_sr)
	s22_sr_ang      = np.unwrap(np.angle(s22_sr))

	# Simu1
	s11_simu_mag    = np.abs(s11_simu)
	s11_simu_ang    = np.unwrap(np.angle(s11_simu))








	# Modeling S11

	f_s11n = (f_s11-120)/60

	fit_s11_LNA_mag     = fit_polynomial_fourier('polynomial', f_s11n, s11_LNA_mag,     19, plot='no')  # 
	fit_s11_LNA_ang     = fit_polynomial_fourier('polynomial', f_s11n, s11_LNA_ang,     19, plot='no')  # 

	fit_s11_amb_mag     = fit_polynomial_fourier('fourier',    f_s11n, s11_amb_mag,     27, plot='no')  # 
	fit_s11_amb_ang     = fit_polynomial_fourier('fourier',    f_s11n, s11_amb_ang,     27, plot='no')  # 

	fit_s11_hot_mag     = fit_polynomial_fourier('fourier',    f_s11n, s11_hot_mag,     27, plot='no')  #  
	fit_s11_hot_ang     = fit_polynomial_fourier('fourier',    f_s11n, s11_hot_ang,     27, plot='no')  # 

	fit_s11_open_mag    = fit_polynomial_fourier('fourier',    f_s11n, s11_open_mag,    85, plot='no')  # 27
	fit_s11_open_ang    = fit_polynomial_fourier('fourier',    f_s11n, s11_open_ang,    85, plot='no')  # 27

	fit_s11_shorted_mag = fit_polynomial_fourier('fourier',    f_s11n, s11_shorted_mag, 85, plot='no')  # 27
	fit_s11_shorted_ang = fit_polynomial_fourier('fourier',    f_s11n, s11_shorted_ang, 85, plot='no')  # 27

	fit_s11_sr_mag      = fit_polynomial_fourier('polynomial', f_s11n, s11_sr_mag,      17, plot='no')  # 
	fit_s11_sr_ang      = fit_polynomial_fourier('polynomial', f_s11n, s11_sr_ang,      17, plot='no')  # 

	fit_s12s21_sr_mag   = fit_polynomial_fourier('polynomial', f_s11n, s12s21_sr_mag,   17, plot='no')  # 
	fit_s12s21_sr_ang   = fit_polynomial_fourier('polynomial', f_s11n, s12s21_sr_ang,   17, plot='no')  # 

	fit_s22_sr_mag      = fit_polynomial_fourier('polynomial', f_s11n, s22_sr_mag,      17, plot='no')  # 
	fit_s22_sr_ang      = fit_polynomial_fourier('polynomial', f_s11n, s22_sr_ang,      17, plot='no')  # 

	fit_s11_simu_mag    = fit_polynomial_fourier('polynomial', f_s11n, s11_simu_mag,    35, plot='no')  # 7
	fit_s11_simu_ang    = fit_polynomial_fourier('polynomial', f_s11n, s11_simu_ang,    35, plot='no')  # 7


	r1  = fit_s11_LNA_mag[1]  - s11_LNA_mag
	r2  = fit_s11_LNA_ang[1]  - s11_LNA_ang
	r3  = fit_s11_amb_mag[1]  - s11_amb_mag
	r4  = fit_s11_amb_ang[1]  - s11_amb_ang
	r5  = fit_s11_hot_mag[1]  - s11_hot_mag
	r6  = fit_s11_hot_ang[1]  - s11_hot_ang
	r7  = fit_s11_open_mag[1] - s11_open_mag
	r8  = fit_s11_open_ang[1] - s11_open_ang
	r9  = fit_s11_shorted_mag[1] - s11_shorted_mag
	r10 = fit_s11_shorted_ang[1] - s11_shorted_ang
	r11 = fit_s11_sr_mag[1] - s11_sr_mag
	r12 = fit_s11_sr_ang[1] - s11_sr_ang
	r13 = fit_s12s21_sr_mag[1] - s12s21_sr_mag
	r14 = fit_s12s21_sr_ang[1] - s12s21_sr_ang
	r15 = fit_s22_sr_mag[1] - s22_sr_mag
	r16 = fit_s22_sr_ang[1] - s22_sr_ang	
	r17 = fit_s11_simu_mag[1] - s11_simu_mag
	r18 = fit_s11_simu_ang[1] - s11_simu_ang	
	
	

	# Saving output parameters
	if save == 'yes':


		# Average spectra data in frequency range selected
		spectra = np.array([fe, sa, sh, so, ss, ss1]).T		

		# RMS residuals
		RMS_spectra = np.zeros((5,1))
		RMS_s11     = np.zeros((18,1))

		# Spectra
		RMS_spectra[0, 0] = fit_spec_ambient[2]
		RMS_spectra[1, 0] = fit_spec_hot[2]
		RMS_spectra[2, 0] = fit_spec_open[2]
		RMS_spectra[3, 0] = fit_spec_shorted[2]
		RMS_spectra[4, 0] = fit_spec_sim[2]


		# S11
		RMS_s11[0, 0]  = fit_s11_LNA_mag[2]
		RMS_s11[1, 0]  = fit_s11_LNA_ang[2]
		RMS_s11[2, 0]  = fit_s11_amb_mag[2]
		RMS_s11[3, 0]  = fit_s11_amb_ang[2]
		RMS_s11[4, 0]  = fit_s11_hot_mag[2]
		RMS_s11[5, 0]  = fit_s11_hot_ang[2]
		RMS_s11[6, 0]  = fit_s11_open_mag[2]
		RMS_s11[7, 0]  = fit_s11_open_ang[2]
		RMS_s11[8, 0]  = fit_s11_shorted_mag[2]
		RMS_s11[9, 0]  = fit_s11_shorted_ang[2]
		RMS_s11[10, 0] = fit_s11_sr_mag[2]
		RMS_s11[11, 0] = fit_s11_sr_ang[2]
		RMS_s11[12, 0] = fit_s12s21_sr_mag[2]
		RMS_s11[13, 0] = fit_s12s21_sr_ang[2]
		RMS_s11[14, 0] = fit_s22_sr_mag[2]
		RMS_s11[15, 0] = fit_s22_sr_ang[2]
		RMS_s11[16, 0] = fit_s11_simu_mag[2]
		RMS_s11[17, 0] = fit_s11_simu_ang[2]



		# Formating fit parameters

		# Physical temperature
		phys_temp = np.zeros((5,1))
		phys_temp[0,0] = phys_temp_ambient
		phys_temp[1,0] = phys_temp_hot
		phys_temp[2,0] = phys_temp_open
		phys_temp[3,0] = phys_temp_shorted
		phys_temp[4,0] = phys_temp_sim



		# Spectra
		par_spec_ambient    = np.reshape(fit_spec_ambient[0],    (-1,1))
		par_spec_hot        = np.reshape(fit_spec_hot[0],        (-1,1))
		par_spec_open       = np.reshape(fit_spec_open[0],       (-1,1))
		par_spec_shorted    = np.reshape(fit_spec_shorted[0],    (-1,1))
		par_spec_sim        = np.reshape(fit_spec_sim[0],        (-1,1))



		# S11
		par_s11_LNA_mag     = np.reshape(fit_s11_LNA_mag[0],     (-1,1))
		par_s11_LNA_ang     = np.reshape(fit_s11_LNA_ang[0],     (-1,1))
		par_s11_amb_mag     = np.reshape(fit_s11_amb_mag[0],     (-1,1))
		par_s11_amb_ang     = np.reshape(fit_s11_amb_ang[0],     (-1,1))
		par_s11_hot_mag     = np.reshape(fit_s11_hot_mag[0],     (-1,1))
		par_s11_hot_ang     = np.reshape(fit_s11_hot_ang[0],     (-1,1))
		par_s11_open_mag    = np.reshape(fit_s11_open_mag[0],    (-1,1))
		par_s11_open_ang    = np.reshape(fit_s11_open_ang[0],    (-1,1))
		par_s11_shorted_mag = np.reshape(fit_s11_shorted_mag[0], (-1,1))
		par_s11_shorted_ang = np.reshape(fit_s11_shorted_ang[0], (-1,1))

		par_s11_sr_mag      = np.reshape(fit_s11_sr_mag[0],      (-1,1))
		par_s11_sr_ang      = np.reshape(fit_s11_sr_ang[0],      (-1,1))
		par_s12s21_sr_mag   = np.reshape(fit_s12s21_sr_mag[0],   (-1,1))
		par_s12s21_sr_ang   = np.reshape(fit_s12s21_sr_ang[0],   (-1,1))
		par_s22_sr_mag      = np.reshape(fit_s22_sr_mag[0],      (-1,1))
		par_s22_sr_ang      = np.reshape(fit_s22_sr_ang[0],      (-1,1))

		par_s11_simu_mag    = np.reshape(fit_s11_simu_mag[0],    (-1,1))
		par_s11_simu_ang    = np.reshape(fit_s11_simu_ang[0],    (-1,1))



		# Saving

		np.savetxt(path_data + 'average_spectra_300_350.txt', spectra)

		np.savetxt(path_par_temp    + 'physical_temperatures.txt', phys_temp)

		np.savetxt(path_par_spectra + 'par_spec_amb.txt',     par_spec_ambient)
		np.savetxt(path_par_spectra + 'par_spec_hot.txt',     par_spec_hot)
		np.savetxt(path_par_spectra + 'par_spec_open.txt',    par_spec_open)
		np.savetxt(path_par_spectra + 'par_spec_shorted.txt', par_spec_shorted)
		np.savetxt(path_par_spectra + 'par_spec_simu.txt',    par_spec_sim)
		np.savetxt(path_par_spectra + 'RMS_spec.txt',         RMS_spectra)

		np.savetxt(path_par_s11 + 'par_s11_LNA_mag.txt',      par_s11_LNA_mag)
		np.savetxt(path_par_s11 + 'par_s11_LNA_ang.txt',      par_s11_LNA_ang)
		np.savetxt(path_par_s11 + 'par_s11_amb_mag.txt',      par_s11_amb_mag)
		np.savetxt(path_par_s11 + 'par_s11_amb_ang.txt',      par_s11_amb_ang)
		np.savetxt(path_par_s11 + 'par_s11_hot_mag.txt',      par_s11_hot_mag)
		np.savetxt(path_par_s11 + 'par_s11_hot_ang.txt',      par_s11_hot_ang)
		np.savetxt(path_par_s11 + 'par_s11_open_mag.txt',     par_s11_open_mag)
		np.savetxt(path_par_s11 + 'par_s11_open_ang.txt',     par_s11_open_ang)
		np.savetxt(path_par_s11 + 'par_s11_shorted_mag.txt',  par_s11_shorted_mag)
		np.savetxt(path_par_s11 + 'par_s11_shorted_ang.txt',  par_s11_shorted_ang)

		np.savetxt(path_par_s11 + 'par_s11_sr_mag.txt',       par_s11_sr_mag)
		np.savetxt(path_par_s11 + 'par_s11_sr_ang.txt',       par_s11_sr_ang)
		np.savetxt(path_par_s11 + 'par_s12s21_sr_mag.txt',    par_s12s21_sr_mag)
		np.savetxt(path_par_s11 + 'par_s12s21_sr_ang.txt',    par_s12s21_sr_ang)
		np.savetxt(path_par_s11 + 'par_s22_sr_mag.txt',       par_s22_sr_mag)
		np.savetxt(path_par_s11 + 'par_s22_sr_ang.txt',       par_s22_sr_ang)

		np.savetxt(path_par_s11 + 'par_s11_simu_mag.txt',     par_s11_simu_mag)
		np.savetxt(path_par_s11 + 'par_s11_simu_ang.txt',     par_s11_simu_ang)		



		np.savetxt(path_par_s11 + 'RMS_s11.txt',       	      RMS_s11)


	# End
	print('Files processed.')


	# f_s11, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18
	return 0   #fe, sa, sh, so, ss, ss1














def calibration_file_computation(FMIN, FMAX, cterms, wterms, save='no', save_flag=''):
	
	path_save = '/home/ramo7131/DATA/EDGES/calibration/receiver_calibration/mid_band/2018_01_25C/results/nominal/calibration_files/'
	
	
	
	
	# Spectra
	Tunc = np.genfromtxt('/home/ramo7131/DATA/EDGES/calibration/receiver_calibration/mid_band/2018_01_25C/results/nominal/data/average_spectra_300_350.txt')
	ff    = Tunc[:,0]
	TTae  = Tunc[:,1]
	TThe  = Tunc[:,2]
	TToe  = Tunc[:,3]
	TTse  = Tunc[:,4]
	TTsse = Tunc[:,5]
	
	
	f    = ff[(ff>=FMIN) & (ff<=FMAX)]
	Tae  = TTae[(ff>=FMIN) & (ff<=FMAX)]
	The  = TThe[(ff>=FMIN) & (ff<=FMAX)]
	Toe  = TToe[(ff>=FMIN) & (ff<=FMAX)]
	Tse  = TTse[(ff>=FMIN) & (ff<=FMAX)]
	Tsse = TTsse[(ff>=FMIN) & (ff<=FMAX)]
	
	
	
	
	
	
	
	
	# Physical temperature
	Tphys = np.genfromtxt('/home/ramo7131/DATA/EDGES/calibration/receiver_calibration/mid_band/2018_01_25C/results/nominal/temp/physical_temperatures.txt')
	Ta   = Tphys[0]
	Th   = Tphys[1]
	To   = Tphys[2]
	Ts   = Tphys[3]
	Tsim = Tphys[4]
	
	
	
	
	
	path_s11 = '/home/ramo7131/DATA/EDGES/calibration/receiver_calibration/mid_band/2018_01_25C/results/nominal/s11/'
	fn  = (f-120)/60
	
	par     = np.genfromtxt(path_s11 + 'par_s11_LNA_mag.txt')	
	rl_mag  = model_evaluate('polynomial', par, fn)
	par     = np.genfromtxt(path_s11 + 'par_s11_LNA_ang.txt')
	rl_ang  = model_evaluate('polynomial', par, fn)	
	rl      = rl_mag * (np.cos(rl_ang) + 1j*np.sin(rl_ang))

	
	par     = np.genfromtxt(path_s11 + 'par_s11_amb_mag.txt')
	ra_mag  = model_evaluate('fourier', par, fn)	
	par     = np.genfromtxt(path_s11 + 'par_s11_amb_ang.txt')
	ra_ang  = model_evaluate('fourier', par, fn)
	ra      = ra_mag * (np.cos(ra_ang) + 1j*np.sin(ra_ang))
	
	
	par     = np.genfromtxt(path_s11 + 'par_s11_hot_mag.txt')
	rh_mag  = model_evaluate('fourier', par, fn)
	par     = np.genfromtxt(path_s11 + 'par_s11_hot_ang.txt')
	rh_ang  = model_evaluate('fourier', par, fn)
	rh      = rh_mag * (np.cos(rh_ang) + 1j*np.sin(rh_ang))
	
	
	par     = np.genfromtxt(path_s11 + 'par_s11_open_mag.txt')
	ro_mag  = model_evaluate('fourier', par, fn)	
	par     = np.genfromtxt(path_s11 + 'par_s11_open_ang.txt')
	ro_ang  = model_evaluate('fourier', par, fn)
	ro      = ro_mag * (np.cos(ro_ang) + 1j*np.sin(ro_ang))
	
	
	par     = np.genfromtxt(path_s11 + 'par_s11_shorted_mag.txt')
	rs_mag  = model_evaluate('fourier', par, fn)	
	par     = np.genfromtxt(path_s11 + 'par_s11_shorted_ang.txt')
	rs_ang  = model_evaluate('fourier', par, fn)		
	rs      = rs_mag * (np.cos(rs_ang) + 1j*np.sin(rs_ang))
	
	
	
	
	par         = np.genfromtxt(path_s11 + 'par_s11_sr_mag.txt')
	s11_sr_mag  = model_evaluate('polynomial', par, fn)
	par         = np.genfromtxt(path_s11 + 'par_s11_sr_ang.txt')
	s11_sr_ang  = model_evaluate('polynomial', par, fn)
	s11_sr      = s11_sr_mag * (np.cos(s11_sr_ang) + 1j*np.sin(s11_sr_ang))
	
	
	par            = np.genfromtxt(path_s11 + 'par_s12s21_sr_mag.txt')
	s12s21_sr_mag  = model_evaluate('polynomial', par, fn)	
	par            = np.genfromtxt(path_s11 + 'par_s12s21_sr_ang.txt')
	s12s21_sr_ang  = model_evaluate('polynomial', par, fn)
	s12s21_sr      = s12s21_sr_mag * (np.cos(s12s21_sr_ang) + 1j*np.sin(s12s21_sr_ang))
	
	
	par         = np.genfromtxt(path_s11 + 'par_s22_sr_mag.txt')
	s22_sr_mag  = model_evaluate('polynomial', par, fn)
	par         = np.genfromtxt(path_s11 + 'par_s22_sr_ang.txt')
	s22_sr_ang  = model_evaluate('polynomial', par, fn)
	s22_sr      = s22_sr_mag * (np.cos(s22_sr_ang) + 1j*np.sin(s22_sr_ang))
	
	
	

	par        = np.genfromtxt(path_s11 + 'par_s11_simu_mag.txt')
	rsimu_mag  = model_evaluate('polynomial', par, fn)
	par        = np.genfromtxt(path_s11 + 'par_s11_simu_ang.txt')
	rsimu_ang  = model_evaluate('polynomial', par, fn)
	rsimu      = rsimu_mag * (np.cos(rsimu_ang) + 1j*np.sin(rsimu_ang))






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





	
	
	
	# Calibration quantities
	Tamb_internal = 300
	C1, C2, TU, TC, TS = rcv.calibration_quantities('nothing', fn, Tae, The, Toe, Tse, rl, ra, rh, ro, rs, Ta, Thd, To, Ts, Tamb_internal, cterms, wterms)




	# Cross-check	
	Tac         = eg.calibrated_antenna_temperature(Tae,  ra,  rl, C1, C2, TU, TC, TS, Tamb_internal=300)
	fb, tab, wb = eg.spectral_binning_number_of_samples(f, Tac, np.ones(len(f)), nsamples=64)

	Thc         = eg.calibrated_antenna_temperature(The,  rh,  rl, C1, C2, TU, TC, TS, Tamb_internal=300)
	fb, thb, wb = eg.spectral_binning_number_of_samples(f, Thc, np.ones(len(f)), nsamples=64)

	Toc         = eg.calibrated_antenna_temperature(Toe,  ro,  rl, C1, C2, TU, TC, TS, Tamb_internal=300)
	fb, tob, wb = eg.spectral_binning_number_of_samples(f, Toc, np.ones(len(f)), nsamples=64)

	Tsc         = eg.calibrated_antenna_temperature(Tse,  rs,  rl, C1, C2, TU, TC, TS, Tamb_internal=300)
	fb, tsb, wb = eg.spectral_binning_number_of_samples(f, Tsc, np.ones(len(f)), nsamples=64)

	Tsc         = eg.calibrated_antenna_temperature(Tse,  rs,  rl, C1, C2, TU, TC, TS, Tamb_internal=300)
	fb, tsb, wb = eg.spectral_binning_number_of_samples(f, Tsc, np.ones(len(f)), nsamples=64)


	Tsimuc         = eg.calibrated_antenna_temperature(Tsse,  rsimu,  rl, C1, C2, TU, TC, TS, Tamb_internal=300)
	fb, tsimub, wb = eg.spectral_binning_number_of_samples(f, Tsimuc, np.ones(len(f)), nsamples=64)


	# Why do I NOT get 32 degC for the Ant Sim 3 ???



	if save == 'yes':

		# Array
		save_array = np.zeros((len(f), 8))
		save_array[:,0] = f
		save_array[:,1] = np.real(rl)
		save_array[:,2] = np.imag(rl)
		save_array[:,3] = C1
		save_array[:,4] = C2
		save_array[:,5] = TU
		save_array[:,6] = TC
		save_array[:,7] = TS

		# Save
		np.savetxt(path_save + 'calibration_file_mid_band' + save_flag + '.txt', save_array, fmt='%1.8f')





	return f, Ta, Thd, To, Ts, Tsim, fb, tab, thb, tob, tsb, tsimub






