
import numpy as np
import scipy as sp
import reflection_coefficient as rc
import time  as tt
import edges as eg
import datetime as dt


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








def temperature_thermistor_oven_industries_TR136_170(R, unit):

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









def average_calibration_spectrum(spectrum_files, resistance_file, start_percent=0, plot='no'):
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
	ta_sel = ta[index_start_spectra::,:]
	av_ta = np.mean(ta_sel, axis=0)



	# temperature
	if isinstance(resistance_file, list):
		for i in range(len(resistance_file)):
			if i == 0:
				R = np.genfromtxt(resistance_file[i])
			else:	
				R = np.concatenate((R, np.genfromtxt(resistance_file[i])), axis=0)
	else:
		R = np.genfromtxt(resistance_file)


	temp = temperature_thermistor_oven_industries_TR136_170(R, 'K')
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
		plt.plot(ta_sel[:,30000],'r')
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

	return av_ta, av_temp

















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
	C1, C2, TU, TC, TS = eg.calibration_quantities('nothing', fn, Tae, The, Toe, Tse, rl, ra, rh, ro, rs, Ta, Thd, To, Ts, Tamb_internal, cterms, wterms)




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











def FEKO_mid_band_blade_beam(beam_file=1, frequency_interpolation='no', frequency=np.array([0]), AZ_antenna_axis=0):

	"""

	AZ_antenna_axis = 0    #  Angle of orientation (in integer degrees) of excited antenna panels relative to due North. Best value is around X ???

	"""




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


















def antenna_beam_factor(name_save, beam_file=1, rotation_from_north=90, band_deg=10, index_inband=2.5, index_outband=2.62, reference_frequency=100):



#(band, antenna, name_save, file_high_band_blade_FEKO=1, file_low_band_blade_FEKO=1, reference_frequency=140, rotation_from_north=-5, sky_model='guzman_haslam', band_deg=10, index_inband=2.5, index_outband=2.57):


	band      = 'mid_band'
	sky_model = 'haslam'
	


	# Data paths
	path_data = edges_folder + 'sky_models/'
	path_save = edges_folder + band + '/calibration/beam_factors/raw/'


	# Loading beam	
	AZ_beam  = np.arange(0, 360)
	EL_beam  = np.arange(0, 91)



	# FEKO blade beam
	
	# Fixing rotation angle due to diferent rotation (by 90deg) in Nivedita's map
	if beam_file == 3:
		rotation_from_north = rotation_from_north - 90
		
	beam_all = FEKO_mid_band_blade_beam(beam_file=beam_file, AZ_antenna_axis=rotation_from_north)



	# Frequency array
	if beam_file == 1:
		# ALAN #1
		freq_array = np.arange(50, 201, 2, dtype='uint32')  

	elif beam_file == 2:
		# ALAN #2
		freq_array = np.arange(50, 201, 2, dtype='uint32')  

	elif beam_file == 3:
		# NIVEDITA
		freq_array = np.arange(60, 201, 2, dtype='uint32')  
		
		
	# Index of reference frequency
	index_freq_array = np.arange(len(freq_array))
	irf = index_freq_array[freq_array == reference_frequency]
	print('Reference frequency: ' + str(freq_array[irf][0]) + ' MHz')



	if sky_model == 'haslam':

		# Loading galactic coordinates (the Haslam map is in NESTED Galactic Coordinates)
		coord              = fits.open(path_data + 'coordinate_maps/pixel_coords_map_nested_galactic_res9.fits')
		coord_array        = coord[1].data
		lon                = coord_array['LONGITUDE']
		lat                = coord_array['LATITUDE']
		GALAC_COORD_object = apc.SkyCoord(lon, lat, frame='galactic', unit='deg')  # defaults to ICRS frame

		# Loading Haslam map
		haslam_map = fits.open(path_data + 'haslam_map/lambda_haslam408_dsds.fits')
		haslam408  = (haslam_map[1].data)['temperature']

		# Scaling Haslam map (the map contains the CMB, which has to be removed at 408 MHz, and then added back)
		Tcmb   = 2.725
		T408   = haslam408 - Tcmb
		b0     = band_deg          # default 10 degrees, galactic elevation threshold for different spectral index


		haslam = np.zeros((len(T408), len(freq_array)))
		for i in range(len(freq_array)):

			# Band of the Galactic center, using spectral index
			haslam[(lat >= -b0) & (lat <= b0), i] = T408[(lat >= -b0) & (lat <= b0)] * (freq_array[i]/408)**(-index_inband) + Tcmb

			# Range outside the Galactic center, using second spectral index
			haslam[(lat < -b0) | (lat > b0), i]   = T408[(lat < -b0) | (lat > b0)] * (freq_array[i]/408)**(-index_outband) + Tcmb



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
	for i in range(len(LST)):  # range(1):   #


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



		# Selecting coordinates and sky data above the horizon
		AZ_above_horizon         = AZ[EL>=0]
		EL_above_horizon         = EL[EL>=0]


		if sky_model == 'haslam':
			haslam_above_horizon     = haslam[EL>=0,:]
			haslam_ref_above_horizon = haslam_above_horizon[:, irf].flatten()







		# Arranging AZ and EL arrays corresponding to beam model
		az_array   = np.tile(AZ_beam,91)
		el_array   = np.repeat(EL_beam,360)
		az_el_original      = np.array([az_array, el_array]).T
		az_el_above_horizon = np.array([AZ_above_horizon, EL_above_horizon]).T



		# Precomputation of beam at reference frequency for normalization
		beam_array_v0         = beam_all[irf,:,:].reshape(1,-1)[0]
		beam_above_horizon_v0 = spi.griddata(az_el_original, beam_array_v0, az_el_above_horizon, method='cubic')  # interpolated beam




		# Loop over frequency
		for j in range(len(freq_array)):

			print(name_save + ', Freq: ' + str(j) + ' out of ' + str(len(beam_all[:,0,0])))

			beam_array         = beam_all[j,:,:].reshape(1,-1)[0]
			beam_above_horizon = spi.griddata(az_el_original, beam_array, az_el_above_horizon, method='cubic')  # interpolated beam


			no_nan_array = np.ones(len(AZ_above_horizon)) - np.isnan(beam_above_horizon)
			index_no_nan = np.nonzero(no_nan_array)[0]

	
			if sky_model == 'haslam':				
				convolution_ref[i, j]   = np.sum(beam_above_horizon[index_no_nan]*haslam_ref_above_horizon[index_no_nan])/np.sum(beam_above_horizon[index_no_nan])

				# Antenna temperature
				haslam_above_horizon_ff = haslam_above_horizon[:, j].flatten()
				convolution[i, j]       = np.sum(beam_above_horizon[index_no_nan]*haslam_above_horizon_ff[index_no_nan])/np.sum(beam_above_horizon[index_no_nan])



	if sky_model == 'haslam':
		beam_factor_T = convolution_ref.T/convolution_ref[:,irf].T
		beam_factor   = beam_factor_T.T



	# Saving beam factor
	np.savetxt(path_save + name_save + '_tant.txt', convolution)
	np.savetxt(path_save + name_save + '_data.txt', beam_factor)
	np.savetxt(path_save + name_save + '_LST.txt',  LST)
	np.savetxt(path_save + name_save + '_freq.txt', freq_array)



	return freq_array, LST, convolution, beam_factor
























def antenna_beam_factor_interpolation(band, lst_hires, fnew):

	"""


	"""



	# Mid-Band
	if band == 'mid_band':

		file_path = edges_folder + 'mid_band/calibration/beam_factors/raw/'
		bf_old  = np.genfromtxt(file_path + 'mid_band_50-200MHz_90deg_alan1_haslam_2.5_2.62_reffreq_100MHz_data.txt')
		freq    = np.genfromtxt(file_path + 'mid_band_50-200MHz_90deg_alan1_haslam_2.5_2.62_reffreq_100MHz_freq.txt')
		lst_old = np.genfromtxt(file_path + 'mid_band_50-200MHz_90deg_alan1_haslam_2.5_2.62_reffreq_100MHz_LST.txt')

	


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














def beam_factor_table_computation(f, N_lst, file_name_hdf5):


	# Produce the beam factor at high resolution
	# ------------------------------------------
	#N_lst = 6000   # number of LST points within 24 hours
	
	
	lst_hires = np.arange(0,24,24/N_lst)
	bf        = antenna_beam_factor_interpolation('mid_band', lst_hires, f)
	
	
	
	# Save
	# ----
	file_path = edges_folder + '/mid_band/calibration/beam_factors/raw/'
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
























def corrected_antenna_s11(day, save_name_flag):
	
	# Creating folder if necessary
	path_save = home_folder + '/DATA/EDGES/calibration/antenna_s11/mid_band/s11/corrected/2018_' + str(day) + '/'
	if not exists(path_save):
		makedirs(path_save)	
	





	
	# Mid-Band
	if day == 145:
		path_antenna_s11 = '/home/ramo7131/DATA/EDGES/calibration/antenna_s11/mid_band/s11/raw/mid-recv1-20180525/'	
		date_all = ['145_08_28_27', '145_08_28_27']
	
	
	# Mid-Band
	if day == 147:
		path_antenna_s11 = '/home/ramo7131/DATA/EDGES/calibration/antenna_s11/mid_band/s11/raw/mid-rcv1-20180527/'	
		date_all = ['147_17_03_25', '147_17_03_25']
		
		
	# Mid-Band
	if day == 222:
		path_antenna_s11 = '/home/ramo7131/DATA/EDGES/calibration/antenna_s11/mid_band/s11/raw/mid_20180809/'
		date_all = ['222_05_41_09', '222_05_43_53', '222_05_45_58', '222_05_48_06']
	
			
	# Alan's noise source + 6dB attenuator
	if day == 223:
		path_antenna_s11 = '/home/ramo7131/DATA/EDGES/calibration/antenna_s11/mid_band/s11/raw/mid_20180812_ans1_6db/'
		date_all = ['223_22_40_15', '223_22_41_22', '223_22_42_44']  # Leaving the first one out: '223_22_38_08'


	# Low-Band 3 (Receiver 1, Low-Band 1 antenna, High-Band ground plane with wings)
	if day == 225:  
		path_antenna_s11 = '/home/ramo7131/DATA/EDGES/calibration/antenna_s11/low_band3/s11/raw/low3_20180813_rcv1/'
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
		a = switch_correction_mid_band_2018_01_25C(a_sw_c, f_in = f_m)
		
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

















