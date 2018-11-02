
import numpy as np
import scipy as sp
import reflection_coefficient as rc
import time  as tt
import edges as eg
import datetime as dt

import scipy.io as sio
import scipy.interpolate as spi

import matplotlib.pyplot as plt

import astropy.units as apu
import astropy.time as apt
import astropy.coordinates as apc





from os.path import expanduser
from os.path import exists
from os import listdir, makedirs, system

from astropy.io import fits






# Determining home folder
home_folder = expanduser("~")



















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















def fit_polynomial_fourier(model_type, xdata, ydata, nterms, Weights=1, plot='no', fr=150, df=10, zr=8, dz=2, z_alpha=0, anastasia_model_number=0, jordan_model_number=0, xi_min=0.9, jordan_tau_e_min=0.02, jordan_tau_e_max=0.25, gaussian_flatness_tau=0, gaussian_flatness_tilt=0, external_model_in_K=0):
	"""
	Last modification: May 24, 2015.

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
			AT[i,:] = (xdata**(-2.3))  *  ((np.log(xdata))**i)



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




	data_folder = home_folder + '/DATA/EDGES/calibration/beam/alan/blade_mid_band/'


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
	path_data = home_folder + '/DATA/EDGES/calibration/sky/'
	path_save = home_folder + '/DATA/EDGES/calibration/beam_factors/' + band + '/'


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
	for i in range(len(LST)): # range(1):


		print(name_save + '. LST: ' + str(i) + ' out of 72')


		# Advancing time ( 19:57 minutes UTC correspond to 20 minutes LST )
		minutes_offset = 19
		seconds_offset = 57
		if i > 0:
			Time_iter_dt = Time_iter_dt + dt.timedelta(minutes = minutes_offset, seconds = seconds_offset)
			Time_iter    = np.array([Time_iter_dt.year, Time_iter_dt.month, Time_iter_dt.day, Time_iter_dt.hour, Time_iter_dt.minute, Time_iter_dt.second]) 



		# LST 
		LST[i] = eg.utc2lst(Time_iter, EDGES_lon_deg)



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

























def antenna_beam_factor_interpolation(lst_hires, fnew):

	"""


	"""



	file_path = '/DATA/EDGES/calibration/beam_factors/mid_band/'

	# Low-Band 1, NIVEDITA, Extended Ground Plane
	bf_old  = np.genfromtxt(home_folder + file_path + 'mid_band_50-200MHz_90deg_alan1_haslam_2.5_2.62_reffreq_100MHz_data.txt')
	freq    = np.genfromtxt(home_folder + file_path + 'mid_band_50-200MHz_90deg_alan1_haslam_2.5_2.62_reffreq_100MHz_freq.txt')
	lst_old = np.genfromtxt(home_folder + file_path + 'mid_band_50-200MHz_90deg_alan1_haslam_2.5_2.62_reffreq_100MHz_LST.txt')

	


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


















	
def models_antenna_s11_remove_delay(f_MHz, antenna_s11_day = 145, delay_0=0.17, model_type='polynomial', Nfit=10, plot_fit_residuals='no', MC_mag='no', MC_ang='no', sigma_mag=0.0001, sigma_ang_deg=0.1):	


	


	# Paths
	path_data = home_folder + '/DATA/EDGES/calibration/antenna_s11/mid_band/s11/corrected/'


	# Mid-Band antenna S11
	if (antenna_s11_day == 145):
		d = np.genfromtxt(path_data + '2018_145/antenna_s11_2018_145.txt'); print('Antenna S11: 2018-145')

	if (antenna_s11_day == 147):
		d = np.genfromtxt(path_data + '2018_147/antenna_s11_2018_147.txt'); print('Antenna S11: 2018-147')

	if (antenna_s11_day == 222):
		d = np.genfromtxt(path_data + '2018_222/antenna_s11_2018_222.txt'); print('Antenna S11: 2018-222')





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
	par_re_wd = np.polyfit(f_orig_MHz, re_wd, Nfit-1)
	par_im_wd = np.polyfit(f_orig_MHz, im_wd, Nfit-1)
	
	
	# Evaluating models at original frequency for evaluation
	rX = np.polyval(par_re_wd, f_orig_MHz)
	iX = np.polyval(par_im_wd, f_orig_MHz)

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
	model_re_wd = np.polyval(par_re_wd, f_MHz)
	model_im_wd = np.polyval(par_im_wd, f_MHz)

	model_s11_wd = model_re_wd + 1j*model_im_wd
	ra    = model_s11_wd * np.exp(-1j*delay_0 * f_MHz)







	# Monte Carlo realizations
	# MC_mag='no', MC_ang='no', sigma_mag=0.0001, sigma_ang_deg=0.1
	if (MC_mag == 'yes') or (MC_ang == 'yes'):

		# Magnitude and angle
		abs_s11 = np.abs(ra)
		ang_s11 = np.angle(ra)


		# Uncertainty in magnitude
		if MC_mag == 'yes':
			noise     = np.random.uniform(np.zeros(len(f_MHz)), np.ones(len(f_MHz))) - 0.5
			nterms    = np.random.randint(1,16) # up to 15 terms
			par_poly  = np.polyfit(f_MHz/200, noise, nterms-1)
			poly      = np.polyval(par_poly, f_MHz/200)
			RMS       = np.sqrt(np.sum(np.abs(poly)**2)/len(poly))
			norm_poly = poly/RMS  # normalize to have RMS of ONE

			#sigma_mag = 0.0001	
			abs_s11   = abs_s11 + norm_poly*sigma_mag*np.random.normal()


		# Uncertainty in phase
		if MC_ang == 'yes':
			noise     = np.random.uniform(np.zeros(len(f_MHz)), np.ones(len(f_MHz))) - 0.5
			nterms    = np.random.randint(1,16) # up to 15 terms
			par_poly  = np.polyfit(f_MHz/200, noise, nterms-1)
			poly      = np.polyval(par_poly, f_MHz/200)
			RMS       = np.sqrt(np.sum(np.abs(poly)**2)/len(poly))
			norm_poly = poly/RMS  # normalize to have RMS of ONE

			#sigma_ang_deg = 0.1
			sigma_ang     = (np.pi/180)*sigma_ang_deg
			ang_s11       = ang_s11 + norm_poly*sigma_ang*np.random.normal()


		# MC realization of the antenna reflection coefficient
		ra = abs_s11 * (np.cos(ang_s11) + 1j*np.sin(ang_s11))



	
	return ra
















def sky_data_calibration(ant_s11, FX1, FX2, Nterms_s11, Nterms_spectrum):
	
	
	# if ant_s11 == 145:
		# path_antenna_s11 = '/home/ramo7131/DATA/EDGES/calibration/antenna_s11/mid_band/s11/raw/mid-recv1-20180525/'
		
		# # Four measurements
		# o_m, f_m  = rc.s1p_read(path_antenna_s11 + '2018_145_08_28_27_input1.s1p')
		# s_m, f_m  = rc.s1p_read(path_antenna_s11 + '2018_145_08_28_27_input2.s1p')
		# l_m, f_m  = rc.s1p_read(path_antenna_s11 + '2018_145_08_28_27_input3.s1p')
		# a_m, f_m  = rc.s1p_read(path_antenna_s11 + '2018_145_08_28_27_input4.s1p')
	
	
	# elif ant_s11 == 147:
		# path_antenna_s11 = '/home/ramo7131/DATA/EDGES/calibration/antenna_s11/mid_band/s11/raw/mid-rcv1-20180527/'
		
		# # Four measurements
		# o_m, f_m  = rc.s1p_read(path_antenna_s11 + '2018_147_17_03_25_input1.s1p')
		# s_m, f_m  = rc.s1p_read(path_antenna_s11 + '2018_147_17_03_25_input2.s1p')
		# l_m, f_m  = rc.s1p_read(path_antenna_s11 + '2018_147_17_03_25_input3.s1p')
		# a_m, f_m  = rc.s1p_read(path_antenna_s11 + '2018_147_17_03_25_input4.s1p')	
		
	
	
	
	
	# f_m = f_m/1e6
	
	
	# # Standards assumed at the switch
	# o_sw =  1 * np.ones(len(f_m))
	# s_sw = -1 * np.ones(len(f_m))
	# l_sw =  0 * np.ones(len(f_m))
	
	
	# # Correction at switch
	# a_sw_c, x1, x2, x3  = rc.de_embed(o_sw, s_sw, l_sw, o_m, s_m, l_m, a_m)
	
	
	# # Correction at receiver input
	# out = switch_correction_mid_band_2018_01_25C(a_sw_c, f_in = f_m)
	# a_cc = out[0]
	
	
	
	# # Cutting frequency range of antenna S11
	# fs11 = f_m[(f_m>=FX1) & (f_m<=FX2)]
	# a_c  = a_cc[(f_m>=FX1) & (f_m<=FX2)]
	
	
	
	
	
	
	
	# Receiver calibration file
	#cf = np.genfromtxt('/home/ramo7131/DATA/EDGES/calibration/receiver_calibration/mid_band/2018_01_25C/results/old_nominal/calibration_files/calibration_file_mid_band_60_180_MHz_cfit12_wfit12.txt')
	#cf = np.genfromtxt('/home/ramo7131/DATA/EDGES/calibration/receiver_calibration/mid_band/2018_01_25C/results/old_nominal/calibration_files/calibration_file_mid_band_60_180_MHz_cfit7_wfit7.txt')
	cf = np.genfromtxt('/home/ramo7131/DATA/EDGES/calibration/receiver_calibration/mid_band/2018_01_25C/results/nominal/calibration_files/calibration_file_mid_band_50_180_MHz_cfit6_wfit10.txt')
	
	fk  = cf[:,0]
	rlk = cf[:,1] + 1j*cf[:,2]	
	C1k = cf[:,3]
	C2k = cf[:,4]
	TUk = cf[:,5]
	TCk = cf[:,6]
	TSk = cf[:,7]
	
	
	f  = fk[(fk>=FX1)  & (fk<=FX2)]
	rl = rlk[(fk>=FX1) & (fk<=FX2)]
	C1 = C1k[(fk>=FX1) & (fk<=FX2)]
	C2 = C2k[(fk>=FX1) & (fk<=FX2)]
	TU = TUk[(fk>=FX1) & (fk<=FX2)]
	TC = TCk[(fk>=FX1) & (fk<=FX2)]
	TS = TSk[(fk>=FX1) & (fk<=FX2)]
	
	
	
	
	
	
	# Polynomial interpolation of antenna S11
	# f_m_sel = f_m[(f_m >= FMIN) & (f_m <= FMAX)]
	# a_c_sel = a_c[(f_m >= FMIN) & (f_m <= FMAX)]

	
	p = eg.fit_polynomial_fourier('polynomial', fs11/200, np.real(a_c), Nterms_s11)
	rX_real   = eg.model_evaluate('polynomial', p[0], fs11/200)
	rant_real = eg.model_evaluate('polynomial', p[0], f/200)
	
	p = eg.fit_polynomial_fourier('polynomial', fs11/200, np.imag(a_c), Nterms_s11)
	rX_imag   = eg.model_evaluate('polynomial', p[0], fs11/200)	
	rant_imag = eg.model_evaluate('polynomial', p[0], f/200)
	
	rX = rX_real + 1j*rX_imag
	rant = rant_real + 1j*rant_imag
	
	
	# plt.figure()
	# plt.subplot(3,1,1)
	# plt.plot(fs11, 20*np.log10(np.abs(a_c)) - 20*np.log10(np.abs(rX)))
	# plt.subplot(3,1,2)
	# plt.plot(fs11, (180/np.pi)*np.unwrap(np.angle(a_c)) - (180/np.pi)*np.unwrap(np.angle(rX)))
	
	
	
	
	# Listing files to be processed
	path_spectra_files = '/home/ramo7131/DATA/EDGES/spectra/level2/mid_band/'
	full_list_original = listdir(path_spectra_files)
	full_list_original.sort()
	
	bad_files = ['2018_153_00.hdf5', '2018_154_00.hdf5', '2018_155_00.hdf5', '2018_156_00.hdf5', '2018_158_00.hdf5', '2018_168_00.hdf5', '2018_183_00.hdf5', '2018_194_00.hdf5', '2018_202_00.hdf5', '2018_203_00.hdf5', '2018_206_00.hdf5', '2018_207_00.hdf5']
	full_list = []
	for i in range(len(full_list_original)):
		if full_list_original[i] not in bad_files:
			full_list.append(full_list_original[i])
			
	
	
	
	
	counter = -1
	for i in range(10):  #range(len(full_list)):  # range(10):  #
		
		print(full_list[i])
		
			
		path_file = '/home/ramo7131/DATA/EDGES/spectra/level2/mid_band/' + full_list[i]
		ff, tt, mm, ww = eg.level2read_v2(path_file, print_key='no')
		
		index = np.arange(len(mm[:,0]))
		
		# Nighttime only
		#ix    = index[mm[:,6]<-10]
		
		# Low foreground only
		ix    = index[(mm[:,3]>0) & (mm[:,3]<12)]
		
		
		
		ttt = tt[ix, :]
		mmm = mm[ix, :]
		www = ww[ix, :]
		
		
		#print(len(mmm[:,0]))
	
		f   = ff[(ff>=FX1) & (ff<=FX2)]
		t   = ttt[:,(ff>=FX1) & (ff<=FX2)]
		m   = np.copy(mmm)
		w   = www[:,(ff>=FX1) & (ff<=FX2)]	
			
		tc  = eg.calibrated_antenna_temperature(t, rant, rl, C1, C2, TU, TC, TS, Tamb_internal=300)
		
		
		avt, avw   = eg.spectral_averaging(tc, w)	
		tnr1, wnr1 = eg.RFI_excision_raw_frequency(f, avt, avw)		
		tnr2, wnr2 = eg.RFI_cleaning_sweep(f, tnr1, wnr1, window_width_MHz=4, Npolyterms_block=4, N_choice=20, N_sigma=2.5)
		
		
		fx = np.copy(f)
		tx = np.copy(tnr2)
		wx = np.copy(wnr2)
		
	
		p = eg.fit_polynomial_fourier('LINLOG', fx, tx, Nterms_spectrum, Weights=wx)
		#p = eg.fit_polynomial_fourier('Physical_model', f, tnr, 5, Weights=wnr)
		fb, rb, wb = eg.spectral_binning_number_of_samples(fx, (tx-p[1]), wx)
		
		counter = counter + 1
		if counter == 0:
			res = np.zeros((len(full_list), len(fb)))
			m_all = np.copy(m)
			
		elif counter > 0:
			m_all = np.vstack((m_all, m))
			
		res[counter, :] = rb

		
	
	
	
	
	
	
	# Last part of function:   data_analysis_low_band_spectra_average,   at line 44230 of edges.py
	
	
	
	# # Total average	

	# # Spectral averaging
	# rav_t, wav_t = spectral_averaging(rr, wr)

	# # Spectral binning
	# fb, rb_t, wb_t = spectral_binning_number_of_samples(f, rav_t, wav_t, nsamples=64)

	# # Average model parameters
	# pb_t = np.mean(pr, axis=0)

	# # Evaluating foreground model at binned frequencies
	# tb_fg_t = model_evaluate('EDGES_polynomial', pb_t, fb/200)

	# # Binned total temperature
	# tb_t = tb_fg_t + rb_t

	# # RMS
	# rmsb_t = np.sqrt(np.sum((rb_t[wb_t>0])**2)/len(fb[wb_t>0]))

	# print('RMS of total: ' + str(rmsb_t))




	
	
	
	return full_list, fb, res, m_all



















def data_selection_single_day(date, LST_1=0, LST_2=24, sun_el_max=90, moon_el_max=90, amb_hum_max = 200, min_receiver_temp=0, max_receiver_temp=100):

	"""

	"""



	# Loading data
	path_data = home_folder + '/DATA/EDGES/spectra/level2/mid_band/'


	# Generating index of data within range of LST
	filename    = path_data + date + '.hdf5'
	f, t, m, w  = eg.level2read_v2(filename)


	index       = np.arange(len(m[:,0]))
	index_LST_1 = index[m[:,3]  >= LST_1]
	index_LST_2 = index[m[:,3]  <  LST_2]


	# Applying cuts for Sun elevation, ambient humidity, and receiver temperature
	index_SUN  = index[m[:,6]   <= sun_el_max]
	index_MOON = index[m[:,8]   <= moon_el_max]
	index_HUM  = index[m[:,10]  <= amb_hum_max]
	index_Trec = index[(m[:,11] >= min_receiver_temp) & (m[:,11] <= max_receiver_temp)]   # restricting receiver temperature Trec from thermistor 2 (S11 switch)


	# Combined index
	if LST_1 < LST_2:
		index1    = np.intersect1d(index_LST_1, index_LST_2)

	elif LST_1 > LST_2:
		index1    = np.union1d(index_LST_1, index_LST_2)

	index2    = np.intersect1d(index_SUN, index_MOON)
	index3    = np.intersect1d(index2, index_HUM)
	index4    = np.intersect1d(index3, index_Trec)
	index_all = np.intersect1d(index1, index4)

	print('NUMBER OF TRACES: ' + str(len(index_all)))


	# If there are traces available
	if len(index_all) > 1:

		# Select traces
		t_sel2   = t[index_all, :]
		w_sel2   = w[index_all, :]
		m_sel2   = m[index_all, :]


		# RFI-cleaning set of single-traces
		#t_sel2, w_sel2, t_sets, w_sets = RFI_cleaning_traces(f, t_sel, w_sel, 16)

	else:
		t_sel2  = 0
		w_sel2  = 0
		m_sel2  = 0


	return f, t_sel2, w_sel2, m_sel2


















def low_band_level2_to_level3_FAST(year_day, save='no', save_folder='', save_flag='', LST_1=0, LST_2=24, sun_el_max=-10, moon_el_max=90, amb_hum_max=90, min_receiver_temp=23.4, max_receiver_temp=27.4, ant_s11=145, FLOW=60, FHIGH=120, ant_s11_Nfit=10, fgl=1, glt='value', glp=0.5, fal=1, fbcl=1, receiver_temperature=25, receiver_cal_file=1, beam_correction='yes'):


	"""
	Use this file

	"""

	# CHANGE THIS INFORMATION !!!!!!!

	# Configuration
	# -------------

	#sun_el_max         = -10    # maximum sun elevation
	#moon_el_max        = 90     # maximum moon elevation
	#amb_hum_max        = 90     # maximum ambient humidity
	#min_receiver_temp  = 23.4   # limits of receiver temperature (23degC - 27degC), PLUS correction of 0.4degC for systematic thermistor offset
	#max_receiver_temp  = 27.4   # limits of receiver temperature (23degC - 27degC), PLUS correction of 0.4degC for systematic thermistor offset

	#ant_s11  = 262              # antenna S11 to use    # 'average'

	#fgl      = 1	             # flag ground loss
	#glt      = 'value'          # ground loss type
	#glp      = 0.5              # ground loss percentage
	#fal      = 1                # flag antenna pannel loss
	#fbcl     = 1                # flag balun connector loss 
	#receiver_temperature  = 1, or 25   # 1 for actual measured temperature, or 25 for 25degC

	#low_band_cal_file = 1       #  For Low-Band 1, Original 2015 calibration 50-100 MHz
	#low_band_cal_file = 2       #  For Low-Band 1, Original 2017 calibration 50-100 MHz
	#low_band_cal_file = 3       #  For Low-Band 1, Original 2017 calibration 50-120 MHz
					# Low-Band 2 does not care about calibration file number	
	# beam_correction = 1        # For Low-Band 1, original ground plane, -7 deg, 40-100 MHz
	# beam_correction = 2        # For Low-Band 1, extended ground plane, -7 deg, 40-120 MHz
	# beam_correction = 1        # For Low-Band 2, NS, -2 deg, 40-100 MHz
	# beam_correction = 2        # For Low-Band 2, EW, 87 deg, 40-100 MHz

	# List of files to process
	# ------------------------
	#datelist_raw = data_analysis_date_list(band, 'blade', case=date_list_case) # list of daily measurements
	#datelist_old = data_analysis_daily_spectra_filter(band, datelist_raw)      # selection of daily measurements.
	#datelist     = np.copy(datelist_old)                                       # further selection of daily measurements


	fin  = 0
	tc   = 0
	w_2D = 0
	m_2D = 0

	# Load daily data
	# ---------------	
	fin_X, t_2D_X, w_2D_X, m_2D = data_selection_single_day(year_day, LST_1=LST_1, LST_2=LST_2, sun_el_max=sun_el_max, moon_el_max=moon_el_max, amb_hum_max=amb_hum_max, min_receiver_temp=min_receiver_temp, max_receiver_temp=max_receiver_temp)



		
	# Continue if there are data available
	# ------------------------------------
	if np.sum(t_2D) > 0:
		
		
		# Cut the frequency range
		# -----------------------
		fin  = fin_X[(fin_X>=FLOW) & (fin_X<=FHIGH)]
		t_2D = t_2D_X[:, (fin_X>=FLOW) & (fin_X<=FHIGH)]
		w_2D = w_2D_X[:, (fin_X>=FLOW) & (fin_X<=FHIGH)]
		
		
		
		
		
		# HERE !!!!!!!!!!!!!!!!!!!
		
		
		
		

		# Chromaticity factor, recomputed for every day because the LSTs are different 
		# ----------------------------------------------------------------------------
		if beam_correction != 'no':
			cf = np.zeros((len(m_2D[:,0]), len(fin)))
			for j in range(len(m_2D[:,0])):
				print(j)
				cf[j,:] = antenna_beam_factor_interpolation(band, np.array([m_2D[j,3]]), fin, case_beam_factor=beam_correction)

		# Antenna S11
		# -----------
		#s11_ant = models_antenna_s11(band, 'blade', fin, antenna_s11_day=ant_s11, model_type='polynomial')
		s11_ant = models_antenna_s11_remove_delay(band, 'blade', fin, antenna_s11_day=ant_s11, model_type='polynomial', Nfit=ant_s11_Nfit)

		# Receiver temperature correction
		# -----------------------------
		#if receiver_temperature == 25:
			#RecTemp_base = 25
		#elif receiver_temperature == 1:   # 'Actual'
			#RecTemp_base = m_2D[i,-2]-0.4    # Correction of 0.4 degC is necessary
		#RecTemp = RecTemp_base	

		# Receiver calibration quantities
		# -------------------------------
		print('Receiver calibration')
		s11_LNA, sca, off, TU, TC, TS = receiver_calibration(band, fin, receiver_temperature=receiver_temperature, low_band_cal_file=low_band_cal_file)		

		# Calibrated antenna temperature with losses and beam chromaticity
		# ----------------------------------------------------------------
		tc_with_loss_and_beam = calibrated_antenna_temperature(t_2D, s11_ant, s11_LNA, sca, off, TU, TC, TS)

		# Removing loss
		# -------------
		print('Loss correction')

		# Combined gain (loss), computed only once, at the beginning, same for all days
		# -----------------------------------------------------------------------------
		cg = combined_gain(band, fin, antenna_s11_day=ant_s11, antenna_s11_Nfit=ant_s11_Nfit, flag_ground_loss=fgl, ground_loss_type=glt, ground_loss_percent=glp, flag_antenna_loss=fal, flag_balun_connector_loss=fbcl)		

		Tambient = 273.15 + 25 #m_2D[i,9]		
		tc_with_beam = (tc_with_loss_and_beam - Tambient*(1-cg))/cg

		# Removing beam chromaticity
		# --------------------------
		if beam_correction == 'no':
			print('NO beam correction')
			tc = np.copy(tc_with_beam)
		else:
			print('Beam correction')
			tc = tc_with_beam/cf

		# Save
		# --------------
		if save =='yes':
			path_save = home_folder + '/DATA/EDGES/spectra/level3/' + band + '/' + save_folder + '/'
			save_file = path_save + year_day + save_flag + '.hdf5'	

			with h5py.File(save_file, 'w') as hf:
				hf.create_dataset('frequency',            data = fin)
				hf.create_dataset('antenna_temperature',  data = tc)
				hf.create_dataset('weights',              data = w_2D)
				hf.create_dataset('meta_data',            data = m_2D)

	return fin, tc, w_2D, m_2D

































def plot_daily_residuals(full_list, fb, res):
	
	plt.figure()
	for i in range(len(full_list)):
		if i < 30:
			plt.subplot(1,2,1)
			if i % 2 == 0:
				color = 'r'
			else:
				color = 'b'
			plt.plot(fb, res[i]-i, color)
			plt.xlabel('frequency [MHz]')
			plt.xlim([50, 135])
			plt.xticks([60,70,80,90,100,110,120,130])
			plt.ylim([-31, 1])
			plt.text(52, -i, full_list[i][5:8])
			plt.yticks([])
			
		else:
			plt.subplot(1,2,2)
			if i % 2 == 0:
				color = 'r'
			else:
				color = 'b'			
			plt.plot(fb, res[i]-i+30, color)
			plt.xlabel('frequency [MHz]')
			plt.xlim([50, 135])
			plt.xticks([60,70,80,90,100,110,120,130])
			plt.ylim([-31, 1])
			plt.text(52, -i+30, full_list[i][5:8])	
			plt.yticks([])
		
	return 0









