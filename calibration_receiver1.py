
import numpy as np
import scipy as sp
import reflection_coefficient as rc
import time  as tt
import edges as eg
import datetime as dt
#import receiver_calibration as rcv


import basic as ba
import calibration as cal

import scipy.io as sio
import scipy.interpolate as spi

import matplotlib.pyplot as plt

import astropy.units as apu
import astropy.time as apt
import astropy.coordinates as apc

import h5py

import rfi as rfi




from os.path import expanduser
from os.path import exists
from os import listdir, makedirs, system

from astropy.io import fits






# Determining home folder
home_folder = expanduser("~")

import os
edges_folder       = os.environ['EDGES_vol2']
print('EDGES Folder: ' + edges_folder)

edges_folder_v1       = os.environ['EDGES_vol1']
















def switch_correction_receiver1(ant_s11, f_in = np.zeros([0,1]), case = 1, Nt_s11 = 23, Nt_s12s21 = 23, Nt_s22 = 23):  


	"""
	
	Aug 13, 2018
	
	WRONG!!!!:The characterization of the 4-position switch was done using the male standard of Phil's kit
	
	
	Feb 16, 2019
	
	The characterization of the 4-position switch was done using the EDGES Keysight 85033E Male
	
	
	March 2019
	
	
	April 2019
	
	
	
	
	
	"""

	
	
	
	

	# Loading measurements
	#if case == 1:
		#path_folder     = home_folder + '/DATA/EDGES/mid_band/calibration/receiver_calibration/receiver1/2018_01_25C/data/s11/raw/InternalSwitch/'
		
		#resistance_of_match = 50.12 # EDGES Keysight 85033E Male
		
		#o_in, f = rc.s1p_read(path_folder + 'Open01.s1p')
		#s_in, f = rc.s1p_read(path_folder + 'Short01.s1p')
		#l_in, f = rc.s1p_read(path_folder + 'Match01.s1p')
	
		#o_ex, f = rc.s1p_read(path_folder + 'ExternalOpen01.s1p')
		#s_ex, f = rc.s1p_read(path_folder + 'ExternalShort01.s1p')
		#l_ex, f = rc.s1p_read(path_folder + 'ExternalMatch01.s1p')	
	
	
	#if case == 2:
		#path_folder     = home_folder + '/DATA/EDGES/mid_band/calibration/receiver_calibration/receiver1/2018_01_25C/data/s11/raw/InternalSwitch/'
		
		#resistance_of_match = 50.12 # EDGES Keysight 85033E Male
		
		#o_in, f = rc.s1p_read(path_folder + 'Open02.s1p')
		#s_in, f = rc.s1p_read(path_folder + 'Short02.s1p')
		#l_in, f = rc.s1p_read(path_folder + 'Match02.s1p')
	
		#o_ex, f = rc.s1p_read(path_folder + 'ExternalOpen02.s1p')
		#s_ex, f = rc.s1p_read(path_folder + 'ExternalShort02.s1p')
		#l_ex, f = rc.s1p_read(path_folder + 'ExternalMatch02.s1p')	


		
	# Characterization of Switch at MRO in NOVEMBER 2017
	if case == 0:
		path_folder = edges_folder_v1 + 'calibration/receiver_calibration/mid_band/2017_11_15C_25C_35C/data/s11/raw/25C/receiver_MRO_fieldfox_40-200MHz/'
		
		resistance_of_match = 49.85     # 
		
		
		o_in, f = rc.s1p_read(path_folder + 'B15.s1p')
		s_in, f = rc.s1p_read(path_folder + 'B16.s1p')
		l_in, f = rc.s1p_read(path_folder + 'B17.s1p')
	
		o_ex, f = rc.s1p_read(path_folder + 'B18.s1p')
		s_ex, f = rc.s1p_read(path_folder + 'B19.s1p')
		l_ex, f = rc.s1p_read(path_folder + 'B20.s1p')









	
	if case == 1:
		path_folder     = edges_folder + 'mid_band/calibration/receiver_calibration/receiver1/2018_01_25C/data/s11/raw/InternalSwitch/'
		
		resistance_of_match = 50.027 # male
		#print('50')
		
		o_in, f = rc.s1p_read(path_folder + 'Open01.s1p')
		s_in, f = rc.s1p_read(path_folder + 'Short01.s1p')
		l_in, f = rc.s1p_read(path_folder + 'Match01.s1p')
	
		o_ex, f = rc.s1p_read(path_folder + 'ExternalOpen01.s1p')
		s_ex, f = rc.s1p_read(path_folder + 'ExternalShort01.s1p')
		l_ex, f = rc.s1p_read(path_folder + 'ExternalMatch01.s1p')


	# This is just a test case, using a probably wrong value of resistence of 50.12 ohms
	if case == 105012:
		path_folder     = edges_folder + 'mid_band/calibration/receiver_calibration/receiver1/2018_01_25C/data/s11/raw/InternalSwitch/'
		
		resistance_of_match = 50.12 # male
		#print('50')
		
		o_in, f = rc.s1p_read(path_folder + 'Open01.s1p')
		s_in, f = rc.s1p_read(path_folder + 'Short01.s1p')
		l_in, f = rc.s1p_read(path_folder + 'Match01.s1p')
	
		o_ex, f = rc.s1p_read(path_folder + 'ExternalOpen01.s1p')
		s_ex, f = rc.s1p_read(path_folder + 'ExternalShort01.s1p')
		l_ex, f = rc.s1p_read(path_folder + 'ExternalMatch01.s1p')












	if case == 2:
		path_folder     = edges_folder + 'mid_band/calibration/receiver_calibration/receiver1/2018_01_25C/data/s11/raw/InternalSwitch/'
		
		resistance_of_match = 50.027 # male
		
		o_in, f = rc.s1p_read(path_folder + 'Open02.s1p')
		s_in, f = rc.s1p_read(path_folder + 'Short02.s1p')
		l_in, f = rc.s1p_read(path_folder + 'Match02.s1p')
	
		o_ex, f = rc.s1p_read(path_folder + 'ExternalOpen02.s1p')
		s_ex, f = rc.s1p_read(path_folder + 'ExternalShort02.s1p')
		l_ex, f = rc.s1p_read(path_folder + 'ExternalMatch02.s1p')




	if case == 3:
		path_folder     = edges_folder + 'mid_band/calibration/receiver_calibration/receiver1/2018_01_15C/data/s11/raw/InternalSwitch/'
		
		resistance_of_match = 50.099 # male
		
		o_in, f = rc.s1p_read(path_folder + 'Open01.s1p')
		s_in, f = rc.s1p_read(path_folder + 'Short01.s1p')
		l_in, f = rc.s1p_read(path_folder + 'Match01.s1p')
	
		o_ex, f = rc.s1p_read(path_folder + 'ExternalOpen01.s1p')
		s_ex, f = rc.s1p_read(path_folder + 'ExternalShort01.s1p')
		l_ex, f = rc.s1p_read(path_folder + 'ExternalMatch01.s1p')



	if case == 4:
		path_folder     = edges_folder + 'mid_band/calibration/receiver_calibration/receiver1/2018_01_15C/data/s11/raw/InternalSwitch/'
		
		resistance_of_match = 50.099 # male
		
		o_in, f = rc.s1p_read(path_folder + 'Open02.s1p')
		s_in, f = rc.s1p_read(path_folder + 'Short02.s1p')
		l_in, f = rc.s1p_read(path_folder + 'Match02.s1p')
	
		o_ex, f = rc.s1p_read(path_folder + 'ExternalOpen02.s1p')
		s_ex, f = rc.s1p_read(path_folder + 'ExternalShort02.s1p')
		l_ex, f = rc.s1p_read(path_folder + 'ExternalMatch02.s1p')




	if case == 5:
		path_folder     = edges_folder + 'mid_band/calibration/receiver_calibration/receiver1/2018_01_35C/data/s11/raw/InternalSwitch/'
		
		resistance_of_match = 50.002 # male
		
		o_in, f = rc.s1p_read(path_folder + 'Open01.s1p')
		s_in, f = rc.s1p_read(path_folder + 'Short01.s1p')
		l_in, f = rc.s1p_read(path_folder + 'Match01.s1p')
	
		o_ex, f = rc.s1p_read(path_folder + 'ExternalOpen01.s1p')
		s_ex, f = rc.s1p_read(path_folder + 'ExternalShort01.s1p')
		l_ex, f = rc.s1p_read(path_folder + 'ExternalMatch01.s1p')


	
	




	

	# Calibration 2019-03
	
	# 4 repetitions of the same measurements 
	# -------------------------------------------------------------
	if case == 6:
		path_folder     = edges_folder + 'mid_band/calibration/receiver_calibration/receiver1/2019_03_25C/data/s11/raw/SwitchingState01/'
		
		resistance_of_match = 50.15 # male
		
		o_in, f = rc.s1p_read(path_folder + 'Open01.s1p')
		s_in, f = rc.s1p_read(path_folder + 'Short01.s1p')
		l_in, f = rc.s1p_read(path_folder + 'Match01.s1p')
	
		o_ex, f = rc.s1p_read(path_folder + 'ExternalOpen01.s1p')
		s_ex, f = rc.s1p_read(path_folder + 'ExternalShort01.s1p')
		l_ex, f = rc.s1p_read(path_folder + 'ExternalMatch01.s1p')
	
		
	# This is just a test case, using a probably wrong value of resistence of 50.12 ohms		
	if case == 605012:
		path_folder     = edges_folder + 'mid_band/calibration/receiver_calibration/receiver1/2019_03_25C/data/s11/raw/SwitchingState01/'
		
		resistance_of_match = 50.12 # male
		
		o_in, f = rc.s1p_read(path_folder + 'Open01.s1p')
		s_in, f = rc.s1p_read(path_folder + 'Short01.s1p')
		l_in, f = rc.s1p_read(path_folder + 'Match01.s1p')
	
		o_ex, f = rc.s1p_read(path_folder + 'ExternalOpen01.s1p')
		s_ex, f = rc.s1p_read(path_folder + 'ExternalShort01.s1p')
		l_ex, f = rc.s1p_read(path_folder + 'ExternalMatch01.s1p')		
		
		
		
		
	if case == 7:
		path_folder     = edges_folder + 'mid_band/calibration/receiver_calibration/receiver1/2019_03_25C/data/s11/raw/SwitchingState01/'
		
		resistance_of_match = 50.15 # male
		
		o_in, f = rc.s1p_read(path_folder + 'Open02.s1p')
		s_in, f = rc.s1p_read(path_folder + 'Short02.s1p')
		l_in, f = rc.s1p_read(path_folder + 'Match02.s1p')
	
		o_ex, f = rc.s1p_read(path_folder + 'ExternalOpen02.s1p')
		s_ex, f = rc.s1p_read(path_folder + 'ExternalShort02.s1p')
		l_ex, f = rc.s1p_read(path_folder + 'ExternalMatch02.s1p')

	if case == 8:
		path_folder     = edges_folder + 'mid_band/calibration/receiver_calibration/receiver1/2019_03_25C/data/s11/raw/SwitchingState02/'
		
		resistance_of_match = 50.15 # male
		
		o_in, f = rc.s1p_read(path_folder + 'Open01.s1p')
		s_in, f = rc.s1p_read(path_folder + 'Short01.s1p')
		l_in, f = rc.s1p_read(path_folder + 'Match01.s1p')
	
		o_ex, f = rc.s1p_read(path_folder + 'ExternalOpen01.s1p')
		s_ex, f = rc.s1p_read(path_folder + 'ExternalShort01.s1p')
		l_ex, f = rc.s1p_read(path_folder + 'ExternalMatch01.s1p')

	if case == 9:
		path_folder     = edges_folder + 'mid_band/calibration/receiver_calibration/receiver1/2019_03_25C/data/s11/raw/SwitchingState02/'
		
		resistance_of_match = 50.15 # male
		
		o_in, f = rc.s1p_read(path_folder + 'Open02.s1p')
		s_in, f = rc.s1p_read(path_folder + 'Short02.s1p')
		l_in, f = rc.s1p_read(path_folder + 'Match02.s1p')
	
		o_ex, f = rc.s1p_read(path_folder + 'ExternalOpen02.s1p')
		s_ex, f = rc.s1p_read(path_folder + 'ExternalShort02.s1p')
		l_ex, f = rc.s1p_read(path_folder + 'ExternalMatch02.s1p')
		
	# -------------------------------------------------------------









	# Calibration 2019-04
	
	# 4 repetitions of the same measurements 
	# -------------------------------------------------------------
	if case == 10:
		path_folder     = edges_folder + 'mid_band/calibration/receiver_calibration/receiver1/2019_04_25C/data/s11/raw/SwitchingState01/'
		
		resistance_of_match = 50.15 # male
		
		o_in, f = rc.s1p_read(path_folder + 'Open01.s1p')
		s_in, f = rc.s1p_read(path_folder + 'Short01.s1p')
		l_in, f = rc.s1p_read(path_folder + 'Match01.s1p')
	
		o_ex, f = rc.s1p_read(path_folder + 'ExternalOpen01.s1p')
		s_ex, f = rc.s1p_read(path_folder + 'ExternalShort01.s1p')
		l_ex, f = rc.s1p_read(path_folder + 'ExternalMatch01.s1p')


	if case == 11:
		path_folder     = edges_folder + 'mid_band/calibration/receiver_calibration/receiver1/2019_04_25C/data/s11/raw/SwitchingState01/'
		
		resistance_of_match = 50.15 # male
		
		o_in, f = rc.s1p_read(path_folder + 'Open02.s1p')
		s_in, f = rc.s1p_read(path_folder + 'Short02.s1p')
		l_in, f = rc.s1p_read(path_folder + 'Match02.s1p')
	
		o_ex, f = rc.s1p_read(path_folder + 'ExternalOpen02.s1p')
		s_ex, f = rc.s1p_read(path_folder + 'ExternalShort02.s1p')
		l_ex, f = rc.s1p_read(path_folder + 'ExternalMatch02.s1p')


	if case == 12:
		path_folder     = edges_folder + 'mid_band/calibration/receiver_calibration/receiver1/2019_04_25C/data/s11/raw/SwitchingState02/'
		
		resistance_of_match = 50.15 # male
		
		o_in, f = rc.s1p_read(path_folder + 'Open01.s1p')
		s_in, f = rc.s1p_read(path_folder + 'Short01.s1p')
		l_in, f = rc.s1p_read(path_folder + 'Match01.s1p')
	
		o_ex, f = rc.s1p_read(path_folder + 'ExternalOpen01.s1p')
		s_ex, f = rc.s1p_read(path_folder + 'ExternalShort01.s1p')
		l_ex, f = rc.s1p_read(path_folder + 'ExternalMatch01.s1p')


	if case == 13:
		path_folder     = edges_folder + 'mid_band/calibration/receiver_calibration/receiver1/2019_04_25C/data/s11/raw/SwitchingState02/'
		
		resistance_of_match = 50.15 # male
		
		o_in, f = rc.s1p_read(path_folder + 'Open02.s1p')
		s_in, f = rc.s1p_read(path_folder + 'Short02.s1p')
		l_in, f = rc.s1p_read(path_folder + 'Match02.s1p')
	
		o_ex, f = rc.s1p_read(path_folder + 'ExternalOpen02.s1p')
		s_ex, f = rc.s1p_read(path_folder + 'ExternalShort02.s1p')
		l_ex, f = rc.s1p_read(path_folder + 'ExternalMatch02.s1p')





	# Calibration 2019-10
		
	# 4 repetitions of the same measurements 
	# -------------------------------------------------------------
	if case == 21:
		path_folder     = edges_folder + 'mid_band/calibration/receiver_calibration/receiver1/2019_10_25C/data/s11/raw/InternalSwitch/'
		
		resistance_of_match = 50.15 # male value measured by Nivedita
		
		o_in, f = rc.s1p_read(path_folder + 'Open01.s1p')
		s_in, f = rc.s1p_read(path_folder + 'Short01.s1p')
		l_in, f = rc.s1p_read(path_folder + 'Match01.s1p')
	
		o_ex, f = rc.s1p_read(path_folder + 'ExternalOpen01.s1p')
		s_ex, f = rc.s1p_read(path_folder + 'ExternalShort01.s1p')
		l_ex, f = rc.s1p_read(path_folder + 'ExternalMatch01.s1p')
		
		
		
	if case == 22:
		path_folder     = edges_folder + 'mid_band/calibration/receiver_calibration/receiver1/2019_10_25C/data/s11/raw/InternalSwitch/'
		
		resistance_of_match = 50.15 # male value measured by Nivedita
		
		o_in, f = rc.s1p_read(path_folder + 'Open02.s1p')
		s_in, f = rc.s1p_read(path_folder + 'Short02.s1p')
		l_in, f = rc.s1p_read(path_folder + 'Match02.s1p')
	
		o_ex, f = rc.s1p_read(path_folder + 'ExternalOpen02.s1p')
		s_ex, f = rc.s1p_read(path_folder + 'ExternalShort02.s1p')
		l_ex, f = rc.s1p_read(path_folder + 'ExternalMatch02.s1p')


	if case == 23:
		path_folder     = edges_folder + 'mid_band/calibration/receiver_calibration/receiver1/2019_10_25C/data/s11/raw/InternalSwitch02/'
		
		resistance_of_match = 50.15 # male value measured by Nivedita
		
		o_in, f = rc.s1p_read(path_folder + 'Open01.s1p')
		s_in, f = rc.s1p_read(path_folder + 'Short01.s1p')
		l_in, f = rc.s1p_read(path_folder + 'Match01.s1p')
	
		o_ex, f = rc.s1p_read(path_folder + 'ExternalOpen01.s1p')
		s_ex, f = rc.s1p_read(path_folder + 'ExternalShort01.s1p')
		l_ex, f = rc.s1p_read(path_folder + 'ExternalMatch01.s1p')
		
		
		
	if case == 24:
		path_folder     = edges_folder + 'mid_band/calibration/receiver_calibration/receiver1/2019_10_25C/data/s11/raw/InternalSwitch02/'
		
		resistance_of_match = 50.15 # male value measured by Nivedita
		
		o_in, f = rc.s1p_read(path_folder + 'Open02.s1p')
		s_in, f = rc.s1p_read(path_folder + 'Short02.s1p')
		l_in, f = rc.s1p_read(path_folder + 'Match02.s1p')
	
		o_ex, f = rc.s1p_read(path_folder + 'ExternalOpen02.s1p')
		s_ex, f = rc.s1p_read(path_folder + 'ExternalShort02.s1p')
		l_ex, f = rc.s1p_read(path_folder + 'ExternalMatch02.s1p')











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
	max_f = np.max(f)
	fn    = (f/max_f) - 0.5
	

	if len(f_in) > 10:
		if f_in[0] > 1e5:
			fn_in = f_in/1e6
		elif f_in[-1] < 300:
			fn_in = np.copy(f_in)
		
		fn_in = (fn_in/max_f) - 0.5
			
			
	else:
		fn_in = np.copy(fn)



	# Real-Imaginary parts
	real_s11    = np.real(s11)
	imag_s11    = np.imag(s11)
	real_s12s21 = np.real(s12s21)
	imag_s12s21 = np.imag(s12s21)
	real_s22    = np.real(s22)
	imag_s22    = np.imag(s22)



	# Polynomial fits
	p = ba.fit_polynomial_fourier('fourier', fn, real_s11, Nt_s11)
	fit_real_s11 = ba.model_evaluate('fourier', p[0], fn)
	
	p = ba.fit_polynomial_fourier('fourier', fn, imag_s11, Nt_s11)
	fit_imag_s11 = ba.model_evaluate('fourier', p[0], fn)
	
	
	
	p = ba.fit_polynomial_fourier('fourier', fn, real_s12s21, Nt_s12s21)
	fit_real_s12s21 = ba.model_evaluate('fourier', p[0], fn)
	
	p = ba.fit_polynomial_fourier('fourier', fn, imag_s12s21, Nt_s12s21)
	fit_imag_s12s21 = ba.model_evaluate('fourier', p[0], fn)	
	
	
	
	p = ba.fit_polynomial_fourier('fourier', fn, real_s22, Nt_s22)
	fit_real_s22 = ba.model_evaluate('fourier', p[0], fn)
	
	p = ba.fit_polynomial_fourier('fourier', fn, imag_s22, Nt_s22)
	fit_imag_s22 = ba.model_evaluate('fourier', p[0], fn)
	
	
	
	
	
	#p = np.polyfit(fn, real_s11, Nt_s11-1)
	#fit_real_s11 = np.polyval(p, fn_in)

	#p = np.polyfit(fn, imag_s11, Nt_s11-1)
	#fit_imag_s11 = np.polyval(p, fn_in)


	#p = np.polyfit(fn, real_s12s21, Nt_s12s21-1)
	#fit_real_s12s21 = np.polyval(p, fn_in)

	#p = np.polyfit(fn, imag_s12s21, Nt_s12s21-1)
	#fit_imag_s12s21 = np.polyval(p, fn_in)


	#p = np.polyfit(fn, real_s22, Nt_s22-1)
	#fit_real_s22 = np.polyval(p, fn_in)

	#p = np.polyfit(fn, imag_s22, Nt_s22-1)
	#fit_imag_s22 = np.polyval(p, fn_in)








	fit_s11    = fit_real_s11    + 1j*fit_imag_s11
	fit_s12s21 = fit_real_s12s21 + 1j*fit_imag_s12s21
	fit_s22    = fit_real_s22    + 1j*fit_imag_s22



	# Corrected antenna S11
	corr_ant_s11 = rc.gamma_de_embed(fit_s11, fit_s12s21, fit_s22, ant_s11)






	return (corr_ant_s11, fit_s11, fit_s12s21, fit_s22, f, s11, s12s21, s22)






















def s11_calibration_measurements_mid_band_2018_01_25C(flow=40, fhigh=200, save='no', flag=''):




	# Data paths
	main_path    = edges_folder + 'mid_band/calibration/receiver_calibration/receiver1/2018_01_25C/data/s11/raw/'


	path_LNA     = main_path + 'ReceiverReading01/'    # seems slightly better than the others
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





	# Spectral calibration standards
	# ------------------------------

	# Best switch measurements from 2018 Jan
	SWITCH_CASE = 1
	
	# Number of Fourier terms fitting the switch parameters
	Nt_Fourier = 23




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
	out = switch_correction_receiver1(a_sw_c, f_in = f_a, case = SWITCH_CASE, Nt_s11=Nt_Fourier, Nt_s12s21=Nt_Fourier, Nt_s22=Nt_Fourier)
	a_c = out[0]










	# -----------------------------------------------------------------------------------------
	# Hot load before
	# -------------------
	o_m, f_h = rc.s1p_read(path_hot + 'Open01.s1p')
	s_m, f_h = rc.s1p_read(path_hot + 'Short01.s1p')
	l_m, f_h = rc.s1p_read(path_hot + 'Match01.s1p')
	h_m, f_h = rc.s1p_read(path_hot + 'External01.s1p')


	# Standards assumed at the switch
	o_sw =  1 * np.ones(len(f_h))
	s_sw = -1 * np.ones(len(f_h))
	l_sw =  0 * np.ones(len(f_h))


	# Correction at switch
	h_sw_c, x1, x2, x3  = rc.de_embed(o_sw, s_sw, l_sw, o_m, s_m, l_m, h_m)


	# Correction at receiver input
	out = switch_correction_receiver1(h_sw_c, f_in = f_h, case = SWITCH_CASE, Nt_s11=Nt_Fourier, Nt_s12s21=Nt_Fourier, Nt_s22=Nt_Fourier)
	h_c = out[0]












	# -----------------------------------------------------------------------------------------
	# Open Cable before
	# -------------------
	o_m, f_o = rc.s1p_read(path_open  + 'Open01.s1p')
	s_m, f_o = rc.s1p_read(path_open  + 'Short01.s1p')
	l_m, f_o = rc.s1p_read(path_open  + 'Match01.s1p')
	oc_m, f_o = rc.s1p_read(path_open + 'External01.s1p')


	# Standards assumed at the switch
	o_sw =  1 * np.ones(len(f_o))
	s_sw = -1 * np.ones(len(f_o))
	l_sw =  0 * np.ones(len(f_o))


	# Correction at switch
	oc_sw_c, x1, x2, x3  = rc.de_embed(o_sw, s_sw, l_sw, o_m, s_m, l_m, oc_m)


	# Correction at receiver input
	out = switch_correction_receiver1(oc_sw_c, f_in = f_o, case = SWITCH_CASE, Nt_s11=Nt_Fourier, Nt_s12s21=Nt_Fourier, Nt_s22=Nt_Fourier)
	o_c = out[0]










	# -----------------------------------------------------------------------------------------
	# Shorted Cable before
	# -------------------
	o_m,  f_s = rc.s1p_read(path_shorted + 'Open01.s1p')
	s_m,  f_s = rc.s1p_read(path_shorted + 'Short01.s1p')
	l_m,  f_s = rc.s1p_read(path_shorted + 'Match01.s1p')
	sc_m,  f_s = rc.s1p_read(path_shorted + 'External01.s1p')


	# Standards assumed at the switch
	o_sw =  1 * np.ones(len(f_s))
	s_sw = -1 * np.ones(len(f_s))
	l_sw =  0 * np.ones(len(f_s))


	# Correction at switch
	sc_sw_c, x1, x2, x3  = rc.de_embed(o_sw, s_sw, l_sw, o_m, s_m, l_m, sc_m)


	# Correction at receiver input
	out = switch_correction_receiver1(sc_sw_c, f_in = f_s, case = SWITCH_CASE, Nt_s11=Nt_Fourier, Nt_s12s21=Nt_Fourier, Nt_s22=Nt_Fourier)
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
	out  = switch_correction_receiver1(q_sw_c, f_in = f_q, case = SWITCH_CASE, Nt_s11=Nt_Fourier, Nt_s12s21=Nt_Fourier, Nt_s22=Nt_Fourier)
	q_c  = out[0]	
	









	# S-parameters of semi-rigid cable 
	# ---------------------------------
	d = np.genfromtxt(edges_folder_v1 + 'calibration/receiver_calibration/high_band1/2015_03_25C/data/S11/corrected_original/semi_rigid_s_parameters.txt')
			
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

		save_path       = edges_folder + 'mid_band/calibration/receiver_calibration/receiver1/2018_01_25C/data/s11/corrected/'
		temperature_LNA = '25degC'
		output_file_str = save_path + 's11_calibration_mid_band_LNA' + temperature_LNA + '_' + tt.strftime('%Y-%m-%d-%H-%M-%S') + flag + '.txt'
		np.savetxt(output_file_str, kk)

		print('File saved to: ' + output_file_str)


	return fr, LNAc, a_c, h_c, o_c, s_c, sr_s11r, sr_s11i, sr_s12s21r, sr_s12s21i, sr_s22r, sr_s22i, q_c















def s11_calibration_measurements_mid_band_2019_04_25C(flow=40, fhigh=200, save='no', flag=''):




	# Data paths
	main_path    = edges_folder + 'mid_band/calibration/receiver_calibration/receiver1/2019_04_25C/data/s11/raw/'


	path_LNA     = main_path + 'ReceiverReading02/'    # or ReceiverReading02
	path_ambient = main_path + 'Ambient3/'
	path_hot     = main_path + 'HotLoad2/'
	path_open    = main_path + 'LongCableOpen/'
	path_shorted = main_path + 'LongCableShorted/'
	path_sim2    = main_path + 'AntSim2/'
	path_sim3    = main_path + 'AntSim3/'
	
	SWITCH_CASE = 10
	



	# Receiver reflection coefficient
	# -------------------------------

	# Reading measurements
	o,   fr  = rc.s1p_read(path_LNA + 'Open01.s1p')                # or repetition 02
	s,   fr  = rc.s1p_read(path_LNA + 'Short01.s1p')
	l,   fr  = rc.s1p_read(path_LNA + 'Match01.s1p')
	LNA0, fr = rc.s1p_read(path_LNA + 'ReceiverReading01.s1p')
	
	
	
	


	# Models of standards
	resistance_of_match = 49.99  # female
	md = 1
	oa, sa, la = rc.agilent_85033E(fr, resistance_of_match, md)


	# Correction of measurements
	LNAc, x1, x2, x3   = rc.de_embed(oa, sa, la, o, s, l, LNA0)






	# Calibration loads
	# -----------------



	# -----------------------------------------------------------------------------------------
	# Ambient load before
	# -------------------
	o_m,  f_a = rc.s1p_read(path_ambient + 'Open02.s1p')       # do not use 01, it is bad
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
	out = switch_correction_receiver1(a_sw_c, f_in = f_a, case = SWITCH_CASE)
	a_c = out[0]










	# -----------------------------------------------------------------------------------------
	# Hot load before
	# -------------------
	o_m, f_h = rc.s1p_read(path_hot + 'Open02.s1p')       # do not use 01, it is bad
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
	out = switch_correction_receiver1(h_sw_c, f_in = f_a, case = SWITCH_CASE)
	h_c = out[0]












	# -----------------------------------------------------------------------------------------
	# Open Cable before
	# -------------------
	o_m, f_o = rc.s1p_read(path_open  + 'Open02.s1p')        # 01 seems OK also
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
	out = switch_correction_receiver1(oc_sw_c, f_in = f_a, case = SWITCH_CASE)
	o_c = out[0]










	# -----------------------------------------------------------------------------------------
	# Short Cable before
	# -------------------
	o_m,  f_s = rc.s1p_read(path_shorted + 'Open02.s1p')         # 01 is OK also
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
	out = switch_correction_receiver1(sc_sw_c, f_in = f_a, case = SWITCH_CASE)
	s_c = out[0]	










	# -----------------------------------------------------------------------------------------
	# Antenna Simulator 2
	# --------------------------
	o_m, f_q = rc.s1p_read(path_sim2 + 'Open01.s1p')          # try both, 01 and 02, there are important differences
	s_m, f_q = rc.s1p_read(path_sim2 + 'Short01.s1p')
	l_m, f_q = rc.s1p_read(path_sim2 + 'Match01.s1p')
	q_m, f_q = rc.s1p_read(path_sim2 + 'External01.s1p')


	# Standards assumed at the switch
	o_sw =  1 * np.ones(len(f_q))
	s_sw = -1 * np.ones(len(f_q))
	l_sw =  0 * np.ones(len(f_q))


	# Correction at switch
	q_sw_c, x1, x2, x3  = rc.de_embed(o_sw, s_sw, l_sw, o_m, s_m, l_m, q_m)


	# Correction at receiver input
	out = switch_correction_receiver1(q_sw_c, f_in = f_a, case = SWITCH_CASE)
	q2_c  = out[0]	
	





	# -----------------------------------------------------------------------------------------
	# Antenna Simulator 3
	# --------------------------
	o_m, f_q = rc.s1p_read(path_sim3 + 'Open02.s1p')          # try both, 01 and 02, there are significant differences
	s_m, f_q = rc.s1p_read(path_sim3 + 'Short02.s1p')
	l_m, f_q = rc.s1p_read(path_sim3 + 'Match02.s1p')
	q_m, f_q = rc.s1p_read(path_sim3 + 'External02.s1p')


	# Standards assumed at the switch
	o_sw =  1 * np.ones(len(f_q))
	s_sw = -1 * np.ones(len(f_q))
	l_sw =  0 * np.ones(len(f_q))


	# Correction at switch
	q_sw_c, x1, x2, x3  = rc.de_embed(o_sw, s_sw, l_sw, o_m, s_m, l_m, q_m)


	# Correction at receiver input
	out = switch_correction_receiver1(q_sw_c, f_in = f_a, case = SWITCH_CASE)
	q3_c  = out[0]	
	
	
	
	
	
	
	
	



	# S-parameters of semi-rigid cable 
	# ---------------------------------
	d = np.genfromtxt(edges_folder_v1 + 'calibration/receiver_calibration/high_band1/2015_03_25C/data/S11/corrected_original/semi_rigid_s_parameters.txt')
			
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
	np.real(q2_c),    np.imag(q2_c),
	np.real(q3_c),    np.imag(q3_c)])
	
	temp = tempT.T	
	kk = temp[(fr/1e6 >= flow) & (fr/1e6 <= fhigh), :]











	# -----------------------------------------------------------------------------------------
	# Saving
	if save == 'yes':

		save_path       = edges_folder + 'mid_band/calibration/receiver_calibration/receiver1/2019_04_25C/data/s11/corrected/'
		temperature_LNA = '25degC'
		output_file_str = save_path + 's11_calibration_mid_band_LNA' + temperature_LNA + '_' + tt.strftime('%Y-%m-%d-%H-%M-%S') + flag + '.txt'
		np.savetxt(output_file_str, kk)

		print('File saved to: ' + output_file_str)


	return fr, LNAc, a_c, h_c, o_c, s_c, q2_c, q3_c, sr_s11r, sr_s11i, sr_s12s21r, sr_s12s21i, sr_s22r, sr_s22i





















def s11_calibration_measurements_mid_band_2019_10_25C(flow=40, fhigh=200, save='no', flag=''):




	# Data paths
	main_path    = edges_folder + 'mid_band/calibration/receiver_calibration/receiver1/2019_10_25C/data/s11/raw/'


	path_LNA     = main_path + 'ReceiverReading02/'
	path_ambient = main_path + 'Ambient/'
	path_hot     = main_path + 'HotLoad/'
	path_open    = main_path + 'LongCableOpen/'
	path_shorted = main_path + 'LongCableShorted2/'
	path_sim     = main_path + 'AntSim2/'
	path_sim3    = main_path + 'AntSim3/'



	# Receiver reflection coefficient
	# -------------------------------

	# Reading measurements
	o,   fr  = rc.s1p_read(path_LNA + 'Open01.s1p')    # All the readings done are good. Choosing last one done.
	s,   fr  = rc.s1p_read(path_LNA + 'Short01.s1p')
	l,   fr  = rc.s1p_read(path_LNA + 'Match01.s1p')
	LNA0, fr = rc.s1p_read(path_LNA + 'ReceiverReading01.s1p')


	# Models of standards
	resistance_of_match = 50 # 49.98 # female measured by Nivedita
	md = 1
	oa, sa, la = rc.agilent_85033E(fr, resistance_of_match, md)


	# Correction of measurements
	LNAc, x1, x2, x3   = rc.de_embed(oa, sa, la, o, s, l, LNA0)









	# Spectral calibration standards
	# ------------------------------

	# Best switch measurements from Oct 2019
	SWITCH_CASE = 24
	
	# Number of Fourier terms fitting the switch parameters
	Nt_Fourier = 23
	






	# -----------------------------------------------------------------------------------------
	# Ambient load before
	# -------------------
	o_m,  f_a = rc.s1p_read(path_ambient + 'Open01.s1p')
	s_m,  f_a = rc.s1p_read(path_ambient + 'Short01.s1p')
	l_m,  f_a = rc.s1p_read(path_ambient + 'Match01.s1p')
	a_m,  f_a = rc.s1p_read(path_ambient + 'External01.s1p')


	# Standards assumed at the switch
	o_sw =  1 * np.ones(len(f_a))
	s_sw = -1 * np.ones(len(f_a))
	l_sw =  0 * np.ones(len(f_a))


	# Correction at switch
	a_sw_c, x1, x2, x3  = rc.de_embed(o_sw, s_sw, l_sw, o_m, s_m, l_m, a_m)


	# Correction at receiver input
	out = switch_correction_receiver1(a_sw_c, f_in = f_a, case = SWITCH_CASE, Nt_s11=Nt_Fourier, Nt_s12s21=Nt_Fourier, Nt_s22=Nt_Fourier)
	a_c = out[0]










	# -----------------------------------------------------------------------------------------
	# Hot load before
	# -------------------
	o_m, f_h = rc.s1p_read(path_hot + 'Open01.s1p')
	s_m, f_h = rc.s1p_read(path_hot + 'Short01.s1p')
	l_m, f_h = rc.s1p_read(path_hot + 'Match01.s1p')
	h_m, f_h = rc.s1p_read(path_hot + 'External01.s1p')


	# Standards assumed at the switch
	o_sw =  1 * np.ones(len(f_h))
	s_sw = -1 * np.ones(len(f_h))
	l_sw =  0 * np.ones(len(f_h))


	# Correction at switch
	h_sw_c, x1, x2, x3  = rc.de_embed(o_sw, s_sw, l_sw, o_m, s_m, l_m, h_m)


	# Correction at receiver input
	out = switch_correction_receiver1(h_sw_c, f_in = f_h, case = SWITCH_CASE, Nt_s11=Nt_Fourier, Nt_s12s21=Nt_Fourier, Nt_s22=Nt_Fourier)
	h_c = out[0]












	# -----------------------------------------------------------------------------------------
	# Open Cable before
	# -------------------
	o_m, f_o = rc.s1p_read(path_open  + 'Open01.s1p')
	s_m, f_o = rc.s1p_read(path_open  + 'Short01.s1p')
	l_m, f_o = rc.s1p_read(path_open  + 'Match01.s1p')
	oc_m, f_o = rc.s1p_read(path_open + 'External01.s1p')


	# Standards assumed at the switch
	o_sw =  1 * np.ones(len(f_o))
	s_sw = -1 * np.ones(len(f_o))
	l_sw =  0 * np.ones(len(f_o))


	# Correction at switch
	oc_sw_c, x1, x2, x3  = rc.de_embed(o_sw, s_sw, l_sw, o_m, s_m, l_m, oc_m)


	# Correction at receiver input
	out = switch_correction_receiver1(oc_sw_c, f_in = f_o, case = SWITCH_CASE, Nt_s11=Nt_Fourier, Nt_s12s21=Nt_Fourier, Nt_s22=Nt_Fourier)
	o_c = out[0]










	# -----------------------------------------------------------------------------------------
	# Short Cable before
	# -------------------
	o_m,  f_s = rc.s1p_read(path_shorted + 'Open01.s1p')
	s_m,  f_s = rc.s1p_read(path_shorted + 'Short01.s1p')
	l_m,  f_s = rc.s1p_read(path_shorted + 'Match01.s1p')
	sc_m,  f_s = rc.s1p_read(path_shorted + 'External01.s1p')


	# Standards assumed at the switch
	o_sw =  1 * np.ones(len(f_s))
	s_sw = -1 * np.ones(len(f_s))
	l_sw =  0 * np.ones(len(f_s))


	# Correction at switch
	sc_sw_c, x1, x2, x3  = rc.de_embed(o_sw, s_sw, l_sw, o_m, s_m, l_m, sc_m)


	# Correction at receiver input
	out = switch_correction_receiver1(sc_sw_c, f_in = f_s, case = SWITCH_CASE, Nt_s11=Nt_Fourier, Nt_s12s21=Nt_Fourier, Nt_s22=Nt_Fourier)
	s_c = out[0]	







	# -----------------------------------------------------------------------------------------
	# Antenna Simulator 2
	# --------------------------
	o_m, f_q = rc.s1p_read(path_sim + 'Open01.s1p')
	s_m, f_q = rc.s1p_read(path_sim + 'Short01.s1p')
	l_m, f_q = rc.s1p_read(path_sim + 'Match01.s1p')
	q_m, f_q = rc.s1p_read(path_sim + 'External01.s1p')
	

	# Standards assumed at the switch
	o_sw =  1 * np.ones(len(f_q))
	s_sw = -1 * np.ones(len(f_q))
	l_sw =  0 * np.ones(len(f_q))


	# Correction at switch
	q_sw_c, x1, x2, x3  = rc.de_embed(o_sw, s_sw, l_sw, o_m, s_m, l_m, q_m)


	# Correction at receiver input
	out  = switch_correction_receiver1(q_sw_c, f_in = f_q, case = SWITCH_CASE, Nt_s11=Nt_Fourier, Nt_s12s21=Nt_Fourier, Nt_s22=Nt_Fourier)
	q_c  = out[0]	
	







	# -----------------------------------------------------------------------------------------
	# Antenna Simulator 3
	# --------------------------
	o_m, f_q = rc.s1p_read(path_sim3 + 'Open01.s1p')
	s_m, f_q = rc.s1p_read(path_sim3 + 'Short01.s1p')
	l_m, f_q = rc.s1p_read(path_sim3 + 'Match01.s1p')
	q_m, f_q = rc.s1p_read(path_sim3 + 'External01.s1p')
	

	# Standards assumed at the switch
	o_sw =  1 * np.ones(len(f_q))
	s_sw = -1 * np.ones(len(f_q))
	l_sw =  0 * np.ones(len(f_q))


	# Correction at switch
	q_sw_c, x1, x2, x3  = rc.de_embed(o_sw, s_sw, l_sw, o_m, s_m, l_m, q_m)


	# Correction at receiver input
	out  = switch_correction_receiver1(q_sw_c, f_in = f_q, case = SWITCH_CASE, Nt_s11=Nt_Fourier, Nt_s12s21=Nt_Fourier, Nt_s22=Nt_Fourier)
	q3_c  = out[0]	
	
	
	
	
	
	





	# S-parameters of semi-rigid cable 
	# ---------------------------------
	d = np.genfromtxt(edges_folder_v1 + 'calibration/receiver_calibration/high_band1/2015_03_25C/data/S11/corrected_original/semi_rigid_s_parameters.txt')
			
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
	np.real(q_c),    np.imag(q_c),
	np.real(q3_c),   np.imag(q3_c)])
	
	temp = tempT.T	
	kk = temp[(fr/1e6 >= flow) & (fr/1e6 <= fhigh), :]











	# -----------------------------------------------------------------------------------------
	# Saving
	if save == 'yes':

		save_path       = edges_folder + 'mid_band/calibration/receiver_calibration/receiver1/2019_10_25C/data/s11/corrected/'
		temperature_LNA = '25degC'
		output_file_str = save_path + 's11_calibration_mid_band_LNA' + temperature_LNA + '_' + tt.strftime('%Y-%m-%d-%H-%M-%S') + flag + '.txt'
		np.savetxt(output_file_str, kk)

		print('File saved to: ' + output_file_str)


	return fr, LNAc, a_c, h_c, o_c, s_c, sr_s11r, sr_s11i, sr_s12s21r, sr_s12s21i, sr_s22r, sr_s22i, q_c, q3_c






















































def spectra_temperature_average_mid_band_2018_01_25C(save_folder, FLOW, FHIGH):
	
	"""

	May 21, 2019
	
	"""
	

	
	# Main folder
	main_folder     = edges_folder + 'mid_band/calibration/receiver_calibration/receiver1/2018_01_25C/'


	# Paths for source data
	path_spectra    = main_folder + 'data/spectra/'
	path_resistance = main_folder + 'data/resistance/corrected/'
	
	
	# Output folders
	if not exists(main_folder + 'results/' + save_folder + '/temp/'):
		makedirs(main_folder + 'results/' + save_folder + '/temp/')

	if not exists(main_folder + 'results/' + save_folder + '/data/'):
		makedirs(main_folder + 'results/' + save_folder + '/data/')

	path_par_temp    = main_folder + 'results/' + save_folder + '/temp/'
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
	#spec_shorted  = [file_shorted1]
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
	#ssa,    phys_temp_ambient  = ba.average_calibration_spectrum(spec_ambient, 'mp3139', res_ambient, 1*percent, plot='no')
	#ssh,    phys_temp_hot      = ba.average_calibration_spectrum(spec_hot,     'mp3139', res_hot,     2*percent, plot='no')
	#sso,    phys_temp_open     = ba.average_calibration_spectrum(spec_open,    'mp3139', res_open,    1*percent, plot='no')
	#sss,    phys_temp_shorted  = ba.average_calibration_spectrum(spec_shorted, 'mp3139', res_shorted, 1*percent, plot='no')
	#sss1,   phys_temp_sim      = ba.average_calibration_spectrum(spec_sim,     'on930',  res_sim,     1*percent, plot='no')


	f, avta, avwa, avpa = ba.average_calibration_spectrum(spec_ambient, FLOW, FHIGH, 'mp3139', res_ambient, 1*percent, plot='no')
	f, avth, avwh, avph = ba.average_calibration_spectrum(spec_hot,     FLOW, FHIGH, 'mp3139', res_hot,     2*percent, plot='no')
	f, avto, avwo, avpo = ba.average_calibration_spectrum(spec_open,    FLOW, FHIGH, 'mp3139', res_open,    1*percent, plot='no')
	f, avts, avws, avps = ba.average_calibration_spectrum(spec_shorted, FLOW, FHIGH, 'mp3139', res_shorted, 1*percent, plot='no')
	f, avtq, avwq, avpq = ba.average_calibration_spectrum(spec_sim,     FLOW, FHIGH, 'on930',  res_sim,     1*percent, plot='no')



	# Array of spectra
	specT = np.array([f, avta, avwa, avth, avwh, avto, avwo, avts, avws, avtq, avwq])
	spec  = specT.T

	
	# Array of physical temperatures
	phys      = np.zeros((5,1))
	phys[0,0] = avpa
	phys[1,0] = avph
	phys[2,0] = avpo
	phys[3,0] = avps
	phys[4,0] = avpq

 
	# Saving
	np.savetxt(path_data + 'average_spectra_300_350.txt', spec)
	np.savetxt(path_par_temp    + 'physical_temperatures.txt', phys)
	
	return spec, phys














def new_calibration_processing_mid_band_2018_01_25C(flow=50, fhigh=180, save='no', save_folder=0):


	"""
	
	Modification: May 23, 2019.

	"""	




	# Main folder
	main_folder     = edges_folder + 'mid_band/calibration/receiver_calibration/receiver1/2018_01_25C/'


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









	# Average calibration spectra / physical temperature
	# Percentage of initial data to leave out
	#percent = 5 # 5%
	#ssa,    phys_temp_ambient  = ba.average_calibration_spectrum(spec_ambient, 'mp3139', res_ambient, 1*percent, plot='no')
	#ssh,    phys_temp_hot      = ba.average_calibration_spectrum(spec_hot,     'mp3139', res_hot,     2*percent, plot='no')
	#sso,    phys_temp_open     = ba.average_calibration_spectrum(spec_open,    'mp3139', res_open,    1*percent, plot='no')
	#sss,    phys_temp_shorted  = ba.average_calibration_spectrum(spec_shorted, 'mp3139', res_shorted, 1*percent, plot='no')
	#sss1,   phys_temp_sim      = ba.average_calibration_spectrum(spec_sim,     'on930',  res_sim,     1*percent, plot='no')
	
	
	
	
	
	
	ssp   = np.genfromtxt(main_folder + 'results/nominal_cleaned_60_120MHz/data/average_spectra_300_350.txt')
	#ff   = sp[:,0]
	#ssa  = sp[:,1]
	#ssh  = sp[:,3]
	#sso  = sp[:,5]
	#sss  = sp[:,7]
	#sss1 = sp[:,9]
	
	sp = ssp[(ssp[:,0]>=flow) & (ssp[:,0]<=fhigh), :]
	fe  = sp[:,0]
	sa  = sp[:,1]
	sh  = sp[:,3]
	so  = sp[:,5]
	ss  = sp[:,7]
	ss1 = sp[:,9]	
	
	
	pt = np.genfromtxt(main_folder + 'results/nominal_cleaned_60_120MHz/temp/physical_temperatures.txt')
	phys_temp_ambient = pt[0]
	phys_temp_hot     = pt[1]
	phys_temp_open    = pt[2]
	phys_temp_shorted = pt[3]
	phys_temp_sim     = pt[4]






	## Select frequency range
	##ff, ilow, ihigh = ba.frequency_edges(flow, fhigh)
	#fe    = ff[(ff>=flow) & (ff<=fhigh)]
	#sa    = ssa[(ff>=flow) & (ff<=fhigh)]
	#sh    = ssh[(ff>=flow) & (ff<=fhigh)]
	#so    = sso[(ff>=flow) & (ff<=fhigh)]
	#ss    = sss[(ff>=flow) & (ff<=fhigh)]
	#ss1   = sss1[(ff>=flow) & (ff<=fhigh)]






	# Spectra modeling
	fen = (fe-120)/60
	fit_spec_ambient    = ba.fit_polynomial_fourier('fourier',    fen, sa,     17,  plot='no')
	fit_spec_hot        = ba.fit_polynomial_fourier('fourier',    fen, sh,     17,  plot='no')
	fit_spec_open       = ba.fit_polynomial_fourier('fourier',    fen, so,    121,  plot='no')
	fit_spec_shorted    = ba.fit_polynomial_fourier('fourier',    fen, ss,    121,  plot='no')
	fit_spec_sim        = ba.fit_polynomial_fourier('fourier',    fen, ss1,    37,  plot='no')

	model_spec_ambient  = ba.model_evaluate('fourier', fit_spec_ambient[0],    fen)
	model_spec_hot      = ba.model_evaluate('fourier', fit_spec_hot[0],        fen)
	model_spec_open     = ba.model_evaluate('fourier', fit_spec_open[0],       fen)
	model_spec_shorted  = ba.model_evaluate('fourier', fit_spec_shorted[0],    fen)
	model_spec_sim      = ba.model_evaluate('fourier', fit_spec_sim[0],        fen)



	# Loading S11 data
	#s11_all = np.genfromtxt(path_s11 + 's11_calibration_mid_band_LNA25degC_2018-08-13-22-05-34.txt')
	#s11_all = np.genfromtxt(path_s11 + 's11_calibration_mid_band_LNA25degC_2019-02-16-22-45-49.txt')
	#s11_all = np.genfromtxt(path_s11 + '
	#s11_all = np.genfromtxt(path_s11 + 's11_calibration_mid_band_LNA25degC_2019-05-11-20-34-59_other_s11_LNA.txt')
	#s11_all = np.genfromtxt(path_s11 + 's11_calibration_mid_band_LNA25degC_2019-05-11-20-43-04_other_s11_ambient.txt')
	#s11_all = np.genfromtxt(path_s11 + 's11_calibration_mid_band_LNA25degC_2019-05-11-20-46-12_other_s11_hot.txt')
	#s11_all = np.genfromtxt(path_s11 + 's11_calibration_mid_band_LNA25degC_2019-05-11-20-48-43_other_s11_open.txt')
	s11_all = np.genfromtxt(path_s11 + 's11_calibration_mid_band_LNA25degC_2019-05-11-20-50-27_other_s11_shorted.txt')
	#s11_all = np.genfromtxt(path_s11 + 's11_calibration_mid_band_LNA25degC_2019-05-11-20-52-01_other_s11_antsim3.txt')
	#s11_all = np.genfromtxt(path_s11 + 's11_calibration_mid_band_LNA25degC_2019-05-11-21-04-08_female_standard_resistance_49.98.txt')
	#s11_all = np.genfromtxt(path_s11 + 's11_calibration_mid_band_LNA25degC_2019-05-11-21-08-48_male_standard_resistance_50.00.txt')
	
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

	fit_s11_LNA_mag     = ba.fit_polynomial_fourier('polynomial', f_s11n, s11_LNA_mag,     19, plot='no')  # 
	fit_s11_LNA_ang     = ba.fit_polynomial_fourier('polynomial', f_s11n, s11_LNA_ang,     19, plot='no')  # 
	
	fit_s11_amb_mag     = ba.fit_polynomial_fourier('fourier',    f_s11n, s11_amb_mag,     27, plot='no')  # 
	fit_s11_amb_ang     = ba.fit_polynomial_fourier('fourier',    f_s11n, s11_amb_ang,     27, plot='no')  # 

	fit_s11_hot_mag     = ba.fit_polynomial_fourier('fourier',    f_s11n, s11_hot_mag,     27, plot='no')  #  
	fit_s11_hot_ang     = ba.fit_polynomial_fourier('fourier',    f_s11n, s11_hot_ang,     27, plot='no')  # 

	fit_s11_open_mag    = ba.fit_polynomial_fourier('fourier',    f_s11n, s11_open_mag,    85, plot='no')  # nominal: 85
	fit_s11_open_ang    = ba.fit_polynomial_fourier('fourier',    f_s11n, s11_open_ang,    85, plot='no')  # nominal: 85

	fit_s11_shorted_mag = ba.fit_polynomial_fourier('fourier',    f_s11n, s11_shorted_mag, 85, plot='no')  # nominal: 85
	fit_s11_shorted_ang = ba.fit_polynomial_fourier('fourier',    f_s11n, s11_shorted_ang, 85, plot='no')  # nominal: 85

	fit_s11_sr_mag      = ba.fit_polynomial_fourier('polynomial', f_s11n, s11_sr_mag,      17, plot='no')  # 
	fit_s11_sr_ang      = ba.fit_polynomial_fourier('polynomial', f_s11n, s11_sr_ang,      17, plot='no')  # 

	fit_s12s21_sr_mag   = ba.fit_polynomial_fourier('polynomial', f_s11n, s12s21_sr_mag,   17, plot='no')  # 
	fit_s12s21_sr_ang   = ba.fit_polynomial_fourier('polynomial', f_s11n, s12s21_sr_ang,   17, plot='no')  # 

	fit_s22_sr_mag      = ba.fit_polynomial_fourier('polynomial', f_s11n, s22_sr_mag,      17, plot='no')  # 
	fit_s22_sr_ang      = ba.fit_polynomial_fourier('polynomial', f_s11n, s22_sr_ang,      17, plot='no')  # 

	fit_s11_simu_mag    = ba.fit_polynomial_fourier('polynomial', f_s11n, s11_simu_mag,    35, plot='no')  # 7
	fit_s11_simu_ang    = ba.fit_polynomial_fourier('polynomial', f_s11n, s11_simu_ang,    35, plot='no')  # 7


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
		spectra = np.copy(sp) #np.array([fe, sa, sh, so, ss, ss1]).T		

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

		np.savetxt(path_data + 'average_spectra_300_350.txt', spectra)  # NOOOOO!!!!  

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
	return f_s11, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18    #     0   #fe, sa, sh, so, ss, ss1

































def calibration_processing_mid_band_2018_01_25C(flow=50, fhigh=180, save='no', save_folder=0):


	"""
	
	Modification: Dec 3, 2019.

	"""	




	# Main folder
	main_folder     = edges_folder + 'mid_band/calibration/receiver_calibration/receiver1/2018_01_25C/'


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
	#spec_shorted  = [file_shorted1]
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
	fe, sa,  w1, phys_temp_ambient  = ba.average_calibration_spectrum(spec_ambient, 'no', flow, fhigh, 'mp3139', res_ambient, 1*percent, plot='no')
	fe, sh,  w2, phys_temp_hot      = ba.average_calibration_spectrum(spec_hot,     'no', flow, fhigh, 'mp3139', res_hot,     2*percent, plot='no')
	fe, so,  w3, phys_temp_open     = ba.average_calibration_spectrum(spec_open,    'no', flow, fhigh, 'mp3139', res_open,    1*percent, plot='no')
	fe, ss,  w4, phys_temp_shorted  = ba.average_calibration_spectrum(spec_shorted, 'no', flow, fhigh, 'mp3139', res_shorted, 1*percent, plot='no')
	fe, ss1, w5, phys_temp_sim      = ba.average_calibration_spectrum(spec_sim,     'no', flow, fhigh, 'on930',  res_sim,     1*percent, plot='no')









	## Select frequency range
	#ff, ilow, ihigh = ba.frequency_edges(flow, fhigh)
	#fe    = ff[ilow:ihigh+1]
	#sa    = ssa[ilow:ihigh+1]
	#sh    = ssh[ilow:ihigh+1]
	#so    = sso[ilow:ihigh+1]
	#ss    = sss[ilow:ihigh+1]
	#ss1   = sss1[ilow:ihigh+1]








	# Spectra modeling
	fen = (fe/200)-0.5

	fit_spec_ambient    = ba.fit_polynomial_fourier('fourier',    fen, sa,     17,  plot='no')
	fit_spec_hot        = ba.fit_polynomial_fourier('fourier',    fen, sh,     17,  plot='no')
	fit_spec_open       = ba.fit_polynomial_fourier('fourier',    fen, so,    121,  plot='no')
	fit_spec_shorted    = ba.fit_polynomial_fourier('fourier',    fen, ss,    121,  plot='no')
	fit_spec_sim        = ba.fit_polynomial_fourier('fourier',    fen, ss1,    37,  plot='no')

	model_spec_ambient  = ba.model_evaluate('fourier', fit_spec_ambient[0],    fen)
	model_spec_hot      = ba.model_evaluate('fourier', fit_spec_hot[0],        fen)
	model_spec_open     = ba.model_evaluate('fourier', fit_spec_open[0],       fen)
	model_spec_shorted  = ba.model_evaluate('fourier', fit_spec_shorted[0],    fen)
	model_spec_sim      = ba.model_evaluate('fourier', fit_spec_sim[0],        fen)



	# Loading S11 data
	#s11_all = np.genfromtxt(path_s11 + 's11_calibration_mid_band_LNA25degC_2018-08-13-22-05-34.txt')
	#s11_all = np.genfromtxt(path_s11 + 's11_calibration_mid_band_LNA25degC_2019-02-16-22-45-49.txt')
	#s11_all = np.genfromtxt(path_s11 + '
	#s11_all = np.genfromtxt(path_s11 + 's11_calibration_mid_band_LNA25degC_2019-05-11-20-34-59_other_s11_LNA.txt')
	#s11_all = np.genfromtxt(path_s11 + 's11_calibration_mid_band_LNA25degC_2019-05-11-20-43-04_other_s11_ambient.txt')
	#s11_all = np.genfromtxt(path_s11 + 's11_calibration_mid_band_LNA25degC_2019-05-11-20-46-12_other_s11_hot.txt')
	#s11_all = np.genfromtxt(path_s11 + 's11_calibration_mid_band_LNA25degC_2019-05-11-20-48-43_other_s11_open.txt')
	#s11_all = np.genfromtxt(path_s11 + 's11_calibration_mid_band_LNA25degC_2019-05-11-20-50-27_other_s11_shorted.txt')
	#s11_all = np.genfromtxt(path_s11 + 's11_calibration_mid_band_LNA25degC_2019-05-11-20-52-01_other_s11_antsim3.txt')
	#s11_all = np.genfromtxt(path_s11 + 's11_calibration_mid_band_LNA25degC_2019-05-11-21-04-08_female_standard_resistance_49.98.txt')
	#s11_all = np.genfromtxt(path_s11 + 's11_calibration_mid_band_LNA25degC_2019-05-11-21-08-48_male_standard_resistance_50.00.txt')


	#s11_all = np.genfromtxt(path_s11 + 's11_calibration_mid_band_LNA25degC_2019-12-03-02-11-41_all_rep2.txt')
	
	# Nominal 
	#s11_all = np.genfromtxt(path_s11 + 's11_calibration_mid_band_LNA25degC_2019-12-03-02-37-29_all_rep2_but_shorted.txt')
	
	# BEST SO FAR --->  s11_all = np.genfromtxt(path_s11 + 's11_calibration_mid_band_LNA25degC_2019-12-06-18-51-24_all_rep2_but_shorted_LNA_rep1.txt')
	#s11_all = np.genfromtxt(path_s11 + 's11_calibration_mid_band_LNA25degC_2019-12-06-18-51-53_all_rep2_but_shorted_LNA_rep2.txt')
	
	#s11_all = np.genfromtxt(path_s11 + 's11_calibration_mid_band_LNA25degC_2019-12-06-21-02-05_all_rep2_but_shorted_LNA_rep12.txt')
	
	#s11_all = np.genfromtxt(path_s11 + 's11_calibration_mid_band_LNA25degC_2019-12-07-10-51-01_LNA1_a2_h2_o1_s1_sim2.txt')
	#s11_all = np.genfromtxt(path_s11 + 's11_calibration_mid_band_LNA25degC_2019-12-07-10-56-06_LNA1_a2_h2_o1_s2_sim2.txt')
	#s11_all = np.genfromtxt(path_s11 + 's11_calibration_mid_band_LNA25degC_2019-12-07-10-56-39_LNA1_a2_h2_o2_s1_sim2.txt')
	#s11_all = np.genfromtxt(path_s11 + 's11_calibration_mid_band_LNA25degC_2019-12-07-10-56-56_LNA1_a2_h2_o2_s2_sim2.txt')

	#s11_all = np.genfromtxt(path_s11 + 's11_calibration_mid_band_LNA25degC_2019-12-07-11-10-27_LNA1_a1_h1_o1_s1_sim2.txt')
	#s11_all = np.genfromtxt(path_s11 + 's11_calibration_mid_band_LNA25degC_2019-12-07-11-11-55_LNA1_a1_h2_o1_s1_sim2.txt')
	s11_all = np.genfromtxt(path_s11 + 's11_calibration_mid_band_LNA25degC_2019-12-07-11-12-42_LNA1_a2_h1_o1_s1_sim2.txt')
		
	
	
	
		
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

	# Simu
	s11_simu_mag    = np.abs(s11_simu)
	s11_simu_ang    = np.unwrap(np.angle(s11_simu))








	# Modeling S11

	f_s11n = (f_s11/200) - 0.5

	fit_s11_LNA_mag     = ba.fit_polynomial_fourier('fourier',    f_s11n, s11_LNA_mag,     19, plot='no')  # 
	fit_s11_LNA_ang     = ba.fit_polynomial_fourier('fourier',    f_s11n, s11_LNA_ang,     19, plot='no')  # 

	fit_s11_amb_mag     = ba.fit_polynomial_fourier('fourier',    f_s11n, s11_amb_mag,     15, plot='no')  # 
	fit_s11_amb_ang     = ba.fit_polynomial_fourier('fourier',    f_s11n, s11_amb_ang,     15, plot='no')  # 

	fit_s11_hot_mag     = ba.fit_polynomial_fourier('fourier',    f_s11n, s11_hot_mag,     19, plot='no')  #  
	fit_s11_hot_ang     = ba.fit_polynomial_fourier('fourier',    f_s11n, s11_hot_ang,     19, plot='no')  # 

	fit_s11_open_mag    = ba.fit_polynomial_fourier('fourier',    f_s11n, s11_open_mag,    121, plot='no')  # nominal: 85
	fit_s11_open_ang    = ba.fit_polynomial_fourier('fourier',    f_s11n, s11_open_ang,    121, plot='no')  # nominal: 85

	fit_s11_shorted_mag = ba.fit_polynomial_fourier('fourier',    f_s11n, s11_shorted_mag, 121, plot='no')  # nominal: 85
	fit_s11_shorted_ang = ba.fit_polynomial_fourier('fourier',    f_s11n, s11_shorted_ang, 121, plot='no')  # nominal: 85

	fit_s11_sr_mag      = ba.fit_polynomial_fourier('polynomial', f_s11n, s11_sr_mag,      17, plot='no')  # 
	fit_s11_sr_ang      = ba.fit_polynomial_fourier('polynomial', f_s11n, s11_sr_ang,      17, plot='no')  # 

	fit_s12s21_sr_mag   = ba.fit_polynomial_fourier('polynomial', f_s11n, s12s21_sr_mag,   17, plot='no')  # 
	fit_s12s21_sr_ang   = ba.fit_polynomial_fourier('polynomial', f_s11n, s12s21_sr_ang,   17, plot='no')  # 

	fit_s22_sr_mag      = ba.fit_polynomial_fourier('polynomial', f_s11n, s22_sr_mag,      17, plot='no')  # 
	fit_s22_sr_ang      = ba.fit_polynomial_fourier('polynomial', f_s11n, s22_sr_ang,      17, plot='no')  # 

	fit_s11_simu_mag    = ba.fit_polynomial_fourier('fourier',    f_s11n, s11_simu_mag,    51, plot='no')  # 7
	fit_s11_simu_ang    = ba.fit_polynomial_fourier('fourier',    f_s11n, s11_simu_ang,    51, plot='no')  # 7



	#r1  = 20*np.log10(fit_s11_LNA_mag[1]) - 20*np.log10(s11_LNA_mag)
	#r2  = (180/np.pi)*fit_s11_LNA_ang[1] - (180/np.pi)*s11_LNA_ang
	#r3  = 20*np.log10(fit_s11_amb_mag[1]) - 20*np.log10(s11_amb_mag)
	#r4  = (180/np.pi)*fit_s11_amb_ang[1] - (180/np.pi)*s11_amb_ang
	#r5  = 20*np.log10(fit_s11_hot_mag[1]) - 20*np.log10(s11_hot_mag)
	#r6  = (180/np.pi)*fit_s11_hot_ang[1] - (180/np.pi)*s11_hot_ang
	#r7  = 20*np.log10(fit_s11_open_mag[1]) - 20*np.log10(s11_open_mag)
	#r8  = (180/np.pi)*fit_s11_open_ang[1] - (180/np.pi)*s11_open_ang
	#r9  = 20*np.log10(fit_s11_shorted_mag[1]) - 20*np.log10(s11_shorted_mag)
	#r10 = (180/np.pi)*fit_s11_shorted_ang[1] - (180/np.pi)*s11_shorted_ang
	#r11 = 20*np.log10(fit_s11_sr_mag[1]) - 20*np.log10(s11_sr_mag)
	#r12 = (180/np.pi)*fit_s11_sr_ang[1] - (180/np.pi)*s11_sr_ang
	#r13 = 20*np.log10(fit_s12s21_sr_mag[1]) - 20*np.log10(s12s21_sr_mag)
	#r14 = (180/np.pi)*fit_s12s21_sr_ang[1] - (180/np.pi)*s12s21_sr_ang
	#r15 = 20*np.log10(fit_s22_sr_mag[1]) - 20*np.log10(s22_sr_mag)
	#r16 = (180/np.pi)*fit_s22_sr_ang[1] - (180/np.pi)*s22_sr_ang	
	#r17 = 20*np.log10(fit_s11_simu_mag[1]) - 20*np.log10(s11_simu_mag)
	#r18 = (180/np.pi)*fit_s11_simu_ang[1] - (180/np.pi)*s11_simu_ang	
	





	
	

	# Saving output parameters
	if save == 'yes':


		# Average spectra data in frequency range selected
		spectra = np.array([fe, sa, sh, so, ss, ss1]).T	
		
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




		
		

		# RMS residuals
		
		# Spectra
		RMS_spectra       = np.zeros((5,1))
		RMS_spectra[0, 0] = fit_spec_ambient[2]
		RMS_spectra[1, 0] = fit_spec_hot[2]
		RMS_spectra[2, 0] = fit_spec_open[2]
		RMS_spectra[3, 0] = fit_spec_shorted[2]
		RMS_spectra[4, 0] = fit_spec_sim[2]


		# S11
		RMS_s11        = np.zeros((18,1))
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



		

		# Saving
		# ----------------------------------

		np.savetxt(path_data        + 'average_spectra_300_350.txt', spectra)
		np.savetxt(path_par_temp    + 'physical_temperatures.txt',   phys_temp)

		np.savetxt(path_par_spectra + 'par_spec_amb.txt',     par_spec_ambient)
		np.savetxt(path_par_spectra + 'par_spec_hot.txt',     par_spec_hot)
		np.savetxt(path_par_spectra + 'par_spec_open.txt',    par_spec_open)
		np.savetxt(path_par_spectra + 'par_spec_shorted.txt', par_spec_shorted)
		np.savetxt(path_par_spectra + 'par_spec_simu.txt',    par_spec_sim)
		

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


		np.savetxt(path_par_spectra + 'RMS_spec.txt',         RMS_spectra)
		np.savetxt(path_par_s11     + 'RMS_s11.txt',          RMS_s11)


	# End
	print('Files processed.')



	return 0 #f_s11, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18























def calibration_processing_mid_band_2019_04_25C(flow=50, fhigh=180, save='no', save_folder=0):


	"""
	
	Apr 29, 2019.

	"""	




	# Main folder
	main_folder     = edges_folder + 'mid_band/calibration/receiver_calibration/receiver1/2019_04_25C/'


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
	file_ambient1 = path_spectra + 'level1_AmbientLoad3_2019_107_20_300_350.mat'
	file_ambient2 = path_spectra + 'level1_AmbientLoad3_2019_108_00_300_350.mat'
	file_ambient3 = path_spectra + 'level1_AmbientLoad3_2019_109_00_300_350.mat'
	spec_ambient  = [file_ambient1, file_ambient2, file_ambient3]
	res_ambient   = path_resistance + 'ambient_load3.txt'



	# Hot
	file_hot1 = path_spectra + 'level1_HotLoad2_2019_105_22_300_350.mat'
	file_hot2 = path_spectra + 'level1_HotLoad2_2019_106_00_300_350.mat'
	file_hot3 = path_spectra + 'level1_HotLoad2_2019_107_00_300_350.mat'
	spec_hot  = [file_hot1, file_hot2, file_hot3]
	res_hot   = path_resistance + 'hot_load2.txt'



	# Open Cable
	file_open1 = path_spectra + 'level1_LongCableOpen_2019_113_00_300_350.mat'
	spec_open  = [file_open1]
	res_open   = path_resistance + 'open_cable.txt'



	# Shorted Cable
	file_shorted1 = path_spectra + 'level1_LongCableShorted_2019_113_20_300_350.mat'
	file_shorted2 = path_spectra + 'level1_LongCableShorted_2019_114_00_300_350.mat'
	spec_shorted  = [file_shorted1, file_shorted2]
	res_shorted   = path_resistance + 'shorted_cable.txt'



	# Antenna Simulator 2
	file_sim21 = path_spectra + 'level1_AntSim2_2019_109_20_300_350.mat'
	file_sim22 = path_spectra + 'level1_AntSim2_2019_110_00_300_350.mat'
	file_sim23 = path_spectra + 'level1_AntSim2_2019_111_00_300_350.mat'
	file_sim24 = path_spectra + 'level1_AntSim2_2019_112_00_300_350.mat'
	spec_sim2  = [file_sim21, file_sim22, file_sim23]
	res_sim2   = path_resistance + 'antenna_simulator2.txt'



	# Antenna Simulator 3
	file_sim31 = path_spectra + 'level1_AntSim3_2019_115_00_300_350.mat'
	file_sim32 = path_spectra + 'level1_AntSim3_2019_116_00_300_350.mat'
	spec_sim3  = [file_sim31, file_sim32]
	res_sim3   = path_resistance + 'antenna_simulator3.txt'





	# Average calibration spectra / physical temperature
	# Percentage of initial data to leave out
	percent = 5 # 5%

	fe, sa,  w1, phys_temp_ambient  = ba.average_calibration_spectrum(spec_ambient, 'no', flow, fhigh, 'mp3139', res_ambient, 1*percent, plot='no')
	fe, sh,  w2, phys_temp_hot      = ba.average_calibration_spectrum(spec_hot,     'no', flow, fhigh, 'mp3139', res_hot,     2*percent, plot='no')
	fe, so,  w3, phys_temp_open     = ba.average_calibration_spectrum(spec_open,    'no', flow, fhigh, 'mp3139', res_open,    1*percent, plot='no')
	fe, ss,  w4, phys_temp_shorted  = ba.average_calibration_spectrum(spec_shorted, 'no', flow, fhigh, 'mp3139', res_shorted, 1*percent, plot='no')
	fe, ss2, w5, phys_temp_sim2     = ba.average_calibration_spectrum(spec_sim2,    'no', flow, fhigh, 'mp3139', res_sim2,    1*percent, plot='no')
	fe, ss3, w6, phys_temp_sim3     = ba.average_calibration_spectrum(spec_sim3,    'no', flow, fhigh, 'on930',  res_sim3,    1*percent, plot='no')






	#ssa,    phys_temp_ambient  = ba.average_calibration_spectrum(spec_ambient, res_ambient, 1*percent, plot='no')
	#ssh,    phys_temp_hot      = ba.average_calibration_spectrum(spec_hot,     res_hot,     2*percent, plot='no')
	#sso,    phys_temp_open     = ba.average_calibration_spectrum(spec_open,    res_open,    1*percent, plot='no')
	#sss,    phys_temp_shorted  = ba.average_calibration_spectrum(spec_shorted, res_shorted, 1*percent, plot='no')
	#sss2,   phys_temp_sim2     = ba.average_calibration_spectrum(spec_sim2,    res_sim2,    1*percent, plot='no')
	#sss3,   phys_temp_sim3     = ba.average_calibration_spectrum(spec_sim3,    res_sim3,    1*percent, plot='no')

	## Select frequency range
	#ff, ilow, ihigh = ba.frequency_edges(flow, fhigh)
	#fe    = ff[ilow:ihigh+1]
	#sa    = ssa[ilow:ihigh+1]
	#sh    = ssh[ilow:ihigh+1]
	#so    = sso[ilow:ihigh+1]
	#ss    = sss[ilow:ihigh+1]
	#ss2   = sss2[ilow:ihigh+1]
	#ss3   = sss3[ilow:ihigh+1]


	





	# Spectra modeling
	fen = (fe-120)/60

	fit_spec_ambient    = ba.fit_polynomial_fourier('fourier',    fen, sa,     17,  plot='no')
	fit_spec_hot        = ba.fit_polynomial_fourier('fourier',    fen, sh,     17,  plot='no')
	fit_spec_open       = ba.fit_polynomial_fourier('fourier',    fen, so,    121,  plot='no')
	fit_spec_shorted    = ba.fit_polynomial_fourier('fourier',    fen, ss,    121,  plot='no')
	fit_spec_sim2       = ba.fit_polynomial_fourier('fourier',    fen, ss2,    37,  plot='no')
	fit_spec_sim3       = ba.fit_polynomial_fourier('fourier',    fen, ss3,    17,  plot='no')

	model_spec_ambient  = ba.model_evaluate('fourier', fit_spec_ambient[0],    fen)
	model_spec_hot      = ba.model_evaluate('fourier', fit_spec_hot[0],        fen)
	model_spec_open     = ba.model_evaluate('fourier', fit_spec_open[0],       fen)
	model_spec_shorted  = ba.model_evaluate('fourier', fit_spec_shorted[0],    fen)
	model_spec_sim2     = ba.model_evaluate('fourier', fit_spec_sim2[0],       fen)
	model_spec_sim3     = ba.model_evaluate('fourier', fit_spec_sim3[0],       fen)




	# Loading S11 data
	#s11_all = np.genfromtxt(path_s11 + 's11_calibration_mid_band_LNA25degC_2018-08-13-22-05-34.txt')
	#s11_all = np.genfromtxt(path_s11 + 's11_calibration_mid_band_LNA25degC_2019-02-16-22-45-49.txt')
	
	#s11_all = np.genfromtxt(path_s11 + 's11_calibration_mid_band_LNA25degC_2019-04-29-20-39-54_antsim2_rep1.txt')
	#s11_all = np.genfromtxt(path_s11 + 's11_calibration_mid_band_LNA25degC_2019-04-29-20-40-46_antsim2_rep2.txt')
	s11_all = np.genfromtxt(path_s11 + 's11_calibration_mid_band_LNA25degC_2019-04-29-22-13-29_LNA_rep2.txt')
	
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

	s11_simu2   = s11[:, 17] + 1j*s11[:, 18]   # ant sim 2





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

	# Simu2
	s11_simu2_mag   = np.abs(s11_simu2)
	s11_simu2_ang   = np.unwrap(np.angle(s11_simu2))








	# Modeling S11

	f_s11n = (f_s11-120)/60

	fit_s11_LNA_mag     = ba.fit_polynomial_fourier('polynomial', f_s11n, s11_LNA_mag,     19, plot='no')  # 
	fit_s11_LNA_ang     = ba.fit_polynomial_fourier('polynomial', f_s11n, s11_LNA_ang,     19, plot='no')  # 

	fit_s11_amb_mag     = ba.fit_polynomial_fourier('fourier',    f_s11n, s11_amb_mag,     27, plot='no')  # 
	fit_s11_amb_ang     = ba.fit_polynomial_fourier('fourier',    f_s11n, s11_amb_ang,     27, plot='no')  # 

	fit_s11_hot_mag     = ba.fit_polynomial_fourier('fourier',    f_s11n, s11_hot_mag,     27, plot='no')  #  
	fit_s11_hot_ang     = ba.fit_polynomial_fourier('fourier',    f_s11n, s11_hot_ang,     27, plot='no')  # 

	fit_s11_open_mag    = ba.fit_polynomial_fourier('fourier',    f_s11n, s11_open_mag,    85, plot='no')  # 27
	fit_s11_open_ang    = ba.fit_polynomial_fourier('fourier',    f_s11n, s11_open_ang,    85, plot='no')  # 27

	fit_s11_shorted_mag = ba.fit_polynomial_fourier('fourier',    f_s11n, s11_shorted_mag, 85, plot='no')  # 27
	fit_s11_shorted_ang = ba.fit_polynomial_fourier('fourier',    f_s11n, s11_shorted_ang, 85, plot='no')  # 27

	fit_s11_sr_mag      = ba.fit_polynomial_fourier('polynomial', f_s11n, s11_sr_mag,      17, plot='no')  # 
	fit_s11_sr_ang      = ba.fit_polynomial_fourier('polynomial', f_s11n, s11_sr_ang,      17, plot='no')  # 

	fit_s12s21_sr_mag   = ba.fit_polynomial_fourier('polynomial', f_s11n, s12s21_sr_mag,   17, plot='no')  # 
	fit_s12s21_sr_ang   = ba.fit_polynomial_fourier('polynomial', f_s11n, s12s21_sr_ang,   17, plot='no')  # 

	fit_s22_sr_mag      = ba.fit_polynomial_fourier('polynomial', f_s11n, s22_sr_mag,      17, plot='no')  # 
	fit_s22_sr_ang      = ba.fit_polynomial_fourier('polynomial', f_s11n, s22_sr_ang,      17, plot='no')  # 

	fit_s11_simu2_mag   = ba.fit_polynomial_fourier('polynomial', f_s11n, s11_simu2_mag,   35, plot='no')  # 7
	fit_s11_simu2_ang   = ba.fit_polynomial_fourier('polynomial', f_s11n, s11_simu2_ang,   35, plot='no')  # 7


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
	r17 = fit_s11_simu2_mag[1] - s11_simu2_mag
	r18 = fit_s11_simu2_ang[1] - s11_simu2_ang	
	
	

	# Saving output parameters
	if save == 'yes':


		# Average spectra data in frequency range selected
		spectra = np.array([fe, sa, sh, so, ss, ss2]).T		

		# RMS residuals
		RMS_spectra = np.zeros((5,1))
		RMS_s11     = np.zeros((18,1))

		# Spectra
		RMS_spectra[0, 0] = fit_spec_ambient[2]
		RMS_spectra[1, 0] = fit_spec_hot[2]
		RMS_spectra[2, 0] = fit_spec_open[2]
		RMS_spectra[3, 0] = fit_spec_shorted[2]
		RMS_spectra[4, 0] = fit_spec_sim2[2]


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
		RMS_s11[16, 0] = fit_s11_simu2_mag[2]
		RMS_s11[17, 0] = fit_s11_simu2_ang[2]



		# Formating fit parameters

		# Physical temperature
		phys_temp = np.zeros((5,1))
		phys_temp[0,0] = phys_temp_ambient
		phys_temp[1,0] = phys_temp_hot
		phys_temp[2,0] = phys_temp_open
		phys_temp[3,0] = phys_temp_shorted
		phys_temp[4,0] = phys_temp_sim2



		# Spectra
		par_spec_ambient    = np.reshape(fit_spec_ambient[0],    (-1,1))
		par_spec_hot        = np.reshape(fit_spec_hot[0],        (-1,1))
		par_spec_open       = np.reshape(fit_spec_open[0],       (-1,1))
		par_spec_shorted    = np.reshape(fit_spec_shorted[0],    (-1,1))
		par_spec_sim        = np.reshape(fit_spec_sim2[0],        (-1,1))



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

		par_s11_simu_mag    = np.reshape(fit_s11_simu2_mag[0],    (-1,1))
		par_s11_simu_ang    = np.reshape(fit_s11_simu2_ang[0],    (-1,1))



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
	#  fe, sa, sh, so, ss, ss2, ss3   #
	return f_s11, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18    #     0   #fe, sa, sh, so, ss, ss1



















def calibration_processing_mid_band_2019_10_25C(flow=50, fhigh=180, save='no', save_folder=0):


	"""
	
	October 2019.

	"""	




	# Main folder
	main_folder     = edges_folder + 'mid_band/calibration/receiver_calibration/receiver1/2019_10_25C/'


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
	file_ambient1 = path_spectra + 'level1_Ambient_2019_274_00_15_30_lab_300_350.mat'
	spec_ambient  = [file_ambient1]
	res_ambient   = path_resistance + 'AmbientLoad.txt'



	# Hot
	file_hot1 = path_spectra + 'level1_HotLoad_2019_275_00_00_35_lab_300_350.mat'
	spec_hot  = [file_hot1]
	res_hot   = path_resistance + 'HotLoad.txt'



	# Open Cable
	file_open1 = path_spectra + 'level1_LongCableOpen_2019_277_00_00_30_lab_300_350.mat'
	spec_open  = [file_open1]
	res_open   = path_resistance + 'LongCableOpen.txt'



	# Shorted Cable
	file_shorted1 = path_spectra + 'level1_LongCableShorted2_2019_287_18_31_06_lab_300_350.mat'
	file_shorted2 = path_spectra + 'level1_LongCableShorted2_2019_288_00_00_07_lab_300_350.mat'
	spec_shorted  = [file_shorted1, file_shorted2]
	res_shorted   = path_resistance + 'LongCableShorted2.txt'



	# Antenna Simulator 2
	file_sim21 = path_spectra + 'level1_AntSim2_2019_281_00_00_06_lab_300_350.mat'
	spec_sim2  = [file_sim21]
	res_sim2   = path_resistance + 'AntSim2.txt'



	# Antenna Simulator 3
	file_sim31 = path_spectra + 'level1_AntSim3_2019_281_20_07_40_lab_300_350.mat'
	file_sim32 = path_spectra + 'level1_AntSim3_2019_282_00_00_19_lab_300_350.mat'
	spec_sim3  = [file_sim31, file_sim32]
	res_sim3   = path_resistance + 'AntSim3.txt'





	# Average calibration spectra / physical temperature
	# Percentage of initial data to leave out
	percent = 5 # 5%

	fe, sa,  w1, phys_temp_ambient  = ba.average_calibration_spectrum(spec_ambient, 'no', flow, fhigh, 'mp3139', res_ambient, 1*percent, plot='no')
	fe, sh,  w2, phys_temp_hot      = ba.average_calibration_spectrum(spec_hot,     'no', flow, fhigh, 'mp3139', res_hot,     2*percent, plot='no')
	fe, so,  w3, phys_temp_open     = ba.average_calibration_spectrum(spec_open,    'no', flow, fhigh, 'mp3139', res_open,    1*percent, plot='no')
	fe, ss,  w4, phys_temp_shorted  = ba.average_calibration_spectrum(spec_shorted, 'no', flow, fhigh, 'mp3139', res_shorted, 1*percent, plot='no')
	fe, ss2, w5, phys_temp_sim2     = ba.average_calibration_spectrum(spec_sim2,    'no', flow, fhigh, 'mp3139', res_sim2,    1*percent, plot='no')
	#fe, ss3, w6, phys_temp_sim3     = ba.average_calibration_spectrum(spec_sim3,    'no', flow, fhigh, 'on930',  res_sim3,    1*percent, plot='no')
	fe, ss3, w6, phys_temp_sim3     = ba.average_calibration_spectrum(spec_sim3,    'no', flow, fhigh, 'mp3139',  res_sim3,    1*percent, plot='no')






	##ssa,    phys_temp_ambient  = ba.average_calibration_spectrum(spec_ambient, res_ambient, 1*percent, plot='no')
	##ssh,    phys_temp_hot      = ba.average_calibration_spectrum(spec_hot,     res_hot,     2*percent, plot='no')
	##sso,    phys_temp_open     = ba.average_calibration_spectrum(spec_open,    res_open,    1*percent, plot='no')
	##sss,    phys_temp_shorted  = ba.average_calibration_spectrum(spec_shorted, res_shorted, 1*percent, plot='no')
	##sss2,   phys_temp_sim2     = ba.average_calibration_spectrum(spec_sim2,    res_sim2,    1*percent, plot='no')
	##sss3,   phys_temp_sim3     = ba.average_calibration_spectrum(spec_sim3,    res_sim3,    1*percent, plot='no')

	### Select frequency range
	##ff, ilow, ihigh = ba.frequency_edges(flow, fhigh)
	##fe    = ff[ilow:ihigh+1]
	##sa    = ssa[ilow:ihigh+1]
	##sh    = ssh[ilow:ihigh+1]
	##so    = sso[ilow:ihigh+1]
	##ss    = sss[ilow:ihigh+1]
	##ss2   = sss2[ilow:ihigh+1]
	##ss3   = sss3[ilow:ihigh+1]


	





	# Spectra modeling
	fen = (fe/200)-0.5

	fit_spec_ambient    = ba.fit_polynomial_fourier('fourier',    fen, sa,     17,  plot='no')
	fit_spec_hot        = ba.fit_polynomial_fourier('fourier',    fen, sh,     17,  plot='no')
	fit_spec_open       = ba.fit_polynomial_fourier('fourier',    fen, so,    121,  plot='no')
	fit_spec_shorted    = ba.fit_polynomial_fourier('fourier',    fen, ss,    121,  plot='no')
	fit_spec_sim2       = ba.fit_polynomial_fourier('fourier',    fen, ss2,    37,  plot='no')
	fit_spec_sim3       = ba.fit_polynomial_fourier('fourier',    fen, ss3,    37,  plot='no')

	model_spec_ambient  = ba.model_evaluate('fourier', fit_spec_ambient[0],    fen)
	model_spec_hot      = ba.model_evaluate('fourier', fit_spec_hot[0],        fen)
	model_spec_open     = ba.model_evaluate('fourier', fit_spec_open[0],       fen)
	model_spec_shorted  = ba.model_evaluate('fourier', fit_spec_shorted[0],    fen)
	model_spec_sim2     = ba.model_evaluate('fourier', fit_spec_sim2[0],       fen)
	model_spec_sim3     = ba.model_evaluate('fourier', fit_spec_sim3[0],       fen)









	# Loading S11 data
	# -----------------------------------------------------------
	#s11_all = np.genfromtxt(path_s11 + 's11_calibration_mid_band_LNA25degC_2019-11-09-19-13-48_second_set_of_measurements.txt')
	s11_all = np.genfromtxt(path_s11 + 's11_calibration_mid_band_LNA25degC_2019-11-12-14-09-42_first_set_of_measurements_for_cable_and_simulators.txt')
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

	s11_simu2   = s11[:, 17] + 1j*s11[:, 18]   # ant sim 2
	s11_simu3   = s11[:, 19] + 1j*s11[:, 20]   # ant sim 3





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

	# Simu2
	s11_simu2_mag   = np.abs(s11_simu2)
	s11_simu2_ang   = np.unwrap(np.angle(s11_simu2))

	# Simu3
	s11_simu3_mag   = np.abs(s11_simu3)
	s11_simu3_ang   = np.unwrap(np.angle(s11_simu3))







	# Modeling S11

	f_s11n = (f_s11/200) - 0.5

	fit_s11_LNA_mag     = ba.fit_polynomial_fourier('fourier',    f_s11n, s11_LNA_mag,     19, plot='no')  # 
	fit_s11_LNA_ang     = ba.fit_polynomial_fourier('fourier',    f_s11n, s11_LNA_ang,     19, plot='no')  # 

	fit_s11_amb_mag     = ba.fit_polynomial_fourier('fourier',    f_s11n, s11_amb_mag,     15, plot='no')  # 
	fit_s11_amb_ang     = ba.fit_polynomial_fourier('fourier',    f_s11n, s11_amb_ang,     15, plot='no')  # 

	fit_s11_hot_mag     = ba.fit_polynomial_fourier('fourier',    f_s11n, s11_hot_mag,     19, plot='no')  #  
	fit_s11_hot_ang     = ba.fit_polynomial_fourier('fourier',    f_s11n, s11_hot_ang,     19, plot='no')  # 

	fit_s11_open_mag    = ba.fit_polynomial_fourier('fourier',    f_s11n, s11_open_mag,    225, plot='no')  # 27
	fit_s11_open_ang    = ba.fit_polynomial_fourier('fourier',    f_s11n, s11_open_ang,    225, plot='no')  # 27

	fit_s11_shorted_mag = ba.fit_polynomial_fourier('fourier',    f_s11n, s11_shorted_mag, 225, plot='no')  # 27
	fit_s11_shorted_ang = ba.fit_polynomial_fourier('fourier',    f_s11n, s11_shorted_ang, 225, plot='no')  # 27

	fit_s11_sr_mag      = ba.fit_polynomial_fourier('polynomial', f_s11n, s11_sr_mag,      17, plot='no')  # 
	fit_s11_sr_ang      = ba.fit_polynomial_fourier('polynomial', f_s11n, s11_sr_ang,      17, plot='no')  # 

	fit_s12s21_sr_mag   = ba.fit_polynomial_fourier('polynomial', f_s11n, s12s21_sr_mag,   17, plot='no')  # 
	fit_s12s21_sr_ang   = ba.fit_polynomial_fourier('polynomial', f_s11n, s12s21_sr_ang,   17, plot='no')  # 

	fit_s22_sr_mag      = ba.fit_polynomial_fourier('polynomial', f_s11n, s22_sr_mag,      17, plot='no')  # 
	fit_s22_sr_ang      = ba.fit_polynomial_fourier('polynomial', f_s11n, s22_sr_ang,      17, plot='no')  # 

	fit_s11_simu2_mag   = ba.fit_polynomial_fourier('fourier',    f_s11n, s11_simu2_mag,   35, plot='no')  # 7
	fit_s11_simu2_ang   = ba.fit_polynomial_fourier('fourier',    f_s11n, s11_simu2_ang,   35, plot='no')  # 7

	fit_s11_simu3_mag   = ba.fit_polynomial_fourier('fourier',    f_s11n, s11_simu3_mag,   91, plot='no')  # 7
	fit_s11_simu3_ang   = ba.fit_polynomial_fourier('fourier',    f_s11n, s11_simu3_ang,   91, plot='no')  # 7




	#r1  = 20*np.log10(fit_s11_LNA_mag[1]) - 20*np.log10(s11_LNA_mag)
	#r2  = (180/np.pi)*fit_s11_LNA_ang[1] - (180/np.pi)*s11_LNA_ang
	#r3  = 20*np.log10(fit_s11_amb_mag[1]) - 20*np.log10(s11_amb_mag)
	#r4  = (180/np.pi)*fit_s11_amb_ang[1] - (180/np.pi)*s11_amb_ang
	#r5  = 20*np.log10(fit_s11_hot_mag[1]) - 20*np.log10(s11_hot_mag)
	#r6  = (180/np.pi)*fit_s11_hot_ang[1] - (180/np.pi)*s11_hot_ang
	#r7  = 20*np.log10(fit_s11_open_mag[1]) - 20*np.log10(s11_open_mag)
	#r8  = (180/np.pi)*fit_s11_open_ang[1] - (180/np.pi)*s11_open_ang
	#r9  = 20*np.log10(fit_s11_shorted_mag[1]) - 20*np.log10(s11_shorted_mag)
	#r10 = (180/np.pi)*fit_s11_shorted_ang[1] - (180/np.pi)*s11_shorted_ang
	#r11 = 20*np.log10(fit_s11_sr_mag[1]) - 20*np.log10(s11_sr_mag)
	#r12 = (180/np.pi)*fit_s11_sr_ang[1] - (180/np.pi)*s11_sr_ang
	#r13 = 20*np.log10(fit_s12s21_sr_mag[1]) - 20*np.log10(s12s21_sr_mag)
	#r14 = (180/np.pi)*fit_s12s21_sr_ang[1] - (180/np.pi)*s12s21_sr_ang
	#r15 = 20*np.log10(fit_s22_sr_mag[1]) - 20*np.log10(s22_sr_mag)
	#r16 = (180/np.pi)*fit_s22_sr_ang[1] - (180/np.pi)*s22_sr_ang	
	#r17 = 20*np.log10(fit_s11_simu2_mag[1]) - 20*np.log10(s11_simu2_mag)
	#r18 = (180/np.pi)*fit_s11_simu2_ang[1] - (180/np.pi)*s11_simu2_ang	
	#r19 = 20*np.log10(fit_s11_simu3_mag[1]) - 20*np.log10(s11_simu3_mag)
	#r20 = (180/np.pi)*fit_s11_simu3_ang[1] - (180/np.pi)*s11_simu3_ang	
	

	# Saving output parameters
	if save == 'yes':


		# Average spectra data in frequency range selected
		spectra = np.array([fe, sa, sh, so, ss, ss2, ss3]).T
		
		# Physical temperature
		phys_temp = np.zeros((6,1))
		phys_temp[0,0] = phys_temp_ambient
		phys_temp[1,0] = phys_temp_hot
		phys_temp[2,0] = phys_temp_open
		phys_temp[3,0] = phys_temp_shorted
		phys_temp[4,0] = phys_temp_sim2
		phys_temp[5,0] = phys_temp_sim3		



		# Formating fit parameters
		# --------------------------------------
	
		# Spectra
		par_spec_ambient    = np.reshape(fit_spec_ambient[0],    (-1,1))
		par_spec_hot        = np.reshape(fit_spec_hot[0],        (-1,1))
		par_spec_open       = np.reshape(fit_spec_open[0],       (-1,1))
		par_spec_shorted    = np.reshape(fit_spec_shorted[0],    (-1,1))
		par_spec_sim2       = np.reshape(fit_spec_sim2[0],       (-1,1))
		par_spec_sim3       = np.reshape(fit_spec_sim3[0],       (-1,1))
		
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
	
		par_s11_simu2_mag   = np.reshape(fit_s11_simu2_mag[0],    (-1,1))
		par_s11_simu2_ang   = np.reshape(fit_s11_simu2_ang[0],    (-1,1))
		par_s11_simu3_mag   = np.reshape(fit_s11_simu3_mag[0],    (-1,1))
		par_s11_simu3_ang   = np.reshape(fit_s11_simu3_ang[0],    (-1,1))




		# RMS residuals
		# ---------------------------------
		
		# Spectra
		RMS_spectra       = np.zeros((6,1))
		RMS_spectra[0, 0] = fit_spec_ambient[2]
		RMS_spectra[1, 0] = fit_spec_hot[2]
		RMS_spectra[2, 0] = fit_spec_open[2]
		RMS_spectra[3, 0] = fit_spec_shorted[2]
		RMS_spectra[4, 0] = fit_spec_sim2[2]
		RMS_spectra[5, 0] = fit_spec_sim3[2]		
	
		# S11
		RMS_s11        = np.zeros((20,1))
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
		RMS_s11[16, 0] = fit_s11_simu2_mag[2]
		RMS_s11[17, 0] = fit_s11_simu2_ang[2]
		RMS_s11[18, 0] = fit_s11_simu3_mag[2]
		RMS_s11[19, 0] = fit_s11_simu3_ang[2]






		# Saving
		# ----------------------------------
		np.savetxt(path_data        + 'average_spectra_300_350.txt', spectra)
		np.savetxt(path_par_temp    + 'physical_temperatures.txt',   phys_temp)

		np.savetxt(path_par_spectra + 'par_spec_amb.txt',            par_spec_ambient)
		np.savetxt(path_par_spectra + 'par_spec_hot.txt',            par_spec_hot)
		np.savetxt(path_par_spectra + 'par_spec_open.txt',           par_spec_open)
		np.savetxt(path_par_spectra + 'par_spec_shorted.txt',        par_spec_shorted)
		np.savetxt(path_par_spectra + 'par_spec_sim2.txt',           par_spec_sim2)
		np.savetxt(path_par_spectra + 'par_spec_sim3.txt',           par_spec_sim3)
		
		np.savetxt(path_par_s11     + 'par_s11_LNA_mag.txt',         par_s11_LNA_mag)
		np.savetxt(path_par_s11     + 'par_s11_LNA_ang.txt',         par_s11_LNA_ang)
		np.savetxt(path_par_s11     + 'par_s11_amb_mag.txt',         par_s11_amb_mag)
		np.savetxt(path_par_s11     + 'par_s11_amb_ang.txt',         par_s11_amb_ang)
		np.savetxt(path_par_s11     + 'par_s11_hot_mag.txt',         par_s11_hot_mag)
		np.savetxt(path_par_s11     + 'par_s11_hot_ang.txt',         par_s11_hot_ang)
		np.savetxt(path_par_s11     + 'par_s11_open_mag.txt',        par_s11_open_mag)
		np.savetxt(path_par_s11     + 'par_s11_open_ang.txt',        par_s11_open_ang)
		np.savetxt(path_par_s11     + 'par_s11_shorted_mag.txt',     par_s11_shorted_mag)
		np.savetxt(path_par_s11     + 'par_s11_shorted_ang.txt',     par_s11_shorted_ang)

		np.savetxt(path_par_s11     + 'par_s11_sr_mag.txt',          par_s11_sr_mag)
		np.savetxt(path_par_s11     + 'par_s11_sr_ang.txt',          par_s11_sr_ang)
		np.savetxt(path_par_s11     + 'par_s12s21_sr_mag.txt',       par_s12s21_sr_mag)
		np.savetxt(path_par_s11     + 'par_s12s21_sr_ang.txt',       par_s12s21_sr_ang)
		np.savetxt(path_par_s11     + 'par_s22_sr_mag.txt',          par_s22_sr_mag)
		np.savetxt(path_par_s11     + 'par_s22_sr_ang.txt',          par_s22_sr_ang)

		np.savetxt(path_par_s11     + 'par_s11_simu2_mag.txt',       par_s11_simu2_mag)
		np.savetxt(path_par_s11     + 'par_s11_simu2_ang.txt',       par_s11_simu2_ang)		
		np.savetxt(path_par_s11     + 'par_s11_simu3_mag.txt',       par_s11_simu3_mag)
		np.savetxt(path_par_s11     + 'par_s11_simu3_ang.txt',       par_s11_simu3_ang)

		np.savetxt(path_par_spectra + 'RMS_spec.txt',                RMS_spectra)
		np.savetxt(path_par_s11     + 'RMS_s11.txt',                 RMS_s11)


	# End
	print('Files processed.')


	return 1 #f_s11, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20






























def calibration_file_computation(calibration_date, folder, FMIN, FMAX, cterms_nominal, wterms_nominal, save_nominal='no', save_nominal_flag='', term_sweep='no', panels=4, plot_nominal='no'):
	
	"""
	
	calibration_date:  '2018_01_25C', '2019_04_25C'
	
	folder: 'nominal', or 'using_50.12ohms', others
	
	"""
	
	
	# Location of saved results
	path_save = edges_folder + 'mid_band/calibration/receiver_calibration/receiver1/' + calibration_date + '/results/' + folder + '/calibration_files/'
	
	
		
	# Spectra
	TuncV = np.genfromtxt(edges_folder + 'mid_band/calibration/receiver_calibration/receiver1/' + calibration_date +  '/results/' + folder + '/data/average_spectra_300_350.txt')
		
		
	# Frequency selection
	Tunc = TuncV[(TuncV[:,0]>=FMIN) & (TuncV[:,0]<=FMAX),:]
		
		
	if (calibration_date == '2018_01_25C') and ((folder == 'nominal_cleaned_60_120MHz') or (folder == 'nominal_cleaned_60_90MHz')):
		
		
		# Taking the original data
		ff    = Tunc[:,0]
		
		TTae  = Tunc[:,1]
		WWae  = Tunc[:,2]
		
		TThe  = Tunc[:,3]
		WWhe  = Tunc[:,4]
			
		TToe  = Tunc[:,5]
		WWoe  = Tunc[:,6]
		
		TTse  = Tunc[:,7]
		WWse  = Tunc[:,8]
		
		TTqe  = Tunc[:,9]
		WWqe  = Tunc[:,10]
		
		WW_all = np.zeros(len(ff))




	elif (calibration_date == '2018_01_25C') and ((folder == 'nominal_2019_12_50-150MHz_try1') or (folder == 'nominal_2019_12_50-150MHz_try1_LNA_rep1') or (folder == 'nominal_2019_12_50-150MHz_try1_LNA_rep2') or (folder == 'nominal_2019_12_50-150MHz_try1_LNA_rep12') or (folder == 'nominal_2019_12_50-150MHz_LNA1_a1_h1_o1_s1_sim2') or (folder == 'nominal_2019_12_50-150MHz_LNA1_a1_h2_o1_s1_sim2') or (folder == 'nominal_2019_12_50-150MHz_LNA1_a2_h1_o1_s1_sim2') or (folder == 'nominal_2019_12_50-150MHz_LNA1_a2_h2_o1_s1_sim2') or (folder == 'nominal_2019_12_50-150MHz_LNA1_a2_h2_o1_s2_sim2') or (folder == 'nominal_2019_12_50-150MHz_LNA1_a2_h2_o2_s1_sim2') or (folder == 'nominal_2019_12_50-150MHz_LNA1_a2_h2_o2_s2_sim2')):
		# Taking the original data
		ff    = Tunc[:,0]		
		TTae  = Tunc[:,1]
		TThe  = Tunc[:,2]
		TToe  = Tunc[:,3]
		TTse  = Tunc[:,4]
		TTqe  = Tunc[:,5]
		
		
		WWae  = np.ones(len(ff))
		WWhe  = np.ones(len(ff))
		WWoe  = np.ones(len(ff))
		WWse  = np.ones(len(ff))
		WWqe  = np.ones(len(ff))
		
		WW_all = np.zeros(len(ff))

		
		
		
	
	else:	
		
		# Taking the original data
		ff    = Tunc[:,0]		
		TTae  = Tunc[:,1]
		TThe  = Tunc[:,2]
		TToe  = Tunc[:,3]
		TTse  = Tunc[:,4]
		TTqe  = Tunc[:,6]
		
		
		WWae  = np.ones(len(ff))
		WWhe  = np.ones(len(ff))
		WWoe  = np.ones(len(ff))
		WWse  = np.ones(len(ff))
		WWqe  = np.ones(len(ff))
		
		WW_all = np.zeros(len(ff))
		
		
		
		
		
		
					
	# Now, RFI cleaning
	N_sigma = 4 #3.5
	
	ind       = np.arange(len(ff))
	
	print('RFI cleaning Ambient')
	tax1, wax1     = rfi.cleaning_sweep(ff, TTae, WWae, window_width_MHz=4, Npolyterms_block=4, N_choice=20, N_sigma=N_sigma)
	flip1, flip2   = rfi.cleaning_sweep(ff, np.flip(TTae), np.flip(WWae), window_width_MHz=4, Npolyterms_block=4, N_choice=20, N_sigma=N_sigma)
	tax2           = np.flip(flip1)
	wax2           = np.flip(flip2)

	iax = np.union1d(ind[wax1==0], ind[wax2==0])	


	
	
	print('RFI cleaning Hot')
	thx1, whx1     = rfi.cleaning_sweep(ff, TThe, WWhe, window_width_MHz=4, Npolyterms_block=4, N_choice=20, N_sigma=N_sigma)
	flip1, flip2   = rfi.cleaning_sweep(ff, np.flip(TThe), np.flip(WWhe), window_width_MHz=4, Npolyterms_block=4, N_choice=20, N_sigma=N_sigma)
	thx2           = np.flip(flip1)
	whx2           = np.flip(flip2)

	ihx = np.union1d(ind[whx1==0], ind[whx2==0])		
	
	
	
	
	print('RFI cleaning Open')
	tox1, wox1     = rfi.cleaning_sweep(ff, TToe, WWoe, window_width_MHz=4, Npolyterms_block=4, N_choice=20, N_sigma=N_sigma)
	flip1, flip2   = rfi.cleaning_sweep(ff, np.flip(TToe), np.flip(WWoe), window_width_MHz=4, Npolyterms_block=4, N_choice=20, N_sigma=N_sigma)
	tox2           = np.flip(flip1)
	wox2           = np.flip(flip2)

	iox = np.union1d(ind[wox1==0], ind[wox2==0])		
	
		
	
	
	print('RFI cleaning Shorted')
	tsx1, wsx1     = rfi.cleaning_sweep(ff, TTse, WWse, window_width_MHz=4, Npolyterms_block=4, N_choice=20, N_sigma=N_sigma)
	flip1, flip2   = rfi.cleaning_sweep(ff, np.flip(TTse), np.flip(WWse), window_width_MHz=4, Npolyterms_block=4, N_choice=20, N_sigma=N_sigma)
	tsx2           = np.flip(flip1)
	wsx2           = np.flip(flip2)

	isx = np.union1d(ind[wsx1==0], ind[wsx2==0])		
	
	
	
	print('RFI cleaning Simulator')
	tqx1, wqx1     = rfi.cleaning_sweep(ff, TTqe, WWqe, window_width_MHz=4, Npolyterms_block=4, N_choice=20, N_sigma=N_sigma)
	flip1, flip2   = rfi.cleaning_sweep(ff, np.flip(TTqe), np.flip(WWqe), window_width_MHz=4, Npolyterms_block=4, N_choice=20, N_sigma=N_sigma)
	tqx2           = np.flip(flip1)
	wqx2           = np.flip(flip2)

	iqx = np.union1d(ind[wqx1==0], ind[wqx2==0])		
		
	
	
	
	
	XX         = np.union1d(iax, ihx)
	print(len(XX))
	
	XX         = np.union1d(XX, iox)
	print(len(XX))
	
	XX         = np.union1d(XX, isx)
	print(len(XX))
	
	index_bad  = np.union1d(XX, iqx)
	print(len(index_bad))
	
	index_good = np.setdiff1d(ind, index_bad)
	print(len(index_good))
	
	
	
	# Removing data points with zero weight. But for the rest, not using the weights because they are pretty even across frequency
	f     = ff[index_good]
	Tae   = TTae[index_good]
	The   = TThe[index_good]
	Toe   = TToe[index_good]
	Tse   = TTse[index_good]
	Tqe   = TTqe[index_good]
	
	WW_all[index_good] = 1



	
	#else:
		#TuncV = np.genfromtxt(edges_folder + 'mid_band/calibration/receiver_calibration/receiver1/' + calibration_date +  '/results/' + folder + '/data/average_spectra_300_350.txt')
		
		## Frequency selection
		#Tunc = TuncV[(TuncV[:,0]>=FMIN) & (TuncV[:,0]<=FMAX),:]		
				
		#xff    = Tunc[:,0]
		#xTTae  = Tunc[:,1]
		#xTThe  = Tunc[:,2]
		#xTToe  = Tunc[:,3]
		#xTTse  = Tunc[:,4]
		#xTTqe  = Tunc[:,5]
		
		#ff     = xff[(xff>=FMIN) & (xff<=FMAX)]
		#TTae   = xTTae[(xff>=FMIN) & (xff<=FMAX)]
		#TThe   = xTThe[(xff>=FMIN) & (xff<=FMAX)]
		#TToe   = xTToe[(xff>=FMIN) & (xff<=FMAX)]
		#TTse   = xTTse[(xff>=FMIN) & (xff<=FMAX)]
		#TTqe   = xTTqe[(xff>=FMIN) & (xff<=FMAX)]
			
		#ff    = np.copy(f)
		#TTae  = np.copy(Tae)
		#TThe  = np.copy(The)
		#TToe  = np.copy(Toe)
		#TTse  = np.copy(Tse)
		#TTqe  = np.copy(Tqe)		
		
		#WW_all = np.ones(len(ff))
	
	
	
	# Physical temperature
	Tphys = np.genfromtxt(edges_folder + 'mid_band/calibration/receiver_calibration/receiver1/' + calibration_date + '/results/' + folder + '/temp/physical_temperatures.txt')
	
	if (calibration_date == '2018_01_25C') and ((folder == 'nominal_2019_12_50-150MHz_try1') or (folder == 'nominal_2019_12_50-150MHz_try1_LNA_rep1') or (folder == 'nominal_2019_12_50-150MHz_try1_LNA_rep2') or (folder == 'nominal_2019_12_50-150MHz_try1_LNA_rep12') or (folder == 'nominal_2019_12_50-150MHz_LNA1_a1_h1_o1_s1_sim2') or (folder == 'nominal_2019_12_50-150MHz_LNA1_a1_h2_o1_s1_sim2') or (folder == 'nominal_2019_12_50-150MHz_LNA1_a2_h1_o1_s1_sim2') or (folder == 'nominal_2019_12_50-150MHz_LNA1_a2_h2_o1_s1_sim2') or (folder == 'nominal_2019_12_50-150MHz_LNA1_a2_h2_o1_s2_sim2') or (folder == 'nominal_2019_12_50-150MHz_LNA1_a2_h2_o2_s1_sim2') or (folder == 'nominal_2019_12_50-150MHz_LNA1_a2_h2_o2_s2_sim2')):
		
		Ta    = Tphys[0]
		Th    = Tphys[1]
		To    = Tphys[2]
		Ts    = Tphys[3]
		Tsim  = Tphys[4]
		
	else:
		Ta    = Tphys[0]
		Th    = Tphys[1]
		To    = Tphys[2]
		Ts    = Tphys[3]
		Tsim  = Tphys[5]		

	
	
	
	
	
	# S11
	path_s11 = edges_folder + 'mid_band/calibration/receiver_calibration/receiver1/' + calibration_date + '/results/' + folder + '/s11/'
	


	# S11 at original frequency array
	if (calibration_date == '2019_10_25C'):
		ffn  = (ff/200) - 0.5		
		
		par     = np.genfromtxt(path_s11 + 'par_s11_LNA_mag.txt')	
		rl_mag  = ba.model_evaluate('fourier', par, ffn)
		par     = np.genfromtxt(path_s11 + 'par_s11_LNA_ang.txt')
		rl_ang  = ba.model_evaluate('fourier', par, ffn)	
		rrl     = rl_mag * (np.cos(rl_ang) + 1j*np.sin(rl_ang))
		
		par     = np.genfromtxt(path_s11 + 'par_s11_amb_mag.txt')
		ra_mag  = ba.model_evaluate('fourier', par, ffn)	
		par     = np.genfromtxt(path_s11 + 'par_s11_amb_ang.txt')
		ra_ang  = ba.model_evaluate('fourier', par, ffn)
		rra     = ra_mag * (np.cos(ra_ang) + 1j*np.sin(ra_ang))
			
		par     = np.genfromtxt(path_s11 + 'par_s11_hot_mag.txt')
		rh_mag  = ba.model_evaluate('fourier', par, ffn)
		par     = np.genfromtxt(path_s11 + 'par_s11_hot_ang.txt')
		rh_ang  = ba.model_evaluate('fourier', par, ffn)
		rrh     = rh_mag * (np.cos(rh_ang) + 1j*np.sin(rh_ang))
			
		par     = np.genfromtxt(path_s11 + 'par_s11_open_mag.txt')
		ro_mag  = ba.model_evaluate('fourier', par, ffn)	
		par     = np.genfromtxt(path_s11 + 'par_s11_open_ang.txt')
		ro_ang  = ba.model_evaluate('fourier', par, ffn)
		rro     = ro_mag * (np.cos(ro_ang) + 1j*np.sin(ro_ang))
			
		par     = np.genfromtxt(path_s11 + 'par_s11_shorted_mag.txt')
		rs_mag  = ba.model_evaluate('fourier', par, ffn)	
		par     = np.genfromtxt(path_s11 + 'par_s11_shorted_ang.txt')
		rs_ang  = ba.model_evaluate('fourier', par, ffn)		
		rrs     = rs_mag * (np.cos(rs_ang) + 1j*np.sin(rs_ang))
		
		
			
		par         = np.genfromtxt(path_s11 + 'par_s11_sr_mag.txt')
		s11_sr_mag  = ba.model_evaluate('polynomial', par, ffn)
		par         = np.genfromtxt(path_s11 + 'par_s11_sr_ang.txt')
		s11_sr_ang  = ba.model_evaluate('polynomial', par, ffn)
		s11_sr      = s11_sr_mag * (np.cos(s11_sr_ang) + 1j*np.sin(s11_sr_ang))
			
		par            = np.genfromtxt(path_s11 + 'par_s12s21_sr_mag.txt')
		s12s21_sr_mag  = ba.model_evaluate('polynomial', par, ffn)	
		par            = np.genfromtxt(path_s11 + 'par_s12s21_sr_ang.txt')
		s12s21_sr_ang  = ba.model_evaluate('polynomial', par, ffn)
		s12s21_sr      = s12s21_sr_mag * (np.cos(s12s21_sr_ang) + 1j*np.sin(s12s21_sr_ang))
			
		par         = np.genfromtxt(path_s11 + 'par_s22_sr_mag.txt')
		s22_sr_mag  = ba.model_evaluate('polynomial', par, ffn)
		par         = np.genfromtxt(path_s11 + 'par_s22_sr_ang.txt')
		s22_sr_ang  = ba.model_evaluate('polynomial', par, ffn)
		s22_sr      = s22_sr_mag * (np.cos(s22_sr_ang) + 1j*np.sin(s22_sr_ang))
		
		
		
		par        = np.genfromtxt(path_s11 + 'par_s11_simu3_mag.txt')
		rsimu_mag  = ba.model_evaluate('fourier', par, ffn)
		par        = np.genfromtxt(path_s11 + 'par_s11_simu3_ang.txt')
		rsimu_ang  = ba.model_evaluate('fourier', par, ffn)
		rrsimu     = rsimu_mag * (np.cos(rsimu_ang) + 1j*np.sin(rsimu_ang))



	elif (calibration_date == '2018_01_25C') and ((folder == 'nominal_2019_12_50-150MHz_try1') or (folder == 'nominal_2019_12_50-150MHz_try1_LNA_rep1') or (folder == 'nominal_2019_12_50-150MHz_try1_LNA_rep2') or (folder == 'nominal_2019_12_50-150MHz_try1_LNA_rep12') or (folder == 'nominal_2019_12_50-150MHz_LNA1_a1_h1_o1_s1_sim2') or (folder == 'nominal_2019_12_50-150MHz_LNA1_a1_h2_o1_s1_sim2') or (folder == 'nominal_2019_12_50-150MHz_LNA1_a2_h1_o1_s1_sim2') or (folder == 'nominal_2019_12_50-150MHz_LNA1_a2_h2_o1_s1_sim2') or (folder == 'nominal_2019_12_50-150MHz_LNA1_a2_h2_o1_s2_sim2') or (folder == 'nominal_2019_12_50-150MHz_LNA1_a2_h2_o2_s1_sim2') or (folder == 'nominal_2019_12_50-150MHz_LNA1_a2_h2_o2_s2_sim2')):
		ffn  = (ff/200) - 0.5		
		
		par     = np.genfromtxt(path_s11 + 'par_s11_LNA_mag.txt')	
		rl_mag  = ba.model_evaluate('fourier', par, ffn)
		par     = np.genfromtxt(path_s11 + 'par_s11_LNA_ang.txt')
		rl_ang  = ba.model_evaluate('fourier', par, ffn)	
		rrl     = rl_mag * (np.cos(rl_ang) + 1j*np.sin(rl_ang))
		
		par     = np.genfromtxt(path_s11 + 'par_s11_amb_mag.txt')
		ra_mag  = ba.model_evaluate('fourier', par, ffn)	
		par     = np.genfromtxt(path_s11 + 'par_s11_amb_ang.txt')
		ra_ang  = ba.model_evaluate('fourier', par, ffn)
		rra     = ra_mag * (np.cos(ra_ang) + 1j*np.sin(ra_ang))
			
		par     = np.genfromtxt(path_s11 + 'par_s11_hot_mag.txt')
		rh_mag  = ba.model_evaluate('fourier', par, ffn)
		par     = np.genfromtxt(path_s11 + 'par_s11_hot_ang.txt')
		rh_ang  = ba.model_evaluate('fourier', par, ffn)
		rrh     = rh_mag * (np.cos(rh_ang) + 1j*np.sin(rh_ang))
			
		par     = np.genfromtxt(path_s11 + 'par_s11_open_mag.txt')
		ro_mag  = ba.model_evaluate('fourier', par, ffn)	
		par     = np.genfromtxt(path_s11 + 'par_s11_open_ang.txt')
		ro_ang  = ba.model_evaluate('fourier', par, ffn)
		rro     = ro_mag * (np.cos(ro_ang) + 1j*np.sin(ro_ang))
			
		par     = np.genfromtxt(path_s11 + 'par_s11_shorted_mag.txt')
		rs_mag  = ba.model_evaluate('fourier', par, ffn)	
		par     = np.genfromtxt(path_s11 + 'par_s11_shorted_ang.txt')
		rs_ang  = ba.model_evaluate('fourier', par, ffn)		
		rrs     = rs_mag * (np.cos(rs_ang) + 1j*np.sin(rs_ang))
		
		
			
		par         = np.genfromtxt(path_s11 + 'par_s11_sr_mag.txt')
		s11_sr_mag  = ba.model_evaluate('polynomial', par, ffn)
		par         = np.genfromtxt(path_s11 + 'par_s11_sr_ang.txt')
		s11_sr_ang  = ba.model_evaluate('polynomial', par, ffn)
		s11_sr      = s11_sr_mag * (np.cos(s11_sr_ang) + 1j*np.sin(s11_sr_ang))
			
		par            = np.genfromtxt(path_s11 + 'par_s12s21_sr_mag.txt')
		s12s21_sr_mag  = ba.model_evaluate('polynomial', par, ffn)	
		par            = np.genfromtxt(path_s11 + 'par_s12s21_sr_ang.txt')
		s12s21_sr_ang  = ba.model_evaluate('polynomial', par, ffn)
		s12s21_sr      = s12s21_sr_mag * (np.cos(s12s21_sr_ang) + 1j*np.sin(s12s21_sr_ang))
			
		par         = np.genfromtxt(path_s11 + 'par_s22_sr_mag.txt')
		s22_sr_mag  = ba.model_evaluate('polynomial', par, ffn)
		par         = np.genfromtxt(path_s11 + 'par_s22_sr_ang.txt')
		s22_sr_ang  = ba.model_evaluate('polynomial', par, ffn)
		s22_sr      = s22_sr_mag * (np.cos(s22_sr_ang) + 1j*np.sin(s22_sr_ang))
		
		
		
		par        = np.genfromtxt(path_s11 + 'par_s11_simu_mag.txt')
		rsimu_mag  = ba.model_evaluate('fourier', par, ffn)
		par        = np.genfromtxt(path_s11 + 'par_s11_simu_ang.txt')
		rsimu_ang  = ba.model_evaluate('fourier', par, ffn)
		rrsimu     = rsimu_mag * (np.cos(rsimu_ang) + 1j*np.sin(rsimu_ang))		



	else:
		ffn  = (ff-120)/60		
		
		par     = np.genfromtxt(path_s11 + 'par_s11_LNA_mag.txt')	
		rl_mag  = ba.model_evaluate('polynomial', par, ffn)
		par     = np.genfromtxt(path_s11 + 'par_s11_LNA_ang.txt')
		rl_ang  = ba.model_evaluate('polynomial', par, ffn)	
		rrl     = rl_mag * (np.cos(rl_ang) + 1j*np.sin(rl_ang))
		
		par     = np.genfromtxt(path_s11 + 'par_s11_amb_mag.txt')
		ra_mag  = ba.model_evaluate('fourier', par, ffn)	
		par     = np.genfromtxt(path_s11 + 'par_s11_amb_ang.txt')
		ra_ang  = ba.model_evaluate('fourier', par, ffn)
		rra     = ra_mag * (np.cos(ra_ang) + 1j*np.sin(ra_ang))
			
		par     = np.genfromtxt(path_s11 + 'par_s11_hot_mag.txt')
		rh_mag  = ba.model_evaluate('fourier', par, ffn)
		par     = np.genfromtxt(path_s11 + 'par_s11_hot_ang.txt')
		rh_ang  = ba.model_evaluate('fourier', par, ffn)
		rrh     = rh_mag * (np.cos(rh_ang) + 1j*np.sin(rh_ang))
			
		par     = np.genfromtxt(path_s11 + 'par_s11_open_mag.txt')
		ro_mag  = ba.model_evaluate('fourier', par, ffn)	
		par     = np.genfromtxt(path_s11 + 'par_s11_open_ang.txt')
		ro_ang  = ba.model_evaluate('fourier', par, ffn)
		rro     = ro_mag * (np.cos(ro_ang) + 1j*np.sin(ro_ang))
			
		par     = np.genfromtxt(path_s11 + 'par_s11_shorted_mag.txt')
		rs_mag  = ba.model_evaluate('fourier', par, ffn)	
		par     = np.genfromtxt(path_s11 + 'par_s11_shorted_ang.txt')
		rs_ang  = ba.model_evaluate('fourier', par, ffn)		
		rrs     = rs_mag * (np.cos(rs_ang) + 1j*np.sin(rs_ang))
		
		
			
		par         = np.genfromtxt(path_s11 + 'par_s11_sr_mag.txt')
		s11_sr_mag  = ba.model_evaluate('polynomial', par, ffn)
		par         = np.genfromtxt(path_s11 + 'par_s11_sr_ang.txt')
		s11_sr_ang  = ba.model_evaluate('polynomial', par, ffn)
		s11_sr      = s11_sr_mag * (np.cos(s11_sr_ang) + 1j*np.sin(s11_sr_ang))
			
		par            = np.genfromtxt(path_s11 + 'par_s12s21_sr_mag.txt')
		s12s21_sr_mag  = ba.model_evaluate('polynomial', par, ffn)	
		par            = np.genfromtxt(path_s11 + 'par_s12s21_sr_ang.txt')
		s12s21_sr_ang  = ba.model_evaluate('polynomial', par, ffn)
		s12s21_sr      = s12s21_sr_mag * (np.cos(s12s21_sr_ang) + 1j*np.sin(s12s21_sr_ang))
			
		par         = np.genfromtxt(path_s11 + 'par_s22_sr_mag.txt')
		s22_sr_mag  = ba.model_evaluate('polynomial', par, ffn)
		par         = np.genfromtxt(path_s11 + 'par_s22_sr_ang.txt')
		s22_sr_ang  = ba.model_evaluate('polynomial', par, ffn)
		s22_sr      = s22_sr_mag * (np.cos(s22_sr_ang) + 1j*np.sin(s22_sr_ang))
		
		
		
		par        = np.genfromtxt(path_s11 + 'par_s11_simu_mag.txt')
		rsimu_mag  = ba.model_evaluate('polynomial', par, ffn)
		par        = np.genfromtxt(path_s11 + 'par_s11_simu_ang.txt')
		rsimu_ang  = ba.model_evaluate('polynomial', par, ffn)
		rrsimu     = rsimu_mag * (np.cos(rsimu_ang) + 1j*np.sin(rsimu_ang))














	# Temperature of hot device

	# reflection coefficient of termination
	rht = rc.gamma_de_embed(s11_sr, s12s21_sr, s22_sr, rrh)

	# inverting the direction of the s-parameters,
	# since the port labels have to be inverted to match those of Pozar eqn 10.25
	s11_sr_rev = s22_sr
	s22_sr_rev = s11_sr

	# absolute value of S_21
	abs_s21 = np.sqrt(np.abs(s12s21_sr))

	# available power gain
	GG = ( abs_s21**2 ) * ( 1-np.abs(rht)**2 ) / ( (np.abs(1-s11_sr_rev*rht))**2 * (1-(np.abs(rrh))**2) )

	# temperature
	TThd  = GG*Th + (1-GG)*Ta















	
	# S11 at cleaned frequency array
	if (calibration_date == '2019_10_25C'):
		
		fn   = (f/200) - 0.5
		
		par     = np.genfromtxt(path_s11 + 'par_s11_LNA_mag.txt')	
		rl_mag  = ba.model_evaluate('fourier', par, fn)
		par     = np.genfromtxt(path_s11 + 'par_s11_LNA_ang.txt')
		rl_ang  = ba.model_evaluate('fourier', par, fn)	
		rl      = rl_mag * (np.cos(rl_ang) + 1j*np.sin(rl_ang))
		
		par     = np.genfromtxt(path_s11 + 'par_s11_amb_mag.txt')
		ra_mag  = ba.model_evaluate('fourier', par, fn)	
		par     = np.genfromtxt(path_s11 + 'par_s11_amb_ang.txt')
		ra_ang  = ba.model_evaluate('fourier', par, fn)
		ra      = ra_mag * (np.cos(ra_ang) + 1j*np.sin(ra_ang))
			
		par     = np.genfromtxt(path_s11 + 'par_s11_hot_mag.txt')
		rh_mag  = ba.model_evaluate('fourier', par, fn)
		par     = np.genfromtxt(path_s11 + 'par_s11_hot_ang.txt')
		rh_ang  = ba.model_evaluate('fourier', par, fn)
		rh      = rh_mag * (np.cos(rh_ang) + 1j*np.sin(rh_ang))
			
		par     = np.genfromtxt(path_s11 + 'par_s11_open_mag.txt')
		ro_mag  = ba.model_evaluate('fourier', par, fn)	
		par     = np.genfromtxt(path_s11 + 'par_s11_open_ang.txt')
		ro_ang  = ba.model_evaluate('fourier', par, fn)
		ro      = ro_mag * (np.cos(ro_ang) + 1j*np.sin(ro_ang))
			
		par     = np.genfromtxt(path_s11 + 'par_s11_shorted_mag.txt')
		rs_mag  = ba.model_evaluate('fourier', par, fn)	
		par     = np.genfromtxt(path_s11 + 'par_s11_shorted_ang.txt')
		rs_ang  = ba.model_evaluate('fourier', par, fn)		
		rs      = rs_mag * (np.cos(rs_ang) + 1j*np.sin(rs_ang))
		
		
			
		par         = np.genfromtxt(path_s11 + 'par_s11_sr_mag.txt')
		s11_sr_mag  = ba.model_evaluate('polynomial', par, fn)
		par         = np.genfromtxt(path_s11 + 'par_s11_sr_ang.txt')
		s11_sr_ang  = ba.model_evaluate('polynomial', par, fn)
		s11_sr      = s11_sr_mag * (np.cos(s11_sr_ang) + 1j*np.sin(s11_sr_ang))
			
		par            = np.genfromtxt(path_s11 + 'par_s12s21_sr_mag.txt')
		s12s21_sr_mag  = ba.model_evaluate('polynomial', par, fn)	
		par            = np.genfromtxt(path_s11 + 'par_s12s21_sr_ang.txt')
		s12s21_sr_ang  = ba.model_evaluate('polynomial', par, fn)
		s12s21_sr      = s12s21_sr_mag * (np.cos(s12s21_sr_ang) + 1j*np.sin(s12s21_sr_ang))
			
		par         = np.genfromtxt(path_s11 + 'par_s22_sr_mag.txt')
		s22_sr_mag  = ba.model_evaluate('polynomial', par, fn)
		par         = np.genfromtxt(path_s11 + 'par_s22_sr_ang.txt')
		s22_sr_ang  = ba.model_evaluate('polynomial', par, fn)
		s22_sr      = s22_sr_mag * (np.cos(s22_sr_ang) + 1j*np.sin(s22_sr_ang))
		
		
		
		par        = np.genfromtxt(path_s11 + 'par_s11_simu3_mag.txt')
		rsimu_mag  = ba.model_evaluate('fourier', par, fn)
		par        = np.genfromtxt(path_s11 + 'par_s11_simu3_ang.txt')
		rsimu_ang  = ba.model_evaluate('fourier', par, fn)
		rsimu      = rsimu_mag * (np.cos(rsimu_ang) + 1j*np.sin(rsimu_ang))




	elif (calibration_date == '2018_01_25C') and ((folder == 'nominal_2019_12_50-150MHz_try1') or (folder == 'nominal_2019_12_50-150MHz_try1_LNA_rep1') or (folder == 'nominal_2019_12_50-150MHz_try1_LNA_rep2') or (folder == 'nominal_2019_12_50-150MHz_try1_LNA_rep12') or (folder == 'nominal_2019_12_50-150MHz_LNA1_a1_h1_o1_s1_sim2') or (folder == 'nominal_2019_12_50-150MHz_LNA1_a1_h2_o1_s1_sim2') or (folder == 'nominal_2019_12_50-150MHz_LNA1_a2_h1_o1_s1_sim2') or (folder == 'nominal_2019_12_50-150MHz_LNA1_a2_h2_o1_s1_sim2') or (folder == 'nominal_2019_12_50-150MHz_LNA1_a2_h2_o1_s2_sim2') or (folder == 'nominal_2019_12_50-150MHz_LNA1_a2_h2_o2_s1_sim2') or (folder == 'nominal_2019_12_50-150MHz_LNA1_a2_h2_o2_s2_sim2')):
		
		fn   = (f/200) - 0.5
		
		par     = np.genfromtxt(path_s11 + 'par_s11_LNA_mag.txt')	
		rl_mag  = ba.model_evaluate('fourier', par, fn)
		par     = np.genfromtxt(path_s11 + 'par_s11_LNA_ang.txt')
		rl_ang  = ba.model_evaluate('fourier', par, fn)	
		rl      = rl_mag * (np.cos(rl_ang) + 1j*np.sin(rl_ang))
		
		par     = np.genfromtxt(path_s11 + 'par_s11_amb_mag.txt')
		ra_mag  = ba.model_evaluate('fourier', par, fn)	
		par     = np.genfromtxt(path_s11 + 'par_s11_amb_ang.txt')
		ra_ang  = ba.model_evaluate('fourier', par, fn)
		ra      = ra_mag * (np.cos(ra_ang) + 1j*np.sin(ra_ang))
			
		par     = np.genfromtxt(path_s11 + 'par_s11_hot_mag.txt')
		rh_mag  = ba.model_evaluate('fourier', par, fn)
		par     = np.genfromtxt(path_s11 + 'par_s11_hot_ang.txt')
		rh_ang  = ba.model_evaluate('fourier', par, fn)
		rh      = rh_mag * (np.cos(rh_ang) + 1j*np.sin(rh_ang))
			
		par     = np.genfromtxt(path_s11 + 'par_s11_open_mag.txt')
		ro_mag  = ba.model_evaluate('fourier', par, fn)	
		par     = np.genfromtxt(path_s11 + 'par_s11_open_ang.txt')
		ro_ang  = ba.model_evaluate('fourier', par, fn)
		ro      = ro_mag * (np.cos(ro_ang) + 1j*np.sin(ro_ang))
			
		par     = np.genfromtxt(path_s11 + 'par_s11_shorted_mag.txt')
		rs_mag  = ba.model_evaluate('fourier', par, fn)	
		par     = np.genfromtxt(path_s11 + 'par_s11_shorted_ang.txt')
		rs_ang  = ba.model_evaluate('fourier', par, fn)		
		rs      = rs_mag * (np.cos(rs_ang) + 1j*np.sin(rs_ang))
		
		
			
		par         = np.genfromtxt(path_s11 + 'par_s11_sr_mag.txt')
		s11_sr_mag  = ba.model_evaluate('polynomial', par, fn)
		par         = np.genfromtxt(path_s11 + 'par_s11_sr_ang.txt')
		s11_sr_ang  = ba.model_evaluate('polynomial', par, fn)
		s11_sr      = s11_sr_mag * (np.cos(s11_sr_ang) + 1j*np.sin(s11_sr_ang))
			
		par            = np.genfromtxt(path_s11 + 'par_s12s21_sr_mag.txt')
		s12s21_sr_mag  = ba.model_evaluate('polynomial', par, fn)	
		par            = np.genfromtxt(path_s11 + 'par_s12s21_sr_ang.txt')
		s12s21_sr_ang  = ba.model_evaluate('polynomial', par, fn)
		s12s21_sr      = s12s21_sr_mag * (np.cos(s12s21_sr_ang) + 1j*np.sin(s12s21_sr_ang))
			
		par         = np.genfromtxt(path_s11 + 'par_s22_sr_mag.txt')
		s22_sr_mag  = ba.model_evaluate('polynomial', par, fn)
		par         = np.genfromtxt(path_s11 + 'par_s22_sr_ang.txt')
		s22_sr_ang  = ba.model_evaluate('polynomial', par, fn)
		s22_sr      = s22_sr_mag * (np.cos(s22_sr_ang) + 1j*np.sin(s22_sr_ang))
		
		
		
		par        = np.genfromtxt(path_s11 + 'par_s11_simu_mag.txt')
		rsimu_mag  = ba.model_evaluate('fourier', par, fn)
		par        = np.genfromtxt(path_s11 + 'par_s11_simu_ang.txt')
		rsimu_ang  = ba.model_evaluate('fourier', par, fn)
		rsimu      = rsimu_mag * (np.cos(rsimu_ang) + 1j*np.sin(rsimu_ang))


	else:
		fn  = (f-120)/60
		
		par     = np.genfromtxt(path_s11 + 'par_s11_LNA_mag.txt')	
		rl_mag  = ba.model_evaluate('polynomial', par, fn)
		par     = np.genfromtxt(path_s11 + 'par_s11_LNA_ang.txt')
		rl_ang  = ba.model_evaluate('polynomial', par, fn)	
		rl      = rl_mag * (np.cos(rl_ang) + 1j*np.sin(rl_ang))
		
		par     = np.genfromtxt(path_s11 + 'par_s11_amb_mag.txt')
		ra_mag  = ba.model_evaluate('fourier', par, fn)	
		par     = np.genfromtxt(path_s11 + 'par_s11_amb_ang.txt')
		ra_ang  = ba.model_evaluate('fourier', par, fn)
		ra      = ra_mag * (np.cos(ra_ang) + 1j*np.sin(ra_ang))
			
		par     = np.genfromtxt(path_s11 + 'par_s11_hot_mag.txt')
		rh_mag  = ba.model_evaluate('fourier', par, fn)
		par     = np.genfromtxt(path_s11 + 'par_s11_hot_ang.txt')
		rh_ang  = ba.model_evaluate('fourier', par, fn)
		rh      = rh_mag * (np.cos(rh_ang) + 1j*np.sin(rh_ang))
			
		par     = np.genfromtxt(path_s11 + 'par_s11_open_mag.txt')
		ro_mag  = ba.model_evaluate('fourier', par, fn)	
		par     = np.genfromtxt(path_s11 + 'par_s11_open_ang.txt')
		ro_ang  = ba.model_evaluate('fourier', par, fn)
		ro      = ro_mag * (np.cos(ro_ang) + 1j*np.sin(ro_ang))
			
		par     = np.genfromtxt(path_s11 + 'par_s11_shorted_mag.txt')
		rs_mag  = ba.model_evaluate('fourier', par, fn)	
		par     = np.genfromtxt(path_s11 + 'par_s11_shorted_ang.txt')
		rs_ang  = ba.model_evaluate('fourier', par, fn)		
		rs      = rs_mag * (np.cos(rs_ang) + 1j*np.sin(rs_ang))
		
		
			
		par         = np.genfromtxt(path_s11 + 'par_s11_sr_mag.txt')
		s11_sr_mag  = ba.model_evaluate('polynomial', par, fn)
		par         = np.genfromtxt(path_s11 + 'par_s11_sr_ang.txt')
		s11_sr_ang  = ba.model_evaluate('polynomial', par, fn)
		s11_sr      = s11_sr_mag * (np.cos(s11_sr_ang) + 1j*np.sin(s11_sr_ang))
			
		par            = np.genfromtxt(path_s11 + 'par_s12s21_sr_mag.txt')
		s12s21_sr_mag  = ba.model_evaluate('polynomial', par, fn)	
		par            = np.genfromtxt(path_s11 + 'par_s12s21_sr_ang.txt')
		s12s21_sr_ang  = ba.model_evaluate('polynomial', par, fn)
		s12s21_sr      = s12s21_sr_mag * (np.cos(s12s21_sr_ang) + 1j*np.sin(s12s21_sr_ang))
			
		par         = np.genfromtxt(path_s11 + 'par_s22_sr_mag.txt')
		s22_sr_mag  = ba.model_evaluate('polynomial', par, fn)
		par         = np.genfromtxt(path_s11 + 'par_s22_sr_ang.txt')
		s22_sr_ang  = ba.model_evaluate('polynomial', par, fn)
		s22_sr      = s22_sr_mag * (np.cos(s22_sr_ang) + 1j*np.sin(s22_sr_ang))
		
		
		
		par        = np.genfromtxt(path_s11 + 'par_s11_simu_mag.txt')
		rsimu_mag  = ba.model_evaluate('polynomial', par, fn)
		par        = np.genfromtxt(path_s11 + 'par_s11_simu_ang.txt')
		rsimu_ang  = ba.model_evaluate('polynomial', par, fn)
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






	
		
	# Sweeping parameters and computing RMS
	Tamb_internal = 300
	index_cterms = np.arange(1,15,1)
	index_wterms = np.arange(1,15,1)
	
	
	
	if term_sweep == 'yes':
		RMS = np.zeros((4,len(index_wterms),len(index_cterms)))
		for j in range(len(index_cterms)):
			for i in range(len(index_wterms)):
				C1, C2, TU, TC, TS = cal.calibration_quantities(fn, Tae, The, Toe, Tse, rl, ra, rh, ro, rs, Ta, Thd, To, Ts, Tamb_internal, j+1, i+1, second_frequency_array=ffn)
				
				# Only open cable
				#C1, C2, TU, TC, TS = cal.calibration_quantities(fn, Tae, The, Tse, Tse, rl, ra, rh, rs, rs, Ta, Thd, Ts, Ts, Tamb_internal, j+1, i+1, second_frequency_array=ffn)
				
				print('---------------------------------------------------')
				print(j+1)
				print(i+1)
	
	
				# Cross-check	
				TTac              = cal.calibrated_antenna_temperature(TTae,  rra,  rrl, C1, C2, TU, TC, TS, Tamb_internal=Tamb_internal)
				fb, tab, wb, stdb = ba.spectral_binning_number_of_samples(ff, TTac, WW_all, nsamples=64)
				
				TThc              = cal.calibrated_antenna_temperature(TThe,  rrh,  rrl, C1, C2, TU, TC, TS, Tamb_internal=Tamb_internal)
				TThhc             = (TThc - (1-GG)*Ta)/GG
				fb, thb, wb, stdb = ba.spectral_binning_number_of_samples(ff, TThhc, WW_all, nsamples=64)
				
				TToc              = cal.calibrated_antenna_temperature(TToe,  rro,  rrl, C1, C2, TU, TC, TS, Tamb_internal=Tamb_internal)
				fb, tob, wb, stdb = ba.spectral_binning_number_of_samples(ff, TToc, WW_all, nsamples=64)
				
				TTsc              = cal.calibrated_antenna_temperature(TTse,  rrs,  rrl, C1, C2, TU, TC, TS, Tamb_internal=Tamb_internal)
				fb, tsb, wb, stdb = ba.spectral_binning_number_of_samples(ff, TTsc, WW_all, nsamples=64)
			
				
				
				TTqc              = cal.calibrated_antenna_temperature(TTqe,  rrsimu,  rrl, C1, C2, TU, TC, TS, Tamb_internal=Tamb_internal)
				fb, tqb, wb, stdb = ba.spectral_binning_number_of_samples(ff, TTqc, WW_all, nsamples=64)
				
				
				RMS[0, i, j] = np.std(tab-Ta)
				RMS[1, i, j] = np.std(thb-Th)
				RMS[2, i, j] = np.std(tob-To)
				RMS[3, i, j] = np.std(tsb-Ts)
				
				
				# Save figure 1 (BINNED)
				# ----------------------
				plt.close()
				plt.close()
				plt.close()
				plt.close()
				
				if panels == 4:
					plt.figure(1, figsize=[6,6])
					plt.subplot(4,1,1); plt.plot(fb, tab); plt.plot(fb, Ta*np.ones(len(fb))); plt.xticks(np.arange(FMIN, FMAX+1, 10), labels=[]); plt.ylabel('Tamb\nRMS=' + str(round(np.std(tab-Ta),3))+'K'); plt.title('CTerms='+str(index_cterms[j])+', WTerms='+str(index_wterms[i]))
					plt.subplot(4,1,2); plt.plot(fb, thb); plt.plot(fb, Th*np.ones(len(fb))); plt.xticks(np.arange(FMIN, FMAX+1, 10), labels=[]); plt.ylabel('Thot\nRMS=' + str(round(np.std(thb-Th),3))+'K')
					plt.subplot(4,1,3); plt.plot(fb, tob); plt.plot(fb, To*np.ones(len(fb))); plt.xticks(np.arange(FMIN, FMAX+1, 10), labels=[]); plt.ylabel('Topen\nRMS=' + str(round(np.std(tob-To),2))+'K')
					plt.subplot(4,1,4); plt.plot(fb, tsb); plt.plot(fb, Ts*np.ones(len(fb))); plt.xticks(np.arange(FMIN, FMAX+1, 10)); plt.ylabel('Tshorted\nRMS=' + str(round(np.std(tsb-Ts),2))+'K')
					plt.xlabel('frequency [MHz]')
				
				elif panels == 5:
					plt.figure(1, figsize=[6,9])
					plt.subplot(5,1,1); plt.plot(fb, tab); plt.plot(fb, Ta*np.ones(len(fb))); plt.xticks(np.arange(FMIN, FMAX+1, 10), labels=[]); plt.ylabel('Tamb\nRMS=' + str(round(np.std(tab-Ta),3))+'K'); plt.title('CTerms='+str(index_cterms[j])+', WTerms='+str(index_wterms[i]))
					plt.subplot(5,1,2); plt.plot(fb, thb); plt.plot(fb, Th*np.ones(len(fb))); plt.xticks(np.arange(FMIN, FMAX+1, 10), labels=[]); plt.ylabel('Thot\nRMS=' + str(round(np.std(thb-Th),3))+'K')
					plt.subplot(5,1,3); plt.plot(fb, tob); plt.plot(fb, To*np.ones(len(fb))); plt.xticks(np.arange(FMIN, FMAX+1, 10), labels=[]); plt.ylabel('Topen\nRMS=' + str(round(np.std(tob-To),2))+'K')
					plt.subplot(5,1,4); plt.plot(fb, tsb); plt.plot(fb, Ts*np.ones(len(fb))); plt.xticks(np.arange(FMIN, FMAX+1, 10), labels=[]); plt.ylabel('Tshorted\nRMS=' + str(round(np.std(tsb-Ts),2))+'K')
					plt.subplot(5,1,5); plt.plot(fb, tqb); plt.plot(fb, Tsim*np.ones(len(fb))); plt.xticks(np.arange(FMIN, FMAX+1, 10)); plt.ylabel('Tsimu\nRMS=' + str(round(np.std(tqb-Tsim),2))+'K')
					plt.xlabel('frequency [MHz]')					
				
				# Creating folder if necessary
				path_save_term_sweep = path_save + '/binned_calibration_term_sweep_'+str(FMIN)+'_'+str(FMAX)+'MHz/'
				if not exists(path_save_term_sweep):
					makedirs(path_save_term_sweep)					
				plt.savefig(path_save_term_sweep + 'calibration_term_sweep_'+str(FMIN)+'_'+str(FMAX)+'MHz_cterms'+str(index_cterms[j])+'_wterms'+str(index_wterms[i])+'.png', bbox_inches='tight')
				plt.close()
				
				
				
				
				# Save figure 2 (RAW)
				# -------------------
				plt.close()
				plt.close()
				plt.close()
				plt.close()
				
				if panels == 4:
					plt.figure(1, figsize=[6,6])
					plt.subplot(4,1,1); plt.plot(ff[WW_all>0], TTac[WW_all>0]); plt.plot(fb, Ta*np.ones(len(fb))); plt.xticks(np.arange(FMIN, FMAX+1, 10), labels=[]); plt.ylabel('Tamb\nRMS=' + str(round(np.std(TTac[WW_all>0]-Ta),3))+'K'); plt.title('CTerms='+str(index_cterms[j])+', WTerms='+str(index_wterms[i]))
					
					plt.subplot(4,1,2); plt.plot(ff[WW_all>0], TThhc[WW_all>0]); plt.plot(fb, Th*np.ones(len(fb))); plt.xticks(np.arange(FMIN, FMAX+1, 10), labels=[]); plt.ylabel('Thot\nRMS=' + str(round(np.std(TThhc[WW_all>0]-Th),3))+'K')
					
					plt.subplot(4,1,3); plt.plot(ff[WW_all>0], TToc[WW_all>0]); plt.plot(fb, To*np.ones(len(fb))); plt.xticks(np.arange(FMIN, FMAX+1, 10), labels=[]); plt.ylabel('Topen\nRMS=' + str(round(np.std(TToc[WW_all>0]-To),2))+'K')
					
					plt.subplot(4,1,4); plt.plot(ff[WW_all>0], TTsc[WW_all>0]); plt.plot(fb, Ts*np.ones(len(fb))); plt.xticks(np.arange(FMIN, FMAX+1, 10)); plt.ylabel('Tshorted\nRMS=' + str(round(np.std(TTsc[WW_all>0]-Ts),2))+'K')
					plt.xlabel('frequency [MHz]')
				
				elif panels == 5:
					plt.figure(1, figsize=[6,9])
					plt.subplot(5,1,1); plt.plot(ff[WW_all>0], TTac[WW_all>0]); plt.plot(fb, Ta*np.ones(len(fb))); plt.xticks(np.arange(FMIN, FMAX+1, 10), labels=[]); plt.ylabel('Tamb\nRMS=' + str(round(np.std(TTac[WW_all>0]-Ta),3))+'K'); plt.title('CTerms='+str(index_cterms[j])+', WTerms='+str(index_wterms[i]))
					plt.subplot(5,1,2); plt.plot(ff[WW_all>0], TThhc[WW_all>0]); plt.plot(fb, Th*np.ones(len(fb))); plt.xticks(np.arange(FMIN, FMAX+1, 10), labels=[]); plt.ylabel('Thot\nRMS=' + str(round(np.std(TThhc[WW_all>0]-Th),3))+'K')
					plt.subplot(5,1,3); plt.plot(ff[WW_all>0], TToc[WW_all>0]); plt.plot(fb, To*np.ones(len(fb))); plt.xticks(np.arange(FMIN, FMAX+1, 10), labels=[]); plt.ylabel('Topen\nRMS=' + str(round(np.std(TToc[WW_all>0]-To),2))+'K')
					plt.subplot(5,1,4); plt.plot(ff[WW_all>0], TTsc[WW_all>0]); plt.plot(fb, Ts*np.ones(len(fb))); plt.xticks(np.arange(FMIN, FMAX+1, 10), labels=[]); plt.ylabel('Tshorted\nRMS=' + str(round(np.std(TTsc[WW_all>0]-Ts),2))+'K')
					plt.subplot(5,1,5); plt.plot(ff[WW_all>0], TTqc[WW_all>0]); plt.plot(fb, Tsim*np.ones(len(fb))); plt.xticks(np.arange(FMIN, FMAX+1, 10)); plt.ylabel('Tsimu\nRMS=' + str(round(np.std(TTqc[WW_all>0]-Tsim),2))+'K')
					plt.xlabel('frequency [MHz]')					
				
				# Creating folder if necessary
				path_save_term_sweep = path_save + '/raw_calibration_term_sweep_'+str(FMIN)+'_'+str(FMAX)+'MHz/'
				if not exists(path_save_term_sweep):
					makedirs(path_save_term_sweep)					
				plt.savefig(path_save_term_sweep + 'calibration_term_sweep_'+str(FMIN)+'_'+str(FMAX)+'MHz_cterms'+str(index_cterms[j])+'_wterms'+str(index_wterms[i])+'.png', bbox_inches='tight')
				plt.close()			
				
				
				
				
				
				
				
				
				
				
				
				
				
				
				
				
		# Save
		with h5py.File(path_save_term_sweep + 'calibration_term_sweep_'+str(FMIN)+'_'+str(FMAX)+'MHz.hdf5', 'w') as hf:
			hf.create_dataset('RMS',          data = RMS)
			hf.create_dataset('index_cterms', data = index_cterms)
			hf.create_dataset('index_wterms', data = index_wterms)
			
			
	
	
	
	# Saving nominal case	
	C1, C2, TU, TC, TS = cal.calibration_quantities(fn, Tae, The, Toe, Tse, rl, ra, rh, ro, rs, Ta, Thd, To, Ts, Tamb_internal, cterms_nominal, wterms_nominal, second_frequency_array=ffn)
	#C1, C2, TU, TC, TS = cal.calibration_quantities(fn, Tae, The, Tse, Tse, rl, ra, rh, rs, rs, Ta, Thd, Ts, Ts, Tamb_internal, cterms_nominal, wterms_nominal, second_frequency_array=ffn)
	
	print('LENGTH of fn: ' + str(len(fn)))
	print('LENGTH of ffn: ' + str(len(ffn)))
	print(np.sum(ffn))
	print('LENGTH of Tae: ' + str(len(Tae)))
	print('LENGTH of rl: ' + str(len(rl)))
	print('LENGTH of ra: ' + str(len(ra)))
	print('LENGTH of C1: ' + str(len(C1)))
	
	
	

	# Saving results
	if save_nominal == 'yes':

		# Array
		save_array      = np.zeros((len(ff), 8))
		save_array[:,0] = ff
		save_array[:,1] = np.real(rrl)
		save_array[:,2] = np.imag(rrl)
		save_array[:,3] = C1
		save_array[:,4] = C2
		save_array[:,5] = TU
		save_array[:,6] = TC
		save_array[:,7] = TS

		# Save
		np.savetxt(path_save + 'calibration_file_receiver1' + save_nominal_flag + '.txt', save_array, fmt='%1.8f')


	
	if plot_nominal == 'yes':





		# Plotting cross-check	
		TTac            = cal.calibrated_antenna_temperature(TTae,  rra,  rrl, C1, C2, TU, TC, TS, Tamb_internal=Tamb_internal)
		fb, tab, wb, sb = ba.spectral_binning_number_of_samples(ff, TTac, WW_all, nsamples=64)
		
		TThc            = cal.calibrated_antenna_temperature(TThe,  rrh,  rrl, C1, C2, TU, TC, TS, Tamb_internal=Tamb_internal)
		TThhc           = (TThc - (1-GG)*Ta)/GG
		fb, thb, wb, sb = ba.spectral_binning_number_of_samples(ff, TThhc, WW_all, nsamples=64)
		
		TToc            = cal.calibrated_antenna_temperature(TToe,  rro,  rrl, C1, C2, TU, TC, TS, Tamb_internal=Tamb_internal)
		fb, tob, wb, sb = ba.spectral_binning_number_of_samples(ff, TToc, WW_all, nsamples=64)
		
		TTsc            = cal.calibrated_antenna_temperature(TTse,  rrs,  rrl, C1, C2, TU, TC, TS, Tamb_internal=Tamb_internal)
		fb, tsb, wb, sb = ba.spectral_binning_number_of_samples(ff, TTsc, WW_all, nsamples=64)
	
		
		
		TTqc            = cal.calibrated_antenna_temperature(TTqe,  rrsimu,  rrl, C1, C2, TU, TC, TS, Tamb_internal=Tamb_internal)
		fb, tqb, wb, sb = ba.spectral_binning_number_of_samples(ff, TTqc, WW_all, nsamples=64)
		
		
		
		
		plt.close()
		plt.close()
		plt.close()
		plt.close()
		
		fmin_new = np.copy(FMIN)
		fmax_new = np.copy(FMAX)
		lw = 1
		
		plt.figure(1, figsize=[10, 10])
		plt.subplot(5,1,1); plt.plot(ff[WW_all>0], TTac[WW_all>0], 'g', linewidth=lw); plt.plot(fb, Ta*np.ones(len(fb)), 'k'); plt.xticks(np.arange(FMIN, FMAX+1, 10), labels=[]); plt.ylabel(r'T$_A$ [K]' + '\n' + 'RMS=' + str(round(np.std(TTac[WW_all>0]-Ta),3))+' K', fontsize=13)
		#plt.title('CTerms='+str(cterms_nominal)+', WTerms='+str(wterms_nominal))
		plt.xlim([fmin_new, fmax_new])
		#plt.ylim([295.8, 297.0])
		#plt.ylim([295.2, 297.5])
		
		plt.subplot(5,1,2); plt.plot(ff[WW_all>0], TThhc[WW_all>0], 'g', linewidth=lw); plt.plot(fb, Th*np.ones(len(fb)), 'k'); plt.xticks(np.arange(FMIN, FMAX+1, 10), labels=[]); plt.ylabel(r'T$_H$ [K]' + '\n' + 'RMS=' + str(round(np.std(TThhc[WW_all>0]-Th),3))+' K', fontsize=13)
		#RMS=' + str(round(np.std(TThhc[WW_all>0]-Th),3))+'K')
		plt.xlim([fmin_new, fmax_new])
		#plt.ylim([398.5, 400.9])
		
		plt.subplot(5,1,3); plt.plot(ff[WW_all>0], TToc[WW_all>0], 'g', linewidth=lw); plt.plot(fb, To*np.ones(len(fb)), 'k'); plt.xticks(np.arange(FMIN, FMAX+1, 10), labels=[]); plt.ylabel(r'T$_O$ [K]' + '\n' + 'RMS=' + str(round(np.std(TToc[WW_all>0]-To),2))+' K', fontsize=13)
		plt.xlim([fmin_new, fmax_new])
		#plt.ylim([288, 304])
		#plt.ylim([284, 308])
		
		plt.subplot(5,1,4); plt.plot(ff[WW_all>0], TTsc[WW_all>0], 'g', linewidth=lw); plt.plot(fb, Ts*np.ones(len(fb)), 'k'); plt.xticks(np.arange(FMIN, FMAX+1, 10)); plt.ylabel(r'T$_S$ [K]' + '\n' + 'RMS=' + str(round(np.std(TTsc[WW_all>0]-Ts),2))+' K', fontsize=13)
		plt.xlim([fmin_new, fmax_new])
		#plt.ylim([288, 304])
		#plt.ylim([284, 308])
		plt.xlabel(r'$\nu$ [MHz]', fontsize=13)
	
	
		plt.subplot(5,1,5); plt.plot(ff[WW_all>0], TTqc[WW_all>0], 'g', linewidth=lw);
		#plt.plot(fb, Tsim*np.ones(len(fb)), 'k');
		plt.xticks(np.arange(FMIN, FMAX+1, 10)); plt.ylabel(r'T$_{\mathrm{sim}}$ [K]' + '\n' + 'RMS=' + str(round(np.std(TTqc[WW_all>0]),2))+' K', fontsize=13)
		plt.xlim([fmin_new, fmax_new])
		#plt.ylim([288, 304])
		#plt.ylim([284, 308])
		plt.xlabel(r'$\nu$ [MHz]', fontsize=13)	
	
	
	
		# Creating folder if necessary
					
		#plt.savefig(path_save + 'calibration_crosscheck_60_120MHz_cterms'+str(cterms_nominal)+'_wterms'+str(wterms_nominal)+'.pdf', bbox_inches='tight')
		plt.savefig(path_save + 'calibration_crosscheck_' + str(fmin_new) + '_' + str(fmax_new) + 'MHz_cterms'+str(cterms_nominal)+'_wterms'+str(wterms_nominal)+'.pdf', bbox_inches='tight')
		plt.close()			
		






	return f, Tae, The, Toe, Tse, Tqe, WW_all    #ff, C1, C2, TU, TC, TS 















def calibration_RMS_read(path_file):

	with h5py.File(path_file,'r') as hf:

		hfX    = hf.get('RMS')
		RMS    = np.array(hfX)

		hfX    = hf.get('index_cterms')
		cterms = np.array(hfX)

		hfX    = hf.get('index_wterms')
		wterms = np.array(hfX)


	return RMS, cterms, wterms	

