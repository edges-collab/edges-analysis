
import numpy as np
import basic as ba
import rfi as rfi
import reflection_coefficient as rc

import scipy.io as sio
import scipy.interpolate as spi

import matplotlib.pyplot as plt
import datetime as dt

import astropy.time as apt
import ephem as eph	# to install, at the bash terminal type $ conda install ephem

import calibration as cal

import h5py

from os import makedirs, listdir
from os.path import exists



home_folder = '/data5/raul'
MRO_folder  = '/data5/edges/data/2014_February_Boolardy'



import os, sys
edges_code_folder = os.environ['EDGES_CODE']
sys.path.insert(0, edges_code_folder)

edges_folder       = os.environ['EDGES']
print('EDGES Folder: ' + edges_folder)







def auxiliary_data(weather_file, thermlog_file, band, year, day):
	"""
	"""

	# scp -P 64122 loco@150.101.175.77:/media/DATA/EDGES_data/weather.txt /home/raul/Desktop/
	# scp -P 64122 loco@150.101.175.77:/media/DATA/EDGES_data/thermlog.txt /home/raul/Desktop/

	# OR

	# scp raul@enterprise.sese.asu.edu:/data1/edges/data/2014_February_Boolardy/weather.txt Desktop/
	# scp raul@enterprise.sese.asu.edu:/data1/edges/data/2014_February_Boolardy/thermlog_low.txt Desktop/
	# scp raul@enterprise.sese.asu.edu:/data1/edges/data/2014_February_Boolardy/thermlog.txt Desktop/



	# Gather data from 'weather.txt' file
	f1 = open(weather_file,'r')		
	lines_all_1 = f1.readlines()

	array1        = np.zeros((0,4))

	if year == 2015:
		i1 = 92000   # ~ day 100
	elif year == 2016:
		i1 = 165097  # start of year 2016
	elif (year == 2017) and (day < 330):
		i1 = 261356  # start of year 2017
	elif (year == 2017) and (day > 331):
		i1 = 0  # start of year 2017 in file weather2.txt
	elif year == 2018:
		i1 = 9806  # start of year in file weather2.txt


	line1         = lines_all_1[i1]
	year_iter_1   = int(line1[0:4])
	day_of_year_1 = int(line1[5:8])


	while (day_of_year_1 <= day) and (year_iter_1 <= year):

		if day_of_year_1 == day:
			#print(line1[0:17] + ' ' + line1[59:65] + ' ' + line1[88:93] + ' ' + line1[113:119])

			date_time = line1[0:17]
			ttt = date_time.split(':')			
			seconds = 3600*int(ttt[2]) + 60*int(ttt[3]) + int(ttt[4])			

			try:
				amb_temp  = float(line1[59:65])
			except ValueError:
				amb_temp  = 0

			try:
				amb_hum   = float(line1[87:93])
			except ValueError:
				amb_hum   = 0

			try:
				rec_temp  = float(line1[113:119])
			except ValueError:
				rec_temp  = 0				


			array1_temp1 = np.array([seconds, amb_temp, amb_hum, rec_temp])
			array1_temp2 = array1_temp1.reshape((1,-1))
			array1       = np.append(array1, array1_temp2, axis=0)

			print('weather time: ' + date_time)


		i1            = i1+1
		print(i1)
		if (i1 != 28394) and (i1 != 1768):
			line1         = lines_all_1[i1]
			year_iter_1   = int(line1[0:4])
			day_of_year_1 = int(line1[5:8])




	# gather data from 'thermlog.txt' file
	f2            = open(thermlog_file,'r')
	lines_all_2   = f2.readlines()

	array2        = np.zeros((0,2))


	if (band == 'high_band') and (year == 2015):
		i2 = 24000    # ~ day 108

	elif (band == 'high_band') and (year == 2016):
		i2 = 58702    # beginning of year 2016

	elif (band == 'low_band') and (year == 2015):
		i2 = 0

	elif (band == 'low_band') and (year == 2016):
		i2 = 14920    # beginning of year 2016	

	elif (band == 'low_band') and (year == 2017):
		i2 = 59352    # beginning of year 2017			

	elif (band == 'low_band2') and (year == 2017) and (day < 332):
		return array1, np.array([0])

	elif (band == 'low_band2') and (year == 2017) and (day >= 332):
		i2 = 0

	elif (band == 'low_band2') and (year == 2018):
		i2 = 4768

	elif (band == 'low_band3') and (year == 2018):
		i2 = 0# 33826

	elif (band == 'mid_band') and (year == 2018) and (day <= 171):
		i2 = 5624     # beginning of year 2018, file "thermlog_mid.txt"

	elif (band == 'mid_band') and (year == 2018) and (day >= 172):
		i2 = 16154   


	line2         = lines_all_2[i2]
	year_iter_2   = int(line2[0:4])
	day_of_year_2 = int(line2[5:8])



	while (day_of_year_2 <= day) and (year_iter_2 <= year):


		if day_of_year_2 == day:
			#print(line2[0:17] + ' ' + line2[48:53])

			date_time = line2[0:17]
			ttt = date_time.split(':')			
			seconds = 3600*int(ttt[2]) + 60*int(ttt[3]) + int(ttt[4])			

			try:
				rec_temp  = float(line2[48:53])
			except ValueError:
				rec_temp  = 0



			array2_temp1 = np.array([seconds, rec_temp])
			array2_temp2 = array2_temp1.reshape((1,-1))
			array2       = np.append(array2, array2_temp2, axis=0)

			print('receiver temperature time: ' + date_time)

		i2 = i2+1
		#print(i2)
		if (i2 != 26348):
			line2         = lines_all_2[i2]
			year_iter_2   = int(line2[0:4])
			day_of_year_2 = int(line2[5:8])	



	return array1, array2









def level1_to_level2(band, year, day_hour, low2_flag='_low2'):

	# Paths and files
	path_level1      = home_folder + '/EDGES/spectra/level1/' + band + '/300_350/'
	path_logs        = MRO_folder  
	save_file        = home_folder + '/EDGES/spectra/level2/' + band + '/' + year + '_' + day_hour + '.hdf5'



	if band == 'low_band2':
		#level1_file      = path_level1 + 'level1_' + year + '_' + day_hour + '_300_350.mat'
		level1_file      = path_level1 + 'level1_' + year + '_' + day_hour + low2_flag + '_300_350.mat'

	elif band == 'low_band3':
		#level1_file      = path_level1 + 'level1_' + year + '_' + day_hour + '_low3_backendB_test.acq_300_350.mat'
		#level1_file      = path_level1 + 'level1_' + year + '_' + day_hour + '_low3_backendB_test2_300_350.mat'
		level1_file      = path_level1 + 'level1_' + year + '_' + day_hour + '_low3_300_350.mat'

	elif band == 'mid_band':
		level1_file      = path_level1 + 'level1_' + year + '_' + day_hour + '_mid_300_350.mat'	

	else:
		level1_file      = path_level1 + 'level1_' + year + '_' + day_hour + '_300_350.mat'



	if (int(year) < 2017) or ((int(year)==2017) and (int(day_hour[0:3])<330)):
		weather_file     = path_logs   + '/weather_upto_20171125.txt'

	if (int(year) == 2018) or ((int(year)==2017) and (int(day_hour[0:3])>331)):
		weather_file     = path_logs   + '/weather2.txt'


	if (band == 'high_band'):
		thermlog_file = path_logs + '/thermlog.txt'

	if (band == 'low_band'):
		thermlog_file = path_logs + '/thermlog_low.txt'

	if (band == 'low_band2'):
		thermlog_file = path_logs + '/thermlog_low2.txt'

	if (band == 'mid_band'):
		thermlog_file = path_logs + '/thermlog_mid.txt'

	if (band == 'low_band3'):
		thermlog_file = path_logs + '/thermlog_low3.txt'







	# Loading data

	# Frequency and indices
	if band == 'low_band':
		flow  = 50
		fhigh = 199

	elif band == 'low_band2':
		flow  = 50
		fhigh = 199

	elif band == 'low_band3':
		flow  = 50
		fhigh = 199			

	elif band == 'mid_band':
		flow  = 50
		fhigh = 199

	elif band == 'high_band':
		flow  = 65
		fhigh = 195



	ff, il, ih = ba.frequency_edges(flow, fhigh)
	fe = ff[il:ih+1]

	ds, dd = ba.level1_MAT(level1_file)
	tt     = ds[:,il:ih+1]
	ww     = np.ones((len(tt[:,0]), len(tt[0,:])))



	# ------------ Meta -------------#
	# -------------------------------#


	# Seconds into measurement	
	seconds_data = 3600*dd[:,3].astype(float) + 60*dd[:,4].astype(float) + dd[:,5].astype(float)

	# EDGES coordinates
	EDGES_LAT = -26.7
	EDGES_LON = 116.6

	# LST
	LST = ba.utc2lst(dd, EDGES_LON)
	LST_column      = LST.reshape(-1,1)

	# Year and day
	year_int        = int(year)
	day_int         = int(day_hour[0:3])

	year_column     = year_int * np.ones((len(LST),1))
	day_column      = day_int * np.ones((len(LST),1))

	if len(day_hour) > 3:
		fraction_int    = int(day_hour[4::])
	elif len(day_hour) == 3:
		fraction_int    = 0

	fraction_column = fraction_int * np.ones((len(LST),1))	



	# Galactic Hour Angle
	LST_gc = 17 + (45/60) + (40.04/(60*60))    # LST of Galactic Center
	GHA    = LST - LST_gc
	for i in range(len(GHA)):
		if GHA[i] < -12.0:
			GHA[i] = GHA[i] + 24
	GHA_column      = GHA.reshape(-1,1)



	sun_moon_azel = ba.SUN_MOON_azel(EDGES_LAT, EDGES_LON, dd)				
	# Sun/Moon coordinates




	# Ambient temperature and humidity, and receiver temperature
	#if band == 'low_band3':
		#amb_rec = np.zeros((len(seconds_data), 4))

	#else:
	aux1, aux2   = auxiliary_data(weather_file, thermlog_file, band, year_int, day_int)

	amb_temp_interp  = np.interp(seconds_data, aux1[:,0], aux1[:,1]) - 273.15
	amb_hum_interp   = np.interp(seconds_data, aux1[:,0], aux1[:,2])
	rec1_temp_interp = np.interp(seconds_data, aux1[:,0], aux1[:,3]) - 273.15

	if len(aux2) == 1:
		rec2_temp_interp = 25*np.ones(len(seconds_data))
	else:
		rec2_temp_interp = np.interp(seconds_data, aux2[:,0], aux2[:,1])

	amb_rec = np.array([amb_temp_interp, amb_hum_interp, rec1_temp_interp, rec2_temp_interp])
	amb_rec = amb_rec.T	




	# Meta	
	meta = np.concatenate((year_column, day_column, fraction_column, LST_column, GHA_column, sun_moon_azel, amb_rec), axis=1)



	# Save
	with h5py.File(save_file, 'w') as hf:
		hf.create_dataset('frequency',           data = fe)
		hf.create_dataset('antenna_temperature', data = tt)
		hf.create_dataset('weights',             data = ww)
		hf.create_dataset('metadata',            data = meta)


	return fe, tt, ww, meta









def level2read(path_file, print_key='no'):

	# path_file = home_folder + '/EDGES/spectra/level2/mid_band/file.hdf5'

	# Show keys (array names inside HDF5 file)
	with h5py.File(path_file,'r') as hf:

		if print_key == 'yes':
			print([key for key in hf.keys()])

		hf_freq    = hf.get('frequency')
		freq       = np.array(hf_freq)

		hf_Ta      = hf.get('antenna_temperature')
		Ta         = np.array(hf_Ta)

		hf_meta    = hf.get('metadata')
		meta       = np.array(hf_meta)

		hf_weights = hf.get('weights')
		weights    = np.array(hf_weights)



	return freq, Ta, meta, weights	










def antenna_efficiency(band, f):
		
	return 1




































def level2_to_level3(band, year_day_hdf5, flag_folder='test', receiver_cal_file=1, antenna_s11_year=2018, antenna_s11_day=145, antenna_s11_Nfit=13, beam_correction=1, balun_correction=1, FLOW=50, FHIGH=130, Nfg=5):
	
	"""

	"""


	fin  = 0
	tc   = 0
	w_2D = 0
	m_2D = 0


	# Load daily data
	# ---------------
	path_data = edges_folder + band + '/spectra/level2/'
	filename  = path_data + year_day_hdf5
	fin_X, t_2D_X, m_2D, w_2D_X = level2read(filename)	
	
	
	
		
	# Continue if there are data available
	# ------------------------------------
	if np.sum(t_2D_X) > 0:
		
		
		# Cut the frequency range
		# -----------------------
		fin  = fin_X[(fin_X>=FLOW) & (fin_X<=FHIGH)]
		t_2D = t_2D_X[:, (fin_X>=FLOW) & (fin_X<=FHIGH)]
		w_2D = w_2D_X[:, (fin_X>=FLOW) & (fin_X<=FHIGH)]
		
		
		
		# Beam factor
		# -----------
		# No beam correction
		if beam_correction == 0:
			bf = 1
			print('NO BEAM CORRECTION')
			
		if beam_correction == 1:
			if band == 'mid_band':
				beam_factor_filename = 'table_hires_mid_band_50-200MHz_90deg_alan1_haslam_2.5_2.62_reffreq_100MHz.hdf5'
				
			elif band == 'low_band3':
				beam_factor_filename = 'table_hires_low_band3_50-120MHz_85deg_alan_haslam_2.5_2.62_reffreq_76MHz.hdf5'
				
			f_table, lst_table, bf_table = cal.beam_factor_table_read(edges_folder + band + '/calibration/beam_factors/table/' + beam_factor_filename)
			bf                           = cal.beam_factor_table_evaluate(f_table, lst_table, bf_table, m_2D[:,3])
			
			
		
		
			
		# Antenna S11
		# -----------
		s11_ant = cal.models_antenna_s11_remove_delay(band, fin, year=antenna_s11_year, day=antenna_s11_day, delay_0=0.17, model_type='polynomial', Nfit=antenna_s11_Nfit, plot_fit_residuals='no')
		
		
		
		# Balun+Connector Loss
		# --------------------
		if balun_correction == 0:
			G = 1
			print('NO BALUN CORRECTION')
			
		if balun_correction == 1:
			Gb, Gc = cal.balun_and_connector_loss(band, fin, s11_ant)
			G      = Gb*Gc
		
		
		
		# Receiver calibration quantities
		# -------------------------------
		if (band == 'mid_band') or (band == 'low_band3'):
			
			if receiver_cal_file == 1:
				print('Receiver calibration FILE 1')
				rcv_file = edges_folder + 'mid_band/calibration/receiver_calibration/receiver1/2018_01_25C/results/nominal/calibration_files/calibration_file_mid_band_cfit6_wfit14.txt'
			
			elif receiver_cal_file == 2:
				print('Receiver calibration FILE 2')
				rcv_file = edges_folder + 'mid_band/calibration/receiver_calibration/receiver1/2018_01_25C/results/nominal/calibration_files/calibration_file_mid_band_cfit10_wfit14.txt'
		
		
		
		rcv = np.genfromtxt(rcv_file)
	
		fX      = rcv[:,0]
		rcv2    = rcv[(fX>=FLOW) & (fX<=FHIGH),:]
		s11_LNA = rcv2[:,1] + 1j*rcv2[:,2]
		C1      = rcv2[:,3]
		C2      = rcv2[:,4]
		TU      = rcv2[:,5]
		TC      = rcv2[:,6]
		TS      = rcv2[:,7]

				
	
		# Calibrated antenna temperature with losses and beam chromaticity
		# ----------------------------------------------------------------
		tc_with_loss_and_beam = cal.calibrated_antenna_temperature(t_2D, s11_ant, s11_LNA, C1, C2, TU, TC, TS)
		
		
		
		# Removing loss
		# -------------
		Tamb_2D = np.reshape(273.15 + m_2D[:,9], (-1, 1))
		G_2D    = np.repeat(np.reshape(G, (1, -1)), len(m_2D[:,9]), axis=0)
		
		tc_with_beam = (tc_with_loss_and_beam - Tamb_2D * (1-G_2D)) / G_2D
		
		
		
		# Removing beam chromaticity
		# --------------------------
		tc = tc_with_beam/bf

	
	
		# RFI cleaning
		# ------------
		tt, ww = rfi.excision_raw_frequency(fin, tc, w_2D)
		
		
			
		# Number of spectra
		# -----------------
		lt = len(tt[:,0])
		
		
		
		# Initializing output arrays
		# --------------------------
		t_all   = np.random.rand(lt, len(fin))
		p_all   = np.random.rand(lt, Nfg)
		r_all   = np.random.rand(lt, len(fin))
		w_all   = np.random.rand(lt, len(fin))
		rms_all = np.random.rand(lt, 3)
		
		
		
		# Foreground models and residuals
		# -------------------------------
		for i in range(lt):

			ti  = tt[i,:]
			wi  = ww[i,:]
			
			tti, wwi = rfi.cleaning_polynomial(fin, ti, wi, Nterms_fg=Nfg, Nterms_std=5, Nstd=5)
			
			
			# Fitting foreground model to binned version of spectra
			# -----------------------------------------------------
			Nsamples      = 16 # 48.8 kHz
			fbi, tbi, wbi = ba.spectral_binning_number_of_samples(fin, tti, wwi, nsamples=Nsamples)
			par_fg        = ba.fit_polynomial_fourier('LINLOG', fbi/200, tbi, Nfg, Weights=wbi)
			
			
			# Evaluating foreground model at raw resolution
			# ---------------------------------------------
			model_i       = ba.model_evaluate('LINLOG', par_fg[0], fin/200)  # model_bi = par_fg[1]
			
			
			# Residuals
			# ---------
			rri   =   tti - model_i
			
			
			
							
			# RMS for two halfs of the spectrum
			# ---------------------------------
			IX = int(np.floor(len(fin)/2))
			
			F1 = fin[0:IX]
			R1 = rri[0:IX]
			W1 = wwi[0:IX]
			
			F2 = fin[IX::]
			R2 = rri[IX::]
			W2 = wwi[IX::]	
			
			RMS1 = np.sqrt(np.sum((R1[W1>0])**2)/len(F1[W1>0]))
			RMS2 = np.sqrt(np.sum((R2[W2>0])**2)/len(F2[W2>0]))
		
		
		
		



			# We also compute residuals for 4 terms as an additiobal filter
			# Fitting foreground model to binned version of spectra
			# -----------------------------------------------------			
			par_fg_4t = ba.fit_polynomial_fourier('LINLOG', fbi/200, tbi, 3, Weights=wbi)
		
		
			# Evaluating foreground model at raw resolution
			# ---------------------------------------------			
			model_i_4t = ba.model_evaluate('LINLOG', par_fg_4t[0], fin/200)
		
		
			# Residuals
			# ---------
			rri_4t   = tti - model_i_4t
			
			
			# RMS
			# ---
			RMS3 = np.sqrt(np.sum((rri_4t[wwi>0])**2)/len(fin[wwi>0]))
		
		
		
		
		
		
		
		
		
			
			
			# Store
			# -----
			t_all[i,:]    = tti
			p_all[i,:]    = par_fg[0]
			r_all[i,:]    = rri
			w_all[i,:]    = wwi
			rms_all[i,0]  = RMS1
			rms_all[i,1]  = RMS2
			rms_all[i,2]  = RMS3
			
			
			print(year_day_hdf5 + ': Spectrum number: ' + str(i+1) + ': RMS: ' + str(RMS1) + ', ' + str(RMS2) + ', ' + str(RMS3))
	



	# Save
	# ----
	save_folder = edges_folder + band + '/spectra/level3/' + flag_folder + '/'
	if not exists(save_folder):
		makedirs(save_folder)

	with h5py.File(save_folder + year_day_hdf5, 'w') as hf:
		hf.create_dataset('frequency',           data = fin)
		hf.create_dataset('antenna_temperature', data = t_all)
		hf.create_dataset('parameters',          data = p_all)
		hf.create_dataset('residuals',           data = r_all)
		hf.create_dataset('weights',             data = w_all)
		hf.create_dataset('rms',                 data = rms_all)
		hf.create_dataset('metadata',            data = m_2D)




	return fin, t_all, p_all, r_all, w_all, rms_all, m_2D



	










def level3read(path_file, print_key='no'):

	# path_file = home_folder + '/EDGES/spectra/level3/mid_band/nominal_60_160MHz/file.hdf5'

	# Show keys (array names inside HDF5 file)
	with h5py.File(path_file,'r') as hf:

		if print_key == 'yes':
			print([key for key in hf.keys()])

		hfX    = hf.get('frequency')
		f      = np.array(hfX)

		hfX    = hf.get('antenna_temperature')
		t      = np.array(hfX)

		hfX    = hf.get('parameters')
		p      = np.array(hfX)
		
		hfX    = hf.get('residuals')
		r      = np.array(hfX)
		
		hfX    = hf.get('weights')
		w      = np.array(hfX)		

		hfX    = hf.get('rms')
		rms    = np.array(hfX)		

		hfX    = hf.get('metadata')
		m      = np.array(hfX)



	return f, t, p, r, w, rms, m	








def data_selection(m, GHA_or_LST='GHA', TIME_1=0, TIME_2=24, sun_el_max=90, moon_el_max=90, amb_hum_max = 200, min_receiver_temp=0, max_receiver_temp=100):

	"""

	"""

	# Master index 
	index       = np.arange(len(m[:,0]))


	# Galactic hour angle / LST 
	if GHA_or_LST == 'GHA':
		
		GHA          = m[:,4]
		GHA[GHA<0]   = GHA[GHA<0] + 24
			
		index_TIME_1 = index[GHA >= TIME_1]
		index_TIME_2 = index[GHA <  TIME_2]		
		

	elif GHA_or_LST == 'LST':
		
		index_TIME_1 = index[m[:,3]  >= TIME_1]
		index_TIME_2 = index[m[:,3]  <  TIME_2]		
		


	# Sun elevation, Moon elevation, ambient humidity, and receiver temperature
	index_SUN  = index[m[:,6]   <= sun_el_max]
	index_MOON = index[m[:,8]   <= moon_el_max]
	index_HUM  = index[m[:,10]  <= amb_hum_max]
	index_Trec = index[(m[:,11] >= min_receiver_temp) & (m[:,11] <= max_receiver_temp)]   # restricting receiver temperature Trec from thermistor 2 (S11 switch)


	# Combined index
	if TIME_1 < TIME_2:
		index1    = np.intersect1d(index_TIME_1, index_TIME_2)

	elif TIME_1 > TIME_2:
		index1    = np.union1d(index_TIME_1, index_TIME_2)

	index2    = np.intersect1d(index_SUN, index_MOON)
	index3    = np.intersect1d(index2, index_HUM)
	index4    = np.intersect1d(index3, index_Trec)
	index_all = np.intersect1d(index1, index4)

	#print('NUMBER OF TRACES: ' + str(len(index_all)))



	return index_all










def rms_filter_computation(case, save_parameters='no'):
	"""

	Last modification:  2018-11-07
	
	Computation of the RMS filter for mid-band data
	
	"""
	
	
	
	
	# Listing files available
	# ------------------------
	if case == 0:
		path_files  = edges_folder + '/mid_band/spectra/level3/case0/'
		save_folder = edges_folder + '/mid_band/rms_filters/case0/'
	
	if case == 1:
		path_files  = edges_folder + '/mid_band/spectra/level3/case1/'
		save_folder = edges_folder + '/mid_band/rms_filters/case1/'
	
	if case == 2:
		path_files  = edges_folder + '/mid_band/spectra/level3/case2/'
		save_folder = edges_folder + '/mid_band/rms_filters/case2/'	

		
		
	new_list   = listdir(path_files)
	new_list.sort()
	
	
	# Loading data used to compute filter
	# -----------------------------------
	N_files = 8                  # Only using the first "N_files" to compute the filter
	for i in range(N_files):     # 
		print(new_list[i])
		
		# Loading data
		f, t, p, r, w, rms, m = level3read(path_files + new_list[i])
		
		# Filtering out high humidity
		amb_hum_max = 50
		IX = data_selection(m, GHA_or_LST='GHA', TIME_1=0, TIME_2=24, sun_el_max=90, moon_el_max=90, amb_hum_max=amb_hum_max, min_receiver_temp=0, max_receiver_temp=100)
		
		tx   = t[IX,:]
		px   = p[IX,:]
		rx   = r[IX,:]
		wx   = w[IX,:]
		rmsx = rms[IX,:]
		mx   = m[IX,:]
		
		# Accumulating data
		if i == 0:
			p_all   = np.copy(px)
			r_all   = np.copy(rx)
			w_all   = np.copy(wx)
			rms_all = np.copy(rmsx)
			m_all   = np.copy(mx)
			
		elif i > 0:
			p_all   = np.vstack((p_all, px))
			r_all   = np.vstack((r_all, rx))
			w_all   = np.vstack((w_all, wx))
			rms_all = np.vstack((rms_all, rmsx))
			m_all   = np.vstack((m_all, mx))
	

		
	# Columns necessary for analysis
	# ------------------------------
	GHA        = m_all[:,4]
	GHA[GHA<0] = GHA[GHA<0] + 24	

	RMS1       = rms_all[:,0]
	RMS2       = rms_all[:,1]
	RMS3       = rms_all[:,2]
	
	IN         = np.arange(0,len(GHA))
	
	
	
	# Number of polynomial terms used to fit each 1-hour bins
	# and number of sigma threshold
	# -------------------------------------------------------
	Npoly  = 3
	Nsigma = 3
	
	
	
	
	
	
	
	
	# Analysis for low-frequency half of the spectrum
	# -----------------------------------------------
	
	# Identification of bad data, within 1-hour "bins", across 24 hours
	# -----------------------------------------------------------------
	for i in range(24):
		GHA_x   =  GHA[(GHA>=i) & (GHA<(i+1))]
		RMS_x   = RMS1[(GHA>=i) & (GHA<(i+1))]
		IN_x    =   IN[(GHA>=i) & (GHA<(i+1))]
		
		W         =  np.ones(len(GHA_x))
		bad_old   = -1
		bad       =  0		
		iteration =  0
		
		while bad > bad_old:

			iteration = iteration + 1

			print(' ')
			print('------------')
			print('GHA: ' + str(i) + '-' + str(i+1) + 'hr')
			print('Iteration: ' + str(iteration))
		
			par   = np.polyfit(GHA_x[W>0], RMS_x[W>0], Npoly-1)
			model = np.polyval(par, GHA_x)
			res   = RMS_x - model			
			std   = np.std(res[W>0])
					
			IN_x_bad = IN_x[np.abs(res) > Nsigma*std]
			W[np.abs(res) > Nsigma*std] = 0
			
			bad_old = np.copy(bad)
			bad     = len(IN_x_bad)
			
			print('STD: ' + str(np.round(std,3)) + ' K')
			print('Number of bad points excised: ' + str(bad))


		# Indices of bad data points
		# --------------------------
		if i == 0:
			IN1_bad = np.copy(IN_x_bad)

		else:
			IN1_bad = np.append(IN1_bad, IN_x_bad)
	
	




	
	
	
	# Analysis for high-frequency half of the spectrum
	# ------------------------------------------------
	
	# Identification of bad data, within 1-hour "bins", across 24 hours
	# -----------------------------------------------------------------
	for i in range(24):
		GHA_x  =  GHA[(GHA>=i) & (GHA<(i+1))]
		RMS_x  = RMS2[(GHA>=i) & (GHA<(i+1))]
		IN_x   =   IN[(GHA>=i) & (GHA<(i+1))]
		
		W         =  np.ones(len(GHA_x))
		bad_old   = -1
		bad       =  0	
		iteration =  0
		
		while bad > bad_old:

			iteration = iteration + 1

			print(' ')
			print('------------')
			print('GHA: ' + str(i) + '-' + str(i+1) + 'hr')
			print('Iteration: ' + str(iteration))
		
			par   = np.polyfit(GHA_x[W>0], RMS_x[W>0], Npoly-1)
			model = np.polyval(par, GHA_x)
			res   = RMS_x - model
			std   = np.std(res[W>0])
			
			IN_x_bad = IN_x[np.abs(res) > Nsigma*std]
			W[np.abs(res) > Nsigma*std] = 0
			
			bad_old = np.copy(bad)
			bad     = len(IN_x_bad)
			
			print('STD: ' + str(np.round(std,3)) + ' K')
			print('Number of bad points excised: ' + str(bad))


		# Indices of bad data points
		# --------------------------
		if i == 0:
			IN2_bad = np.copy(IN_x_bad)

		else:
			IN2_bad = np.append(IN2_bad, IN_x_bad)	






	# Analysis for 3-term residuals
	# ------------------------------------------------
	
	# Identification of bad data, within 1-hour "bins", across 24 hours
	# -----------------------------------------------------------------
	for i in range(24):
		GHA_x  =  GHA[(GHA>=i) & (GHA<(i+1))]
		RMS_x  = RMS3[(GHA>=i) & (GHA<(i+1))]
		IN_x   =   IN[(GHA>=i) & (GHA<(i+1))]
		
		W         =  np.ones(len(GHA_x))
		bad_old   = -1
		bad       =  0	
		iteration =  0
		
		while bad > bad_old:

			iteration = iteration + 1

			print(' ')
			print('------------')
			print('GHA: ' + str(i) + '-' + str(i+1) + 'hr')
			print('Iteration: ' + str(iteration))
			
			par   = np.polyfit(GHA_x[W>0], RMS_x[W>0], Npoly-1)
			model = np.polyval(par, GHA_x)
			res   = RMS_x - model
			std   = np.std(res[W>0])
			
			IN_x_bad = IN_x[np.abs(res) > Nsigma*std]
			W[np.abs(res) > Nsigma*std] = 0
			
			bad_old = np.copy(bad)
			bad     = len(IN_x_bad)
			
			print('STD: ' + str(np.round(std,3)) + ' K')
			print('Number of bad points excised: ' + str(bad))


		# Indices of bad data points
		# --------------------------
		if i == 0:
			IN3_bad = np.copy(IN_x_bad)

		else:
			IN3_bad = np.append(IN3_bad, IN_x_bad)	











	# All bad/good spectra indices
	# ----------------------------
	#IN_bad = np.union1d(IN1_bad, IN2_bad)
	#IN_good  = np.setdiff1d(IN, IN_bad)


		
	# Indices of good spectra
	# -----------------------
	IN1_good = np.setdiff1d(IN, IN1_bad)
	IN2_good = np.setdiff1d(IN, IN2_bad)
	IN3_good = np.setdiff1d(IN, IN3_bad)
	
	
	# Number of terms for the polynomial fit of the RMS across 24 hours
	# and number of terms for the polynomial fit of the standard deviation across 24 hours
	# ------------------------------------------------------------------------------------
	Nterms = 16
	Nstd   = 6
	
	
	# Parameters and models from the RMS and STD polynomial fits
	# ----------------------------------------------------------
	par1       = np.polyfit(GHA[IN1_good], RMS1[IN1_good], Nterms-1)
	model1     = np.polyval(par1, GHA)
	abs_res1   = np.abs(RMS1-model1)
	par1_std   = np.polyfit(GHA[IN1_good], abs_res1[IN1_good], Nstd-1)
	model1_std = np.polyval(par1_std, GHA)
	

	par2       = np.polyfit(GHA[IN2_good], RMS2[IN2_good], Nterms-1)
	model2     = np.polyval(par2, GHA)
	abs_res2   = np.abs(RMS2-model2)
	par2_std   = np.polyfit(GHA[IN2_good], abs_res2[IN2_good], Nstd-1)
	model2_std = np.polyval(par2_std, GHA)
	
	
	par3       = np.polyfit(GHA[IN3_good], RMS3[IN3_good], Nterms-1)
	model3     = np.polyval(par3, GHA)
	abs_res3   = np.abs(RMS3-model3)
	par3_std   = np.polyfit(GHA[IN3_good], abs_res3[IN3_good], Nstd-1)
	model3_std = np.polyval(par3_std, GHA)	
	
	
	
	par        = np.array([par1, par2, par3])
	par_std    = np.array([par1_std, par2_std, par3_std])
	

	# Saving polynomial parameters
	# ----------------------------
	if save_parameters == 'yes':
		np.savetxt(save_folder + 'rms_polynomial_parameters.txt', par)
		np.savetxt(save_folder + 'rms_std_polynomial_parameters.txt', par_std)
	
				
			
	return GHA, RMS1, RMS2, RMS3, IN1_good, IN2_good, IN3_good, model1, model2, model3, abs_res1, abs_res2, abs_res3, model1_std, model2_std, model3_std










def rms_filter(case, gx, rms, Nsigma):
	
	if case == 0:
		file_path = edges_folder + 'mid_band/rms_filters/case0/'	
	
	if case == 1:
		file_path = edges_folder + 'mid_band/rms_filters/case1/'
		
	if case == 2:
		file_path = edges_folder + 'mid_band/rms_filters/case2/'


	p    = np.genfromtxt(file_path + 'rms_polynomial_parameters.txt')
	ps   = np.genfromtxt(file_path + 'rms_std_polynomial_parameters.txt')	
	
	rms1 = rms[:,0]
	rms2 = rms[:,1]
	rms3 = rms[:,2]
		
	m1   = np.polyval(p[0,:], gx)
	m2   = np.polyval(p[1,:], gx)
	m3   = np.polyval(p[2,:], gx)
	
	ms1  = np.polyval(ps[0,:], gx)
	ms2  = np.polyval(ps[1,:], gx)
	ms3  = np.polyval(ps[2,:], gx)
	
	index = np.arange(0, len(rms1))
	
	diff1 = np.abs(rms1 - m1)
	diff2 = np.abs(rms2 - m2)
	diff3 = np.abs(rms3 - m3)
	
	index_good_1   = index[diff1 <= Nsigma*ms1]
	index_good_2   = index[diff2 <= Nsigma*ms2]
	index_good_3   = index[diff3 <= Nsigma*ms3]
	
	index_good_A = np.intersect1d(index_good_1, index_good_2)
	index_good   = np.intersect1d(index_good_A, index_good_3)
	
	return index_good, index_good_1, index_good_2, index_good_3









def level3_to_level4(case):
	
	
	# One-hour bins
	# -------------
	GHA_edges = np.arange(0, 25, 1)
	
	
	
	# Listing files available
	# ------------------------
	if case == 0:
		path_files             = edges_folder + 'mid_band/spectra/level3/case0/'
		save_folder            = edges_folder + 'mid_band/spectra/level4/case0/'
		output_file_name_hdf5  = 'case0.hdf5'
			
	if case == 1:
		path_files             = edges_folder + 'mid_band/spectra/level3/case1/'
		save_folder            = edges_folder + 'mid_band/spectra/level4/case1/'
		output_file_name_hdf5  = 'case1.hdf5'
		
	if case == 2:
		path_files             = edges_folder + 'mid_band/spectra/level3/case2/'
		save_folder            = edges_folder + 'mid_band/spectra/level4/case2/'
		output_file_name_hdf5  = 'case2.hdf5'
		


		
		
	new_list   = listdir(path_files)
	new_list.sort()
	
	#index_new_list = np.arange(0,5)
	#index_new_list = index_new_list.astype('int') # [0,1]  # for testing purposes
	index_new_list = range(len(new_list))

	
	# Loading and cleaning data
	# -------------------------
	flag = -1
	
	year_day_all = np.zeros((len(index_new_list), 2))
	
	
	for i in index_new_list: 
		
		year_day_all[i,0] = float(new_list[i][0:4])
		
		if len(new_list[i]) == 8:
			year_day_all[i,1] = float(new_list[i][5::])			
		elif len(new_list[i]) > 8:
			year_day_all[i,1] = float(new_list[i][5:8])
		
		
		
		
		flag = flag + 1
		
		# Loading data
		f, ty, py, ry, wy, rmsy, my = level3read(path_files + new_list[i])
		print('----------------------------------------------')
		
		
		# Filtering out high humidity
		amb_hum_max = 50
		IX          = data_selection(my, GHA_or_LST='GHA', TIME_1=0, TIME_2=24, sun_el_max=90, moon_el_max=90, amb_hum_max=amb_hum_max, min_receiver_temp=0, max_receiver_temp=100)
		
		px   = py[IX,:]
		rx   = ry[IX,:]
		wx   = wy[IX,:]
		rmsx = rmsy[IX,:]
		mx   = my[IX,:]	
		
		
		# Finding index of clean data
		gx         = mx[:,4]
		gx[gx<0]   = gx[gx<0] + 24
		
		Nsigma     = 3
		index_good, i1, i2, i3 = rms_filter(case, gx, rmsx, Nsigma)
		
				
		# Selecting good data
		p   = px[index_good,:]
		r   = rx[index_good,:]
		w   = wx[index_good,:]
		rms = rmsx[index_good,:]
		m   = mx[index_good,:]
		
		
		
		
		
		
		
		# Storing GHA and rms of good data
		GHA        = m[:,4]
		GHA[GHA<0] = GHA[GHA<0] + 24
		
		AT = np.vstack((gx, rmsx.T))
		BT = np.vstack((GHA, rms.T))
		
		A = AT.T
		B = BT.T
		
		if flag == 0:
			avp_all = np.zeros((len(new_list), len(GHA_edges)-1, len(p[0,:])))
			avr_all = np.zeros((len(new_list), len(GHA_edges)-1, len(r[0,:])))
			avw_all = np.zeros((len(new_list), len(GHA_edges)-1, len(w[0,:])))
			
			grx_all = np.copy(A)
			gr_all  = np.copy(B)
			
		if flag > 0:
			grx_all = np.vstack((grx_all, A))
			gr_all  = np.vstack((gr_all, B))
	






		# Averaging data within each GHA bin
		for j in range(len(GHA_edges)-1):			
					
			GHA_LOW  = GHA_edges[j]
			GHA_HIGH = GHA_edges[j+1]
							
			p1 = p[(GHA>=GHA_LOW) & (GHA<GHA_HIGH),:]
			r1 = r[(GHA>=GHA_LOW) & (GHA<GHA_HIGH),:]
			w1 = w[(GHA>=GHA_LOW) & (GHA<GHA_HIGH),:]
			m1 = m[(GHA>=GHA_LOW) & (GHA<GHA_HIGH),:]
			
			print(str(new_list[i]) + '. GHA: ' + str(GHA_LOW) + '-' + str(GHA_HIGH) + ' hr. Number of spectra: ' + str(len(r1)))
		
				
			if len(r1) > 0:
				
				avp        = np.mean(p1, axis=0)
				avr, avw   = ba.weighted_mean(r1, w1)
				
				
				# Final RFI cleaning
				avr[(f>130) & (f<130.4)]   = 0
				avw[(f>130) & (f<130.4)]   = 0
			
				avr[(f>136.1) & (f<138.2)] = 0
				avw[(f>136.1) & (f<138.2)] = 0
			
				avr[(f>145.5) & (f<146)]   = 0
				avw[(f>145.5) & (f<146)]   = 0
				
				avr[(f>149.8) & (f<150.3)] = 0
				avw[(f>149.8) & (f<150.3)] = 0



				# RFI cleaning of average spectra
				avr_no_rfi, avw_no_rfi = rfi.cleaning_sweep(f, avr, avw, window_width_MHz=3, Npolyterms_block=2, N_choice=20, N_sigma=3)				
				
				
				
			
				# Storing averages	
				avp_all[i,j,:] = avp
				avr_all[i,j,:] = avr_no_rfi
				avw_all[i,j,:] = avw_no_rfi
			


	print()
	print()



	
	
	# Producing plots with daily residuals
	# ------------------------------------
	Ngha = len(GHA_edges)-1
	
	# Loop over days
	for i in index_new_list:
		
		# Loop over number of foreground terms
		for Nfg in [3,4,5]:
			
			# Loop over GHA
			for j in range(Ngha):
				
				print('Nfg: ' + str(Nfg) + '. GHA: ' + str(GHA_edges[j]) + '-' + str(GHA_edges[j+1]) + ' hr')
				
				yp = avp_all[i,j,:]
				yr = avr_all[i,j,:]
				yw = avw_all[i,j,:]
							
				fb, yrb, ywb = ba.spectral_binning_number_of_samples(f, yr, yw)
				
				
				# Creating arrays with residuals to plot
				if j == 0:
					qrb_all = np.zeros((Ngha, len(fb)))
					qwb_all = np.zeros((Ngha, len(fb)))					
					
				
				if np.sum(yw>0):
					
					model        = ba.model_evaluate('LINLOG', yp, fb/200)
					ytb          = model + yrb
					
					ytb[ywb==0]  = 0
									
					par  = ba.fit_polynomial_fourier('LINLOG', fb/200, ytb, Nfg, Weights=ywb)
					qrb  = ytb - par[1]
					
					qrb_all[j,:] = qrb
					qwb_all[j,:] = ywb
			
			
					
			# Plotting residuals for each day
			
			# Settings
			# ----------------------------------
			LST_text    = ['GHA=' + str(GHA_edges[k]) + '-' + str(GHA_edges[k+1]) + ' hr' for k in range(Ngha)]
			
			if Nfg == 3:
				DY = 8
				
			elif Nfg == 4:
				DY = 4
				
			elif Nfg == 5:
				DY = 3
			
			FLOW_plot   =  30
			FHIGH_plot  = 165
			XTICKS      = np.arange(60, 161, 20)
			XTEXT       =  32
			YLABEL      = str(DY)  + ' K per division'
			TITLE       = str(Nfg) + ' LINLOG terms'
			FIGURE_FORMAT = 'png'
			
			
			# Creating folder
			figure_save_path_subfolder = save_folder + '/plots/Nfg_' + str(Nfg) + '/'
			if not exists(figure_save_path_subfolder):
				makedirs(figure_save_path_subfolder)
			
			figure_save_name = new_list[i][:-5]
			
			
			# Plotting
			x = plot_residuals(fb, qrb_all, qwb_all, LST_text, DY=DY, FLOW=FLOW_plot, FHIGH=FHIGH_plot, XTICKS=XTICKS, XTEXT=XTEXT, YLABEL=YLABEL, TITLE=TITLE, save='yes', figure_path=figure_save_path_subfolder, figure_name=figure_save_name, figure_format=FIGURE_FORMAT)
			
						
			
	

		
	
	
	## Producing total integrated average
	#for i in range(Ngha):
		
		#print('GHA bin ' + str(i+1) + ' of ' + str(len(GHA_edges)-1))
		
		
		#zp = np.mean(avp_all[:,i,:], axis=0)
		
		#zr, zw = ba.weighted_mean(avr_all[:,i,:], avw_all[:,i,:])
		
		#print(np.sum(zw))
		#avr_no_rfi, avw_no_rfi = rfi.cleaning_sweep(f, zr, zw, window_width_MHz=5, Npolyterms_block=3, N_choice=20, N_sigma=3)
		
		
		#fb, rb, wb = ba.spectral_binning_number_of_samples(f, avr_no_rfi, avw_no_rfi)
		#model      = ba.model_evaluate('LINLOG', zp, fb/200)
		#tb         = model + rb
	
		#tb[wb==0]  = 0
		
		#if i == 0:
			#tb_all = np.zeros((len(GHA_edges)-1, len(tb)))
			#wb_all = np.zeros((len(GHA_edges)-1, len(wb)))
			
			
		#tb_all[i,:] = tb
		#wb_all[i,:] = wb


	## Formatting output data
	#gha_edges_column = np.reshape(GHA_edges, -1, 1)
	
	#dataT = np.array([fb, tb_all[0,:], wb_all[0,:]])
	#for i in range(len(tb_all[:,0])-1):
		#dataT = np.vstack((dataT, tb_all[i+1,:], wb_all[i+1,:]))
		
	#data = dataT.T
					
	#np.savetxt(data_save_path + data_save_name + '_gha_edges' + '.txt', gha_edges_column,  header = 'GHA edges of integrated spectra [hr]')
	#np.savetxt(data_save_path + data_save_name + '_data' + '.txt',      data,              header = 'Frequency [MHz],\t Temperature [K],\t Weights')
	
	
	
	
	# Save
	# ----
	if not exists(save_folder):
		makedirs(save_folder)
	with h5py.File(save_folder + output_file_name_hdf5, 'w') as hf:
		hf.create_dataset('frequency',    data = f)
		hf.create_dataset('parameters',   data = avp_all)
		hf.create_dataset('residuals',    data = avr_all)
		hf.create_dataset('weights',      data = avw_all)
		hf.create_dataset('gha_edges',    data = GHA_edges)
		hf.create_dataset('year_day',     data = year_day_all)
		
		
	return f, avp_all, avr_all, avw_all, GHA_edges, year_day_all  #, f, avp_all, avr_all, avw_all








def level4read(path_file):

	with h5py.File(path_file,'r') as hf:

		hfX    = hf.get('frequency')
		f      = np.array(hfX)

		hfX    = hf.get('parameters')
		p_all  = np.array(hfX)

		hfX    = hf.get('residuals')
		r_all  = np.array(hfX)

		hfX    = hf.get('weights')
		w_all  = np.array(hfX)		

		hfX    = hf.get('gha_edges')
		gha    = np.array(hfX)		

		hfX    = hf.get('year_day')
		yd     = np.array(hfX)



	return f, p_all, r_all, w_all, gha, yd	






def one_hour_filter(year, day, gha):
	
	bad = np.array([
	        [2018, 146, 6],
	        [2018, 146, 7],
	        [2018, 146, 8],
	        [2018, 146, 9],
	        [2018, 146, 10],
	        [2018, 151, 9],
	        [2018, 157, 7],
	        [2018, 159, 4],
	        [2018, 159, 7],
	        [2018, 159, 8],
	        [2018, 159, 9],
	        [2018, 164, 19],
	        [2018, 170, 16],
	        [2018, 170, 17],
	        [2018, 170, 23],
	        [2018, 184, 11],
	        [2018, 184, 12],
	        [2018, 184, 13],
	        [2018, 184, 14],
	        [2018, 184, 15],
	        [2018, 184, 16],
	        [2018, 184, 17],
	        [2018, 185, 0],
	        [2018, 185, 1],
	        [2018, 185, 2],
	        [2018, 185, 6],
	        [2018, 185, 7],
	        [2018, 185, 8],
	        [2018, 185, 13],
	        [2018, 185, 14],
	        [2018, 185, 15],
	        [2018, 186, 9],
	        [2018, 186, 10],
	        [2018, 186, 11],
	        [2018, 186, 12],
	        [2018, 186, 13],
	        [2018, 186, 14],
	        [2018, 186, 15],
	        [2018, 186, 16],
	        [2018, 186, 17],
	        [2018, 190, 11],
	        [2018, 190, 12],
	        [2018, 190, 13],
	        [2018, 190, 14],
	        [2018, 190, 15],
	        [2018, 190, 16],
	        [2018, 190, 17],
	        [2018, 191, 8],
	        [2018, 191, 9],
	        [2018, 191, 10],
	        [2018, 191, 11],
	        [2018, 191, 12],
	        [2018, 191, 13],
	        [2018, 191, 14],
	        [2018, 191, 15],
	        [2018, 191, 16],
	        [2018, 192, 10],
	        [2018, 192, 11],
	        [2018, 192, 12],
	        [2018, 192, 13],
	        [2018, 192, 14],
	        [2018, 192, 15],
	        [2018, 192, 16],
	        [2018, 192, 17],
	        [2018, 192, 18],
	        [2018, 192, 19],
	        [2018, 195, 6],
	        [2018, 195, 7],
	        [2018, 195, 8],
	        [2018, 195, 9],
	        [2018, 195, 10],
	        [2018, 195, 11],
	        [2018, 195, 12],
	        [2018, 195, 13],
	        [2018, 196,  8],
	        [2018, 196,  9],
	        [2018, 196,  10],
	        [2018, 196,  11],
	        [2018, 196,  12],
	        [2018, 196,  13],
	        [2018, 196,  14],
	        [2018, 196,  15],
	        [2018, 196,  16],
	        [2018, 196,  17],
	        [2018, 199,  14],
	        [2018, 204,  5],
	        [2018, 204,  6],
	        [2018, 204,  7],
	        [2018, 204,  8],
	        [2018, 204,  9],
	        [2018, 204,  10],
	        [2018, 204,  11],
	        [2018, 204,  12],
	        [2018, 208,  13],
	        [2018, 208,  14],
	        [2018, 208,  15],
	        [2018, 209,  12],
	        [2018, 209,  13],
	        [2018, 209,  14],
	        [2018, 209,  15],
	        [2018, 209,  16],
	        [2018, 209,  17],
	        [2018, 209,  18],
	        [2018, 211,  20],
	        [2018, 212,  17],
	        [2018, 213,  0],
	        [2018, 213,  1],
	        [2018, 213,  2],
	        [2018, 213,  3],
	        [2018, 213,  4],
	        [2018, 213,  5],
	        [2018, 213,  6],
	        [2018, 213,  7],
	        [2018, 213,  8],
	        [2018, 213,  9],
	        [2018, 213,  10],
	        [2018, 213,  11],
	        [2018, 213,  12],
	        [2018, 213,  13],
	        [2018, 213,  14],
	        [2018, 213,  15],
	        [2018, 213,  16],
	        [2018, 213,  17],
	        [2018, 213,  18],
	        [2018, 213,  19],
	        [2018, 213,  20],
	        [2018, 213,  21],
	        [2018, 213,  22],
	        [2018, 213,  23],
	        [2018, 214,  13],
	        [2018, 214,  14],
	        [2018, 214,  15],
	        [2018, 214,  16],
	        [2018, 214,  17],
	        [2018, 214,  18],
	        [2018, 215,  0],
	        [2018, 215,  1],
	        [2018, 215,  2],
	        [2018, 215,  3],	        
	        [2018, 215,  21],
	        [2018, 215,  22],
	        [2018, 215,  23],	        	        
	        [2018, 216,  9],
	        [2018, 216,  10],
	        [2018, 216,  11],
	        [2018, 216,  12],
	        [2018, 216,  13],
	        [2018, 216,  14],
	        [2018, 216,  15],	        
	        [2018, 219,  15],
	        [2018, 219,  16],
	        [2018, 219,  17],
	        [2018, 219,  18],
	        [2018, 219,  19],	
	        [2018, 221,  0],
	        [2018, 221,  1],
	        [2018, 221,  2],
	        [2018, 221,  3],
	        [2018, 221,  4],
	        [2018, 221,  5],
	        [2018, 221,  6],
	        [2018, 221,  7],
	        [2018, 221,  8],
	        [2018, 221,  9],
	        [2018, 221,  10],
	        [2018, 221,  11],
	        [2018, 221,  12],
	        [2018, 221,  13],
	        [2018, 221,  14],
	        [2018, 221,  15],
	        [2018, 221,  16],
	        [2018, 221,  17],
	        [2018, 221,  18],
	        [2018, 221,  19],
	        [2018, 221,  20],
	        [2018, 221,  21],
	        [2018, 221,  22],
	        [2018, 221,  23],	
	        [2018, 222,  0],
	        [2018, 222,  1],
	        [2018, 222,  2],
	        [2018, 222,  3],
	        [2018, 222,  4],
	        [2018, 222,  5],
	        [2018, 222,  6],
	        [2018, 222,  7],
	        [2018, 222,  8],
	        [2018, 222,  9],
	        [2018, 222,  10],
	        [2018, 222,  11],
	        [2018, 222,  12],
	        [2018, 222,  13],
	        [2018, 222,  14],
	        [2018, 222,  15],
	        [2018, 222,  16],
	        [2018, 222,  17],
	        [2018, 222,  18],
	        [2018, 222,  19],
	        [2018, 222,  20],
	        [2018, 222,  21],
	        [2018, 222,  22],
	        [2018, 222,  23]	        
	        
	        ])
	
	
	keep = 1
	for i in range(len(bad)):
		if (year == bad[i,0]) and (day == bad[i,1]) and (gha == bad[i,2]):
			keep = 0
			
		
	return keep








def season_integrated_spectra_GHA(band, case, new_gha_edges=np.arange(0,25,2), data_save_name = 'caseX_1hr_average'):
	
	
	if case == 0:
		case_str = 'case0'
	
	if case == 1:
		case_str = 'case1'
		
	if case == 2:
		case_str = 'case2'



	data_save_path = edges_folder + band + '/spectra/level5/' + case_str + '/'
	
	
	# Loading level4 data
	f, p_all, r_all, w_all, gha_edges, yd = level4read(edges_folder + band + '/spectra/level4/' + case_str + '/' + case_str + '.hdf5')


	# Creating intermediate 1hr-average arrays
	pr_all = np.zeros((len(gha_edges)-1, len(p_all[0,0,:])))
	rr_all = np.zeros((len(gha_edges)-1, len(f)))
	wr_all = np.zeros((len(gha_edges)-1, len(f)))	


	
	# Looping over every 1hr GHA
	for j in range(len(gha_edges)-1):
		
		# Looping over day
		counter = 0
		for i in range(len(yd)):
			
			# Returns a 1 if the 1hr average tested is good quality, and a 0 if it is not
			keep = one_hour_filter(yd[i,0], yd[i,1], gha_edges[j])
			print(yd[i,1])
			print(gha_edges[j])
			
			# Index of good spectra
			if keep == 1:
				if counter == 0:
					index_good = np.array([i])
					counter    = counter+1
				
				elif counter > 0:
					index_good = np.append(index_good, i)
		
		
		# Selecting good parameters and spectra		
		pp  = p_all[index_good, j, :]
		rr  = r_all[index_good, j, :]
		ww  = w_all[index_good, j, :]
		
		
		# Average parameters and spectra
		avp      = np.mean(pp, axis=0)
		avr, avw = ba.weighted_mean(rr, ww)
		
		
		# RFI cleaning of 1-hr season average spectra
		avr_no_rfi, avw_no_rfi = rfi.cleaning_sweep(f, avr, avw, window_width_MHz=3, Npolyterms_block=2, N_choice=20, N_sigma=2.5) # 3
		
		
		# Storing season 1hr-average spectra
		pr_all[j,:] = avp			
		rr_all[j,:] = avr_no_rfi
		wr_all[j,:] = avw_no_rfi
		
		
	
	
	
	
	
	print('-------------------------------')
		
		
	
	# Averaging data within new gha edges
	for j in range(len(new_gha_edges)-1):
		new_gha_start  = new_gha_edges[j]
		new_gha_end    = new_gha_edges[j+1]
		
		print(str(new_gha_start) + ' ' + str(new_gha_end))
		
		counter = 0
		for i in range(len(gha_edges)-1):
			if (new_gha_start < new_gha_end):
				if (gha_edges[i] >= new_gha_start) and (gha_edges[i] < new_gha_end):
					
					print(gha_edges[i])
					if counter == 0:
						px_all = pr_all[i,:]
						rx_all = rr_all[i,:]
						wx_all = wr_all[i,:]
						counter = counter + 1
					
					elif counter > 0:
						px_all = np.vstack((px_all, pr_all[i,:]))
						rx_all = np.vstack((rx_all, rr_all[i,:]))
						wx_all = np.vstack((wx_all, wr_all[i,:]))
						
			elif (new_gha_start > new_gha_end):
				if (gha_edges[i] >= new_gha_start) or (gha_edges[i] < new_gha_end):
					
					print(gha_edges[i])
					if counter == 0:
						px_all = pr_all[i,:]
						rx_all = rr_all[i,:]
						wx_all = wr_all[i,:]
						counter = counter + 1
					
					elif counter > 0:
						px_all = np.vstack((px_all, pr_all[i,:]))
						rx_all = np.vstack((rx_all, rr_all[i,:]))
						wx_all = np.vstack((wx_all, wr_all[i,:]))


					
		
					
		avpx       = np.mean(px_all, axis=0)		
		avrx, avwx = ba.weighted_mean(rx_all, wx_all)
		
		avrx_no_rfi, avwx_no_rfi = rfi.cleaning_sweep(f, avrx, avwx, window_width_MHz=3, Npolyterms_block=2, N_choice=20, N_sigma=2.5)  # 3
				
				
		## Storing average spectra
		#pry_all[j,:] = avpx		
		#rry_all[j,:] = avrx_no_rfi
		#wry_all[j,:] = avwx_no_rfi
		
		
		
		
		# Frequency binning
		fb, rbx, wbx = ba.spectral_binning_number_of_samples(f, avrx_no_rfi, avwx_no_rfi)
		modelx       = ba.model_evaluate('LINLOG', avpx, fb/200)
		tbx          = modelx + rbx
		tbx[wbx==0]  = 0
		
		
		# Storing binned average spectra
		if j == 0:
			tbx_all = np.zeros((len(new_gha_edges)-1, len(fb)))
			wbx_all = np.zeros((len(new_gha_edges)-1, len(fb)))
						
		tbx_all[j,:] = tbx
		wbx_all[j,:] = wbx			
				





		## Formatting output data
		#new_gha_edges_column = np.reshape(new_gha_edges, -1, 1)
		
		#dataT = np.array([fb, tbx_all[0,:], wbx_all[0,:]])
		#for i in range(len(tbx_all[:,0])-1):
			#dataT = np.vstack((dataT, tbx_all[i+1,:], wbx_all[i+1,:]))
			
		#data = dataT.T
		
		
		# Saving data				
		np.savetxt(data_save_path + data_save_name + '_gha_edges.txt',    new_gha_edges,  header = 'GHA edges of integrated spectra [hr].')
		np.savetxt(data_save_path + data_save_name + '_frequency.txt',    fb,             header = 'Frequency [MHz].')
		np.savetxt(data_save_path + data_save_name + '_temperature.txt',  tbx_all,        header = 'Rows correspond to different GHAs. Columns correspond to frequency.')
		np.savetxt(data_save_path + data_save_name + '_weights.txt',      wbx_all,        header = 'Rows correspond to different GHAs. Columns correspond to frequency.')
		



	return fb, tbx_all, wbx_all



















def integrated_residuals_GHA(file_data, flow, fhigh, Nfg):
	
	d = np.genfromtxt(file_data)
	
	fx = d[:,0]
	
	for i in range(int((len(d[0,:])-1)/2)):
		index_t = 2*(i+1)-1
		index_w = 2*(i+1)
		tx = d[:,index_t]
		wx = d[:,index_w]
		
		f = fx[(fx>=flow) & (fx<=fhigh)]
		t = tx[(fx>=flow) & (fx<=fhigh)]
		w = wx[(fx>=flow) & (fx<=fhigh)]
		
		#print(index_t)
		#print(index_w)
		
		par = ba.fit_polynomial_fourier('LINLOG', f/200, t, Nfg, Weights=w)
		
		r = t - par[1]
		
		if i == 0:
			r_all = np.copy(r)
			w_all = np.copy(w)
			
		elif i > 0:
			r_all = np.vstack((r_all, r))
			w_all = np.vstack((w_all, w))
		
		
		
		
	return f, r_all, w_all















def spectra_to_residuals(fx, tx_2D, wx_2D, flow, fhigh, Nfg):
	
	f    = fx[(fx>=flow) & (fx<=fhigh)]
	t_2D = tx_2D[:, (fx>=flow) & (fx<=fhigh)]
	w_2D = wx_2D[:, (fx>=flow) & (fx<=fhigh)]	

	
	for i in range(len(t_2D[:, 0])):
		t = t_2D[i, :]
		w = w_2D[i, :]
		
		par = ba.fit_polynomial_fourier('LINLOG', f/200, t, Nfg, Weights=w)
		
		r = t - par[1]
		
		if i == 0:
			r_all = np.copy(r)
			w_all = np.copy(w)
			
		elif i > 0:
			r_all = np.vstack((r_all, r))
			w_all = np.vstack((w_all, w))
		
		
	return f, r_all, w_all


























def daily_residuals_LST(file_name, LST_boundaries=np.arange(0,25,2), FLOW=60, FHIGH=150, Nfg=5, SUN_EL_max=90, MOON_EL_max=90):
	
	flag_folder = 'nominal_60_160MHz_fullcal'	
	
	
	# Listing files to be processed
	path_files = home_folder + '/EDGES/spectra/level3/mid_band/' + flag_folder + '/'
	
	f, t, p, r, w, rms, m = level3read(path_files + file_name)
	
	
	
	flag = 0
	for i in range(len(LST_boundaries)-1):
		
		IX = data_selection(m, LST_1=LST_boundaries[i], LST_2=LST_boundaries[i+1], sun_el_max=SUN_EL_max, moon_el_max=MOON_EL_max, amb_hum_max = 200, min_receiver_temp=0, max_receiver_temp=100)
		
		
		if len(IX)>0:
			RX = r[IX,:]
			WX = w[IX,:]
			PX = p[IX,:]
			
			avr, avw = ba.spectral_averaging(RX, WX)
			fb, rb, wb = ba.spectral_binning_number_of_samples(f, avr, avw, nsamples=64)
			
			avp = np.mean(PX, axis=0)
				
			mb = ba.model_evaluate('LINLOG', avp, fb/200)
			tb = mb + rb
			
			
			fb_x   = fb[(fb >=FLOW) & (fb<=FHIGH)]	
			tb_x   = tb[(fb >=FLOW) & (fb<=FHIGH)]
			wb_x   = wb[(fb >=FLOW) & (fb<=FHIGH)]			
			
			
			
			
			par_x = ba.fit_polynomial_fourier('LINLOG', fb_x/200, tb_x, Nfg, Weights=wb_x)
			rb_x  = tb_x - par_x[1]
			
					
			if flag == 0:
				
				rb_x_all = np.zeros((len(LST_boundaries)-1, len(fb_x)))
				wb_x_all = np.zeros((len(LST_boundaries)-1, len(fb_x)))
			
					
			rb_x_all[i,:] = rb_x
			wb_x_all[i,:] = wb_x
			
			flag = flag + 1
			
	if flag == 0:
		fb, rb, wb = ba.spectral_binning_number_of_samples(f, r[0,:], w[0,:], nsamples=64)
		fb_x       = fb[(fb >=FLOW) & (fb<=FHIGH)]
		rb_x_all   = np.zeros((len(LST_boundaries)-1, len(fb_x)))
		wb_x_all   = np.zeros((len(LST_boundaries)-1, len(fb_x)))
			
	
	
	return fb_x, rb_x_all, wb_x_all









def plot_residuals(f, r, w, list_names, FIG_SX=7, FIG_SY=12, DY=2, FLOW=50, FHIGH=180, XTICKS=np.arange(60, 180+1, 20), XTEXT=160, YLABEL='ylabel', TITLE='hello', save='no', figure_path='/home/raul/Desktop/', figure_name='2018_150_00', figure_format='png'):
	
	# Nspectra_column=35,
	#Ncol_real = len(r[:,0])/Nspectra_columns
	#Ncol_int  = int(np.ceil(Ncol_real))

	
	N_spec = len(r[:,0])
		
	plt.close()
	plt.close()	
	plt.figure(figsize=(FIG_SX, FIG_SY))
	
	for i in range(len(list_names)):
		print(i)
	
		if i % 2 == 0:
			color = 'r'
		else:
			color = 'b'
			
		plt.plot(f[w[i]>0], (r[i]-i*DY)[w[i]>0], color)
		plt.text(XTEXT, -i*DY, list_names[i])
		
	plt.xlim([FLOW, FHIGH])
	plt.ylim([-DY*(N_spec), DY])
	
	plt.grid()
	
	plt.xticks(XTICKS)
	plt.yticks([])

	plt.xlabel('frequency [MHz]')
	plt.ylabel(YLABEL)

	plt.title(TITLE)	
	
	
		
	#else:
		#plt.subplot(1, Ncol_int, 2)
		#if i % 2 == 0:
			#color = 'r'
		#else:
			#color = 'b'
			
		#plt.plot(f, r[i]+(-i+Nspectra_column)*DY, color)
		
		#plt.xlim([FLOW, FHIGH])
		#plt.xticks(np.arange(FLOW, FHIGH+1, 10))
		#plt.ylim([-(Nspectra_column+1)*DY, DY])
		#plt.text(XTEXT, (-i+Nspectra_column)*DY, list_file_names[i])	
		#plt.yticks([])
		
		#plt.xlabel('frequency [MHz]')

	
	if save == 'yes':		
		plt.savefig(figure_path + figure_name + '.' + figure_format, bbox_inches='tight')
		plt.close()
		plt.close()
			
	return 0



































def plots_level3_rms_folder(band, case, YTOP_LOW=10, YTOP_HIGH=50, YBOTTOM_LOW=10, YBOTTOM_HIGH=30):
	
	"""	
	This function plots the RMS of residuals of Level3 data
	
	YTOP_LOW/_HIGH:      y-limits of top panel
	YBOTTOM_LOW/_HIGH:   y-limits of bottom panel
	"""
		
	# Case selection
	if case == 1:
		flag_folder = 'nominal_60_160MHz'	
		
	# Listing files to be processed
	path_files = home_folder + '/EDGES/spectra/level3/' + band + '/' + flag_folder + '/'
	list_files = listdir(path_files)
	list_files.sort()
	
	# Folder to save plots
	path_plots = path_files + 'plots_residuals_rms/'
	if not exists(path_plots):
		makedirs(path_plots)
		
	
	# Loading data and plotting RMS
	lf = len(list_files)
	for i in range(lf):
		print(i+1)
		f, t, p, r, w, rms, m = level3read(path_files + list_files[i], print_key='no')
				
		plt.figure()
		
		plt.subplot(2,1,1)
		plt.plot(m[:,3], rms[:,0], '.')
		plt.xticks(np.arange(0, 25, 2))
		plt.grid()
		plt.xlim([0, 24])
		plt.ylim([YTOP_LOW, YTOP_HIGH])
		
		plt.ylabel('RMS [K]')
		plt.title(list_files[i] + ':  Low-frequency Half')
	
		plt.subplot(2,1,2)
		plt.plot(m[:,3], rms[:,1], '.r')
		plt.xticks(np.arange(0, 25, 2))
		plt.grid()
		plt.xlim([0, 24])
		plt.ylim([YBOTTOM_LOW, YBOTTOM_HIGH])
		
		plt.ylabel('RMS [K]')
		plt.xlabel('LST [Hr]')
		plt.title(list_files[i] + ':  High-frequency Half')

		if len(list_files[0]) > 12:
			file_name = list_files[i][0:11]
			
		elif len(list_files[0]) == 12:
			file_name = list_files[i][0:8]
		
		plt.savefig(path_plots + file_name + '.png', bbox_inches='tight')
		plt.close()
		plt.close()


	return 0









#def rms_filter(rms, gha, Nterms=9, Nstd=3.5):
	
	#"""
	
	#rms: N x 2. First (second) column is the rms of the first (second) part of the spectral residuals.
	#gha: N.     1D array with GHA of spectra.
	#Nterms: number. Number of polynomial terms to model the rms and the rms std as a function of positive/negative GHA.
	#Nstd: number. Number of sigmas beyond which spectra are flagged.
	
	#"""
	
	
		
	## Index of spectra
	## ----------------
	#I  = np.arange(len(gha))
	
	
	
	## Separate RMS 0 (first half of spectrum) and 1 (second half of spectrum) 
	## based on positive (p) and negative (n) GHA 
	## -----------------------------------------------------------------------
	
	#Ip = I[gha>=0]
	#In = I[gha<0]
	
	#Gp = gha[gha>=0]
	#Gn = gha[gha<0]
		
	
	#Rp0 = rms[gha>=0,0]			
	#Rn0 = rms[gha<0,0]
	
	#Wp0 = np.ones(len(Gp))
	#Wn0 = np.ones(len(Gn))
	
		
	#Rp1 = rms[gha>=0,1]			
	#Rn1 = rms[gha<0,1]
	
	#Wp1 = np.ones(len(Gp))
	#Wn1 = np.ones(len(Gn))
	
	
	
	## Identify bad spectra based on RMS
	## ---------------------------------
	#RXp0, WXp0 = rfi.cleaning_polynomial(Gp, Rp0, Wp0, Nterms_fg=Nterms, Nterms_std=Nterms, Nstd=Nstd)
	#RXn0, WXn0 = rfi.cleaning_polynomial(-Gn, Rn0, Wn0, Nterms_fg=Nterms, Nterms_std=Nterms, Nstd=Nstd)
	
	#RXp1, WXp1 = rfi.cleaning_polynomial(Gp, Rp1, Wp1, Nterms_fg=Nterms, Nterms_std=Nterms, Nstd=Nstd)
	#RXn1, WXn1 = rfi.cleaning_polynomial(-Gn, Rn1, Wn1, Nterms_fg=Nterms, Nterms_std=Nterms, Nstd=Nstd)	
	
	

	## Assign zero weight to index of bad spectra
	## ------------------------------------------
	#Wrms0 = np.ones(len(t[:,0]))
	#Wrms0[Ip[WXp0==0]] = 0
	#Wrms0[In[WXn0==0]] = 0	
	
	#Wrms1 = np.ones(len(t[:,0]))
	#Wrms1[Ip[WXp1==0]] = 0
	#Wrms1[In[WXn1==0]] = 0
	
	
	
	#return Wrms0, Wrms1











def average_level3_mid_band(case, LST_1=0, LST_2=24, sun_el_max=90, moon_el_max=90):
	
	# Case selection
	if case == 1:
		flag_folder = 'nominal_60_160MHz'	
	
	if case == 3:
		flag_folder = 'nominal_60_160MHz_fullcal'
	
	
	
	
	# Listing files to be processed
	path_files = home_folder + '/EDGES/spectra/level3/mid_band/' + flag_folder + '/'
	list_files = listdir(path_files)
	list_files.sort()
	
	
	# 
	lf = len(list_files)

	flag = 0
	
	for i in range(10): #(lf):
		print(str(i+1) + ' of ' + str(lf))
		f, t, p, r, w, rms, m = level3read(path_files + list_files[i], print_key='no')
		

		if i == 0:
			RX_all = np.zeros((0, len(f)))
			WX_all = np.zeros((0, len(f)))
			PX_all = np.zeros((0, len(p[0,:])))

		#Wrms0, Wrms1 = rms_filter(rms, m[:,4], Nterms=9, Nstd=3.5)
		IX = data_selection(m, LST_1=LST_1, LST_2=LST_2, sun_el_max=sun_el_max, moon_el_max=moon_el_max, amb_hum_max = 200, min_receiver_temp=0, max_receiver_temp=100)
		
		if len(IX)>0:
			RX = r[IX,:]
			WX = w[IX,:]
			PX = p[IX,:]
	
			avr, avw = ba.spectral_averaging(RX, WX)
			
			fb, rb, wb = ba.spectral_binning_number_of_samples(f, avr, avw, nsamples=64)
			
			RX_all = np.vstack((RX_all, RX))
			WX_all = np.vstack((WX_all, WX))
			PX_all = np.vstack((PX_all, PX))
			
			if flag == 0:
				rb_all = np.zeros((lf, len(fb)))
				wb_all = np.zeros((lf, len(fb)))
				
				flag = 1
			
			rb_all[i,:] = rb
			wb_all[i,:] = wb		
			
		
		

		
		
	
	
	return fb, rb_all, wb_all, list_files, f, RX_all, WX_all, PX_all










## Load data
## ---------
#path_data = home_folder + '/EDGES/spectra/level3/' + band + '/' + flag_folder + '/'
#filename  = path_data + year_day_hdf5
#f, t, p, r, w, rms, m = level3read(filename)



































