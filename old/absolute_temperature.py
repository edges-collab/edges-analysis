

import numpy as np
import matplotlib.pyplot as plt
import edges as eg
import datetime as dt

import scipy.interpolate as spi

import astropy.units as apu
import astropy.time as apt
import astropy.coordinates as apc

from os.path import expanduser
from astropy.io import fits


# Determining home folder
home_folder = expanduser("~")









def data_analysis_low_band_60MHz(case='low1'):

	"""
	Use this file instead of the version without FAST

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




	# Settings
	# ----------------------------------------
	
	
	# Low-Band 1 -----------------------------
	
	if case == 'low1':
		band              = 'low_band_2015'
		date_list_case    = 1
		ant_s11           = 243
		
		recv_temp = 25
		
		
		
	
	# Low-Band 2 -----------------------------
	
	# NS
	if case == 'low2_NS':
		band              = 'low_band2_2017'
		date_list_case    = 1
		ant_s11           = 87		
		recv_temp = 'recv_temp'
		
		
			
	# EW with Shield
	if case == 'low2_EW_shield':
		band              = 'low_band2_2017'
		date_list_case    = 2
		ant_s11           = 153.2
		recv_temp = 'recv_temp'

	# EW no Shield
	if case == 'low2_EW_no_shield':
		band              = 'low_band2_2017'
		date_list_case    = 3
		ant_s11           = 180
		recv_temp = 'recv_temp'
		
	
	
	
	
	
	antenna           = 'blade'
	
	
	LST_1 = 0
	LST_2 = 24
	
	
	sun_el_max  = 90
	moon_el_max = 90
	amb_hum_max = 90
	
	min_receiver_temp = -50
	max_receiver_temp =  5000
	

		
	ant_s11_Nfit      = 10
	
	fgl   = 1
	glt   = 'value'
	glp   = 0.5
	fal   = 1
	fbcl  = 1
	
	low_band_cal_file    = 1
	

	


	# List of files to process
	# ------------------------
	datelist_raw   = eg.data_analysis_date_list(band, antenna, case=date_list_case) # list of daily measurements
	datelist       = eg.data_analysis_daily_spectra_filter(band, datelist_raw)      # selection of daily measurements



	# Output arrays
	# -----------------------
	time_bin     = 2/6   # 20 minutes
	time_centers = np.arange((time_bin/2), 24, time_bin)
	
	v0       = 60 # frequency
	dv       = 1        # channel width    
	dv_noise = 0.055    # 9 samples is channel width of 55 kHz	
	
	
	TA_all  = np.zeros((len(datelist), len(time_centers)))
	WA_all  = np.zeros((len(datelist), len(time_centers)))
	
	ambT_all  = np.zeros((len(datelist), len(time_centers)))
	ambH_all  = np.zeros((len(datelist), len(time_centers)))
	recT2_all = np.zeros((len(datelist), len(time_centers)))
	recT1_all = np.zeros((len(datelist), len(time_centers)))

	sunel_all  = np.zeros((len(datelist), len(time_centers)))
	moonel_all = np.zeros((len(datelist), len(time_centers)))
	
	
	

	# Process files
	# ----------------------------
	flag_j = 0

	for j in range(len(datelist)):   # range(5):  # 
		
		

		fin  = 0
		tc   = 0
		w_2D = 0
		m_2D = 0



		

		fin, t_2D, w_2D, m_2D = eg.data_selection_single_day_v3(band, datelist[j], LST_1, LST_2, sun_el_max=sun_el_max, moon_el_max=moon_el_max, amb_hum_max=amb_hum_max, min_receiver_temp=min_receiver_temp, max_receiver_temp=max_receiver_temp)
	
	


		# Continue if there are data available
		# ------------------------------------
		if np.sum(t_2D) > 0:
	
	
			flag_j = flag_j + 1
			
			
					
			# Continue if there are data available
			# ------------------------------------
			if np.sum(t_2D) > 0:
		
		
				print(' ')
				print(' ')
				print('File ' + str(int(flag_j)) + ' of ' + str(int(len(datelist))))
				print(datelist[j])
		
		
				
				# Cut the data at 100 MHz in cases where the calibration goes only to 100 MHz
				# ---------------------------------------------------------------------------
				if (band == 'low_band_2015') and (low_band_cal_file <= 2):
					if np.max(fin) > 100:
						fin  = fin[fin<=100]
						t_2D = t_2D[:, fin<=100]
						w_2D = w_2D[:, fin<=100]				
							
				
				
		
				# Antenna S11
				# -----------
				#s11_ant = models_antenna_s11(band, 'blade', fin, antenna_s11_day=ant_s11, model_type='polynomial')
				s11_ant = eg.models_antenna_s11_remove_delay(band, 'blade', fin, antenna_s11_day=ant_s11, model_type='polynomial', Nfit=ant_s11_Nfit)
		
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
				if recv_temp == 25:
					s11_LNA, sca, off, TU, TC, TS = eg.receiver_calibration(band, fin, receiver_temperature=recv_temp, low_band_cal_file=low_band_cal_file)
					
				elif recv_temp == 'recv_temp':
					ld = len(m_2D[:,0])
					s11_LNA = np.zeros((ld, len(fin)))
					sca = np.zeros((ld, len(fin)))
					off = np.zeros((ld, len(fin)))
					TU  = np.zeros((ld, len(fin)))
					TC  = np.zeros((ld, len(fin)))
					TS  = np.zeros((ld, len(fin)))
					
					for i in range(ld):
						#print(i)
						x0, x1, x2, x3, x4, x5 = eg.receiver_calibration_fast(band, fin, receiver_temperature=m_2D[i,-2])
						s11_LNA[i,:] = x0
						sca[i,:]     = x1
						off[i,:]     = x2
						TU[i,:]      = x3
						TC[i,:]      = x4
						TS[i,:]      = x5
						
				#s11_LNA, sca, off, TU, TC, TS
				#s11_LNA, sca, off, TU, TC, TS = eg.receiver_calibration(band, fin, receiver_temperature=recv_temp, low_band_cal_file=low_band_cal_file)
				
				
				
				
		
				# Calibrated antenna temperature with losses and beam chromaticity
				# ----------------------------------------------------------------
				TA_with_loss_and_beam = eg.calibrated_antenna_temperature(t_2D, s11_ant, s11_LNA, sca, off, TU, TC, TS)
		
				# Removing loss
				# -------------
				print('Loss correction')
		
				# Combined gain (loss), computed only once, at the beginning, same for all days
				# -----------------------------------------------------------------------------
				cg = eg.combined_gain(band, fin, antenna_s11_day=ant_s11, antenna_s11_Nfit=ant_s11_Nfit, flag_ground_loss=fgl, ground_loss_type=glt, ground_loss_percent=glp, flag_antenna_loss=fal, flag_balun_connector_loss=fbcl)		
		
				Tambient = 273.15 + 25 #m_2D[i,9]		
				TA       = (TA_with_loss_and_beam - Tambient*(1-cg))/cg
				
				
				
			
				# Binning data contained in the only pixel at ref frequency
				# ---------------------------------------------------------
				TC   = TA[:, (fin >= (v0-(dv/2))) & (fin <= (v0+(dv/2)))]
				WC   = w_2D[:, (fin >= (v0-(dv/2))) & (fin <= (v0+(dv/2)))]	
				
				sum_TC = np.sum(WC * TC, axis=1)
				sum_WC = np.sum(WC, axis=1)
				
				LST    = m_2D[:,3]
				index  = np.floor(LST/time_bin)
				
				# print(index)
				
				
				TC_sum_all = np.zeros(int(24/time_bin))
				WC_sum_all = np.zeros(int(24/time_bin))
				
				
				
				ambT_sum_all  = np.zeros(int(24/time_bin))
				wambT_sum_all = np.zeros(int(24/time_bin))
				
				ambH_sum_all  = np.zeros(int(24/time_bin))
				wambH_sum_all = np.zeros(int(24/time_bin))

				recT2_sum_all  = np.zeros(int(24/time_bin))
				wrecT2_sum_all = np.zeros(int(24/time_bin))
				
				recT1_sum_all  = np.zeros(int(24/time_bin))
				wrecT1_sum_all = np.zeros(int(24/time_bin))
				
				
				
				sunel_sum_all  = np.zeros(int(24/time_bin))
				wsunel_sum_all = np.zeros(int(24/time_bin))
				
				
				moonel_sum_all  = np.zeros(int(24/time_bin))
				wmoonel_sum_all = np.zeros(int(24/time_bin))
				
				
				for i in range(len(LST)):
					
					TC_sum_all[int(index[i])]     =  TC_sum_all[int(index[i])]  +  sum_TC[i]
					WC_sum_all[int(index[i])]     =  WC_sum_all[int(index[i])]  +  sum_WC[i]
					
					ambT_sum_all[int(index[i])]   = ambT_sum_all[int(index[i])] + m_2D[i,-4]
					wambT_sum_all[int(index[i])]  = wambT_sum_all[int(index[i])] + 1
					
					ambH_sum_all[int(index[i])]   = ambH_sum_all[int(index[i])] + m_2D[i,-3]
					wambH_sum_all[int(index[i])]  = wambH_sum_all[int(index[i])] + 1					
					
					recT2_sum_all[int(index[i])]  = recT2_sum_all[int(index[i])] + m_2D[i,-2]
					wrecT2_sum_all[int(index[i])] = wrecT2_sum_all[int(index[i])] + 1						
					
					recT1_sum_all[int(index[i])]  = recT1_sum_all[int(index[i])] + m_2D[i,-1]
					wrecT1_sum_all[int(index[i])] = wrecT1_sum_all[int(index[i])] + 1
					
					sunel_sum_all[int(index[i])]  = sunel_sum_all[int(index[i])] + m_2D[i,6]
					wsunel_sum_all[int(index[i])] = wsunel_sum_all[int(index[i])] + 1
					
					moonel_sum_all[int(index[i])]  = moonel_sum_all[int(index[i])] + m_2D[i,8]
					wmoonel_sum_all[int(index[i])] = wmoonel_sum_all[int(index[i])] + 1					
					
					
					
				TA_all[j,:]    = TC_sum_all / WC_sum_all
				WA_all[j,:]    = WC_sum_all
					
				ambT_all[j,:]  = ambT_sum_all / wambT_sum_all
				ambH_all[j,:]  = ambH_sum_all / wambH_sum_all
				recT2_all[j,:] = recT2_sum_all / wrecT2_sum_all
				recT1_all[j,:] = recT1_sum_all / wrecT1_sum_all
				
				sunel_all[j,:]  = sunel_sum_all / wsunel_sum_all
				moonel_all[j,:] = moonel_sum_all / wmoonel_sum_all





	return TA_all, WA_all, m_2D, ambT_all, ambH_all, recT2_all, recT1_all, sunel_all, moonel_all
















def data_analysis_low_band2_NS_60MHz():

	"""
	Use this file instead of the version without FAST

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




	# Settings
	# ----------------------
	band              = 'low_band_2015'
	antenna           = 'blade'
	date_list_case    = 1
	
	LST_1 = 0
	LST_2 = 24
	
	
	sun_el_max  = 90
	moon_el_max = 90
	amb_hum_max = 90
	
	min_receiver_temp = -50
	max_receiver_temp =  5000
	
	ant_s11           = 243
	ant_s11_Nfit      = 10
	
	fgl   = 1
	glt   = 'value'
	glp   = 0.5
	fal   = 1
	fbcl  = 1
	
	receiver_temperature = 25
	low_band_cal_file    = 1
	

	


	# List of files to process
	# ------------------------
	datelist_raw   = eg.data_analysis_date_list(band, antenna, case=date_list_case) # list of daily measurements
	datelist       = eg.data_analysis_daily_spectra_filter(band, datelist_raw)      # selection of daily measurements



	# Output arrays
	# -----------------------
	time_bin     = 2/6   # 20 minutes
	time_centers = np.arange((time_bin/2), 24, time_bin)
	
	v0       = 60 # frequency
	dv       = 1        # channel width    
	dv_noise = 0.055    # 9 samples is channel width of 55 kHz	
	
	
	TA_all  = np.zeros((len(datelist), len(time_centers)))
	WA_all  = np.zeros((len(datelist), len(time_centers)))
	
	ambT_all  = np.zeros((len(datelist), len(time_centers)))
	ambH_all  = np.zeros((len(datelist), len(time_centers)))
	recT2_all = np.zeros((len(datelist), len(time_centers)))
	recT1_all = np.zeros((len(datelist), len(time_centers)))

	sunel_all  = np.zeros((len(datelist), len(time_centers)))
	moonel_all = np.zeros((len(datelist), len(time_centers)))
	
	
	

	# Process files
	# ----------------------------
	flag_j = 0

	for j in range(3):  #(len(datelist)):

		fin  = 0
		tc   = 0
		w_2D = 0
		m_2D = 0


		fin, t_2D, w_2D, m_2D = eg.data_selection_single_day_v3(band, datelist[j], LST_1, LST_2, sun_el_max=sun_el_max, moon_el_max=moon_el_max, amb_hum_max=amb_hum_max, min_receiver_temp=min_receiver_temp, max_receiver_temp=max_receiver_temp)
	
	


		# Continue if there are data available
		# ------------------------------------
		if np.sum(t_2D) > 0:
	
	
			flag_j = flag_j + 1
			
			
					
			# Continue if there are data available
			# ------------------------------------
			if np.sum(t_2D) > 0:
		
				
				# Cut the data at 100 MHz in cases where the calibration goes only to 100 MHz
				# ---------------------------------------------------------------------------
				if (band == 'low_band_2015') and (low_band_cal_file <= 2):
					if np.max(fin) > 100:
						fin  = fin[fin<=100]
						t_2D = t_2D[:, fin<=100]
						w_2D = w_2D[:, fin<=100]				
							
				
				
		
				# Antenna S11
				# -----------
				#s11_ant = models_antenna_s11(band, 'blade', fin, antenna_s11_day=ant_s11, model_type='polynomial')
				s11_ant = eg.models_antenna_s11_remove_delay(band, 'blade', fin, antenna_s11_day=ant_s11, model_type='polynomial', Nfit=ant_s11_Nfit)
		
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
				s11_LNA, sca, off, TU, TC, TS = eg.receiver_calibration(band, fin, receiver_temperature=receiver_temperature, low_band_cal_file=low_band_cal_file)		
		
				# Calibrated antenna temperature with losses and beam chromaticity
				# ----------------------------------------------------------------
				TA_with_loss_and_beam = eg.calibrated_antenna_temperature(t_2D, s11_ant, s11_LNA, sca, off, TU, TC, TS)
		
				# Removing loss
				# -------------
				print('Loss correction')
		
				# Combined gain (loss), computed only once, at the beginning, same for all days
				# -----------------------------------------------------------------------------
				cg = eg.combined_gain(band, fin, antenna_s11_day=ant_s11, antenna_s11_Nfit=ant_s11_Nfit, flag_ground_loss=fgl, ground_loss_type=glt, ground_loss_percent=glp, flag_antenna_loss=fal, flag_balun_connector_loss=fbcl)		
		
				Tambient = 273.15 + 25 #m_2D[i,9]		
				TA       = (TA_with_loss_and_beam - Tambient*(1-cg))/cg
				
				
				
			
				# Binning data contained in the only pixel at ref frequency
				# ---------------------------------------------------------
				TC   = TA[:, (fin >= (v0-(dv/2))) & (fin <= (v0+(dv/2)))]
				WC   = w_2D[:, (fin >= (v0-(dv/2))) & (fin <= (v0+(dv/2)))]	
				
				sum_TC = np.sum(WC * TC, axis=1)
				sum_WC = np.sum(WC, axis=1)
				
				LST    = m_2D[:,3]
				index  = np.floor(LST/time_bin)
				
				# print(index)
				
				
				TC_sum_all = np.zeros(int(24/time_bin))
				WC_sum_all = np.zeros(int(24/time_bin))
				
				
				
				ambT_sum_all  = np.zeros(int(24/time_bin))
				wambT_sum_all = np.zeros(int(24/time_bin))
				
				ambH_sum_all  = np.zeros(int(24/time_bin))
				wambH_sum_all = np.zeros(int(24/time_bin))

				recT2_sum_all  = np.zeros(int(24/time_bin))
				wrecT2_sum_all = np.zeros(int(24/time_bin))
				
				recT1_sum_all  = np.zeros(int(24/time_bin))
				wrecT1_sum_all = np.zeros(int(24/time_bin))
				
				
				
				sunel_sum_all  = np.zeros(int(24/time_bin))
				wsunel_sum_all = np.zeros(int(24/time_bin))
				
				
				moonel_sum_all  = np.zeros(int(24/time_bin))
				wmoonel_sum_all = np.zeros(int(24/time_bin))
				
				
				for i in range(len(LST)):
					
					TC_sum_all[int(index[i])]     =  TC_sum_all[int(index[i])]  +  sum_TC[i]
					WC_sum_all[int(index[i])]     =  WC_sum_all[int(index[i])]  +  sum_WC[i]
					
					ambT_sum_all[int(index[i])]   = ambT_sum_all[int(index[i])] + m_2D[i,-4]
					wambT_sum_all[int(index[i])]  = wambT_sum_all[int(index[i])] + 1
					
					ambH_sum_all[int(index[i])]   = ambH_sum_all[int(index[i])] + m_2D[i,-3]
					wambH_sum_all[int(index[i])]  = wambH_sum_all[int(index[i])] + 1					
					
					recT2_sum_all[int(index[i])]  = recT2_sum_all[int(index[i])] + m_2D[i,-2]
					wrecT2_sum_all[int(index[i])] = wrecT2_sum_all[int(index[i])] + 1						
					
					recT1_sum_all[int(index[i])]  = recT1_sum_all[int(index[i])] + m_2D[i,-1]
					wrecT1_sum_all[int(index[i])] = wrecT1_sum_all[int(index[i])] + 1
					
					sunel_sum_all[int(index[i])]  = sunel_sum_all[int(index[i])] + m_2D[i,6]
					wsunel_sum_all[int(index[i])] = wsunel_sum_all[int(index[i])] + 1
					
					moonel_sum_all[int(index[i])]  = moonel_sum_all[int(index[i])] + m_2D[i,8]
					wmoonel_sum_all[int(index[i])] = wmoonel_sum_all[int(index[i])] + 1					
					
					
					
				TA_all[j,:]    = TC_sum_all / WC_sum_all
				WA_all[j,:]    = WC_sum_all
					
				ambT_all[j,:]  = ambT_sum_all / wambT_sum_all
				ambH_all[j,:]  = ambH_sum_all / wambH_sum_all
				recT2_all[j,:] = recT2_sum_all / wrecT2_sum_all
				recT1_all[j,:] = recT1_sum_all / wrecT1_sum_all
				
				sunel_all[j,:]  = sunel_sum_all / wsunel_sum_all
				moonel_all[j,:] = moonel_sum_all / wmoonel_sum_all





	return TA_all, WA_all, m_2D, ambT_all, ambH_all, recT2_all, recT1_all, sunel_all, moonel_all
















def data_analysis_high_band_140MHz(save, MC='no', name_prefix=''):

	"""
	save: 'yes', 'no'

	"""


	# Settings
	# --------------------
	band              = 'high_band_2015'
	antenna           = 'blade'
	date_list_case    = 0


	LST_1             =   0
	LST_2             =  24

	if MC == 'yes':
		sun_el_max        = 90  # All LSTs allowed for the one day used for MC propagation (day 70)
	elif MC == 'no':
		sun_el_max        = -10

	moon_el_max       = 90
	amb_hum_max       = 90
	min_receiver_temp = 23.4
	max_receiver_temp = 27.4

	cwterms           = 7 

	ant_s11           = 262
	ant_s11_Nfit      =   9

	fgl  = 1                  # 1
	glt  = 'value' 
	glp  = 0.5                # 0.5
	fal  = 1                  # 1   
	fbcl = 1                  # 1	


	receiver_temperature = 'actual'
	MC_recv_temp         = 'no'



	if MC == 'no':
		MC_ant_s11_abs = 'no'
		MC_ant_s11_ang = 'no'

		MC_ground_loss  = 'no'
		MC_antenna_loss = 'no'
		MC_balun_connector_loss = [0,0,0,0,0,0,0,0]

		MC_sigma_ambient_temperature = 0  # K, Gaussian standard deviation

		MC_recv_temp = 0
		MC_recv_calibration = 'no'

		MC_recv_s11 = 'no'



	elif MC == 'yes':
		MC_ant_s11_abs = 'yes'
		MC_ant_s11_ang = 'yes'

		MC_ground_loss  = 'yes'
		MC_antenna_loss = 'yes'
		MC_balun_connector_loss = [1,1,1,1,1,1,1,1]

		MC_sigma_ambient_temperature = 1*np.random.normal()  # 1 K Gaussian standard deviation

		MC_recv_temp = 0.1*np.random.normal()  # 0.1 K Gaussian standard deviation
		MC_recv_calibration = 'yes' # 'yes'

		MC_recv_s11 = 'yes'        # 'yes'  




	# List of files to process
	# ------------------------
	datelist_raw   = data_analysis_date_list(band, antenna, case=date_list_case) # list of daily measurements
	datelist       = data_analysis_daily_spectra_filter(band, datelist_raw)      # selection of daily measurements
	#datelist       = datelist[0:10] #['2016_107_00']                                             #      #      #, '2015_207_00']  #  ['2015_252_00']




	# LST arrays
	# -----------------------
	time_bin     = 2/6   # 20 minutes
	time_centers = np.arange((time_bin/2), 24, time_bin)
	time_edges   = np.arange(0, 24+(0.9*time_bin), time_bin)	


	# Output arrays
	# -----------------------
	v0       = 140 # frequency
	dv       = 1        # channel width    
	dv_noise = 0.055    # 9 samples is channel width of 55 kHz
	TA_all  = np.zeros((len(datelist), len(time_centers)))
	WA_all  = np.zeros((len(datelist), len(time_centers)))
	std_all = np.zeros((len(datelist), len(time_centers)))

	ambT_all  = np.zeros((len(datelist), len(time_centers)))
	ambH_all  = np.zeros((len(datelist), len(time_centers)))
	recT1_all = np.zeros((len(datelist), len(time_centers)))
	recT2_all = np.zeros((len(datelist), len(time_centers)))

	year_day_all = np.zeros((len(datelist), 3))



	# Process files
	# ----------------------------
	flag_j = 0

	for j in range(len(datelist)):


		# Load daily data
		# ---------------
		if (MC == 'no') or ((MC == 'yes') and (j == 70)):   # 70 is a reference day for doing MC error propagation 

			fin, t_2D, w_2D, m_2D = data_selection_single_day_v3(band, datelist[j], LST_1, LST_2, sun_el_max=sun_el_max, moon_el_max=moon_el_max, amb_hum_max=amb_hum_max, min_receiver_temp=min_receiver_temp, max_receiver_temp=max_receiver_temp)



			# Continue if there are data available
			# ------------------------------------
			if np.sum(t_2D) > 0:





				# Same antenna S11 and Losses are applied to all days
				flag_j = flag_j + 1		
				if flag_j == 1:

					# Antenna S11
					# --------------
					s11_ant = models_antenna_s11_remove_delay(band, antenna, fin, antenna_s11_day = ant_s11, model_type='polynomial', Nfit=ant_s11_Nfit, MC_mag=MC_ant_s11_abs, MC_ang=MC_ant_s11_ang, sigma_mag=0.0001, sigma_ang_deg=0.1)

					# Losses
					# ----------------------
					cg = combined_gain(band, fin, antenna_s11_day=ant_s11, antenna_s11_Nfit=ant_s11_Nfit, flag_ground_loss=fgl, ground_loss_type=glt, ground_loss_percent=glp, flag_antenna_loss=fal, flag_balun_connector_loss=fbcl, MC_ground_loss=MC_ground_loss, MC_antenna_loss=MC_antenna_loss, MC_balun_connector_loss=MC_balun_connector_loss)


					# Receiver lab calibration uncertainty
					# ------------------------------------
					s11_LNA, sca, off, TU, TC, TS = receiver_calibration_fast(band, fin, receiver_temperature = 25, cwterms=cwterms, MC='no')
					s11_LNA_MC, sca_MC, off_MC, TU_MC, TC_MC, TS_MC = receiver_calibration_fast(band, fin, receiver_temperature = 25, cwterms=cwterms, MC='yes')

					d_s11_LNA = s11_LNA_MC - s11_LNA
					d_sca = sca_MC - sca
					d_off = off_MC - off
					d_TU  = TU_MC  - TU
					d_TC  = TC_MC  - TC
					d_TS  = TS_MC  - TS




				print('-------------------')



				# Temporal arrays
				# ---------------
				TC_sum_all      = np.zeros(len(time_centers))
				WC_sum_all      = np.zeros(len(time_centers))
				inv_var_sum_all = np.zeros(len(time_centers))
				

				ambT_sum_all  = np.zeros(len(time_centers))
				wambT_sum_all = np.zeros(len(time_centers))
				ambH_sum_all  = np.zeros(len(time_centers))
				wambH_sum_all = np.zeros(len(time_centers))
				recT1_sum_all  = np.zeros(len(time_centers))
				wrecT1_sum_all = np.zeros(len(time_centers))
				recT2_sum_all  = np.zeros(len(time_centers))
				wrecT2_sum_all = np.zeros(len(time_centers))			


				# Year and day
				# ------------
				year = m_2D[0,0]
				day  = m_2D[0,1]
				frac = m_2D[0,2]
				year_day_all[j,0] = year
				year_day_all[j,1] = day
				year_day_all[j,2] = frac	


				# Calibrate every raw trace
				# -------------------------
				lt = len(t_2D[:,0])   # Number of traces in every daily file
				flag_i = 0

				for i in range(lt):

					flag_i = flag_i + 1
					print(datelist[j] + ' --- spectrum: ' + str(i+1) + ' of ' + str(lt))


					# Receiver temperature correction
					# -----------------------------
					if receiver_temperature == '25':
						RecTemp_base = 25
					elif receiver_temperature == 'actual':
						RecTemp_base = m_2D[i,-2]-0.4    # Correction of 0.4 degC is necessary


					RecTemp = RecTemp_base + MC_recv_temp    # Assigning a 1-sigma of 100 mK to the receiver temperature uncertainty, applied on 39-sec scales


					# Receiver calibration quantities
					# -----------------------------------------------			
					s11_LNA, sca, off, TU, TC, TS = receiver_calibration_fast(band, fin, receiver_temperature = RecTemp, cwterms=cwterms, MC='no')

					if MC_recv_calibration == 'yes':
						sca = sca + d_sca
						off = off + d_off
						TU  = TU + d_TU
						TC  = TC + d_TC
						TS  = TS + d_TS

					if MC_recv_s11 == 'yes':
						s11_LNA = s11_LNA + d_s11_LNA




					# Calibrated antenna temperature with losses and beam chromaticity
					# ----------------------------------------------------------------
					TAL = calibrated_antenna_temperature(t_2D[i,:], s11_ant, s11_LNA, sca, off, TU, TC, TS)			



					# Removing loss
					# -------------
					Tambient = 273.15 + m_2D[i,9] + MC_sigma_ambient_temperature
					TA = (TAL - Tambient*(1-cg))/cg
					TA[w_2D[i,:] == 0] = 0



					# Binning data contained in the only pixel at ref frequency
					# ---------------------------------------------------------
					TC   = TA[(fin >= (v0-(dv/2))) & (fin <= (v0+(dv/2)))]
					WC   = w_2D[i, (fin >= (v0-(dv/2))) & (fin <= (v0+(dv/2)))]
					
					TA0  = np.sum(WC*TC)/np.sum(WC) # average temperature per bin per trace
					
					TCn  = TA[(fin >= (v0-(dv_noise/2))) & (fin <= (v0+(dv_noise/2)))]
					std0 = np.std(TCn)               # standard deviation per bin per trace
					LST  = m_2D[i,3]
					
					


					# Accumulating temperature average and weights
					# --------------------------------------------
					
					index                  = int(np.floor(LST/time_bin))
					TC_sum_all[index]      = TC_sum_all[index]      + np.sum(WC)*TA0
					WC_sum_all[index]      = WC_sum_all[index]      + np.sum(WC)
					inv_var_sum_all[index] = inv_var_sum_all[index] + (1/std0**2)
					

					if m_2D[i,-4] != 0:
						ambT_sum_all[index]  = ambT_sum_all[index] + m_2D[i,-4]
						wambT_sum_all[index] = wambT_sum_all[index] + 1

					if m_2D[i,-3] != 0:
						ambH_sum_all[index]  = ambH_sum_all[index] + m_2D[i,-3]
						wambH_sum_all[index] = wambH_sum_all[index] + 1				

					if m_2D[i,-2] != 0:
						recT2_sum_all[index]  = recT2_sum_all[index] + m_2D[i,-2]
						wrecT2_sum_all[index] = wrecT2_sum_all[index] + 1

					if (m_2D[i,-1] >= 15) and (m_2D[i,-1] <= 30):
						recT1_sum_all[index]  = recT1_sum_all[index] + m_2D[i,-1]
						wrecT1_sum_all[index] = wrecT1_sum_all[index] + 1




				# Weighted average temperature, along with weights, at all LSTs
				# --------------------------------------------------------------

				for k in range(len(time_centers)):
					if (TC_sum_all[k] > 0) and (WC_sum_all[k] > 0):
						#print(TC_sum_all[k])
						#print(WC_sum_all[k])
						TA_all[j, k]  = TC_sum_all[k] / WC_sum_all[k]
						WA_all[j, k]  = WC_sum_all[k]
						std_all[j, k] = np.sqrt(1 / inv_var_sum_all[k])

					if (ambT_sum_all[k] != 0) and (wambT_sum_all[k] > 0):
						ambT_all[j, k] = ambT_sum_all[k] / wambT_sum_all[k]

					if (ambH_sum_all[k] != 0) and (wambH_sum_all[k] > 0):
						ambH_all[j, k] = ambH_sum_all[k] / wambH_sum_all[k]

					if (recT1_sum_all[k] != 0) and (wrecT1_sum_all[k] > 0):
						recT1_all[j, k] = recT1_sum_all[k] / wrecT1_sum_all[k]

					if (recT2_sum_all[k] != 0) and (wrecT2_sum_all[k] > 0):
						recT2_all[j, k] = recT2_sum_all[k] / wrecT2_sum_all[k]



	# Saving data
	# --------------------------

	if save == 'yes':

		path = home_folder + '/DATA/EDGES/results/high_band/products/temperature_at_140MHz/'

		if MC == 'no':
			path = path + 'nominal_with_std/'

		elif MC == 'yes':
			path = path + 'MC/'

		tc_T = np.reshape(time_centers, (-1,1))
		np.savetxt(path + name_prefix + 'LST.txt', tc_T)	

		np.savetxt(path + name_prefix + 'temperature.txt', TA_all)
		np.savetxt(path + name_prefix + 'standard_deviation.txt', std_all)
		np.savetxt(path + name_prefix + 'weights.txt', WA_all)
		np.savetxt(path + name_prefix + 'year_day.txt', year_day_all)

		np.savetxt(path + name_prefix + 'ambient_temp.txt', ambT_all)
		np.savetxt(path + name_prefix + 'ambient_hum.txt', ambH_all)
		np.savetxt(path + name_prefix + 'receiver_temp1.txt', recT1_all)
		np.savetxt(path + name_prefix + 'receiver_temp2.txt', recT2_all)


	return time_centers, year_day_all, TA_all, WA_all, std_all, ambT_all, ambH_all, recT1_all, recT2_all








def batch_MC_high_band_analysis_140MHz():

	N = 2000
	for i in range(N):
		print ('---------------------------------------------------------------------------------------- ' + str(i+1) + ' of ' + str(N))
		time_centers, year_day, TA, WA, ambT, ambH, recT1, recT2 = data_analysis_high_band_antenna_temperature_140MHz('no', MC='yes', name_prefix='') #'MC_' + str(i+1))
		if i == 0:
			TA_MC = np.zeros((N, len(TA[70,:])))
		TA_MC[i,:] = TA[70,:]

	path = home_folder + '/DATA/EDGES/results/high_band/products/temperature_at_140MHz/MC/'
	np.savetxt(path + 'temperature_MC.txt', TA_MC.T)

	return TA_MC















def data_selection_140MHz():


	"""

	This function selects the right measurements at 140 MHz to report

	"""


	LST      = np.genfromtxt(home_folder + '/DATA/EDGES/results/high_band/products/temperature_at_140MHz/nominal/LST.txt')
	year_day = np.genfromtxt(home_folder + '/DATA/EDGES/results/high_band/products/temperature_at_140MHz/nominal/year_day.txt')
	temp     = np.genfromtxt(home_folder + '/DATA/EDGES/results/high_band/products/temperature_at_140MHz/nominal/temperature.txt')
	weights  = np.genfromtxt(home_folder + '/DATA/EDGES/results/high_band/products/temperature_at_140MHz/nominal/weights.txt')

	temp2          = np.copy(temp)
	temp2[141::,:] = 0
	
	temp2[0:33, 0] = 0; temp2[73::, 0] = 0
	temp2[0:33, 1] = 0; temp2[73::, 1] = 0
	temp2[0:34, 2] = 0; temp2[73::, 2] = 0
	temp2[0:33, 3] = 0; temp2[73::, 3] = 0
	temp2[0:33, 4] = 0; temp2[73::, 4] = 0
	temp2[0:33, 5] = 0; temp2[73::, 5] = 0
	temp2[0:33, 6] = 0; temp2[73::, 6] = 0

	temp2[128::, 7] = 0
	temp2[130::, 8] = 0
	temp2[135::, 9] = 0

	temp2[0:26, 10] = 0; temp2[143::, 10] = 0
	temp2[0:25, 11] = 0; temp2[143::, 11] = 0
	temp2[0:26, 12] = 0; temp2[140::, 12] = 0

	temp2[0:34, 13] = 0
	temp2[0:40, 14] = 0
	temp2[0:47, 15] = 0
	temp2[0:53, 16] = 0
	temp2[0:58, 17] = 0
	temp2[0:64, 18] = 0
	temp2[0:69, 19] = 0
	temp2[0:73, 20] = 0
	temp2[0:77, 21] = 0
	temp2[0:79, 22] = 0
	temp2[0:84, 23] = 0
	temp2[0:89, 24] = 0
	temp2[0:95, 25] = 0
	temp2[0:99, 26] = 0
	temp2[0:103, 27] = 0
	temp2[0:108, 28] = 0
	temp2[0:112, 29] = 0
	temp2[0:116, 30] = 0
	temp2[0:119, 31] = 0
	temp2[0:124, 32] = 0
	temp2[0:128, 33] = 0
	temp2[0:131, 34] = 0
	temp2[0:134, 35] = 0

	temp2[0::, 36:45] = 0
	temp2[73::, 36::] = 0
	
	temp2[6::,  45] = 0; temp2[0:2, 45] = 0
	temp2[10::, 46] = 0; temp2[0:2, 46] = 0; temp2[8, 46]   = 0
	temp2[14::, 47] = 0; temp2[0:3, 47] = 0; temp2[8, 47]   = 0
	temp2[18::, 48] = 0; temp2[0:5, 48] = 0
	temp2[22::, 49] = 0; temp2[0:8, 49] = 0; temp2[11, 49] = 0
	temp2[28::, 50] = 0; temp2[[0,1,2,6,9,12,15], 50] = 0 
	temp2[31::, 51] = 0; temp2[[0,1,9,11,12,14],  51] = 0
	temp2[0:2,  52] = 0
			
	temp2[37::, 53] = 0; temp2[0:2, 53] = 0
	
	temp2[0:33, 54::] = 0
	
	temp2[40::, 54] = 0
	temp2[44::, 55] = 0 
	temp2[47::, 56] = 0
	temp2[52::, 57] = 0
	temp2[56::, 58] = 0
	temp2[58::, 59] = 0
	temp2[62::, 60] = 0
	temp2[66::, 61] = 0
	temp2[70::, 62] = 0
	temp2[72::, 63] = 0

	temp2[0:33, 71] = 0; temp2[73::, 71] = 0




	


	# Compute mean and std for every LST in the range 23:50 and 11:30 hh:mm, by iterating 10 times and removing points that are beyond 3 sigma.
	# ------------------------------
	mean_all  = np.zeros(len(LST))
	std_all   = np.zeros(len(LST))	
	for i in range(len(LST)):

		for j in range(10):
			mean_all[i] = np.mean(temp2[temp2[:,i]>0,i])
			std_all[i]  = np.std(temp2[temp2[:,i]>0,i])

			for k in range(len(temp2[:,i])):
				if temp2[k,i] > 0:
					diff = temp2[k,i] - mean_all[i]
					if np.abs(diff) > 3.0 * std_all[i]:
						temp2[k,i] = 0


	# Saving mean and std
	# ----------------------------
	results = np.hstack((LST.reshape(-1,1), mean_all.reshape(-1,1), std_all.reshape(-1,1)))
	np.savetxt(home_folder + '/DATA/EDGES/results/high_band/products/temperature_at_140MHz/nominal/LST_mean_std_20180114.txt', results)





	# Figure 1
	# -------------------------------------
	plt.close()
	plt.close()
	index_all    = np.arange(len(temp[:,0]))
	
	size_x = 4.5
	size_y = 10
	x0=0.16
	y0=0.1
	dx=0.8
	dy=0.85
	
	f1 = plt.figure(figsize=(size_x, size_y))
	ax = f1.add_axes([x0, y0, dx, dy])
	
	dT = 2
	year_day2 = np.copy(year_day)
	year_day2[year_day2[:,0]==2016,1] = year_day2[year_day2[:,0]==2016,1] + 365
	for i in range(72):
		ax.plot(year_day2[temp2[:,i]>0,1], temp2[temp2[:,i]>0,i] - mean_all[i] - i*dT, '.', markersize=4)
		
	ax.set_xlim([200, 390])
	ax.set_ylim([-144, 2])
	
	xt = np.arange(200,401,25)
	ax.set_xticks(xt)
	ax.set_xticklabels(['2015-200', '2015-225', '2015-250', '2015-275', '2015-300', '2015-325', '2015-350', '2016-10', '2016-35'], rotation=65)
	
	yt = np.arange(-142,1,2)
	ytl = []
	for i in range(len(yt)):
		if (-(yt[i]/2)-1)%3 == 0:
			hh = int((-(yt[i]/2)-1)/3)
			ytl.append(str(hh).zfill(2) + ':30')
		else:
			ytl.append('')
			
	ax.set_yticks(yt)
	ax.set_yticklabels(ytl)
	
	ax.grid()
	ax.plot([365.5, 365.5], [-144, 2], 'k--')

	ax.set_xlabel('date [yyyy-ddd]')
	ax.set_ylabel('[2 K per division]')
	ax.text(176,7,'LST')
	ax.text(167,4,'[hh:mm]')
				

	# Save plot
	# --------------------------------------
	plt.savefig(home_folder + '/DATA/EDGES/results/plots/20180113/temperature_at_140MHz.pdf', bbox_inches='tight')
	plt.close()	



	return LST, mean_all, std_all












def plot_LST_140MHz():

	d  = np.genfromtxt(home_folder + '/DATA/EDGES/results/high_band/products/temperature_at_140MHz/nominal/LST_mean_std_20180114.txt')
	MC = np.genfromtxt(home_folder + '/DATA/EDGES/results/high_band/products/temperature_at_140MHz/MC/temperature_MC.txt')

	plt.close()
	plt.close()
	
	plt.figure(figsize=[5, 5])

	plt.subplot(2,1,1)
	plt.plot(d[:,0], d[:,1], 'b.-')
	plt.xlim([0, 24])
	plt.ylim([0, 1250])
	plt.xticks([0,4,8,12,16,20,24])
	plt.yticks([0,250,500,750,1000,1250])
	plt.grid()
	plt.ylabel(r'$\rm{T_A}$ [K]')
	
	plt.subplot(2,1,2)
	syst = np.std(MC, axis=1)
	syst[(d[:,0]>19.9) & (d[:,0]<20.4)] = 0.82
	syst[(d[:,0]>22.9) & (d[:,0]<23.4)] = 0.27571337500000004
	syst[np.isnan(d[:,1])==True]=np.nan
	
	scatter = 0.5*np.ones(len(d[:,0]))
	scatter[np.isnan(d[:,1])==True]=np.nan
	
	total = np.sqrt(syst**2 + scatter**2)
	
	
	plt.plot(d[:,0], syst, 'r.-')
	plt.plot(d[:,0], scatter, 'g.-')
	plt.plot(d[:,0], total, 'b.-')
	plt.xlim([0, 24])
	plt.ylim([0, 1.25])
	plt.xticks([0,4,8,12,16,20,24])
	plt.yticks([0,0.250,0.500,0.750,1.000,1.250])
	plt.grid()
	
	plt.xlabel('LST [hr]')
	plt.ylabel(r'$\sigma(\rm{T_A})$ [K]')
	plt.legend(['calibration uncertainty','day-to-day scatter','total uncertainty'])
	

	# Save plot
	plt.savefig(home_folder + '/DATA/EDGES/results/plots/20180113/LST.pdf', bbox_inches='tight')
	plt.close()	

	# Save systematic standard deviation from MC
	resultT = np.array([d[:,0], syst])
	result  = resultT.T
	np.savetxt(home_folder + '/DATA/EDGES/results/high_band/products/temperature_at_140MHz/MC/systematic_std_dev.txt', result)


	# Save total uncertainty
	totalT = np.array([d[:,0], total])
	total  = totalT.T
	np.savetxt(home_folder + '/DATA/EDGES/results/high_band/products/temperature_at_140MHz/MC/total_uncertainty.txt', total)


	

	return 0 #total
























def plot_beam_projection_140MHz():


	plt.close()
	plt.close()
	plt.close()

	path_data = home_folder + '/DATA/EDGES/calibration/sky/'


	# Loading galactic coordinates (the Haslam map is in NESTED Galactic Coordinates)
	coord              = fits.open(path_data + 'coordinate_maps/pixel_coords_map_nested_galactic_res9.fits')
	coord_array        = coord[1].data
	lon                = coord_array['LONGITUDE']
	lat                = coord_array['LATITUDE']
	GALAC_COORD_object = apc.SkyCoord(lon, lat, frame='galactic', unit='deg')  # defaults to ICRS frame


	# Loading Haslam map
	haslam_map = fits.open(path_data + 'haslam_map/lambda_haslam408_dsds.fits')
	haslam408  = (haslam_map[1].data)['temperature']


	# Spectral index in HEALPix RING Galactic Coordinates, nside=512
	beta_file = path_data + 'spectral_index/sky_spectral_index_original_45_408_MHz_maps_galactic_coordinates_nside_512_ring_3Ksubtracted.hdf5'
	with h5py.File(beta_file, 'r') as hf:			
		hf_beta   = hf.get('spectral_index')
		beta_ring = np.array(hf_beta)


	# Convert beta to NESTED format
	beta = hp.reorder(beta_ring, r2n=True)


	# Loading celestial coordinates to fill in the spectral index hole around the north pole
	coord              = fits.open(path_data + 'coordinate_maps/pixel_coords_map_nested_celestial_res9.fits')
	coord_array        = coord[1].data
	RA                 = coord_array['LONGITUDE']
	DEC                = coord_array['LATITUDE']


	# Filling the hole
	beta[DEC>68] = np.mean(beta[(DEC>60) & (DEC<68)])


	# Remove offsets
	Tcmb = 2.725
	Tzlc = -3.46		
	T408 = haslam408 - 3    # - Tcmb - Tzlc   # Corrections necessary


	# Produce map at 140 MHz
	m140 = T408 * (140/408)**(-beta) + 3










	beam     = FEKO_high_band_blade_beam_plus_shaped_finite_ground_plane(AZ_antenna_axis=-5)
	beam140  = beam[12,:,:]
	beam140n = beam140/np.max(beam140)

	FWHM = np.zeros((360, 2))
	EL_raw    = np.arange(0,91,1)
	EL_new    = np.arange(0,90.01, 0.01)		

	for j in range(len(beam140[0,:])): # Loop over AZ
		#print(j)

		func      = spi.interp1d(EL_raw, beam140n[:,j])
		beam140n_interp = func(EL_new)

		minDiff = 100
		for i in range(len(EL_new)):	
			Diff    = np.abs(beam140n_interp[i] - 0.5)
			if Diff < minDiff:
				#print(90-EL_new[i])
				minDiff   = np.copy(Diff)
				FWHM[j,0] = j
				FWHM[j,1] = 90 - EL_new[i]








	# Reference location
	EDGES_lat_deg  = -26.714778
	EDGES_lon_deg  = 116.605528 
	EDGES_location = apc.EarthLocation(lat=EDGES_lat_deg*apu.deg, lon=EDGES_lon_deg*apu.deg)

	# Numpy arrays of time
	Time_iter_UTC_1 = np.array([2014, 1, 1,  3,  0, 50])   # 17.76  Obtained manually using LST = eg.utc2lst(Time_iter_UTC_start, EDGES_lon_deg)
	Time_iter_UTC_2 = np.array([2014, 1, 1,  8,  0,  0])   # 22.50  (22h : 30m)
	Time_iter_UTC_3 = np.array([2014, 1, 1, 11, 59, 20])   # 2.50   (2h : 30m) 
	Time_iter_UTC_4 = np.array([2014, 1, 1, 16, 18, 25])   # 6.83   (6h : 50m)
	Time_iter_UTC_5 = np.array([2014, 1, 1, 20, 58, 00])   # 11.50  (11h : 30m)	
	
	RA_center_1 = 17.50
	RA_center_2 = 22.50
	RA_center_3 =  2.50
	RA_center_4 =  6.83
	RA_center_5 = 11.50
	
	
	
	l_center_1 = 0.03051160
	b_center_1 = 4.10569938

	l_center_2 = 25.44699470
	b_center_2 = -58.57554080	
	
	l_center_3 = 217.51791982-360
	b_center_3 = -68.06121337
			
	l_center_4 = 236.90465820-360
	b_center_4 = -12.16152815

	l_center_5 = 281.25772313-360
	b_center_5 = 32.70644579
		






	# Transforming Numpy arrays to Datetime objects
	Time_iter_UTC_1_dt = dt.datetime(Time_iter_UTC_1[0], Time_iter_UTC_1[1], Time_iter_UTC_1[2], Time_iter_UTC_1[3], Time_iter_UTC_1[4], Time_iter_UTC_1[5]) 
	Time_iter_UTC_2_dt = dt.datetime(Time_iter_UTC_2[0], Time_iter_UTC_2[1], Time_iter_UTC_2[2], Time_iter_UTC_2[3], Time_iter_UTC_2[4], Time_iter_UTC_2[5])
	Time_iter_UTC_3_dt = dt.datetime(Time_iter_UTC_3[0], Time_iter_UTC_3[1], Time_iter_UTC_3[2], Time_iter_UTC_3[3], Time_iter_UTC_3[4], Time_iter_UTC_3[5]) 
	Time_iter_UTC_4_dt = dt.datetime(Time_iter_UTC_4[0], Time_iter_UTC_4[1], Time_iter_UTC_4[2], Time_iter_UTC_4[3], Time_iter_UTC_4[4], Time_iter_UTC_4[5])
	Time_iter_UTC_5_dt = dt.datetime(Time_iter_UTC_5[0], Time_iter_UTC_5[1], Time_iter_UTC_5[2], Time_iter_UTC_5[3], Time_iter_UTC_5[4], Time_iter_UTC_5[5])











	# Converting Beam Contours from Local to Equatorial and Galactic coordinates
	AltAz_1 = apc.SkyCoord(alt = (90-FWHM[:,1])*apu.deg, az = FWHM[:,0]*apu.deg, frame = 'altaz', obstime = apt.Time(Time_iter_UTC_1_dt, format='datetime'), location = EDGES_location)
	AltAz_2 = apc.SkyCoord(alt = (90-FWHM[:,1])*apu.deg, az = FWHM[:,0]*apu.deg, frame = 'altaz', obstime = apt.Time(Time_iter_UTC_2_dt, format='datetime'), location = EDGES_location)
	AltAz_3 = apc.SkyCoord(alt = (90-FWHM[:,1])*apu.deg, az = FWHM[:,0]*apu.deg, frame = 'altaz', obstime = apt.Time(Time_iter_UTC_3_dt, format='datetime'), location = EDGES_location)
	AltAz_4 = apc.SkyCoord(alt = (90-FWHM[:,1])*apu.deg, az = FWHM[:,0]*apu.deg, frame = 'altaz', obstime = apt.Time(Time_iter_UTC_4_dt, format='datetime'), location = EDGES_location)
	AltAz_5 = apc.SkyCoord(alt = (90-FWHM[:,1])*apu.deg, az = FWHM[:,0]*apu.deg, frame = 'altaz', obstime = apt.Time(Time_iter_UTC_5_dt, format='datetime'), location = EDGES_location)








	# Beam in Equatorial coordinates
	RaDec_1 = AltAz_1.icrs	
	Ra_1    = np.asarray(RaDec_1.ra)
	Dec_1   = np.asarray(RaDec_1.dec)
	
	RaDec_2 = AltAz_2.icrs	
	Ra_2    = np.asarray(RaDec_2.ra)
	Dec_2   = np.asarray(RaDec_2.dec)	
	
	RaDec_3 = AltAz_3.icrs	
	Ra_3    = np.asarray(RaDec_3.ra)
	Dec_3   = np.asarray(RaDec_3.dec)
	
	RaDec_4 = AltAz_4.icrs	
	Ra_4    = np.asarray(RaDec_4.ra)
	Dec_4   = np.asarray(RaDec_4.dec)
	
	RaDec_5 = AltAz_5.icrs	
	Ra_5    = np.asarray(RaDec_5.ra)
	Dec_5   = np.asarray(RaDec_5.dec)
	
		



	RaWrap_1 = np.copy(Ra_1)
	RaWrap_1[Ra_1>(17.76/24)*360] = RaWrap_1[Ra_1>(17.76/24)*360] - 360
	
	RaWrap_11  = RaWrap_1[RaWrap_1>0]
	DecWrap_11 = Dec_1[RaWrap_1>0]
	RaWrap_12  = RaWrap_1[RaWrap_1<0]
	DecWrap_12 = Dec_1[RaWrap_1<0]
	
	RaWrap_11n  = RaWrap_11[7::]
	DecWrap_11n = DecWrap_11[7::]
	
	RaWrap_11n2  = np.append(RaWrap_11n, RaWrap_11[0:7])
	DecWrap_11n2 = np.append(DecWrap_11n, DecWrap_11[0:7])
	
	RaWrap_11  = np.copy(RaWrap_11n2)
	DecWrap_11 = np.copy(DecWrap_11n2)
	
	
	

	RaWrap_2 = np.copy(Ra_2)
	RaWrap_2[Ra_2>(12/24)*360] = RaWrap_2[Ra_2>(12/24)*360] - 360
	
	RaWrap_3 = np.copy(Ra_3)
	RaWrap_3[Ra_3>(12/24)*360] = RaWrap_3[Ra_3>(12/24)*360] - 360
	
	RaWrap_4 = np.copy(Ra_4)
	RaWrap_4[Ra_4>(12/24)*360] = RaWrap_4[Ra_4>(12/24)*360] - 360
	
	RaWrap_5 = np.copy(Ra_5)









	# Beam in Galactic coordinates
	lb_1 = AltAz_1.galactic	
	l_1  = np.asarray(lb_1.l)
	b_1  = np.asarray(lb_1.b)
	
	lb_2 = AltAz_2.galactic	
	l_2  = np.asarray(lb_2.l)
	b_2  = np.asarray(lb_2.b)
	
	lb_3 = AltAz_3.galactic	
	l_3  = np.asarray(lb_3.l)
	b_3  = np.asarray(lb_3.b)
	
	lb_4 = AltAz_4.galactic	
	l_4  = np.asarray(lb_4.l)
	b_4  = np.asarray(lb_4.b)
	
	lb_5 = AltAz_5.galactic	
	l_5  = np.asarray(lb_5.l)
	b_5  = np.asarray(lb_5.b)
	
	
	
	lWrap_1 = np.copy(l_1)	
	lWrap_1[l_1>180] = l_1[l_1>180] - 360
	

	lWrap_2          = np.copy(l_2)
	lWrap_2[l_2>74]  = l_2[l_2>74] - 360
	
	lWrap_21         = lWrap_2[lWrap_2>-180]
	bWrap_21         = b_2[lWrap_2>-180]
	
	lWrap_22         = lWrap_2[lWrap_2<-180] + 360
	bWrap_22         = b_2[lWrap_2<-180]
	
	
	
	lWrap_3             = np.copy(l_3)
	lWrap_3[l_3>159.5]  = l_3[l_3>159.5] - 360

	lWrap_31         = lWrap_3[lWrap_3>-180]
	bWrap_31         = b_3[lWrap_3>-180]
	
	lWrap_32         = lWrap_3[lWrap_3<-180] + 360
	bWrap_32         = b_3[lWrap_3<-180]
	
		
	lWrap_4          = np.copy(l_4)
	lWrap_4[l_4>180] = l_4[l_4>180] - 360
	
	
	lWrap_5          = np.copy(l_5)
	lWrap_5[l_4>180] = l_5[l_5>180] - 360	
	
	
	
	





	# Plot 1 in Equatorial Coordinates
	hp.cartview(np.log10(m140), nest='true', coord=('G', 'C'), flip='geo', title='', notext='true', rot=((17.76/24)*360 + 180, 0, 0), min=2.2, max=3.9, unit=r'log($T_{\mathrm{sky}}$)', cmap=plt.get_cmap('gray'))
	hp.graticule()	

	plt.plot(np.arange(-180,181,1), -26.7*np.ones(361), 'r--', linewidth=2)

	plt.plot(-180 + ((24-17.76)/24)*360 + RaWrap_12, DecWrap_12, 'y--', linewidth=2)
	plt.plot(-180 + ((24-17.76)/24)*360 + RaWrap_2, Dec_2, 'w',   linewidth=2)
	plt.plot(-180 + ((24-17.76)/24)*360 + RaWrap_3, Dec_3, 'y',   linewidth=2)
	plt.plot(-180 + ((24-17.76)/24)*360 + RaWrap_4, Dec_4, 'w--', linewidth=2)
	plt.plot(-180 + ((24-17.76)/24)*360 + RaWrap_5, Dec_5, 'w:',  linewidth=2)
	plt.plot(-180 + ((24-17.76)/24)*360 + RaWrap_11, DecWrap_11, 'y--', linewidth=2)
	
	plt.plot(-180 + ((24-17.76)/24)*360 + (RA_center_1/24)*360,       -26.7, 'x', color='y', markersize=5, mew=2)
	plt.plot(-180 + ((24-17.76)/24)*360 + (RA_center_2/24)*360 - 360, -26.7, 'x', color='1', markersize=5, mew=2)
	plt.plot(-180 + ((24-17.76)/24)*360 + (RA_center_3/24)*360,       -26.7, 'x', color='y', markersize=5, mew=2)
	plt.plot(-180 + ((24-17.76)/24)*360 + (RA_center_4/24)*360,       -26.7, 'x', color='1', markersize=5, mew=2)
	plt.plot(-180 + ((24-17.76)/24)*360 + (RA_center_5/24)*360,       -26.7, 'x', color='1', markersize=5, mew=2)



	off_x = -7
	off_y = -12
	factor1 = 0.5
	plt.text(-180 + ((18-17.76)/24)*360 + off_x, -90+off_y, '18')
	plt.text(-150 + ((18-17.76)/24)*360 + off_x, -90+off_y, '20')
	plt.text(-120 + ((18-17.76)/24)*360 + off_x, -90+off_y, '22')
	plt.text( -90 + ((18-17.76)/24)*360 + factor1*off_x, -90+off_y, '0')
	plt.text( -60 + ((18-17.76)/24)*360 + factor1*off_x, -90+off_y, '2')
	plt.text( -30 + ((18-17.76)/24)*360 + factor1*off_x, -90+off_y, '4')
	plt.text(  -0 + ((18-17.76)/24)*360 + factor1*off_x, -90+off_y, '6')

	plt.text( 30 + ((18-17.76)/24)*360 + factor1*off_x, -90+off_y, '8')
	plt.text( 60 + ((18-17.76)/24)*360 + off_x, -90+off_y, '10')
	plt.text( 90 + ((18-17.76)/24)*360 + off_x, -90+off_y, '12')
	plt.text(120 + ((18-17.76)/24)*360 + off_x, -90+off_y, '14')
	plt.text(150 + ((18-17.76)/24)*360 + off_x, -90+off_y, '16')
	#plt.text(180 + ((18-17.76)/24)*360 + off_x, -90+off_y, '12')		

	plt.text(-76, -115, 'local sidereal time [hours]')



	off_x1 = -15
	off_x2 = -10
	off_x3 = -19
	off_y  = -3
	plt.text(-180+off_x1,  90+off_y,  '90')
	plt.text(-180+off_x1,  60+off_y,  '60')
	plt.text(-180+off_x1,  30+off_y,  '30')
	plt.text(-180+off_x2,   0+off_y,   '0')
	plt.text(-180+off_x3, -30+off_y, '-30')
	plt.text(-180+off_x3, -60+off_y, '-60')
	plt.text(-180+off_x3, -90+off_y, '-90')

	plt.text(-210, 45, 'declination [degrees]', rotation=90)









	# Plot 2, Galactic coordinates
	hp.cartview(np.log10(m140), nest='true', coord=('G'), flip='geo', title='', notext='true', min=2.2, max=3.9, unit=r'log($T_{\mathrm{sky}}$)', cmap=plt.get_cmap('gray'))   # rot=((17.76/24)*360 + 180, 0, 0) #,  , min=2.2, max=3.9 unit=r'log($T_{sky}$)' title='', min=2.3, max=2.6, unit=r'$\beta$'), , rot=[180,0,0]
	hp.graticule()


	plt.plot(lWrap_1, b_1, 'y--',     linewidth=2)
	plt.plot(lWrap_21, bWrap_21, 'w', linewidth=2)
	plt.plot(lWrap_22, bWrap_22, 'w', linewidth=2)
	plt.plot(lWrap_31, bWrap_31, 'y', linewidth=2)
	plt.plot(lWrap_32, bWrap_32, 'y', linewidth=2)
	plt.plot(lWrap_4, b_4, 'w--',     linewidth=2)
	plt.plot(lWrap_5, b_5, 'w:',      linewidth=2)
	
	plt.plot(l_center_1, b_center_1, 'x', color='y', markersize=5, mew=2)
	plt.plot(l_center_2, b_center_2, 'x', color='w', markersize=5, mew=2)
	plt.plot(l_center_3, b_center_3, 'x', color='y', markersize=5, mew=2)
	plt.plot(l_center_4, b_center_4, 'x', color='w', markersize=5, mew=2)
	plt.plot(l_center_5, b_center_5, 'x', color='w', markersize=5, mew=2)
	
	

	off_x = -15
	off_y = -16   
	factor1 = 0.7
	factor2 = 0.2
	factor3 = 0.4
	factor4 = 0.65
	plt.text(-180 + off_x,         -90+off_y, '-180')
	plt.text(-150 + off_x,         -90+off_y, '-150')
	plt.text(-120 + off_x,         -90+off_y, '-120')
	plt.text( -90 + factor1*off_x, -90+off_y, '-90')
	plt.text( -60 + factor1*off_x, -90+off_y, '-60')
	plt.text( -30 + factor1*off_x, -90+off_y, '-30')
	plt.text(  -0 + factor2*off_x, -90+off_y, '0')

	plt.text( 30 + factor3*off_x,  -90+off_y, '30')
	plt.text( 60 + factor3*off_x,  -90+off_y, '60')
	plt.text( 90 + factor3*off_x,  -90+off_y, '90')
	plt.text(120 + factor4*off_x,  -90+off_y, '120')
	plt.text(150 + factor4*off_x,  -90+off_y, '150')
	plt.text(180 + factor4*off_x,  -90+off_y, '180')		

	plt.text(-81, -125, 'galactic longitude [degrees]')



	off_x1 = -15
	off_x2 = -10
	off_x3 = -19
	off_y  = -3
	plt.text(-180+off_x1,  90+off_y,  '90')
	plt.text(-180+off_x1,  60+off_y,  '60')
	plt.text(-180+off_x1,  30+off_y,  '30')
	plt.text(-180+off_x2,   0+off_y,   '0')
	plt.text(-180+off_x3, -30+off_y, '-30')
	plt.text(-180+off_x3, -60+off_y, '-60')
	plt.text(-180+off_x3, -90+off_y, '-90')

	plt.text(-210, 68, 'galactic latitude [degrees]', rotation=90)



	return Ra_1, Dec_1, RaWrap_1, RaWrap_11, DecWrap_11, RaWrap_12, DecWrap_12












def simulated_antenna_temperature(name_save, band, beam_file=1, rotation_from_north=-5, sky_map_option='haslam', reference_frequency = 140, band_deg=10, index_inband=2.5, index_outband=2.57):



	# Data paths
	path_data = home_folder + '/DATA/EDGES/calibration/sky/'
	path_save = home_folder + '/DATA/EDGES/results/low_band/products/temperature_at_60MHz/'


	# Loading beam
	AZ_beam  = np.arange(0, 360)
	EL_beam  = np.arange(0, 91)
	
	
	# Beam model
	if band == 'high_band':
		# beam_file = 1:   nominal from Alan, plus shaped ground plane, 80-200 MHz, every 5 MHz
		# rotation from north = -5 (NS)		
		beam_all   = eg.FEKO_high_band_blade_beam_plus_shaped_finite_ground_plane(beam_file=beam_file, AZ_antenna_axis=rotation_from_north)		
		freq_array = np.arange(80, 201, 5, dtype='uint32')
		
		
	if band == 'low_band':
		# beam_file = 0:   beam from Nivedita, extended ground plane, 40-100 MHz, every 2 MHz
		# beam_file = 2:   beam from Alan, extended ground plane, 40-100 MHz, every 2 MHz	
		# rotation from north. NS (low1=-7deg, low2=-2deg), EW (low2=87deg)
		beam_all   = eg.FEKO_low_band_blade_beam(beam_file=beam_file, AZ_antenna_axis=rotation_from_north)
		freq_array = np.arange(40, 101, 2) 
	
	
	


	# Beam map at the desired frequency
	beam_map = beam_all[freq_array==reference_frequency,:,:].reshape(1,-1)[0]
	
	


	#offset = np.arange(0, 61, 2)
	#scale  = np.arange(1, 1.205, 0.01)

	offset = 0
	scale  = 1
	
	#offset_scale_array = np.zeros((len(offset)*len(scale), 2))
	
	
	
	#k = -1
	
	# Convolving beam with extrapolated Sky Map, also sweeping over Map Offset and Gain
	# for x1 in range(len(offset)):
		# for x2 in range(len(scale)):

	#k = k + 1
	
	#offset_scale_array[k, 0] =offset[x1]
	#offset_scale_array[k, 1] = scale[x2]
	


	#print('OFFSET: ' + str(offset[x1]) + '. SCALE: ' + str(scale[x2]))

	if sky_map_option == 'haslam': 
	
		# Loading galactic coordinates (the Haslam map is in NESTED Galactic Coordinates)
		coord              = fits.open(path_data + 'coordinate_maps/pixel_coords_map_nested_galactic_res9.fits')
		coord_array        = coord[1].data
		lon                = coord_array['LONGITUDE']
		lat                = coord_array['LATITUDE']
		GALAC_COORD_object = apc.SkyCoord(lon, lat, frame='galactic', unit='deg')  # defaults to ICRS frame
	
		# Loading Haslam map
		#haslam_map = fits.open(path_data + 'haslam_map/lambda_haslam408_dsds.fits')
		#haslam408  = (haslam_map[1].data)['temperature']
		
		haslam_map = fits.open(path_data + 'haslam_map/lambda_haslam408_nofilt.fits')   # Here we use the raw map, without destriping
		haslam408  = (haslam_map[1].data)['temperature']/1000   # dividing by 1000 to make it K, because this raw map version is in mK
	
		# Scaling Haslam map (the map contains the CMB, which has to be removed at 408 MHz, and then added back)
		Tcmb   = 2.725
		T408   = (scale*haslam408 + offset) - Tcmb
		b0     = band_deg          # default 10 degrees, galactic elevation threshold for different spectral index
	
	
	
		sky_map = np.zeros(len(T408))
	
		# Band of the Galactic center, using spectral index
		sky_map[(lat >= -b0) & (lat <= b0)] = T408[(lat >= -b0) & (lat <= b0)] * (reference_frequency/408)**(-index_inband) + Tcmb
	
		# Range outside the Galactic center, using second spectral index
		sky_map[(lat < -b0) | (lat > b0)]   = T408[(lat < -b0) | (lat > b0)] * (reference_frequency/408)**(-index_outband) + Tcmb






	

	elif sky_map_option == 'LW': 
	
		# Loading galactic coordinates (the Haslam map is in NESTED Galactic Coordinates)
		coord              = fits.open(path_data + 'coordinate_maps/pixel_coords_map_nested_galactic_res8.fits')
		coord_array        = coord[1].data
		lon                = coord_array['LONGITUDE']
		lat                = coord_array['LATITUDE']
		GALAC_COORD_object = apc.SkyCoord(lon, lat, frame='galactic', unit='deg')  # defaults to ICRS frame
	
		# Loading Haslam map
		#haslam_map = fits.open(path_data + 'haslam_map/lambda_haslam408_dsds.fits')
		#haslam408  = (haslam_map[1].data)['temperature']
		

		LW150 = np.genfromtxt(path_data + 'LW_150MHz_map/150_healpix_gal_nested_R8.txt')   # Here we use the raw map, without destriping
		
		# Scaling Haslam map (the map contains the CMB, which has to be removed at 408 MHz, and then added back)
		Tcmb = 2.725
		T150 = (scale*LW150 + offset) - Tcmb
		b0   = band_deg          # default 10 degrees, galactic elevation threshold for different spectral index
	
	
	
		sky_map = np.zeros(len(T150))
	
		# Band of the Galactic center, using spectral index
		sky_map[(lat >= -b0) & (lat <= b0)] = T150[(lat >= -b0) & (lat <= b0)] * (reference_frequency/150)**(-index_inband) + Tcmb
	
		# Range outside the Galactic center, using second spectral index
		sky_map[(lat < -b0) | (lat > b0)]   = T150[(lat < -b0) | (lat > b0)] * (reference_frequency/150)**(-index_outband) + Tcmb
	
		
		print(len(LW150))
		print(len(sky_map))







	elif sky_map_option == 'guzman': 
	
		# Loading galactic coordinates (the Haslam map is in NESTED Galactic Coordinates)
		coord              = fits.open(path_data + 'coordinate_maps/pixel_coords_map_nested_galactic_res9.fits')
		coord_array        = coord[1].data
		lon                = coord_array['LONGITUDE']
		lat                = coord_array['LATITUDE']
		GALAC_COORD_object = apc.SkyCoord(lon, lat, frame='galactic', unit='deg')  # defaults to ICRS frame
	
	
		# Loading Guzman map
		guzman45 = eg.guzman_45MHz_map()
	
	
		# Scaling Haslam map (the map contains the CMB, which has to be removed at 45 MHz, and then added back)
		Tcmb   = 2.725
		T45    = (scale*guzman45 + offset) - Tcmb
		b0     = band_deg          # default 10 degrees, galactic elevation threshold for different spectral index
	
	
	
		sky_map = np.zeros(len(T45))
	
		# Band of the Galactic center, using spectral index
		sky_map[(lat >= -b0) & (lat <= b0)] = T45[(lat >= -b0) & (lat <= b0)] * (reference_frequency/45)**(-index_inband) + Tcmb
	
		# Range outside the Galactic center, using second spectral index
		sky_map[(lat < -b0) | (lat > b0)]   = T45[(lat < -b0) | (lat > b0)] * (reference_frequency/45)**(-index_outband) + Tcmb







	
	# EDGES location	
	EDGES_lat_deg  = -26.714778
	EDGES_lon_deg  = 116.605528 
	EDGES_location = apc.EarthLocation(lat=EDGES_lat_deg*apu.deg, lon=EDGES_lon_deg*apu.deg)


	# Reference UTC observation time. At this time, the LST is 0.1666 (00:10 Hrs LST) at the EDGES location (it was wrong before, now it is correct)
	#Time_iter    = np.array([2014, 1, 1, 9, 39, 42]) 
	
	# Reference UTC observation time. At this time, the LST is 0.000 (00:00 Hrs LST) at the EDGES location
	Time_iter    = np.array([2014, 1, 1, 9, 29, 45])
	
	
	
	Time_iter_dt = dt.datetime(Time_iter[0], Time_iter[1], Time_iter[2], Time_iter[3], Time_iter[4], Time_iter[5]) 


	# Looping over LST
	LST             = np.zeros(72)
	#LST             = np.zeros(24*60)  # 1440 minutes over 24 hours
	convolution     = np.zeros(len(LST))


	for i in range(len(LST)):
	#for i in range(1):


		#if i == 0:
		#	convolution_array  = np.zeros((len(offset)*len(scale), len(LST)))



		print(name_save + ', LST: ' + str(i+1) + ' out of ' + str(len(LST)))


		# Advancing time ( 19:57 minutes UTC correspond to 20 minutes LST )
		minutes_offset = 19
		seconds_offset = 57
		
		# Advancing time ( 1 minutes UTC correspond to <1 minute LST )
		# minutes_offset = 1
		# seconds_offset = 0
		
		
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
		sky_map_above_horizon    = sky_map[EL>=0]

		
		# Re-structure beam coordinates
		az_array   = np.tile(AZ_beam,91)
		el_array   = np.repeat(EL_beam,360)
		az_el_original      = np.array([az_array, el_array]).T
		
		
		# Re-structure sky coordinates
		az_el_above_horizon = np.array([AZ_above_horizon, EL_above_horizon]).T


		# Interpolate beam coordinates to coordinates of the sky above the horizon
		beam_above_horizon = spi.griddata(az_el_original, beam_map, az_el_above_horizon, method='cubic')  # interpolated beam


		no_nan_array = np.ones(len(AZ_above_horizon)) - np.isnan(beam_above_horizon)
		index_no_nan = np.nonzero(no_nan_array)[0]


		# Antenna temperature
		convolution[i] = np.sum(beam_above_horizon[index_no_nan]*sky_map_above_horizon[index_no_nan])/np.sum(beam_above_horizon[index_no_nan])



	#convolution_array[k,:] = convolution



	# # Saving beam factor
	#np.savetxt(path_save + name_save + '_tant.txt', convolution_array)
	np.savetxt(path_save + name_save + '_tant.txt', convolution)
	np.savetxt(path_save + name_save + '_LST.txt',  LST)
	
	#np.savetxt(path_save + name_save + '_offset_scale.txt', offset_scale_array)
	
	



	return LST, convolution #convolution_array   # offset_scale_array, 














def isotropic_extragalactic_contribution(v_eval_MHz, model='gervasi2008'):
	
	
	if (model == 'gervasi2008') or (model == 'seiffert2011'):
		t_ref_K   =  0.88
		v_ref_MHz =  610
		beta      = -2.7
		
		t_eval_K  = t_ref_K * (v_eval_MHz/v_ref_MHz)**beta



	if model == 'guzman2011':
		t_ref_K   =  10**(7.66)
		v_ref_MHz =  1  
		beta      = -2.79
		
		t_eval_K  = t_ref_K * (v_eval_MHz/v_ref_MHz)**beta



	if model == 'vernstrom2011':     # Conservative power law
		t_ref_K   =  0.11
		v_ref_MHz =  1400  
		beta      = -2.28
		
		t_eval_K  = t_ref_K * (v_eval_MHz/v_ref_MHz)**beta		


		
	if model == 'subrahmanyan2013':    # MCMC results
		v_MHz          = np.array([150, 408, 1420])
		t_K            = np.array([28, 2.5, 0.12])
		par            = np.polyfit(np.log10(v_MHz), np.log10(t_K), 1)
		log10_t_eval_K = np.polyval(par, np.log10(v_eval_MHz))
		t_eval_K       = 10**(log10_t_eval_K)


	
	if model == 'seiffert2011':
		t_gervasi_K = np.copy(t_eval_K)
		
		t_ref_K   =  18.4
		v_ref_MHz =  310
		beta      = -2.57
		
		t_excess_K  = t_ref_K * (v_eval_MHz/v_ref_MHz)**beta
		t_eval_K    = t_gervasi_K + t_excess_K
		
		
		
	return t_eval_K









def chi_square_minimization():
	
	#data =  np.genfromtxt('')
	#data = np.
	simulation = np.genfromtxt(home_folder + '/DATA/EDGES/results/low_band/products/temperature_at_60MHz/low_minus2deg_guzman_60MHz_band10deg_inband2.5_outband2.6_tant.txt')
	
	sigma  = 10  # K
	noise  = np.random.normal(np.zeros(len(simulation)), sigma*np.ones(len(simulation)))
	data   = simulation + noise
	
	
	gain   = 1 + np.arange(-0.01, 0.011, 0.001)
	offset = np.arange(-10, 10.01, 0.01)
	X2     = np.zeros((len(gain), len(offset)))
	
	
	for j in range(len(gain)):
		for i in range(len(offset)):
			
			print(gain[j])
			
			model   = gain[j] * simulation + offset[i]
			X2[j,i] = np.sum(((data - model)/sigma)**2)
			
						
	return simulation, data, X2














