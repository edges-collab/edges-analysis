

import numpy as np
import matplotlib.pyplot as plt
import reflection_coefficient as rc
import edges as eg

from os import listdir, makedirs
from os.path import exists

import h5py



def SPAread(file_name):

	# Initial values
	power = np.zeros(0)
	freq  = np.zeros(0)

	DATEX         = 0
	GPS_TIME      = 0
	GPS_LON       = 0
	GPS_LAT       = 0
	RBW           = 0
	VBW           = 0
	INPUT_ATTEN   = 0
	PREAMP_SET    = 0
	DETECTION     = 0
	TRACE_AVERAGE = 0
	FREQ_STEP     = 0


	# Flags
	flag_DATETIME      = 0
	flag_GPS_DATETIME  = 0
	flag_GPS_LON       = 0
	flag_GPS_LAT       = 0
	flag_RBW           = 0
	flag_VBW           = 0 
	flag_INPUT_ATTEN   = 0
	flag_PREAMP_SET    = 0
	flag_DETECTION     = 0
	flag_TRACE_AVERAGE = 0
	flag_FREQ_STEP     = 0    


	with open(file_name) as f:

		for tline in f:
			# print(tline)


			# Settings

			# DATETIME
			if (len(tline) > 4) and (tline[0:4]=='DATE') and (flag_DATETIME == 0):

				for i in range(len(tline)):
					if tline[i] == '=':
						start = i + 1

				DATEX = tline[start:-1]
				flag_DATETIME = 1


			# GPS DATETIME
			if (len(tline) > 12) and (tline[0:12]=='GPS_FIX_TIME') and (flag_GPS_DATETIME == 0):

				for i in range(len(tline)):
					if tline[i] == '=':
						start = i + 1

				GPS_TIME = tline[start:-1]
				flag_GPS_DATETIME = 1           



			# GPS LON
			if (len(tline) > 17) and (tline[0:17]=='GPS_FIX_LONGITUDE') and (flag_GPS_LON == 0):

				for i in range(len(tline)):
					if tline[i] == '=':
						start = i + 1

				GPS_LON = tline[start:-1]
				flag_GPS_LON = 1            


			# GPS LAT
			if (len(tline) > 16) and (tline[0:16]=='GPS_FIX_LATITUDE') and (flag_GPS_LAT == 0):

				for i in range(len(tline)):
					if tline[i] == '=':
						start = i + 1

				GPS_LAT = tline[start:-1]
				flag_GPS_LAT = 1            


			# RBW
			if (len(tline) > 3) and (tline[0:3]=='RBW') and (flag_RBW == 0):

				for i in range(len(tline)):
					if tline[i] == '=':
						start = i + 1

				RBW = 1e6*float(tline[start:-1])
				flag_RBW = 1            


			# VBW
			if (len(tline) > 3) and (tline[0:3]=='VBW') and (flag_VBW == 0):

				for i in range(len(tline)):
					if tline[i] == '=':
						start = i + 1

				VBW = 1e6*float(tline[start:-1])
				flag_VBW = 1  


			# INPUT ATTEN
			if (len(tline) > 11) and (tline[0:11]=='INPUT_ATTEN') and (flag_INPUT_ATTEN == 0):

				for i in range(len(tline)):
					if tline[i] == '=':
						start = i + 1

				INPUT_ATTEN = float(tline[start:-1])
				flag_INPUT_ATTEN = 1  


			# PREAMP SET
			if (len(tline) > 10) and (tline[0:10]=='PREAMP_SET') and (flag_PREAMP_SET == 0):

				for i in range(len(tline)):
					if tline[i] == '=':
						start = i + 1

				PREAMP_SET = float(tline[start:-1])
				flag_PREAMP_SET = 1  


			# DETECTION
			if (len(tline) > 9) and (tline[0:9]=='DETECTION') and (flag_DETECTION == 0):

				for i in range(len(tline)):
					if tline[i] == '=':
						start = i + 1

				DETECTION = float(tline[start:-1])

				if DETECTION == 0:
					DETECTION = 'PEAK'
				elif DETECTION == 2:
					DETECTION = 'RMS'

				flag_DETECTION = 1  



			# TRACE_AVERAGE
			if (len(tline) > 13) and (tline[0:13]=='TRACE_AVERAGE') and (flag_TRACE_AVERAGE == 0):

				for i in range(len(tline)):
					if tline[i] == '=':
						start = i + 1

				TRACE_AVERAGE = float(tline[start:-1])        
				flag_TRACE_AVERAGE = 1  



			# FREQ_STEP
			if (len(tline) > 9) and (tline[0:9]=='FREQ_STEP') and (flag_FREQ_STEP == 0):

				for i in range(len(tline)):
					if tline[i] == '=':
						start = i + 1

				FREQ_STEP = float(tline[start:-1])        
				flag_FREQ_STEP = 1  






			# Power and frequency
			if (len(tline) > 0) and (tline[0:2]=='P_'):
				for i in range(len(tline)):
					if tline[i] == '=':
						start1 = i + 1
					elif tline[i] == ',':
						end1   = i - 1
						start2 = i + 1
					elif tline[i] == 'M':
						end2 = i - 1                   

				power_i = tline[start1:end1+1];
				freq_i  = tline[start2:end2+1];

				power = np.append(power, float(power_i))
				freq  = np.append(freq, float(freq_i))



	# Output
	data = np.array([freq, power])
	data = data.T

	meta_labels = ['LOCAL DATE', 'GPS_TIME', 'GPS_LON', 'GPS_LAT', 'RBW', 'VBW', 'INPUT_ATTEN', 'PREAMP_SET', 'DETECTION', 'TRACE_AVERAGE', 'FREQ_STEP']
	meta_values = [DATEX, GPS_TIME, GPS_LON, GPS_LAT, RBW, VBW, INPUT_ATTEN, PREAMP_SET, DETECTION, TRACE_AVERAGE, FREQ_STEP]
	meta        = [meta_labels, meta_values]

	return data, meta





def s11_antenna(antenna_file, f):

	# '/home/raul/DATA/MARI/etapa1/calibration_data/s-parameters_antenna/beam_vertical.s2p'
	# f in MHz

	# Loading measurement
	ant_raw  = np.genfromtxt(antenna_file,  skip_header=23)
	fs11     = ant_raw[:,0]*1e3
	ant_raw  = ant_raw[:,1] + 1j*ant_raw[:,2]

	ant_raw_abs = np.abs(ant_raw)
	ant_raw_ang = np.unwrap(np.angle(ant_raw))


	# Interpolation to desired frequency vector
	ant_abs = np.interp(f, fs11, ant_raw_abs)
	ant_ang = np.interp(f, fs11, ant_raw_ang)
	ant     = ant_abs * (np.cos(ant_ang) + 1j*np.sin(ant_ang)) 


	# S-parameters of cable + filter
	s11c, s12s21c, s22c, s11f, s12s21f, s22f = s_parameters_cable_filter(f)


	# Adding cable and filter to antenna S11
	ant_cable        = rc.gamma_shifted(s11c, s12s21c, s22c, ant)
	ant_cable_filter = rc.gamma_shifted(s11f, s12s21f, s22f, ant_cable)

	return ant_cable_filter








def s11_antenna_CHIME_and_AZUL(antenna_file, f):

	# '/home/raul/DATA/MARI/etapa1/calibration_data/s-parameters_antenna/beam_vertical.s2p'
	# f in MHz

	# Loading measurement
	ant_raw  = np.genfromtxt(antenna_file,  skip_header=23)
	fs11     = ant_raw[:,0]*1e3
	ant_raw  = ant_raw[:,1] + 1j*ant_raw[:,2]

	ant_raw_abs = np.abs(ant_raw)
	ant_raw_ang = np.unwrap(np.angle(ant_raw))


	# Interpolation to desired frequency vector
	ant_abs = np.interp(f, fs11, ant_raw_abs)
	ant_ang = np.interp(f, fs11, ant_raw_ang)
	ant     = ant_abs * (np.cos(ant_ang) + 1j*np.sin(ant_ang)) 

	return ant







def s_parameters_cable_filter(f):


	# Measurements at VNA input
	open_vna  = np.genfromtxt('/home/raul/DATA/MARI/etapa1/calibration_data/s-parameters_VNA/open_vna.s2p',  skip_header=23)
	short_vna = np.genfromtxt('/home/raul/DATA/MARI/etapa1/calibration_data/s-parameters_VNA/short_vna.s2p', skip_header=23)
	load_vna  = np.genfromtxt('/home/raul/DATA/MARI/etapa1/calibration_data/s-parameters_VNA/load_vna.s2p',  skip_header=23)

	fs11      = open_vna[:,0]*1e3
	open_vna  = open_vna[:,1]  + 1j*open_vna[:,2]
	short_vna = short_vna[:,1] + 1j*short_vna[:,2]
	load_vna  = load_vna[:,1]  + 1j*load_vna[:,2]




	# Measurements at end of Cable
	open_cable  = np.genfromtxt('/home/raul/DATA/MARI/etapa1/calibration_data/s-parameters_cable/open_cable_ant.s2p',  skip_header=23)
	short_cable = np.genfromtxt('/home/raul/DATA/MARI/etapa1/calibration_data/s-parameters_cable/short_cable_ant.s2p', skip_header=23)
	load_cable  = np.genfromtxt('/home/raul/DATA/MARI/etapa1/calibration_data/s-parameters_cable/load_cable_ant.s2p',  skip_header=23)

	open_cable  = open_cable[:,1]  + 1j*open_cable[:,2]
	short_cable = short_cable[:,1] + 1j*short_cable[:,2]
	load_cable  = load_cable[:,1]  + 1j*load_cable[:,2]




	# Measurements at end of Filter
	open_filter  = np.genfromtxt('/home/raul/DATA/MARI/etapa1/calibration_data/s-parameters_filter/open_filter.s2p',  skip_header=23)
	short_filter = np.genfromtxt('/home/raul/DATA/MARI/etapa1/calibration_data/s-parameters_filter/short_filter.s2p', skip_header=23)
	load_filter  = np.genfromtxt('/home/raul/DATA/MARI/etapa1/calibration_data/s-parameters_filter/load_filter.s2p',  skip_header=23)

	open_filter  = open_filter[:,1]  + 1j*open_filter[:,2]
	short_filter = short_filter[:,1] + 1j*short_filter[:,2]
	load_filter  = load_filter[:,1]  + 1j*load_filter[:,2]	



	# S-parameters Cable
	x, s11c, s12s21c, s22c = rc.de_embed(open_vna, short_vna, load_vna, open_cable, short_cable, load_cable, open_cable)


	# S-parameters Filter
	x, s11f, s12s21f, s22f = rc.de_embed(open_vna, short_vna, load_vna, open_filter, short_filter, load_filter, open_filter)



	# Interpolation to desired frequency vector
	s11c_abs_interp = np.interp(f, fs11, np.abs(s11c))
	s11c_ang_interp = np.interp(f, fs11, np.unwrap(np.angle(s11c)))
	s11c_interp     = s11c_abs_interp * (np.cos(s11c_ang_interp) + 1j*np.sin(s11c_ang_interp))

	s12s21c_abs_interp = np.interp(f, fs11, np.abs(s12s21c))
	s12s21c_ang_interp = np.interp(f, fs11, np.unwrap(np.angle(s12s21c)))
	s12s21c_interp     = s12s21c_abs_interp * (np.cos(s12s21c_ang_interp) + 1j*np.sin(s12s21c_ang_interp))

	s22c_abs_interp = np.interp(f, fs11, np.abs(s22c))
	s22c_ang_interp = np.interp(f, fs11, np.unwrap(np.angle(s22c)))
	s22c_interp     = s22c_abs_interp * (np.cos(s22c_ang_interp) + 1j*np.sin(s22c_ang_interp))	


	s11f_abs_interp = np.interp(f, fs11, np.abs(s11f))
	s11f_ang_interp = np.interp(f, fs11, np.unwrap(np.angle(s11f)))
	s11f_interp     = s11f_abs_interp * (np.cos(s11f_ang_interp) + 1j*np.sin(s11f_ang_interp))

	s12s21f_abs_interp = np.interp(f, fs11, np.abs(s12s21f))
	s12s21f_ang_interp = np.interp(f, fs11, np.unwrap(np.angle(s12s21f)))
	s12s21f_interp     = s12s21f_abs_interp * (np.cos(s12s21f_ang_interp) + 1j*np.sin(s12s21f_ang_interp))

	s22f_abs_interp = np.interp(f, fs11, np.abs(s22f))
	s22f_ang_interp = np.interp(f, fs11, np.unwrap(np.angle(s22f)))
	s22f_interp     = s22f_abs_interp * (np.cos(s22f_ang_interp) + 1j*np.sin(s22f_ang_interp))


	return s11c_interp, s12s21c_interp, s22c_interp, s11f_interp, s12s21f_interp, s22f_interp











def raw2level1(device, raw_path, normal_shifted='normal'):  

	#device: 'ant' or 'amb' or 'hot'

	#raw_path = '/home/raul/DATA/MARI/etapa1/raw/day1_paranal/beam_vertical_EW/'
	#cal_path = '/home/raul/DATA/MARI/etapa1/raw/day1_paranal/calibration/'

	# normal_shifted='shifted'  applies from Etapa2_3 on


	# Listing data
	full_list = listdir(raw_path)
	full_list.sort()

	n_50_70   = []
	n_70_90   = []
	n_90_110  = []
	n_110_130 = []
	n_130_150 = []
	n_150_170 = []
	n_170_190 = []
	n_190_210 = []   # only up to 200 MHz anyways

	#for i in range(10):
	#	print(i)

	for i in range(len(full_list)):
		#print(i)
		if (device in full_list[i]) and ('50-70' in full_list[i]):
			n_50_70.append(full_list[i])

		if (device in full_list[i]) and ('70-90' in full_list[i]):
			n_70_90.append(full_list[i])

		if (device in full_list[i]) and ('90-110' in full_list[i]):
			n_90_110.append(full_list[i])

		if (device in full_list[i]) and ('110-130' in full_list[i]):
			n_110_130.append(full_list[i])			

		if (device in full_list[i]) and ('130-150' in full_list[i]):
			n_130_150.append(full_list[i])	

		if (device in full_list[i]) and ('150-170' in full_list[i]):
			n_150_170.append(full_list[i])				

		if (device in full_list[i]) and ('170-190' in full_list[i]):
			n_170_190.append(full_list[i])		

		if (device in full_list[i]) and ('190-210' in full_list[i]):
			n_190_210.append(full_list[i])			


	l = min([len(n_50_70), len(n_70_90), len(n_90_110), len(n_110_130), len(n_130_150), len(n_150_170), len(n_170_190), len(n_190_210)])




	if l > 0:


		# Loading data
		
		# Array of dates from the 50-70 file
		date_array = np.zeros((l,6))
				
		for i in range(l):
					
			if i == 0:
				
				# 50-70 MHz
				d_50_70, m50 = SPAread(raw_path + n_50_70[i])
				f_50_70 = d_50_70[:,0]
				p_50_70 = d_50_70[:,1]
				p_50_70 = np.reshape(p_50_70, (1,-1))
				
				
				# 70-90 MHz
				d_70_90, m = SPAread(raw_path + n_70_90[i])
				f_70_90 = d_70_90[:,0]
				p_70_90 = d_70_90[:,1]
				p_70_90 = np.reshape(p_70_90, (1,-1))
				

				# 90-110 MHz
				d_90_110, m = SPAread(raw_path + n_90_110[i])
				f_90_110 = d_90_110[:,0]
				p_90_110 = d_90_110[:,1]
				p_90_110 = np.reshape(p_90_110, (1,-1))
				

				# 110-130 MHz
				d_110_130, m = SPAread(raw_path + n_110_130[i])
				f_110_130 = d_110_130[:,0]
				p_110_130 = d_110_130[:,1]
				p_110_130 = np.reshape(p_110_130, (1,-1))
				

				# 130-150 MHz
				d_130_150, m = SPAread(raw_path + n_130_150[i])
				f_130_150 = d_130_150[:,0]
				p_130_150 = d_130_150[:,1]
				p_130_150 = np.reshape(p_130_150, (1,-1))
				

				# 150-170 MHz
				d_150_170, m = SPAread(raw_path + n_150_170[i])
				f_150_170 = d_150_170[:,0]
				p_150_170 = d_150_170[:,1]
				p_150_170 = np.reshape(p_150_170, (1,-1))
				

				# 170-190 MHz
				d_170_190, m = SPAread(raw_path + n_170_190[i])
				f_170_190 = d_170_190[:,0]
				p_170_190 = d_170_190[:,1]
				p_170_190 = np.reshape(p_170_190, (1,-1))
				

				# 190-210 MHz
				d_190_210, m = SPAread(raw_path + n_190_210[i])
				f_190_210 = d_190_210[:,0]
				p_190_210 = d_190_210[:,1]
				p_190_210 = np.reshape(p_190_210, (1,-1))


			if i > 0:

				# 50-70
				d, m50 = SPAread(raw_path + n_50_70[i]);
				p = d[:,1]			
				p_50_70 = np.append(p_50_70, np.reshape(p,(1,-1)), axis=0)
								

				# 70-90
				d, m = SPAread(raw_path + n_70_90[i]);
				p = d[:,1]			
				p_70_90 = np.append(p_70_90, np.reshape(p,(1,-1)), axis=0)
				

				# 90-110
				d, m = SPAread(raw_path + n_90_110[i]);
				p = d[:,1]			
				p_90_110 = np.append(p_90_110, np.reshape(p,(1,-1)), axis=0)
				

				# 110-130
				d, m = SPAread(raw_path + n_110_130[i]);
				p = d[:,1]			
				p_110_130 = np.append(p_110_130, np.reshape(p,(1,-1)), axis=0)
				

				# 130-150
				d, m = SPAread(raw_path + n_130_150[i]);
				p = d[:,1]			
				p_130_150 = np.append(p_130_150, np.reshape(p,(1,-1)), axis=0)
				

				# 150-170
				d, m = SPAread(raw_path + n_150_170[i]);
				p = d[:,1]			
				p_150_170 = np.append(p_150_170, np.reshape(p,(1,-1)), axis=0)
				

				# 170-190
				d, m = SPAread(raw_path + n_170_190[i]);
				p = d[:,1]			
				p_170_190 = np.append(p_170_190, np.reshape(p,(1,-1)), axis=0)
				

				# 190-210
				d, m = SPAread(raw_path + n_190_210[i]);
				p = d[:,1]			
				p_190_210 = np.append(p_190_210, np.reshape(p,(1,-1)), axis=0)
				
			
			
			# Storing data information	
			date   = m50[1][0]
			year   = int(date[0:4])
			month  = int(date[5:7])
			day    = int(date[8:10])
			hour   = int(date[14:16])
			minute = int(date[17:19])
			second = int(date[20::])
			
		
			date_array[i,0] = year
			date_array[i,1] = month
			date_array[i,2] = day
			date_array[i,3] = hour
			date_array[i,4] = minute
			date_array[i,5] = second
			
			

		# Data concatenated and selected in range 50 - 200 MHz
		if normal_shifted == 'normal': # This applies to Etapa1, and Etapa 2_1, and 2_2
			power = np.hstack((p_50_70[:,0:-1], p_70_90[:,0:-1], p_90_110[:,0:-1], p_110_130[:,0:-1], p_130_150[:,0:-1], p_150_170[:,0:-1], p_170_190[:,0:-1], p_190_210[:,0:276]))
			freq  = np.hstack((f_50_70[0:-1], f_70_90[0:-1], f_90_110[0:-1], f_110_130[0:-1], f_130_150[0:-1], f_150_170[0:-1], f_170_190[0:-1], f_190_210[0:276]))

		if normal_shifted == 'shifted': # This applies to Etapa2_3 on
			power = np.hstack((p_50_70[:,15:], p_70_90[:,16:], p_90_110[:,16:], p_110_130[:,16:], p_130_150[:,16:], p_150_170[:,16:], p_170_190[:,16:], p_190_210[:,16:396]))
			freq  = np.hstack((f_50_70[15:], f_70_90[16:], f_90_110[16:], f_110_130[16:], f_130_150[16:], f_150_170[16:], f_170_190[16:], f_190_210[16:396]))		



	if l == 0:		

		print('NO DATA FOR ' + device)
		power = 0
		freq  = 0
		date_array = 0




	return freq, power, date_array

























def raw2level1_mari2_6_CHIME(device, raw_path):

	# (device, raw_path, normal_shifted='normal'):

	
	full_list = listdir(raw_path)
	full_list.sort()

	n_400_440 = []
	n_440_480 = []
	n_480_520 = []
	n_520_560 = []
	n_560_600 = []
	n_600_640 = []
	n_640_680 = []
	n_680_720 = []
	n_720_760 = []
	n_760_800 = []
	#n_800_840 = []
	#n_840_880 = []
	
	
	
	
	for i in range(len(full_list)):
		#print(i)
		if (device in full_list[i]) and ('400-440' in full_list[i]):
			n_400_440.append(full_list[i])

		if (device in full_list[i]) and ('440-480' in full_list[i]):
			n_440_480.append(full_list[i])

		if (device in full_list[i]) and ('480-520' in full_list[i]):
			n_480_520.append(full_list[i])

		if (device in full_list[i]) and ('520-560' in full_list[i]):
			n_520_560.append(full_list[i])			

		if (device in full_list[i]) and ('560-600' in full_list[i]):
			n_560_600.append(full_list[i])	

		if (device in full_list[i]) and ('600-640' in full_list[i]):
			n_600_640.append(full_list[i])				

		if (device in full_list[i]) and ('640-680' in full_list[i]):
			n_640_680.append(full_list[i])		

		if (device in full_list[i]) and ('680-720' in full_list[i]):
			n_680_720.append(full_list[i])
			
		if (device in full_list[i]) and ('720-760' in full_list[i]):
			n_720_760.append(full_list[i])
			
		if (device in full_list[i]) and ('760-800' in full_list[i]):
			n_760_800.append(full_list[i])
			
		#if (device in full_list[i]) and ('800-840' in full_list[i]):
		#	n_800_840.append(full_list[i])
			
		#if (device in full_list[i]) and ('840-880' in full_list[i]):
		#	n_840_880.append(full_list[i])


	l = min([len(n_400_440), len(n_440_480), len(n_480_520), len(n_520_560), len(n_560_600), len(n_600_640), len(n_640_680), len(n_680_720), len(n_720_760), len(n_760_800)]) #, len(n_800_840), len(n_840_880)
	


	if l > 0:

	
		# Loading data
		
		# Array of dates from the 400-440 file
		date_array = np.zeros((l,6))
				
		for i in range(l):
					
			if i == 0:
				
				# 400-440 MHz
				d_400_440, m400 = SPAread(raw_path + n_400_440[i])
				f_400_440 = d_400_440[:,0]
				p_400_440 = d_400_440[:,1]
				p_400_440 = np.reshape(p_400_440, (1,-1))
				
				# 440-480 MHz
				d_440_480, m = SPAread(raw_path + n_440_480[i])
				f_440_480 = d_440_480[:,0]
				p_440_480 = d_440_480[:,1]
				p_440_480 = np.reshape(p_440_480, (1,-1))
				
				# 480-520 MHz
				d_480_520, m = SPAread(raw_path + n_480_520[i])
				f_480_520 = d_480_520[:,0]
				p_480_520 = d_480_520[:,1]
				p_480_520 = np.reshape(p_480_520, (1,-1))
				
				# 520-560 MHz
				d_520_560, m = SPAread(raw_path + n_520_560[i])
				f_520_560 = d_520_560[:,0]
				p_520_560 = d_520_560[:,1]
				p_520_560 = np.reshape(p_520_560, (1,-1))
				
				# 560-600 MHz
				d_560_600, m = SPAread(raw_path + n_560_600[i])
				f_560_600 = d_560_600[:,0]
				p_560_600 = d_560_600[:,1]
				p_560_600 = np.reshape(p_560_600, (1,-1))
				
				# 600-640 MHz
				d_600_640, m = SPAread(raw_path + n_600_640[i])
				f_600_640 = d_600_640[:,0]
				p_600_640 = d_600_640[:,1]
				p_600_640 = np.reshape(p_600_640, (1,-1))
				
				# 640-680 MHz
				d_640_680, m = SPAread(raw_path + n_640_680[i])
				f_640_680 = d_640_680[:,0]
				p_640_680 = d_640_680[:,1]
				p_640_680 = np.reshape(p_640_680, (1,-1))
				
				# 680-720 MHz
				d_680_720, m = SPAread(raw_path + n_680_720[i])
				f_680_720 = d_680_720[:,0]
				p_680_720 = d_680_720[:,1]
				p_680_720 = np.reshape(p_680_720, (1,-1))
	
				# 720-760 MHz
				d_720_760, m = SPAread(raw_path + n_720_760[i])
				f_720_760 = d_720_760[:,0]
				p_720_760 = d_720_760[:,1]
				p_720_760 = np.reshape(p_720_760, (1,-1))
				
				# 760-800 MHz
				d_760_800, m = SPAread(raw_path + n_760_800[i])
				f_760_800 = d_760_800[:,0]
				p_760_800 = d_760_800[:,1]
				p_760_800 = np.reshape(p_760_800, (1,-1))
	
				## 800-840 MHz
				#d_800_840, m = SPAread(raw_path + n_800_840[i])
				#f_800_840 = d_800_840[:,0]
				#p_800_840 = d_800_840[:,1]
				#p_800_840 = np.reshape(p_800_840, (1,-1))
	
				## 840-880 MHz
				#d_840_880, m = SPAread(raw_path + n_840_880[i])
				#f_840_880 = d_840_880[:,0]
				#p_840_880 = d_840_880[:,1]
				#p_840_880 = np.reshape(p_840_880, (1,-1))
	

	
			if i > 0:
	
				# 400-440
				d, m400 = SPAread(raw_path + n_400_440[i]);
				p = d[:,1]			
				p_400_440 = np.append(p_400_440, np.reshape(p,(1,-1)), axis=0)	
	
				# 440-480
				d, m = SPAread(raw_path + n_440_480[i]);
				p = d[:,1]			
				p_440_480 = np.append(p_440_480, np.reshape(p,(1,-1)), axis=0)		

				# 480-520
				d, m = SPAread(raw_path + n_480_520[i]);
				p = d[:,1]			
				p_480_520 = np.append(p_480_520, np.reshape(p,(1,-1)), axis=0)		
	
				# 520-560
				d, m = SPAread(raw_path + n_520_560[i]);
				p = d[:,1]			
				p_520_560 = np.append(p_520_560, np.reshape(p,(1,-1)), axis=0)
				
				# 560-600
				d, m = SPAread(raw_path + n_560_600[i]);
				p = d[:,1]			
				p_560_600 = np.append(p_560_600, np.reshape(p,(1,-1)), axis=0)
				
				# 600-640
				d, m = SPAread(raw_path + n_600_640[i]);
				p = d[:,1]			
				p_600_640 = np.append(p_600_640, np.reshape(p,(1,-1)), axis=0)
				
				# 640-680
				d, m = SPAread(raw_path + n_640_680[i]);
				p = d[:,1]			
				p_640_680 = np.append(p_640_680, np.reshape(p,(1,-1)), axis=0)	
				
				# 680-720
				d, m = SPAread(raw_path + n_680_720[i]);
				p = d[:,1]			
				p_680_720 = np.append(p_680_720, np.reshape(p,(1,-1)), axis=0)	
	
				# 720-760
				d, m = SPAread(raw_path + n_720_760[i]);
				p = d[:,1]			
				p_720_760 = np.append(p_720_760, np.reshape(p,(1,-1)), axis=0)	
	
				# 760-800
				d, m = SPAread(raw_path + n_760_800[i]);
				p = d[:,1]			
				p_760_800 = np.append(p_760_800, np.reshape(p,(1,-1)), axis=0)	

				## 800-840
				#d, m = SPAread(raw_path + n_800_840[i]);
				#p = d[:,1]			
				#p_800_840 = np.append(p_800_840, np.reshape(p,(1,-1)), axis=0)	
	
				## 840-880
				#d, m = SPAread(raw_path + n_840_880[i]);
				#p = d[:,1]			
				#p_840_880 = np.append(p_840_880, np.reshape(p,(1,-1)), axis=0)	


	
			# Storing data information	
			date   = m400[1][0]
			year   = int(date[0:4])
			month  = int(date[5:7])
			day    = int(date[8:10])
			hour   = int(date[14:16])
			minute = int(date[17:19])
			second = int(date[20::])
			
		
			date_array[i,0] = year
			date_array[i,1] = month
			date_array[i,2] = day
			date_array[i,3] = hour
			date_array[i,4] = minute
			date_array[i,5] = second

		power = np.hstack((p_400_440[:,15:], p_440_480[:,16:], p_480_520[:,16:], p_520_560[:,16:], p_560_600[:,16:], p_600_640[:,16:], p_640_680[:,16:], p_680_720[:,16:], p_720_760[:,16:], p_760_800[:,16:])) #, p_800_840[:,16:], p_840_880[:,16:]
		freq  = np.hstack((f_400_440[15:], f_440_480[16:], f_480_520[16:], f_520_560[16:], f_560_600[16:], f_600_640[16:], f_640_680[16:], f_680_720[16:], f_720_760[16:], f_760_800[16:])) #, f_800_840[16:], f_840_880[16:]
		
		


	if l == 0:		

		print('NO DATA FOR ' + device)
		power = 0
		freq  = 0
		date_array = 0




	return freq, power, date_array









def raw2level1_mari2_6_AZUL(device, raw_path):

	
	full_list = listdir(raw_path)
	full_list.sort()

	n_750_790   = []
	n_790_830   = []
	n_830_870   = []
	n_870_910   = []
	n_910_950   = []
	n_950_990   = []
	n_990_1030  = []
	n_1030_1070 = []
	n_1070_1110 = []
	n_1110_1150 = []
	n_1150_1190 = []
	n_1190_1230 = []
	n_1230_1270 = []
	n_1270_1310 = []
	n_1310_1350 = []
	n_1350_1390 = []
	n_1390_1430 = []
	n_1430_1470 = []
	n_1470_1510 = []


	
	for i in range(len(full_list)):
		#print(i)
		if (device in full_list[i]) and ('750-790' in full_list[i]):
			n_750_790.append(full_list[i])
			
		if (device in full_list[i]) and ('790-830' in full_list[i]):
			n_790_830.append(full_list[i])
			
		if (device in full_list[i]) and ('830-870' in full_list[i]):
			n_830_870.append(full_list[i])
			
		if (device in full_list[i]) and ('870-910' in full_list[i]):
			n_870_910.append(full_list[i])
			
		if (device in full_list[i]) and ('910-950' in full_list[i]):
			n_910_950.append(full_list[i])
			
		if (device in full_list[i]) and ('950-990' in full_list[i]):
			n_950_990.append(full_list[i])
			
		if (device in full_list[i]) and ('990-1030' in full_list[i]):
			n_990_1030.append(full_list[i])
		
		if (device in full_list[i]) and ('1030-1070' in full_list[i]):
			n_1030_1070.append(full_list[i])
			
		if (device in full_list[i]) and ('1070-1110' in full_list[i]):
			n_1070_1110.append(full_list[i])
			
		if (device in full_list[i]) and ('1110-1150' in full_list[i]):
			n_1110_1150.append(full_list[i])
			
		if (device in full_list[i]) and ('1150-1190' in full_list[i]):
			n_1150_1190.append(full_list[i])
			
		if (device in full_list[i]) and ('1190-1230' in full_list[i]):
			n_1190_1230.append(full_list[i])
			
		if (device in full_list[i]) and ('1230-1270' in full_list[i]):
			n_1230_1270.append(full_list[i])
		
		if (device in full_list[i]) and ('1270-1310' in full_list[i]):
			n_1270_1310.append(full_list[i])
			
		if (device in full_list[i]) and ('1310-1350' in full_list[i]):
			n_1310_1350.append(full_list[i])
			
		if (device in full_list[i]) and ('1350-1390' in full_list[i]):
			n_1350_1390.append(full_list[i])
			
		if (device in full_list[i]) and ('1390-1430' in full_list[i]):
			n_1390_1430.append(full_list[i])
			
		if (device in full_list[i]) and ('1430-1470' in full_list[i]):
			n_1430_1470.append(full_list[i])
			
		if (device in full_list[i]) and ('1470-1510' in full_list[i]):
			n_1470_1510.append(full_list[i])


	l = min([len(n_750_790), len(n_790_830), len(n_830_870), len(n_870_910), len(n_910_950), len(n_950_990), len(n_990_1030), len(n_1030_1070), len(n_1070_1110), len(n_1110_1150), len(n_1150_1190), len(n_1190_1230), len(n_1230_1270), len(n_1270_1310), len(n_1310_1350), len(n_1350_1390), len(n_1390_1430), len(n_1430_1470), len(n_1470_1510)])
	


	if l > 0:

	
		# Loading data
		
		# Array of dates from the 750-790 file
		date_array = np.zeros((l,6))
				
		for i in range(l):
					
			if i == 0:
				
				# 750-790 MHz
				d_750_790, m750 = SPAread(raw_path + n_750_790[i])
				f_750_790 = d_750_790[:,0]
				p_750_790 = d_750_790[:,1]
				p_750_790 = np.reshape(p_750_790, (1,-1))
				
				# 790-830 MHz
				d_790_830, m = SPAread(raw_path + n_790_830[i])
				f_790_830 = d_790_830[:,0]
				p_790_830 = d_790_830[:,1]
				p_790_830 = np.reshape(p_790_830, (1,-1))

				# 830-870 MHz
				d_830_870, m = SPAread(raw_path + n_830_870[i])
				f_830_870 = d_830_870[:,0]
				p_830_870 = d_830_870[:,1]
				p_830_870 = np.reshape(p_830_870, (1,-1))
				
				# 870-910 MHz
				d_870_910, m = SPAread(raw_path + n_870_910[i])
				f_870_910 = d_870_910[:,0]
				p_870_910 = d_870_910[:,1]
				p_870_910 = np.reshape(p_870_910, (1,-1))

				# 910-950 MHz
				d_910_950, m = SPAread(raw_path + n_910_950[i])
				f_910_950 = d_910_950[:,0]
				p_910_950 = d_910_950[:,1]
				p_910_950 = np.reshape(p_910_950, (1,-1))
				
				# 950-990 MHz
				d_950_990, m = SPAread(raw_path + n_950_990[i])
				f_950_990 = d_950_990[:,0]
				p_950_990 = d_950_990[:,1]
				p_950_990 = np.reshape(p_950_990, (1,-1))

				# 990-1030 MHz
				d_990_1030, m = SPAread(raw_path + n_990_1030[i])
				f_990_1030 = d_990_1030[:,0]
				p_990_1030 = d_990_1030[:,1]
				p_990_1030 = np.reshape(p_990_1030, (1,-1))
				
				# 1030-1070 MHz
				d_1030_1070, m = SPAread(raw_path + n_1030_1070[i])
				f_1030_1070 = d_1030_1070[:,0]
				p_1030_1070 = d_1030_1070[:,1]
				p_1030_1070 = np.reshape(p_1030_1070, (1,-1))
				
				# 1070-1110 MHz
				d_1070_1110, m = SPAread(raw_path + n_1070_1110[i])
				f_1070_1110 = d_1070_1110[:,0]
				p_1070_1110 = d_1070_1110[:,1]
				p_1070_1110 = np.reshape(p_1070_1110, (1,-1))
				
				# 1110-1150 MHz
				d_1110_1150, m = SPAread(raw_path + n_1110_1150[i])
				f_1110_1150 = d_1110_1150[:,0]
				p_1110_1150 = d_1110_1150[:,1]
				p_1110_1150 = np.reshape(p_1110_1150, (1,-1))
				
				# 1150-1190 MHz
				d_1150_1190, m = SPAread(raw_path + n_1150_1190[i])
				f_1150_1190 = d_1150_1190[:,0]
				p_1150_1190 = d_1150_1190[:,1]
				p_1150_1190 = np.reshape(p_1150_1190, (1,-1))

				# 1190-1230 MHz
				d_1190_1230, m = SPAread(raw_path + n_1190_1230[i])
				f_1190_1230 = d_1190_1230[:,0]
				p_1190_1230 = d_1190_1230[:,1]
				p_1190_1230 = np.reshape(p_1190_1230, (1,-1))
				
				# 1230-1270 MHz
				d_1230_1270, m = SPAread(raw_path + n_1230_1270[i])
				f_1230_1270 = d_1230_1270[:,0]
				p_1230_1270 = d_1230_1270[:,1]
				p_1230_1270 = np.reshape(p_1230_1270, (1,-1))
				
				# 1270-1310 MHz
				d_1270_1310, m = SPAread(raw_path + n_1270_1310[i])
				f_1270_1310 = d_1270_1310[:,0]
				p_1270_1310 = d_1270_1310[:,1]
				p_1270_1310 = np.reshape(p_1270_1310, (1,-1))
				
				# 1310-1350 MHz
				d_1310_1350, m = SPAread(raw_path + n_1310_1350[i])
				f_1310_1350 = d_1310_1350[:,0]
				p_1310_1350 = d_1310_1350[:,1]
				p_1310_1350 = np.reshape(p_1310_1350, (1,-1))
				
				# 1350-1390 MHz
				d_1350_1390, m = SPAread(raw_path + n_1350_1390[i])
				f_1350_1390 = d_1350_1390[:,0]
				p_1350_1390 = d_1350_1390[:,1]
				p_1350_1390 = np.reshape(p_1350_1390, (1,-1))
				
				# 1390-1430 MHz
				d_1390_1430, m = SPAread(raw_path + n_1390_1430[i])
				f_1390_1430 = d_1390_1430[:,0]
				p_1390_1430 = d_1390_1430[:,1]
				p_1390_1430 = np.reshape(p_1390_1430, (1,-1))
				
				# 1430-1470 MHz
				d_1430_1470, m = SPAread(raw_path + n_1430_1470[i])
				f_1430_1470 = d_1430_1470[:,0]
				p_1430_1470 = d_1430_1470[:,1]
				p_1430_1470 = np.reshape(p_1430_1470, (1,-1))
				
				# 1470-1510 MHz
				d_1470_1510, m = SPAread(raw_path + n_1470_1510[i])
				f_1470_1510 = d_1470_1510[:,0]
				p_1470_1510 = d_1470_1510[:,1]
				p_1470_1510 = np.reshape(p_1470_1510, (1,-1))				



	
			if i > 0:
	
				# 750-790
				d, m750 = SPAread(raw_path + n_750_790[i]);
				p = d[:,1]			
				p_750_790 = np.append(p_750_790, np.reshape(p,(1,-1)), axis=0)	
	
				# 790-830
				d, m = SPAread(raw_path + n_790_830[i]);
				p = d[:,1]			
				p_790_830 = np.append(p_790_830, np.reshape(p,(1,-1)), axis=0)
				
				# 830-870
				d, m = SPAread(raw_path + n_830_870[i]);
				p = d[:,1]			
				p_830_870 = np.append(p_830_870, np.reshape(p,(1,-1)), axis=0)
				
				# 870-910
				d, m = SPAread(raw_path + n_870_910[i]);
				p = d[:,1]			
				p_870_910 = np.append(p_870_910, np.reshape(p,(1,-1)), axis=0)				
				
				# 910-950
				d, m = SPAread(raw_path + n_910_950[i]);
				p = d[:,1]			
				p_910_950 = np.append(p_910_950, np.reshape(p,(1,-1)), axis=0)
				
				# 950-990
				d, m = SPAread(raw_path + n_950_990[i]);
				p = d[:,1]			
				p_950_990 = np.append(p_950_990, np.reshape(p,(1,-1)), axis=0)				
				
				# 990-1030
				d, m = SPAread(raw_path + n_990_1030[i]);
				p = d[:,1]			
				p_990_1030 = np.append(p_990_1030, np.reshape(p,(1,-1)), axis=0)				
				
				# 1030-1070
				d, m = SPAread(raw_path + n_1030_1070[i]);
				p = d[:,1]			
				p_1030_1070 = np.append(p_1030_1070, np.reshape(p,(1,-1)), axis=0)				
				
				# 1070-1110
				d, m = SPAread(raw_path + n_1070_1110[i]);
				p = d[:,1]			
				p_1070_1110 = np.append(p_1070_1110, np.reshape(p,(1,-1)), axis=0)				

				# 1110-1150
				d, m = SPAread(raw_path + n_1110_1150[i]);
				p = d[:,1]			
				p_1110_1150 = np.append(p_1110_1150, np.reshape(p,(1,-1)), axis=0)

				# 1150-1190
				d, m = SPAread(raw_path + n_1150_1190[i]);
				p = d[:,1]			
				p_1150_1190 = np.append(p_1150_1190, np.reshape(p,(1,-1)), axis=0)

				# 1190-1230
				d, m = SPAread(raw_path + n_1190_1230[i]);
				p = d[:,1]			
				p_1190_1230 = np.append(p_1190_1230, np.reshape(p,(1,-1)), axis=0)

				# 1230-1270
				d, m = SPAread(raw_path + n_1230_1270[i]);
				p = d[:,1]			
				p_1230_1270 = np.append(p_1230_1270, np.reshape(p,(1,-1)), axis=0)

				# 1270-1310
				d, m = SPAread(raw_path + n_1270_1310[i]);
				p = d[:,1]			
				p_1270_1310 = np.append(p_1270_1310, np.reshape(p,(1,-1)), axis=0)

				# 1310-1350
				d, m = SPAread(raw_path + n_1310_1350[i]);
				p = d[:,1]			
				p_1310_1350 = np.append(p_1310_1350, np.reshape(p,(1,-1)), axis=0)

				# 1350-1390
				d, m = SPAread(raw_path + n_1350_1390[i]);
				p = d[:,1]			
				p_1350_1390 = np.append(p_1350_1390, np.reshape(p,(1,-1)), axis=0)

				# 1390-1430
				d, m = SPAread(raw_path + n_1390_1430[i]);
				p = d[:,1]			
				p_1390_1430 = np.append(p_1390_1430, np.reshape(p,(1,-1)), axis=0)

				# 1430-1470
				d, m = SPAread(raw_path + n_1430_1470[i]);
				p = d[:,1]			
				p_1430_1470 = np.append(p_1430_1470, np.reshape(p,(1,-1)), axis=0)

				# 1470-1510
				d, m = SPAread(raw_path + n_1470_1510[i]);
				p = d[:,1]			
				p_1470_1510 = np.append(p_1470_1510, np.reshape(p,(1,-1)), axis=0)



			# Storing data information	
			date   = m750[1][0]
			year   = int(date[0:4])
			month  = int(date[5:7])
			day    = int(date[8:10])
			hour   = int(date[14:16])
			minute = int(date[17:19])
			second = int(date[20::])
			
		
			date_array[i,0] = year
			date_array[i,1] = month
			date_array[i,2] = day
			date_array[i,3] = hour
			date_array[i,4] = minute
			date_array[i,5] = second
			
		power = np.hstack((p_750_790[:,15:], p_790_830[:,16:], p_830_870[:,16:], p_870_910[:,16:], p_910_950[:,16:], p_950_990[:,16:], p_990_1030[:,16:], p_1030_1070[:,16:], p_1070_1110[:,16:], p_1110_1150[:,16:], p_1150_1190[:,16:], p_1190_1230[:,16:], p_1230_1270[:,16:], p_1270_1310[:,16:], p_1310_1350[:,16:], p_1350_1390[:,16:], p_1390_1430[:,16:], p_1430_1470[:,16:], p_1470_1510[:,16:]))
		
		freq = np.hstack((f_750_790[15:], f_790_830[16:], f_830_870[16:], f_870_910[16:], f_910_950[16:], f_950_990[16:], f_990_1030[16:], f_1030_1070[16:], f_1070_1110[16:], f_1110_1150[16:], f_1150_1190[16:], f_1190_1230[16:], f_1230_1270[16:], f_1270_1310[16:], f_1310_1350[16:], f_1350_1390[16:], f_1390_1430[16:], f_1430_1470[16:], f_1470_1510[16:]))	



			
	if l == 0:		

		print('NO DATA FOR ' + device)
		power = 0
		freq  = 0
		date_array = 0			
			
			
	return freq, power, date_array
			

			
			
			
			




def calibration_script(path_data, path_calibration, path_antenna_file, path_save_file, save='no', normal_shifted_data='normal', normal_shifted_calibration='normal'):

	#freq, Pant_dBm = raw2level1('ant', '/home/raul/DATA/MARI/etapa1/raw/day1_paranal/beam_vertical_EW/')
	#freq, Pamb_dBm = raw2level1('amb', '/home/raul/DATA/MARI/etapa1/raw/day1_paranal/calibration/')
	#freq, Phot_dBm = raw2level1('hot', '/home/raul/DATA/MARI/etapa1/raw/day1_paranal/calibration/')

	


	# Data are received in the range 50 - 200 MHz
	# path_data = '/home/raul/DATA/MARI/etapa2_3/raw/beam_vertical_NS_2/'
	# path_calibration = '/home/raul/DATA/MARI/etapa2_3/raw/calibracion/'
	freq, Pant_dBm, date_ant = raw2level1('ant', path_data,        normal_shifted=normal_shifted_data)
	freq, Pamb_dBm, date_amb = raw2level1('amb', path_calibration, normal_shifted=normal_shifted_calibration)
	freq, Phot_dBm, date_hot = raw2level1('hot', path_calibration, normal_shifted=normal_shifted_calibration)	


	# dBm to linear power
	Pant = 10**(Pant_dBm/10)
	Pamb = 10**(Pamb_dBm/10)
	Phot = 10**(Phot_dBm/10)


	# Averaging calibration measurements
	Pamb_av = np.mean(Pamb, axis=0)
	Phot_av = np.mean(Phot, axis=0)


	# Model of hot load
	parameters_hot = np.genfromtxt('/home/raul/DATA/MARI/etapa1/calibration_data/temp/parameters_hot_mari.txt');
	Th             = np.polyval(parameters_hot, freq/200);


	# Conversion of antenna power to temperature
	Ta = 283
	X  = (Th - Ta) * (Pant - Pamb_av)/(Phot_av - Pamb_av)    +    Ta



	# S11 of LNA
	d = np.genfromtxt('/home/raul/DATA/MARI/etapa1/calibration_data/s-parameters_LNA/MARI_LNA.S2P', skip_header=8)
	fs11            = d[:,0]/1e6
	raw_s11_LNA_abs = 10**(d[:,1]/20)
	raw_s11_LNA_ang = d[:,2]

	s11_LNA_abs = np.interp(freq, fs11, raw_s11_LNA_abs)
	s11_LNA_ang = np.interp(freq, fs11, raw_s11_LNA_ang)

	s11_LNA = s11_LNA_abs * (np.cos((np.pi/180)*s11_LNA_ang) + 1j*np.sin((np.pi/180)*s11_LNA_ang))



	# S11 of antenna
	# path_antenna_file = '/home/raul/DATA/MARI/etapa1/calibration_data/s-parameters_antenna/beam_vertical.s2p'
	# path_antenna_file = '/home/raul/DATA/MARI/etapa2_3/raw/beam_vertical_ns_mari-2_3.s2p'
	s11_ant = s11_antenna(path_antenna_file, freq)
	s11_ant_abs = np.abs(s11_ant)



	# Calibration
	F = np.sqrt(1-np.abs(s11_LNA)**2) / (1 - s11_LNA*s11_ant)
	G = 1 - np.abs(s11_LNA)**2
	Tant = X * G/((1 - np.abs(s11_ant)**2) * np.abs(F)**2)



	# Plot from ipython
	# im = plt.imshow(ta, interpolation='none', aspect='auto', extent=[50,200,len(ta[:,1]),0]);im.set_clim([0,5000]);cb = plt.colorbar(im); cb.set_label('antenna temperature [K]');plt.xlabel('frequency [MHz]');plt.ylabel('trace')
	
	
	# Save data
	if save == 'yes':
		
		with h5py.File(path_save_file, 'w') as hf:
			hf.create_dataset('frequency',           data=freq)
			hf.create_dataset('antenna_temperature', data=Tant)
			hf.create_dataset('date',                data=date_ant)
			hf.create_dataset('power_antenna',       data=Pant_dBm)
			hf.create_dataset('power_ambient',       data=Pamb_dBm)
			hf.create_dataset('power_hot',           data=Phot_dBm)
			hf.create_dataset('s11_antenna',         data=s11_ant)

	return freq, Tant, date_ant














def calibration_script_mari2_6_CHIME(path_data, path_calibration, path_save_file, save='no'):


	# Data are received in the range 400-800 MHz
	# path_data        = '/home/raul/DATA/MARI/etapa2_6/raw/Conicyt_site/CHIME/'
	# path_calibration = '/home/raul/DATA/MARI/etapa2_6/raw/Conicyt_site/CHIME/calibration/'
	freq, Pant_dBm_all, date_ant_all = raw2level1_mari2_6_CHIME('ant', path_data)
	freq, Pamb_dBm, date_amb         = raw2level1_mari2_6_CHIME('amb', path_calibration)
	freq, Phot_dBm, date_hot         = raw2level1_mari2_6_CHIME('hot', path_calibration)
	
	
	# Removing first and last traces because of open cover of Faraday cage
	Pant_dBm = Pant_dBm_all[3:-3,:]
	date_ant = date_ant_all[3:-3,:]



	# dBm to linear power
	Pant = 10**(Pant_dBm/10)
	Pamb = 10**(Pamb_dBm/10)
	Phot = 10**(Phot_dBm/10)



	# Averaging calibration measurements
	Pamb_av = np.mean(Pamb, axis=0)
	Phot_av = np.mean(Phot, axis=0)


	# Conversion of antenna power to temperature
	Ta  = 283
	LdB = 10 # loss of attenuator in dB
	G   = 10**(-LdB/10)
	Tns = 4866.8 # noise figure of HP noise source ~ 12.6 dB at ~ 500 MHz 
	Th  = G*Tns + (1-G)*Ta
	X   = (Th - Ta) * (Pant - Pamb_av)/(Phot_av - Pamb_av)    +    Ta



	# Calibration
	
	# LNA
	# better than (because of the 1-dB attenuator)
	# at 400 MHz, 0.083, linear units 
	# at 800 MHz, 0.130, linear units
	s11_LNA = 0 # approximation
	
	# Antenna
	path_antenna_file = '/home/raul/DATA/MARI/etapa2_6/raw/radiosky/ant_chime_400-800.s2p'
	s11_ant           = s11_antenna_CHIME_and_AZUL(path_antenna_file, freq)
	
	# Calibration (first-order)	
	F = np.sqrt(1-np.abs(s11_LNA)**2) / (1 - s11_LNA*s11_ant)
	G = 1 - np.abs(s11_LNA)**2
	Tant = X * G/((1 - np.abs(s11_ant)**2) * np.abs(F)**2)


	# Removing self_generated RFI
	Tant_clean = self_RFI_remove(freq, Tant)



	# Save data
	if save == 'yes':
		
		with h5py.File(path_save_file, 'w') as hf:
			hf.create_dataset('frequency',           data=freq)
			hf.create_dataset('antenna_temperature', data=Tant_clean)
			hf.create_dataset('antenna_temperature_with_self_RFI', data=Tant)	
			hf.create_dataset('date',                data=date_ant)
			hf.create_dataset('power_antenna',       data=Pant_dBm)
			hf.create_dataset('power_ambient',       data=Pamb_dBm)
			hf.create_dataset('power_hot',           data=Phot_dBm)
			hf.create_dataset('s11_antenna',         data=s11_ant)	


	return freq, Tant, date_ant


















def calibration_script_mari2_6_AZUL(path_data, path_calibration, path_save_file, site, save='no'):


	# Data are received in the range 750-1500 MHz
	# path_data        = '/home/raul/DATA/MARI/etapa2_6/raw/Conicyt_site/Azul/'
	# path_calibration = '/home/raul/DATA/MARI/etapa2_6/raw/Conicyt_site/Azul/calibration/'
	freq, Pant_dBm_all, date_ant_all = raw2level1_mari2_6_AZUL('ant', path_data)
	freq, Pamb_dBm, date_amb         = raw2level1_mari2_6_AZUL('amb', path_calibration)
	freq, Phot_dBm, date_hot         = raw2level1_mari2_6_AZUL('hot', path_calibration)
	
	
	# Removing first and last traces because of open cover of Faraday cage
	if site == 'MARI':
		index    = np.array([1,2,6,7,8,9,10,11,12,13])
		Pant_dBm = Pant_dBm_all[index,:]
		date_ant = date_ant_all[index,:]		
		
	elif site == 'Conicyt':
		Pant_dBm = Pant_dBm_all[1::,:]
		date_ant = date_ant_all[1::,:]		



	# dBm to linear power
	Pant = 10**(Pant_dBm/10)
	Pamb = 10**(Pamb_dBm/10)
	Phot = 10**(Phot_dBm/10)



	# Averaging calibration measurements
	Pamb_av = np.mean(Pamb, axis=0)
	Phot_av = np.mean(Phot, axis=0)


	# Conversion of antenna power to temperature
	Ta  = 283
	LdB = 10 # loss of attenuator in dB
	G   = 10**(-LdB/10)
	Tns = 4819.5 # noise figure of HP noise source 12.56 dB at 1000 MHz 
	Th  = G*Tns + (1-G)*Ta
	X   = (Th - Ta) * (Pant - Pamb_av)/(Phot_av - Pamb_av)    +    Ta



	# Calibration
	
	# LNA
	# better than (because of the 1-dB attenuator)
	# at 400 MHz, 0.083, linear units 
	# at 800 MHz, 0.130, linear units
	s11_LNA = 0 # approximation
	
	# Antenna
	path_antenna_file = '/home/raul/DATA/MARI/etapa2_6/raw/radiosky/ant_azul_700-1500.s2p'
	s11_ant           = s11_antenna_CHIME_and_AZUL(path_antenna_file, freq)
	
	# Calibration (first-order)	
	F = np.sqrt(1-np.abs(s11_LNA)**2) / (1 - s11_LNA*s11_ant)
	G = 1 - np.abs(s11_LNA)**2
	Tant = X * G/((1 - np.abs(s11_ant)**2) * np.abs(F)**2)


	
	

	# Save data
	if save == 'yes':
		
		with h5py.File(path_save_file, 'w') as hf:
			hf.create_dataset('frequency',           data=freq)
			hf.create_dataset('antenna_temperature', data=Tant)
			hf.create_dataset('date',                data=date_ant)
			hf.create_dataset('power_antenna',       data=Pant_dBm)
			hf.create_dataset('power_ambient',       data=Pamb_dBm)
			hf.create_dataset('power_hot',           data=Phot_dBm)
			hf.create_dataset('s11_antenna',         data=s11_ant)	


	return freq, Tant, date_ant



























def batch_MARI2_calibration():
	
	# Etapa 2-1, part1
	path_data         = '/home/raul/DATA/MARI/etapa2_1/raw/part1/'
	path_calibration  = '/home/raul/DATA/MARI/etapa2_3/raw/calibration/'
	path_antenna_file = '/home/raul/DATA/MARI/etapa2_1/raw/beam_horizontal_tara.s2p'
	path_save_file    = '/home/raul/DATA/MARI/etapa2_1/level1/etapa2_1_horizontal_part1.h5'
	
	f, Ta, date = calibration_script(path_data, path_calibration, path_antenna_file, path_save_file, save='yes', normal_shifted_data='normal', normal_shifted_calibration='shifted')
	
		
		
	# Etapa 2-1, part2
	path_data         = '/home/raul/DATA/MARI/etapa2_1/raw/part2/'
	path_calibration  = '/home/raul/DATA/MARI/etapa2_3/raw/calibration/'
	path_antenna_file = '/home/raul/DATA/MARI/etapa2_1/raw/beam_horizontal_tara.s2p'
	path_save_file    = '/home/raul/DATA/MARI/etapa2_1/level1/etapa2_1_horizontal_part2.h5'
	
	f, Ta, date = calibration_script(path_data, path_calibration, path_antenna_file, path_save_file, save='yes', normal_shifted_data='normal', normal_shifted_calibration='shifted')
	
	
	
	# Etapa 2-1, part3
	path_data         = '/home/raul/DATA/MARI/etapa2_1/raw/part3/'
	path_calibration  = '/home/raul/DATA/MARI/etapa2_3/raw/calibration/'
	path_antenna_file = '/home/raul/DATA/MARI/etapa2_1/raw/beam_horizontal_tara.s2p'
	path_save_file    = '/home/raul/DATA/MARI/etapa2_1/level1/etapa2_1_horizontal_part3.h5'
	
	f, Ta, date = calibration_script(path_data, path_calibration, path_antenna_file, path_save_file, save='yes', normal_shifted_data='normal', normal_shifted_calibration='shifted')
		
		

	# Etapa 2-1, part4
	path_data         = '/home/raul/DATA/MARI/etapa2_1/raw/part4/'
	path_calibration  = '/home/raul/DATA/MARI/etapa2_3/raw/calibration/'
	path_antenna_file = '/home/raul/DATA/MARI/etapa2_1/raw/beam_horizontal_tara.s2p'
	path_save_file    = '/home/raul/DATA/MARI/etapa2_1/level1/etapa2_1_horizontal_part4.h5'
	
	f, Ta, date = calibration_script(path_data, path_calibration, path_antenna_file, path_save_file, save='yes', normal_shifted_data='normal', normal_shifted_calibration='shifted')
		
	
	
	
	
	
	
	





	# Etapa 2-2, part1
	path_data         = '/home/raul/DATA/MARI/etapa2_2/raw/part1/'
	path_calibration  = '/home/raul/DATA/MARI/etapa2_3/raw/calibration/'
	path_antenna_file = '/home/raul/DATA/MARI/etapa2_2/raw/tara2_beam_vertical_ns.s2p'
	path_save_file    = '/home/raul/DATA/MARI/etapa2_2/level1/etapa2_2_verticalNS_part1.h5'
	
	f, Ta, date = calibration_script(path_data, path_calibration, path_antenna_file, path_save_file, save='yes', normal_shifted_data='normal', normal_shifted_calibration='shifted')
	

	# Etapa 2-2, part2
	path_data         = '/home/raul/DATA/MARI/etapa2_2/raw/part2/'
	path_calibration  = '/home/raul/DATA/MARI/etapa2_3/raw/calibration/'
	path_antenna_file = '/home/raul/DATA/MARI/etapa2_2/raw/tara2_beam_vertical_ns.s2p'
	path_save_file    = '/home/raul/DATA/MARI/etapa2_2/level1/etapa2_2_verticalNS_part2.h5'
	
	f, Ta, date = calibration_script(path_data, path_calibration, path_antenna_file, path_save_file, save='yes', normal_shifted_data='normal', normal_shifted_calibration='shifted')
	

	# Etapa 2-2, part3
	path_data         = '/home/raul/DATA/MARI/etapa2_2/raw/part3/'
	path_calibration  = '/home/raul/DATA/MARI/etapa2_3/raw/calibration/'
	path_antenna_file = '/home/raul/DATA/MARI/etapa2_2/raw/tara2_beam_vertical_ns.s2p'
	path_save_file    = '/home/raul/DATA/MARI/etapa2_2/level1/etapa2_2_verticalNS_part3.h5'
	
	f, Ta, date = calibration_script(path_data, path_calibration, path_antenna_file, path_save_file, save='yes', normal_shifted_data='normal', normal_shifted_calibration='shifted')
	

	# Etapa 2-2, part4
	path_data         = '/home/raul/DATA/MARI/etapa2_2/raw/part4/'
	path_calibration  = '/home/raul/DATA/MARI/etapa2_3/raw/calibration/'
	path_antenna_file = '/home/raul/DATA/MARI/etapa2_2/raw/tara2_beam_vertical_ns.s2p'
	path_save_file    = '/home/raul/DATA/MARI/etapa2_2/level1/etapa2_2_verticalNS_part4.h5'
	
	f, Ta, date = calibration_script(path_data, path_calibration, path_antenna_file, path_save_file, save='yes', normal_shifted_data='normal', normal_shifted_calibration='shifted')
	

	# Etapa 2-2, part5
	path_data         = '/home/raul/DATA/MARI/etapa2_2/raw/part5/'
	path_calibration  = '/home/raul/DATA/MARI/etapa2_3/raw/calibration/'
	path_antenna_file = '/home/raul/DATA/MARI/etapa2_2/raw/tara2_beam_vertical_ns.s2p'
	path_save_file    = '/home/raul/DATA/MARI/etapa2_2/level1/etapa2_2_verticalNS_part5.h5'
	
	f, Ta, date = calibration_script(path_data, path_calibration, path_antenna_file, path_save_file, save='yes', normal_shifted_data='normal', normal_shifted_calibration='shifted')
	







	
	
	
	
	
	
	
	
	
	
	# Etapa 2-3, part1
	path_data         = '/home/raul/DATA/MARI/etapa2_3/raw/part1/'
	path_calibration  = '/home/raul/DATA/MARI/etapa2_3/raw/calibration/'
	path_antenna_file = '/home/raul/DATA/MARI/etapa2_3/raw/beam_vertical_ns_mari-2_3.s2p'
	path_save_file    = '/home/raul/DATA/MARI/etapa2_3/level1/etapa2_3_verticalNS_part1.h5'
	
	f, Ta, date = calibration_script(path_data, path_calibration, path_antenna_file, path_save_file, save='yes', normal_shifted_data='shifted', normal_shifted_calibration='shifted')
	
		
	# Etapa 2-3, part2
	path_data         = '/home/raul/DATA/MARI/etapa2_3/raw/part2/'
	path_calibration  = '/home/raul/DATA/MARI/etapa2_3/raw/calibration/'
	path_antenna_file = '/home/raul/DATA/MARI/etapa2_3/raw/beam_vertical_ns_mari-2_3.s2p'
	path_save_file    = '/home/raul/DATA/MARI/etapa2_3/level1/etapa2_3_verticalNS_part2.h5'
	
	f, Ta, date = calibration_script(path_data, path_calibration, path_antenna_file, path_save_file, save='yes', normal_shifted_data='shifted', normal_shifted_calibration='shifted')
	
	
	
	
	
	
	
	
	
	
	
	
	# Etapa 2-4, part1
	path_data         = '/home/raul/DATA/MARI/etapa2_4/raw/part1/'
	path_calibration  = '/home/raul/DATA/MARI/etapa2_4/raw/calibration/'
	path_antenna_file = '/home/raul/DATA/MARI/etapa2_4/raw/tara_horizontal.s2p'
	path_save_file    = '/home/raul/DATA/MARI/etapa2_4/level1/etapa2_4_horizontal_part1.h5'
	
	f, Ta, date = calibration_script(path_data, path_calibration, path_antenna_file, path_save_file, save='yes', normal_shifted_data='shifted', normal_shifted_calibration='shifted')
	
	
	# Etapa 2-4, part2
	path_data         = '/home/raul/DATA/MARI/etapa2_4/raw/part2/'
	path_calibration  = '/home/raul/DATA/MARI/etapa2_4/raw/calibration/'
	path_antenna_file = '/home/raul/DATA/MARI/etapa2_4/raw/tara_horizontal.s2p'
	path_save_file    = '/home/raul/DATA/MARI/etapa2_4/level1/etapa2_4_horizontal_part2.h5'
	
	f, Ta, date = calibration_script(path_data, path_calibration, path_antenna_file, path_save_file, save='yes', normal_shifted_data='shifted', normal_shifted_calibration='shifted')
		
	
	# Etapa 2-4, part3
	path_data         = '/home/raul/DATA/MARI/etapa2_4/raw/part3/'
	path_calibration  = '/home/raul/DATA/MARI/etapa2_4/raw/calibration/'
	path_antenna_file = '/home/raul/DATA/MARI/etapa2_4/raw/tara_horizontal.s2p'
	path_save_file    = '/home/raul/DATA/MARI/etapa2_4/level1/etapa2_4_horizontal_part3.h5'
	
	f, Ta, date = calibration_script(path_data, path_calibration, path_antenna_file, path_save_file, save='yes', normal_shifted_data='shifted', normal_shifted_calibration='shifted')
		



	
	
	
	
	
	# Etapa 2-5, part1 
	path_data         = '/home/raul/DATA/MARI/etapa2_5/raw/part1/'
	path_calibration  = '/home/raul/DATA/MARI/etapa2_5/raw/calibration/'
	path_antenna_file = '/home/raul/DATA/MARI/etapa2_5/raw/beam_vertical_ew.s2p'
	path_save_file    = '/home/raul/DATA/MARI/etapa2_5/level1/etapa2_5_verticalEW_part1_test.h5'
	
	f, Ta, date = calibration_script(path_data, path_calibration, path_antenna_file, path_save_file, save='yes', normal_shifted_data='shifted', normal_shifted_calibration='shifted')
		

	# Etapa 2-5, part2
	path_data         = '/home/raul/DATA/MARI/etapa2_5/raw/part2/'
	path_calibration  = '/home/raul/DATA/MARI/etapa2_5/raw/calibration/'
	path_antenna_file = '/home/raul/DATA/MARI/etapa2_5/raw/beam_vertical_ew.s2p'
	path_save_file    = '/home/raul/DATA/MARI/etapa2_5/level1/etapa2_5_verticalEW_part2.h5'
	
	f, Ta, date = calibration_script(path_data, path_calibration, path_antenna_file, path_save_file, save='yes', normal_shifted_data='shifted', normal_shifted_calibration='shifted')
		

	# Etapa 2-5, part3
	path_data         = '/home/raul/DATA/MARI/etapa2_5/raw/part3/'
	path_calibration  = '/home/raul/DATA/MARI/etapa2_5/raw/calibration/'
	path_antenna_file = '/home/raul/DATA/MARI/etapa2_5/raw/beam_vertical_ew.s2p'
	path_save_file    = '/home/raul/DATA/MARI/etapa2_5/level1/etapa2_5_verticalEW_part3.h5'
	
	f, Ta, date = calibration_script(path_data, path_calibration, path_antenna_file, path_save_file, save='yes', normal_shifted_data='shifted', normal_shifted_calibration='shifted')


	# Etapa 2-5, part4
	path_data         = '/home/raul/DATA/MARI/etapa2_5/raw/part4/'
	path_calibration  = '/home/raul/DATA/MARI/etapa2_5/raw/calibration/'
	path_antenna_file = '/home/raul/DATA/MARI/etapa2_5/raw/beam_vertical_ew.s2p'
	path_save_file    = '/home/raul/DATA/MARI/etapa2_5/level1/etapa2_5_verticalEW_part4.h5'
	
	f, Ta, date = calibration_script(path_data, path_calibration, path_antenna_file, path_save_file, save='yes', normal_shifted_data='shifted', normal_shifted_calibration='shifted')


	
	return 1










def batch_MARI2_6_calibration():

	
	# CHIME, MARI Site
	path_data         = '/home/raul/DATA/MARI/etapa2_6/raw/MARI_site/CHIME/'
	path_calibration  = '/home/raul/DATA/MARI/etapa2_6/raw/MARI_site/CHIME/calibration/'
	path_save_file    = '/home/raul/DATA/MARI/etapa2_6/level1/etapa2_6_CHIME_mari.h5'
	
	f, Ta, date = calibration_script_mari2_6_CHIME(path_data, path_calibration, path_save_file, save='yes')
	
	
	
	# CHIME, Conicyt Site
	path_data         = '/home/raul/DATA/MARI/etapa2_6/raw/Conicyt_site/CHIME/'
	path_calibration  = '/home/raul/DATA/MARI/etapa2_6/raw/Conicyt_site/CHIME/calibration/'
	path_save_file    = '/home/raul/DATA/MARI/etapa2_6/level1/etapa2_6_CHIME_conicyt.h5'
	
	f, Ta, date = calibration_script_mari2_6_CHIME(path_data, path_calibration, path_save_file, save='yes')




	# Azul, MARI Site
	path_data         = '/home/raul/DATA/MARI/etapa2_6/raw/MARI_site/Azul/'
	path_calibration  = '/home/raul/DATA/MARI/etapa2_6/raw/MARI_site/Azul/calibration/'
	path_save_file    = '/home/raul/DATA/MARI/etapa2_6/level1/etapa2_6_AZUL_mari.h5'

	f, Ta, date = calibration_script_mari2_6_AZUL(path_data, path_calibration, path_save_file, 'MARI', save='yes')



	# Azul, Conicyt Site
	path_data         = '/home/raul/DATA/MARI/etapa2_6/raw/Conicyt_site/Azul/'
	path_calibration  = '/home/raul/DATA/MARI/etapa2_6/raw/Conicyt_site/Azul/calibration/'
	path_save_file    = '/home/raul/DATA/MARI/etapa2_6/level1/etapa2_6_AZUL_conicyt.h5'
	
	f, Ta, date = calibration_script_mari2_6_AZUL(path_data, path_calibration, path_save_file, 'Conicyt', save='yes')



	return 1
	




























def read_level1(path_file, CHIME_self_RFI = 'no'):
	
	# path_file    = '/home/raul/DATA/MARI/etapa2_3/level1/etapa2_3_verticalNS_part1.h5'
	
	
	if CHIME_self_RFI == 'no':
		
		# Show keys (array names inside HDF5 file)
		with h5py.File(path_file,'r') as hf:
			print([key for key in hf.keys()])
			
			hf_freq = hf.get('frequency')
			freq    = np.array(hf_freq)
			
			hf_Ta   = hf.get('antenna_temperature')
			Ta      = np.array(hf_Ta)
			
			hf_date = hf.get('date')
			date    = np.array(hf_date)
			
			hf_pant = hf.get('power_antenna')
			pant    = np.array(hf_pant)
			
			hf_pamb = hf.get('power_ambient')
			pamb    = np.array(hf_pamb)
			
			hf_phot = hf.get('power_hot')
			phot    = np.array(hf_phot)
			
			hf_s11  = hf.get('s11_antenna')
			s11_ant = np.array(hf_s11)
			
			
		return freq, Ta, date, pant, pamb, phot, s11_ant
	
	
	elif CHIME_self_RFI == 'yes':
		
		# Show keys (array names inside HDF5 file)
		with h5py.File(path_file,'r') as hf:
			print([key for key in hf.keys()])
			
			hf_freq = hf.get('frequency')
			freq    = np.array(hf_freq)
			
			hf_Ta   = hf.get('antenna_temperature')
			Ta      = np.array(hf_Ta)
			
			hf_Ta_self_RFI = hf.get('antenna_temperature_with_self_RFI')
			Ta_self_RFI = np.array(hf_Ta_self_RFI)			
			
			hf_date = hf.get('date')
			date    = np.array(hf_date)
			
			hf_pant = hf.get('power_antenna')
			pant    = np.array(hf_pant)
			
			hf_pamb = hf.get('power_ambient')
			pamb    = np.array(hf_pamb)
			
			hf_phot = hf.get('power_hot')
			phot    = np.array(hf_phot)
			
			hf_s11  = hf.get('s11_antenna')
			s11_ant = np.array(hf_s11)
			
			
		return freq, Ta, Ta_self_RFI, date, pant, pamb, phot, s11_ant		
	






def CHIME_plot_average():


	D1 = read_level1('/home/raul/DATA/MARI/etapa2_6/level1/etapa2_6_CHIME_mari.h5',    CHIME_self_RFI='yes')
	D2 = read_level1('/home/raul/DATA/MARI/etapa2_6/level1/etapa2_6_CHIME_conicyt.h5', CHIME_self_RFI='yes')
	
	
	f1 = plt.figure(1)
	freq    = D1[0]
	mari    = 10*np.log10( np.mean(10**(D1[4]/10), axis=0))
	conicyt = 10*np.log10( np.mean(10**(D2[4]/10), axis=0))
	
	plt.subplot(2,1,1)
	plt.plot(freq, mari, 'b')
	plt.title('MARI Site')
	plt.ylabel('power [dBm]')
	plt.ylim([-110, -70])
	plt.grid()
	
	plt.subplot(2,1,2)
	plt.plot(freq, conicyt, 'r')
	plt.title('Conicyt Site')
	plt.ylabel('power [dBm]')
	plt.xlabel('frequency [MHz]')
	plt.ylim([-110, -70])
	plt.grid()
	
	
	
	
	f2 = plt.figure(2)
	
	plt.subplot(2,2,1)
	plt.plot(D1[0], np.mean(D1[2], axis=0), 'b');plt.ylim([0, 6000])
	plt.grid()
	plt.ylabel('antenna temperature [K]')
	plt.title('MARI Site')
	
	plt.subplot(2,2,3)
	plt.plot(D1[0], np.mean(D1[1], axis=0), 'b');plt.ylim([0, 6000])
	plt.grid()
	plt.xlabel('frequency [MHz]')
	plt.ylabel('antenna temperature [K]')
	
	plt.subplot(2,2,2)
	plt.plot(D2[0], np.mean(D2[2], axis=0), 'r');plt.ylim([0, 6000])
	plt.grid()
	plt.title('Conicyt Site')
	
	plt.subplot(2,2,4)
	plt.plot(D2[0], np.mean(D2[1], axis=0), 'r');plt.ylim([0, 6000])
	plt.grid()
	plt.xlabel('frequency [MHz]')
	
	
	return 1






def self_RFI_remove(f, Ta):
	
	Ta_clean = np.copy(Ta)
	df = 0.04 # MHz
	
	f0 = 520
	f1 = 572
	f2 = 624
	f3 = 650
	f4 = 676
	f5 = 700
	f6 = 702
	f7 = 728
	f8 = 754
	f9 = 780
	
	for i in range(len(f)):
		
		if (f[i] > (f0 - df)) and (f[i] < (f0 + df)):
			Ta_clean[:, i] = 0

			
		if (f[i] > (f1 - df)) and (f[i] < (f1 + df)):
			Ta_clean[:, i] = 0

			
		if (f[i] > (f2 - df)) and (f[i] < (f2 + df)):
			Ta_clean[:, i] = 0		
			
		
		if (f[i] > (f3 - df)) and (f[i] < (f3 + df)):
			Ta_clean[:, i] = 0			
	

		if (f[i] > (f4 - df)) and (f[i] < (f4 + df)):
			Ta_clean[:, i] = 0

			
		if (f[i] > (f5 - df)) and (f[i] < (f5 + df)):
			Ta_clean[:, i] = 0


		if (f[i] > (f6 - df)) and (f[i] < (f6 + df)):
			Ta_clean[:, i] = 0
	

		if (f[i] > (f7 - df)) and (f[i] < (f7 + df)):
			Ta_clean[:, i] = 0
			
		
		if (f[i] > (f8 - df)) and (f[i] < (f8 + df)):
			Ta_clean[:, i] = 0
			
			
		if (f[i] > (f9 - df)) and (f[i] < (f9 + df)):
			Ta_clean[:, i] = 0

	
	return Ta_clean







def plot_etapa1_snapshot():
	
	file_d1_H  = '/home/raul/DATA/MARI/etapa1/level2/day1_paranal/day1_paranal_H.txt'
	file_d1_NS = '/home/raul/DATA/MARI/etapa1/level2/day1_paranal/day1_paranal_NS.txt'
	file_d1_EW = '/home/raul/DATA/MARI/etapa1/level2/day1_paranal/day1_paranal_EW.txt'
	
	file_d2_H  = '/home/raul/DATA/MARI/etapa1/level2/day2_sierra_amarilla/day2_sierra_amarilla_H.txt'
	file_d2_NS = '/home/raul/DATA/MARI/etapa1/level2/day2_sierra_amarilla/day2_sierra_amarilla_NS.txt'
	file_d2_EW = '/home/raul/DATA/MARI/etapa1/level2/day2_sierra_amarilla/day2_sierra_amarilla_EW.txt'	
	
	file_d3_H  = '/home/raul/DATA/MARI/etapa1/level2/day3_llullaillaco/day3_llullaillaco_H.txt'
	file_d3_NS = '/home/raul/DATA/MARI/etapa1/level2/day3_llullaillaco/day3_llullaillaco_NS.txt'
	file_d3_EW = '/home/raul/DATA/MARI/etapa1/level2/day3_llullaillaco/day3_llullaillaco_EW.txt'	
	
	file_d4_H  = '/home/raul/DATA/MARI/etapa1/level2/day4_pampa_la_bola/day4_pampa_la_bola_H.txt'
	file_d4_NS = '/home/raul/DATA/MARI/etapa1/level2/day4_pampa_la_bola/day4_pampa_la_bola_NS.txt'
	file_d4_EW = '/home/raul/DATA/MARI/etapa1/level2/day4_pampa_la_bola/day4_pampa_la_bola_EW.txt'
	
	file_d5_H  = '/home/raul/DATA/MARI/etapa1/level2/day5_quisiquiro/day5_quisiquiro_H.txt'
	file_d5_NS = '/home/raul/DATA/MARI/etapa1/level2/day5_quisiquiro/day5_quisiquiro_NS.txt'
	file_d5_EW = '/home/raul/DATA/MARI/etapa1/level2/day5_quisiquiro/day5_quisiquiro_EW.txt'
	
	file_d6_H  = '/home/raul/DATA/MARI/etapa1/level2/day6_sasal_el_laco/day6_sasal_el_laco_H.txt'
	file_d6_NS = '/home/raul/DATA/MARI/etapa1/level2/day6_sasal_el_laco/day6_sasal_el_laco_NS.txt'
	file_d6_EW = '/home/raul/DATA/MARI/etapa1/level2/day6_sasal_el_laco/day6_sasal_el_laco_EW.txt'
	
	file_d7_H  = '/home/raul/DATA/MARI/etapa1/level2/day7_radio_sky/day7_radio_sky_H.txt'
	file_d7_NS = '/home/raul/DATA/MARI/etapa1/level2/day7_radio_sky/day7_radio_sky_NS.txt'
	file_d7_EW = '/home/raul/DATA/MARI/etapa1/level2/day7_radio_sky/day7_radio_sky_EW.txt'
	
	
	
	
	
	
	d1_H = np.genfromtxt(file_d1_H)
	d1_NS = np.genfromtxt(file_d1_NS)
	d1_EW = np.genfromtxt(file_d1_EW)
	
	d2_H_raw  = np.genfromtxt(file_d2_H)
	d2_NS     = np.genfromtxt(file_d2_NS)
	d2_EW_raw = np.genfromtxt(file_d2_EW)
	
	d3_H = np.genfromtxt(file_d3_H)
	d3_NS = np.genfromtxt(file_d3_NS)
	d3_EW = np.genfromtxt(file_d3_EW)
	
	d4_H      = np.genfromtxt(file_d4_H)
	d4_NS_raw = np.genfromtxt(file_d4_NS)
	d4_EW     = np.genfromtxt(file_d4_EW)
	
	d5_H = np.genfromtxt(file_d5_H)
	d5_NS = np.genfromtxt(file_d5_NS)
	d5_EW = np.genfromtxt(file_d5_EW)
	
	d6_H = np.genfromtxt(file_d6_H)
	d6_NS = np.genfromtxt(file_d6_NS)
	d6_EW = np.genfromtxt(file_d6_EW)
	
	d7_H = np.genfromtxt(file_d7_H)
	d7_NS = np.genfromtxt(file_d7_NS)
	d7_EW = np.genfromtxt(file_d7_EW)
	
	
	
	# Eliminating bad traces
	d2_H  = d2_H_raw[:,[0]+list(range(2,len(d2_H_raw[0,:])))]
	d2_EW = d2_EW_raw[:,[0,1,2,3,4,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]]
	
	d4_NS = d4_NS_raw[:,[0,1,2,3,4,5,6,7,9,10,11,12]]
	
	
	
	# Eliminating glitches
	index = list(range(0,2198))+list(range(2215,len(d1_H[:,0])))
	d1_H =  d1_H[index, :]
	d1_NS = d1_NS[index, :]
	d1_EW = d1_EW[index, :]
	
	 
	index = list(range(0,1380)) + list(range(1440,1644)) + list(range(1663,2750)) + list(range(2764,len(d7_H[:,0])))
	d7_H =  d7_H[index, :]
	d7_NS = d7_NS[index, :]
	d7_EW = d7_EW[index, :]
	
	
	index = list(range(0,2750)) + list(range(2764,len(d4_H[:,0])))
	d4_H =  d4_H[index, :]
	d4_NS = d4_NS[index, :]
	d4_EW = d4_EW[index, :]

	








	# Plot
	size_x = 12
	size_y = 15
	f1  = plt.figure(num=1, figsize=(size_x, size_y))
	c = 'r'
	
	plt.subplot(7,3,1)
	plt.plot(d1_H[:,0], np.mean(d1_H[:,1::], axis=1), c)
	plt.xlim([50,200])
	plt.ylim([0,6000])
	plt.xticks([50,75,100,125,150,175,200],'')
	plt.yticks([0,1000,2000,3000,4000,5000],['0','1000','2000','3000','4000','5000'])
	plt.ylabel('VLT\nT [K]')
	plt.grid()
	plt.title('Vertical')
	
	plt.subplot(7,3,2)
	plt.plot(d1_NS[:,0], np.mean(d1_NS[:,1::], axis=1), c)
	plt.xlim([50,200])
	plt.ylim([0,6000])
	plt.xticks([50,75,100,125,150,175,200],'')
	plt.yticks([0,1000,2000,3000,4000,5000],'')
	plt.grid()
	plt.title('Horizontal E-W')

	plt.subplot(7,3,3)
	plt.plot(d1_EW[:,0], np.mean(d1_EW[:,1::], axis=1), c)
	plt.xlim([50,200])
	plt.ylim([0,6000])	
	plt.xticks([50,75,100,125,150,175,200],'')
	plt.yticks([0,1000,2000,3000,4000,5000],'')
	plt.grid()
	plt.title('Horizontal N-S')







	plt.subplot(7,3,4)
	plt.plot(d2_H[:,0], np.mean(d2_H[:,1::], axis=1), c)
	plt.xlim([50,200])
	plt.ylim([0,6000])
	plt.xticks([50,75,100,125,150,175,200],'')
	plt.yticks([0,1000,2000,3000,4000,5000],['0','1000','2000','3000','4000','5000'])
	plt.ylabel('Sierra Amarilla\nT [K]')
	plt.grid()
	
	plt.subplot(7,3,5)
	plt.plot(d2_NS[:,0], np.mean(d2_NS[:,1::], axis=1), c)
	plt.xlim([50,200])
	plt.ylim([0,6000])	
	plt.xticks([50,75,100,125,150,175,200],'')
	plt.yticks([0,1000,2000,3000,4000,5000],'')
	plt.grid()
	
	plt.subplot(7,3,6)
	plt.plot(d2_EW[:,0], np.mean(d2_EW[:,1::], axis=1), c)
	plt.xlim([50,200])
	plt.ylim([0,6000])	
	plt.xticks([50,75,100,125,150,175,200],'')
	plt.yticks([0,1000,2000,3000,4000,5000],'')
	plt.grid()








	plt.subplot(7,3,7)
	plt.plot(d3_H[:,0], np.mean(d3_H[:,1::], axis=1), c)
	plt.xlim([50,200])
	plt.ylim([0,6000])
	plt.xticks([50,75,100,125,150,175,200],'')
	plt.yticks([0,1000,2000,3000,4000,5000],['0','1000','2000','3000','4000','5000'])
	plt.ylabel('Llullaillaco\nT [K]')
	plt.grid()
	
	plt.subplot(7,3,8)
	plt.plot(d3_NS[:,0], np.mean(d3_NS[:,1::], axis=1), c)
	plt.xlim([50,200])
	plt.ylim([0,6000])	
	plt.xticks([50,75,100,125,150,175,200],'')
	plt.yticks([0,1000,2000,3000,4000,5000],'')
	plt.grid()
	
	plt.subplot(7,3,9)
	plt.plot(d3_EW[:,0], np.mean(d3_EW[:,1::], axis=1), c)
	plt.xlim([50,200])
	plt.ylim([0,6000])	
	plt.xticks([50,75,100,125,150,175,200],'')
	plt.yticks([0,1000,2000,3000,4000,5000],'')
	plt.grid()


	
	
	
	
	
	
	plt.subplot(7,3,10)
	plt.plot(d7_H[:,0], np.mean(d7_H[:,1::], axis=1), c)
	plt.xlim([50,200])
	plt.ylim([0,6000])
	plt.xticks([50,75,100,125,150,175,200],'')
	plt.yticks([0,1000,2000,3000,4000,5000],['0','1000','2000','3000','4000','5000'])
	plt.ylabel('San Pedro de A.\nT [K]')
	plt.grid()
	
	plt.subplot(7,3,11)
	plt.plot(d7_NS[:,0], np.mean(d7_NS[:,1::], axis=1), c)
	plt.xlim([50,200])
	plt.ylim([0,6000])	
	plt.xticks([50,75,100,125,150,175,200],'')
	plt.yticks([0,1000,2000,3000,4000,5000],'')
	plt.grid()
	
	plt.subplot(7,3,12)
	plt.plot(d7_EW[:,0], np.mean(d7_EW[:,1::], axis=1), c)
	plt.xlim([50,200])
	plt.ylim([0,6000])	
	plt.xticks([50,75,100,125,150,175,200],'')
	plt.yticks([0,1000,2000,3000,4000,5000],'')
	plt.grid()	
	
	
	
	
	
	
	plt.subplot(7,3,13)
	plt.plot(d4_H[:,0], np.mean(d4_H[:,1::], axis=1), c)
	plt.xlim([50,200])
	plt.ylim([0,6000])
	plt.xticks([50,75,100,125,150,175,200],'')
	plt.yticks([0,1000,2000,3000,4000,5000],['0','1000','2000','3000','4000','5000'])
	plt.ylabel('ALMA\nT [K]')
	plt.grid()
	
	plt.subplot(7,3,14)
	plt.plot(d4_NS[:,0], np.mean(d4_NS[:,1::], axis=1), c)
	plt.xlim([50,200])
	plt.ylim([0,6000])	
	plt.xticks([50,75,100,125,150,175,200],'')
	plt.yticks([0,1000,2000,3000,4000,5000],'')
	plt.grid()
	
	plt.subplot(7,3,15)
	plt.plot(d4_EW[:,0], np.mean(d4_EW[:,1::], axis=1), c)
	plt.xlim([50,200])
	plt.ylim([0,6000])		
	plt.xticks([50,75,100,125,150,175,200],'')
	plt.yticks([0,1000,2000,3000,4000,5000],'')
	plt.grid()
	
	
	
	
	
	
	
	


	plt.subplot(7,3,16)
	plt.plot(d6_H[:,0], np.mean(d6_H[:,1::], axis=1), c)
	plt.xlim([50,200])
	plt.ylim([0,6000])
	plt.xticks([50,75,100,125,150,175,200],'')
	plt.yticks([0,1000,2000,3000,4000,5000],['0','1000','2000','3000','4000','5000'])
	plt.ylabel('Sasal el Laco\nT [K]')
	plt.grid()
	
	plt.subplot(7,3,17)
	plt.plot(d6_NS[:,0], np.mean(d6_NS[:,1::], axis=1), c)
	plt.xlim([50,200])
	plt.ylim([0,6000])	
	plt.xticks([50,75,100,125,150,175,200],'')
	plt.yticks([0,1000,2000,3000,4000,5000],'')
	plt.grid()
	
	plt.subplot(7,3,18)
	plt.plot(d6_EW[:,0], np.mean(d6_EW[:,1::], axis=1), c)
	plt.xlim([50,200])
	plt.ylim([0,6000])	
	plt.xticks([50,75,100,125,150,175,200],'')
	plt.yticks([0,1000,2000,3000,4000,5000],'')
	plt.grid()














	plt.subplot(7,3,19)
	plt.plot(d5_H[:,0], np.mean(d5_H[:,1::], axis=1), c)
	plt.xlim([50,200])
	plt.ylim([0,6000])
	plt.xticks([50,75,100,125,150,175,200])
	plt.yticks([0,1000,2000,3000,4000,5000],['0','1000','2000','3000','4000','5000'])
	plt.ylabel('Quisiquiro\nT [K]')
	plt.xlabel('frequency [MHz]')
	plt.grid()
	
	plt.subplot(7,3,20)
	plt.plot(d5_NS[:,0], np.mean(d5_NS[:,1::], axis=1), c)
	plt.xlim([50,200])
	plt.ylim([0,6000])	
	plt.xticks([50,75,100,125,150,175,200])
	plt.yticks([0,1000,2000,3000,4000,5000],'')
	plt.xlabel('frequency [MHz]')
	plt.grid()
	
	plt.subplot(7,3,21)
	plt.plot(d5_EW[:,0], np.mean(d5_EW[:,1::], axis=1), c)
	plt.xlim([50,200])
	plt.ylim([0,6000])	
	plt.xticks([50,75,100,125,150,175,200])
	plt.yticks([0,1000,2000,3000,4000,5000],'')
	plt.xlabel('frequency [MHz]')
	plt.grid()
	
	
	path_plot_save = '/home/raul/DATA/MARI/plots/'
	plt.savefig(path_plot_save + 'etapa1_snapshot.pdf', bbox_inches='tight')
	plt.close()	
	plt.close()	
	
	
	
	
	
	
	
	
	
	
	# Plot
	size_x2 = 7
	size_y2 = 8
	f2  = plt.figure(num=2, figsize=(size_x2, size_y2))
	c = 'r'
	
	plt.subplot(4,1,1)
	plt.plot(d4_H[:,0], np.mean(d4_H[:,1::], axis=1), c)
	plt.xlim([50,200])
	plt.ylim([0,6000])
	plt.xticks([50,75,100,125,150,175,200],'')
	plt.yticks([0,1000,2000,3000,4000,5000],['0','1000','2000','3000','4000','5000'])
	plt.ylabel('ALMA\nT [K]')
	plt.grid()
	plt.text(151, 4200, 'VERTICAL\nPOLARIZATION')
	
	plt.subplot(4,1,2)
	plt.plot(d4_NS[:,0], np.mean(d4_NS[:,1::], axis=1), c)
	plt.xlim([50,200])
	plt.ylim([0,6000])	
	plt.xticks([50,75,100,125,150,175,200],'')
	plt.yticks([0,1000,2000,3000,4000,5000],['0','1000','2000','3000','4000','5000'])
	plt.ylabel('ALMA\nT [K]')
	plt.grid()
	plt.text(151, 4200, 'HORIZONAL E-W\nPOLARIZATION')
	
	plt.subplot(4,1,3)
	plt.plot(d5_H[:,0], np.mean(d5_H[:,1::], axis=1), c)
	plt.xlim([50,200])
	plt.ylim([0,6000])
	plt.xticks([50,75,100,125,150,175,200])
	plt.yticks([0,1000,2000,3000,4000,5000],['0','1000','2000','3000','4000','5000'])
	plt.ylabel('Quisiquiro\nT [K]')
	plt.xlabel('frequency [MHz]')
	plt.grid()
	plt.text(76, 4200, 'VERTICAL\nPOLARIZATION')
	
	plt.subplot(4,1,4)
	plt.plot(d5_NS[:,0], np.mean(d5_NS[:,1::], axis=1), c)
	plt.xlim([50,200])
	plt.ylim([0,6000])	
	plt.xticks([50,75,100,125,150,175,200])
	plt.yticks([0,1000,2000,3000,4000,5000],['0','1000','2000','3000','4000','5000'])
	plt.ylabel('Quisiquiro\nT [K]')
	plt.xlabel('frequency [MHz]')
	plt.grid()
	plt.text(76, 4200, 'HORIZONAL E-W\nPOLARIZATION')
	
	path_plot_save = '/home/raul/DATA/MARI/plots/'
	plt.savefig(path_plot_save + 'etapa1_snapshot_2x2.pdf', bbox_inches='tight')
	plt.close()	
	plt.close()	
	
	
	
	return 1

























def plot_etapa2():
	
	
	# None is good
	file_11 = '/home/raul/DATA/MARI/etapa2_1/level1/etapa2_1_horizontal_part1.h5'
	file_12 = '/home/raul/DATA/MARI/etapa2_1/level1/etapa2_1_horizontal_part2.h5'
	file_13 = '/home/raul/DATA/MARI/etapa2_1/level1/etapa2_1_horizontal_part3.h5'
	file_14 = '/home/raul/DATA/MARI/etapa2_1/level1/etapa2_1_horizontal_part4.h5'
		
	file_21 = '/home/raul/DATA/MARI/etapa2_2/level1/etapa2_2_verticalNS_part1.h5'
	file_22 = '/home/raul/DATA/MARI/etapa2_2/level1/etapa2_2_verticalNS_part2.h5'
	file_23 = '/home/raul/DATA/MARI/etapa2_2/level1/etapa2_2_verticalNS_part3.h5'
	file_24 = '/home/raul/DATA/MARI/etapa2_2/level1/etapa2_2_verticalNS_part4.h5'	
	file_25 = '/home/raul/DATA/MARI/etapa2_2/level1/etapa2_2_verticalNS_part4.h5'
	
	file_31 = '/home/raul/DATA/MARI/etapa2_3/level1/etapa2_3_verticalNS_part1.h5'
	file_32 = '/home/raul/DATA/MARI/etapa2_3/level1/etapa2_3_verticalNS_part2.h5'
	
	file_41 = '/home/raul/DATA/MARI/etapa2_4/level1/etapa2_4_horizontal_part1.h5'
	file_42 = '/home/raul/DATA/MARI/etapa2_4/level1/etapa2_4_horizontal_part2.h5'
	file_43 = '/home/raul/DATA/MARI/etapa2_4/level1/etapa2_4_horizontal_part3.h5'
	
	file_51 = '/home/raul/DATA/MARI/etapa2_5/level1/etapa2_5_verticalEW_part1.h5'
	file_52 = '/home/raul/DATA/MARI/etapa2_5/level1/etapa2_5_verticalEW_part2.h5'
	file_53 = '/home/raul/DATA/MARI/etapa2_5/level1/etapa2_5_verticalEW_part3.h5'
	file_54 = '/home/raul/DATA/MARI/etapa2_5/level1/etapa2_5_verticalEW_part4.h5'
	
	
	
	
	freq, T11, date11, pant, pamb, phot, s11_ant = read_level1(file_11)
	freq, T12, date12, pant, pamb, phot, s11_ant = read_level1(file_12)
	freq, T13, date13, pant, pamb, phot, s11_ant = read_level1(file_13)
	freq, T14, date14, pant, pamb, phot, s11_ant = read_level1(file_14)
	
	freq, T21, date21, pant, pamb, phot, s11_ant = read_level1(file_21)
	freq, T22, date22, pant, pamb, phot, s11_ant = read_level1(file_22)
	freq, T23, date23, pant, pamb, phot, s11_ant = read_level1(file_23)
	freq, T24, date24, pant, pamb, phot, s11_ant = read_level1(file_24)	
	freq, T25, date25, pant, pamb, phot, s11_ant = read_level1(file_25)
	
	freq, T31, date31, pant, pamb, phot, s11_ant = read_level1(file_31)
	freq, T32, date32, pant, pamb, phot, s11_ant = read_level1(file_32)	
	
	freq, T41, date41, pant, pamb, phot, s11_ant = read_level1(file_41)
	freq, T42, date42, pant, pamb, phot, s11_ant = read_level1(file_42)	
	freq, T43, date43, pant, pamb, phot, s11_ant = read_level1(file_43)
	
	freq, T51, date51, pant, pamb, phot, s11_ant = read_level1(file_51)
	freq, T52, date52, pant, pamb, phot, s11_ant = read_level1(file_52)	
	freq, T53, date53, pant, pamb, phot, s11_ant = read_level1(file_53)	
	freq, T54, date54, pant, pamb, phot, s11_ant = read_level1(file_54)
	
	
	
	
	
	
	
	# Etapa2-2
	specT21 = T21[275::,:]
	#avT21   = np.mean(T21[275::,:], axis=0)
	
	index   = list(range(0,150)) + list(range(370,len(T22[:,0])))
	specT22 = T22[index,:]	
	#avT22 = np.mean(T22[index,:], axis=0)
	
	specT23 = T23[0:120,:]
	#avT23 = np.mean(T23[0:120,:], axis=0)
	
	index   = list(range(70,120)) + list(range(390,len(T24[:,0])))
	specT24 = T24[index,:]
	#avT24 = np.mean(T24[index,:], axis=0)
	
	index   = list(range(70,130)) + list(range(380,len(T25[:,0])))
	specT25 = T25[index,:]
	#avT25 = np.mean(T25[index,:], axis=0)	
	
	
	

	# Etapa2-3
	specT31 = T31[390::,:]
	#avT31 = np.mean(T31[390::,:], axis=0)
	
	index = list(range(0,150)) + list(range(450,700)) + list(range(1050,1300))
	specT32 = T32[index,:]
	#avT32 = np.mean(T32[index,:], axis=0)
	





	# Etapa2-5
	#avT51 = np.mean(T51[120:390,:], axis=0)
	#avT52 = np.mean(T52[230:470,:], axis=0)
	#avT53 = np.mean(T53[230:400,:], axis=0)
	
	specT51 = T51[120:390,:]
	specT52 = T52[230:470,:]
	specT53 = T53[230:400,:]
	
	
	
	index = list(range(100,320)) + list(range(830,950))
	#avT54 = np.mean(T54[index,:], axis=0)
	
	specT54 = T54[120:390,:]
	
	
	
	TT = np.concatenate((specT21, specT22, specT23, specT24, specT25, specT31, specT32, specT51, specT52, specT53, specT54), axis=0)
	
	# avT21, avT22, avT23, avT24, avT25, avT31, avT32,
	#kk  = np.array([avT51, avT52, avT53, avT54])
	#kkm = np.mean(kk, axis=0)
	#plt.plot(freq, avT1)
	
	
	
	TT_noRFI = np.zeros((len(TT[:,0]), len(TT[0,:])))
	WW = np.zeros((len(TT[:,0]), len(TT[0,:])))
	for i in range(len(TT[:,0])):
		print(i)
		TT_noRFI[i,:], WW[i,:] = eg.RFI_cleaning_std(freq, TT[i,:], np.ones(len(freq)), df_MHz = 5, npar = 4, n_sigma = 4)
		
		
	#TT_EW_noRFI = np.zeros((len(TT_EW[:,0]), len(TT_EW[0,:])))
	#WW_EW = np.zeros((len(TT_EW[:,0]), len(TT_EW[0,:])))
	#for i in range(len(TT_EW[:,0])):
		#print(i)
		#TT_EW_noRFI[i,:], WW_EW[i,:] = eg.RFI_cleaning_std(freq, TT_EW[i,:], np.ones(len(freq)), df_MHz = 5, npar = 4, n_sigma = 4)	

	
	
	plt.close()
	plt.close()
	plt.close()
	plt.close()
	plt.close()
	

	
	# Plot
	size_x = 7
	size_y = 5.5
	f1  = plt.figure(num=1, figsize=(size_x, size_y))
	c = 'r'
		
	plt.subplot(2,1,1)
	pp     = eg.fit_polynomial_fourier('fourier', freq/200, TT_noRFI[1211,:], 451, Weights=np.ones(len(freq)))
	model  = eg.model_evaluate('fourier', pp[0], freq/200)
	R      = TT_noRFI[1211,:] - model
	RR     = R[15::]
	freqRR = freq[15::]
	RR[2384] = 0
	RR[2386] = 0
	RR[2397] = 0
	plt.plot(freqRR, RR+90)
	plt.xlim([50,200])
	plt.ylim([-2000, 2000])
	plt.xticks([50,75,100,125,150,175,200])
	plt.grid()
	plt.ylabel('single-spectrum noise [K]')
	
	plt.subplot(2,1,2)
	plt.semilogy(freq, 100*np.sum(np.abs(WW-1)/len(WW[:,0]), axis=0) )
	plt.xlim([50,200])
	plt.ylim([0.5, 100])
	plt.xticks([50,75,100,125,150,175,200])
	plt.yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100])
	plt.grid()
	plt.xlabel('frequency [MHz]')
	plt.ylabel('occupancy [%]')
	

	
	path_plot_save = '/home/raul/DATA/MARI/plots/'
	plt.savefig(path_plot_save + 'MARI_occupancy.pdf', bbox_inches='tight')
	plt.close()	
	plt.close()	
	
	
	return freq, TT, WW, RR