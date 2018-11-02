
import numpy as np
import scipy.io as sio

import matplotlib.pyplot as plt
import datetime as dt

import astropy.time as apt
import ephem as eph	# to install, at the bash terminal type $ conda install ephem










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









def utc2lst(utc_time_array, LONG_float):
	"""
	Last modification: May 29, 2015.

	This function converts an Nx6 array of floats or integers representing UTC date/time, to a 1D array of LST date/time.

	Definition:
	LST = utc2lst(utc_time_array, LONG_float)

	Input parameters:
	utc_time_array: Nx6 array of floats or integers, where each row is of the form [yyyy, mm, dd, HH, MM, SS]. It can also be a 6-element 1D array.
	LONG_float: terrestrial longitude of observatory (float) in degrees.

	Output parameters:
	LST: 1D array of LST dates/times

	Usage:
	LST = utc2lst(utc_time_array, -27.5)
	"""

	# converting input array to "int"
	uta = utc_time_array.astype(int)



	# compatibility for 1D inputs (only 1 array of date/time), and 2D (N arrays of date/time)
	if uta.ndim == 1:
		len_array = 1
	if uta.ndim == 2:
		len_array = len(uta)



	# converting UTC to LST
	LST = np.zeros(len_array)

	for i in range(len_array):
		if uta.ndim == 1:
			yyyy = uta[0]
			mm   = uta[1]
			dd   = uta[2]
			HH   = uta[3]
			MM   = uta[4]
			SS   = uta[5]

		elif uta.ndim == 2:
			yyyy = uta[i,0]
			mm   = uta[i,1]
			dd   = uta[i,2]
			HH   = uta[i,3]
			MM   = uta[i,4]
			SS   = uta[i,5]


		# time stamp in python "datetime" format
		udt = dt.datetime(yyyy, mm, dd, HH, MM, SS)

		# python "datetime" to astropy "Time" format
		t = apt.Time(udt, format='datetime', scale='utc')

		# necessary approximation to compute sidereal time
		t.delta_ut1_utc = 0

		# LST at longitude LONG_float, in degrees
		LST_object = t.sidereal_time('apparent', str(LONG_float)+'d', model='IAU2006A')
		LST[i] = LST_object.value


	return LST










def SUN_MOON_azel(LAT, LON, UTC_array):
	# 
	# Local coordinates of the Sun using Astropy
	#
	# EDGES_lat_deg = -26.7
	# EDGES_lon_deg = 116.6
	#	


	# Observation coordinates
	OBS_lat_deg = str(LAT) 
	OBS_lon_deg = str(LON) 
	#print(' ')
	#print('Observation Coordinates: ' + 'LAT: ' + OBS_lat_deg + ' LON: ' + OBS_lon_deg)
	#print('------------------------')

	OBS_location     = eph.Observer()
	OBS_location.lat = OBS_lat_deg 
	OBS_location.lon = OBS_lon_deg


	# Compute local coordinates of Sun and Moon
	SH = UTC_array.shape

	if len(SH) == 1:
		coord    = np.zeros(4)

		OBS_location.date = dt.datetime(UTC_array[0], UTC_array[1], UTC_array[2], UTC_array[3], UTC_array[4], UTC_array[5])
		Sun  = eph.Sun(OBS_location)
		Moon = eph.Moon(OBS_location)

		coord[0] = (180/np.pi)*eph.degrees(Sun.az)
		coord[1] = (180/np.pi)*eph.degrees(Sun.alt)		
		coord[2] = (180/np.pi)*eph.degrees(Moon.az)
		coord[3] = (180/np.pi)*eph.degrees(Moon.alt)


		#print('Sun AZ:  ' + str(round(coord[0],2))  + '      Sun EL:  ' + str(round(coord[1],2)))
		#print('Moon AZ: ' + str(round(coord[2],2))  + '      Moon EL: ' + str(round(coord[3],2)))

	elif len(SH) == 2:
		coord    = np.zeros((SH[0],4))
		for i in range(SH[0]):
			OBS_location.date = dt.datetime(UTC_array[i,0], UTC_array[i,1], UTC_array[i,2], UTC_array[i,3], UTC_array[i,4], UTC_array[i,5])
			Sun  = eph.Sun(OBS_location)
			Moon = eph.Moon(OBS_location)

			coord[i,0] = (180/np.pi)*eph.degrees(Sun.az)
			coord[i,1] = (180/np.pi)*eph.degrees(Sun.alt)		
			coord[i,2] = (180/np.pi)*eph.degrees(Moon.az)
			coord[i,3] = (180/np.pi)*eph.degrees(Moon.alt)


			#print('Sun AZ:  ' + str(round(coord[i,0],2))  + '      Sun EL:  ' + str(round(coord[i,1],2)))
			#print('Moon AZ: ' + str(round(coord[i,2],2))  + '      Moon EL: ' + str(round(coord[i,3],2)))
			#print('-------------------------------------')

	return coord












def level1_to_level2_v2(band, year, day_hour, folder_data_save, folder_plot_save):
	
	
	
	
	
	
	# Paths and files
	path_level1      = home_folder + '/DATA/EDGES/spectra/level1/' + band + '/300_350/'
	path_logs        = home_folder + '/DATA/EDGES/spectra/auxiliary/'
	save_file        = home_folder + '/DATA/EDGES/spectra/level2/' + band_save + '/' + folder_data_save + '/' + year + '_' + day_hour + '.hdf5'





	if band == 'low_band2':
		level1_file      = path_level1 + 'level1_' + year + '_' + day_hour + '_low2_300_350.mat'
		
	elif band == 'mid_band':
		level1_file      = path_level1 + 'level1_' + year + '_' + day_hour + '_mid_300_350.mat'	

	else:
		level1_file      = path_level1 + 'level1_' + year + '_' + day_hour + '_300_350.mat'
		



	weather_file     = path_logs   + 'weather.txt'	

	if (band == 'low_band') or (band == 'mid_band'):
		thermlog_file = path_logs + 'thermlog_low.txt'

	if (band == 'high_band') or (band == 'low_band2'):
		thermlog_file = path_logs + 'thermlog.txt'










	# Loading data
	
	# Frequency and indices
	if band == 'low_band':
		flow  = 50
		fhigh = 120

	elif band == 'low_band2':
		flow  = 50
		fhigh = 100

	elif band == 'high_band':
		flow  = 65
		fhigh = 195

	elif band == 'mid_band':
		flow  = 50
		fhigh = 200

	
	ff, il, ih = frequency_edges(flow, fhigh)
	fe = ff[il:ih+1]
	
	ds, dd = level1_MAT(level1_file)
	tt     = ds[:,il:ih+1]
	ww     = np.ones((len(tt[:,0]), len(tt[0,:])))
	
	
	

	
	
	
	
	
	
	
	# ------------ Meta -------------#
	# -------------------------------#
		
	# Seconds into measurement
	seconds_data = 3600*dd[:,3].astype(float) + 60*dd[:,4].astype(float) + dd[:,5].astype(float)

	# Year and day
	year_int        = int(year)
	day_int         = int(day_hour[0:3])
	fraction_int    = int(day_hour[4::])
	
	year_column     = year_int * np.ones((len(LST),1))
	day_column      = day_int * np.ones((len(LST),1))
	fraction_column = fraction_int * np.ones((len(LST),1))	
	
	
	
	# EDGES coordinates
	EDGES_LAT = -26.7
	EDGES_LON = 116.6
	
	
	
	# LST
	LST = utc2lst(dd, EDGES_LON)
	LST_column      = LST.reshape(-1,1)



	# Galactic Hour Angle
	LST_gc = 17 + (45/60) + (40.04/(60*60))    # LST of Galactic Center
	GHA    = LST - LST_gc
	for i in range(len(GHA)):
		if GHA[i] < -12.0:
			GHA[i] = GHA[i] + 24
	GHA_column      = GHA.reshape(-1,1)
		
			

	# Sun/Moon coordinates
	sun_moon_azel = SUN_MOON_azel(EDGES_LAT, EDGES_LON, dd)				




	# Ambient temperature and humidity, and receiver temperature
	aux1, aux2   = auxiliary_data(weather_file, thermlog_file, band, year_int, day_int)
	
	amb_temp_interp  = np.interp(seconds_data, aux1[:,0], aux1[:,1]) - 273.15
	amb_hum_interp   = np.interp(seconds_data, aux1[:,0], aux1[:,2])
	rec1_temp_interp = np.interp(seconds_data, aux1[:,0], aux1[:,3]) - 273.15
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
	
	

