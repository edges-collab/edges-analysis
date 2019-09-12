


import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


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

import os, sys
edges_folder       = os.environ['EDGES_vol2']
print('EDGES Folder: ' + edges_folder)





def gsm2019_maps():

	"""
	This function will return the GSM2019 maps in NESTED Galactic Coordinates

	freq: 22, 35, 45, 60, 80, 150, 408 MHz

	"""


	# Loading NESTED galactic coordinates for Nside=128
	# -------------------------------------------------
	coord              = fits.open(edges_folder + '/sky_models/coordinate_maps/pixel_coords_map_nested_galactic_res7.fits')
	coord_array        = coord[1].data
	lon_raw            = coord_array['LONGITUDE']
	lat_raw            = coord_array['LATITUDE']
	GALAC_COORD_object = apc.SkyCoord(lon_raw, lat_raw, frame='galactic', unit='deg')  # defaults to ICRS frame		





	# Loading maps
	# ------------
	d1 = np.load('/home/raul/DATA2/EDGES_vol2/sky_models/gsm2019/low_egsm_bfit_kelvin.npz')

	f           = d1['freqs']
	maps_ring   = d1['data']
	errors_ring = d1['error']





	# Add 75 MHz
	# ----------
	d2 = np.load('/home/raul/DATA2/EDGES_vol2/sky_models/gsm2019/egsm_pred_75mhz_kelvin.npz')
	#d2 = np.load('/home/raul/DATA2/EDGES_vol2/sky_models/gsm2019/egsm_low_betapred.npz')

	f_75MHz          = d2['freqs']
	map_75MHz_ring   = d2['data']
	error_75MHz_ring = d2['error']

	#print(maps_ring.shape)
	#print(map_75MHz_ring.shape)

	f           = np.insert(f, 4, f_75MHz[0])
	maps_ring   = np.insert(maps_ring, 4, map_75MHz_ring, axis=0)
	errors_ring = np.insert(errors_ring, 4, error_75MHz_ring, axis=0)





	# Converting map from RING to NESTED
	# ----------------------------------
	maps_nestT   = np.zeros((len(maps_ring[:,0]), len(maps_ring[0,:])))
	errors_nestT = np.zeros((len(maps_ring[:,0]), len(maps_ring[0,:])))

	for i in range(len(f)):
		maps_nestT[i,:]   = hp.reorder(maps_ring[i,:], r2n=True)
		errors_nestT[i,:] = hp.reorder(errors_ring[i,:], r2n=True)

	maps_nest   = maps_nestT.T
	errors_nest = errors_nestT.T


	print('Output maps are in NEST format')

	return f, maps_nest, errors_nest, lon_raw, lat_raw, GALAC_COORD_object











def gsm2019_model(f, error_factor_map=0, error_factor_specindex=0):


	# Loading NESTED galactic coordinates for Nside=128
	# -------------------------------------------------
	coord              = fits.open(edges_folder + '/sky_models/coordinate_maps/pixel_coords_map_nested_galactic_res7.fits')
	coord_array        = coord[1].data
	lon_raw            = coord_array['LONGITUDE']
	lat_raw            = coord_array['LATITUDE']
	GALAC_COORD_object = apc.SkyCoord(lon_raw, lat_raw, frame='galactic', unit='deg')  # defaults to ICRS frame		




	# 75 MHz Map
	# ----------
	d = np.load('/home/raul/DATA2/EDGES_vol2/sky_models/gsm2019/egsm_pred_75mhz_kelvin.npz')
	f_75MHz          = d['freqs']
	map_ring       = d['data']
	error_map_ring = d['error']

	map_nest       = hp.reorder(map_ring, r2n=True)
	error_map_nest = hp.reorder(error_map_ring, r2n=True)	



	# Spectral Index
	# --------------
	s = np.load('/home/raul/DATA2/EDGES_vol2/sky_models/gsm2019/egsm_low_betapred.npz')
	specindex_ring = s['data']
	error_specindex_ring = s['error']

	specindex_nest       = hp.reorder(specindex_ring, r2n=True)
	error_specindex_nest = hp.reorder(error_specindex_ring, r2n=True)	



	# Compute maps at other frequencies using a power law model
	# ---------------------------------------------------------
	m = np.zeros((len(map_nest), len(f))) 
	for i in range(len(f)):
		m[:,i] = (map_nest + error_factor_map*error_map_nest) * ((f[i]/75)**(specindex_nest + error_factor_specindex*error_specindex_nest))




	return m, lon_raw, lat_raw, GALAC_COORD_object #map_nest, error_map_nest, specindex_nest, error_specindex_nest




















def antenna_temperature_gsm2019(case, save_filename, reference_frequency=76, error_factor=0):

	"""
	
	error_factor: multiple of temperature "error" added to nominal map. If '0', computation is done for nominal map. If '+1 (-1)', computation is done for nominal +(-) error.  This factor can take any value.
	
	
	"""




	# Sky maps (in NESTED coordinates)
	# ------------------------------------------------------------------
	
	if case == 1:
		f_maps, sky_maps_nominal, sky_errors, lon, lat, GALAC_COORD_object = gsm2019_maps()	
		sky_maps = sky_maps_nominal + error_factor*sky_errors   # error_factor: is an input parameter
		output_file_header = 'LST [hr], Tant at 22MHz [K], 35MHz [K], 45MHz [K], 60MHz [K], 80MHz [K], 150MHz [K], 408MHz [K]'
		path_save = edges_folder + 'others/antenna_temperature_gsm2019/test1/'
		
			
	if case == 2:
		f_maps, sky_maps_nominal, sky_errors, lon, lat, GALAC_COORD_object = gsm2019_maps()
		sky_maps_T         = np.array([sky_maps_nominal[:,4], sky_maps_nominal[:,4] - sky_errors[:,4], sky_maps_nominal[:,4] + sky_errors[:,4]])
		sky_maps           = sky_maps_T.T
		f_maps             = np.array([75, 75, 75])
		output_file_header = 'LST [hr], nominal [K], nominal-error [K], nominal+error [K]'
		path_save = edges_folder + 'others/antenna_temperature_gsm2019/test2/'
	

	if case == 3:
		f_maps   = np.arange(50,101,10)
		sky_maps, lon, lat, GALAC_COORD_object = gsm2019_model(f_maps, error_factor_map=0, error_factor_specindex=error_factor)
		output_file_header = 'LST [hr], Tant at 50MHz [K], 60MHz [K], 70MHz [K], 80MHz [K], 90MHz [K], 100MHz [K]'
		path_save = edges_folder + 'others/antenna_temperature_gsm2019/test3/'
	
	
	









	# Antenna beam
	# ---------------------------------------------------------------------------
	AZ_beam  = np.arange(0, 360)
	EL_beam  = np.arange(0, 91)


	# FEKO blade beam	
	# Fixing rotation angle due to diferent rotation (by 90deg) in Nivedita's map
	#if beam_file == 100:
	#	rotation_from_north = rotation_from_north - 90
	
	# Best case, Feb 20, 2019
	

	beam_all = oeg.FEKO_low_band_blade_beam(beam_file=2, frequency_interpolation='no', frequency=np.array([0]), AZ_antenna_axis=0)
	freq_array = np.arange(40,121,2)	


	## Frequency array
	# beam_all = FEKO_blade_beam('mid_band', beam_file, AZ_antenna_axis=rotation_from_north)
	#if beam_file == 0:  # Best case, Feb 20, 2019
		## ALAN #0
		#freq_array = np.arange(50, 201, 2, dtype='uint32') 		
	
	#elif beam_file == 1:
		## ALAN #1
		#freq_array = np.arange(50, 201, 2, dtype='uint32')  

	#elif beam_file == 2:
		## ALAN #2
		#freq_array = np.arange(50, 201, 2, dtype='uint32')  

	#elif beam_file == 100:
		## NIVEDITA
		#freq_array = np.arange(60, 201, 2, dtype='uint32')	

			
			
			
			
			
			
					
	# Index of beam frequency
	index_freq_array = np.arange(len(freq_array))
	irf = index_freq_array[freq_array == reference_frequency]
	print('Reference frequency: ' + str(freq_array[irf][0]) + ' MHz')


	# Beam at reference frequency
	beam = beam_all[irf,:,:]




	
	


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
	output = np.zeros((72, 1+len(f_maps)))	



	for i in range(72):
	#for i in range(len(LST)):


		print('LST: ' + str(i+1) + ' out of 72')


		# Advancing time ( 19:57 minutes UTC correspond to 20 minutes LST )
		minutes_offset = 19
		seconds_offset = 57
		if i > 0:
			Time_iter_dt = Time_iter_dt + dt.timedelta(minutes = minutes_offset, seconds = seconds_offset)
			Time_iter    = np.array([Time_iter_dt.year, Time_iter_dt.month, Time_iter_dt.day, Time_iter_dt.hour, Time_iter_dt.minute, Time_iter_dt.second]) 



		# LST 
		output[i,0] = ba.utc2lst(Time_iter, EDGES_lon_deg)



		# Transforming Galactic coordinates of Sky to Local coordinates		
		altaz          = GALAC_COORD_object.transform_to(apc.AltAz(location=EDGES_location, obstime=apt.Time(Time_iter_dt, format='datetime')))
		AZ             = np.asarray(altaz.az)
		EL             = np.asarray(altaz.alt)



		# Selecting coordinates above the horizon
		AZ_above_horizon         = AZ[EL>=0]
		EL_above_horizon         = EL[EL>=0]



		# Selecting sky data above the horizon
		sky_above_horizon     = sky_maps[EL>=0,:]   # .flatten()
		



		# Arranging AZ and EL arrays corresponding to beam model
		az_array   = np.tile(AZ_beam,91)
		el_array   = np.repeat(EL_beam,360)
		az_el_original      = np.array([az_array, el_array]).T
		az_el_above_horizon = np.array([AZ_above_horizon, EL_above_horizon]).T



		# Loop over frequency
		for j in range(len(f_maps)):

			print('Freq: ' + str(j+1) + ' out of ' + str(len(f_maps)))

			beam_array         = beam.reshape(1,-1)[0]
			beam_above_horizon = spi.griddata(az_el_original, beam_array, az_el_above_horizon, method='cubic')  # interpolated beam


			no_nan_array       = np.ones(len(AZ_above_horizon)) - np.isnan(beam_above_horizon)
			index_no_nan       = np.nonzero(no_nan_array)[0]

	
			# Convolution between (beam at all frequencies) and (sky at reference frequency)
			output[i, j+1]   = np.sum(beam_above_horizon[index_no_nan]*sky_above_horizon[index_no_nan, j])/np.sum(beam_above_horizon[index_no_nan])


			## Antenna temperature, i.e., Convolution between (beam at all frequencies) and (sky at all frequencies)
			#sky_above_horizon_ff    = sky_above_horizon[:, j].flatten()
			#convolution[i, j]       = np.sum(beam_above_horizon[index_no_nan]*sky_above_horizon_ff[index_no_nan])/np.sum(beam_above_horizon[index_no_nan])



	
	#beam_factor_T = convolution_ref.T/convolution_ref[:,irf].T
	#beam_factor   = beam_factor_T.T





	# Saving
	# ---------------------------------------------------------
	np.savetxt(path_save + '/' + save_filename + '.txt', output, header=output_file_header)



	return output




















def fit_spectral_index(case, folder, save_filename):
	
	if case == 1:
		d = np.genfromtxt(edges_folder + 'others/antenna_temperature_gsm2019/test3/gsm2019_nominal.txt')
		
	if case == 2:
		d = np.genfromtxt(edges_folder + 'others/antenna_temperature_gsm2019/test3/gsm2019_minus_specindex_error.txt')
		
	if case == 3:
		d = np.genfromtxt(edges_folder + 'others/antenna_temperature_gsm2019/test3/gsm2019_plus_specindex_error.txt')
		
		
	
	f = np.arange(50,101,10)
	p = np.zeros((len(d[:,0]), 4))
	
	for i in range(len(d[:,0])):
		p[i,0] = np.copy(d[i,0])
		PP     = np.polyfit(np.log10(f/75), np.log10(d[i,1::]), 1)
		p[i,1] = 10**PP[1]
		p[i,2] = PP[0]
		
		log_model = np.polyval(PP, np.log10(f/75))
		model = 10**(log_model)
		
		RMS = np.std(d[i,1::] - model)
		p[i,3] = RMS
		print(RMS)
	


	# Saving
	# ---------------------------------------------------------
	path_save = edges_folder + 'others/antenna_temperature_gsm2019/' + folder + '/' + save_filename + '.txt'
	np.savetxt(path_save, p, header='LST [hr], Tant at 75 MHz [K], Spectral Index, Fit RMS [K]')
	
	
	return p





