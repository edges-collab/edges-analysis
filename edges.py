
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

import h5py

from os import makedirs, listdir
from os.path import exists



home_folder = '/data5/raul'
MRO_folder  = '/data5/edges/data/2014_February_Boolardy'



import os, sys
edges_code_folder = os.environ['EDGES_CODE']
sys.path.insert(0, edges_code_folder)

data_folder       = os.environ['EDGES_DATA']
print('EDGES Data Folder: ' + data_folder)




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
				amb_hum   = float(line1[88:93])
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



	# Sun/Moon coordinates
	sun_moon_azel = ba.SUN_MOON_azel(EDGES_LAT, EDGES_LON, dd)				




	# Ambient temperature and humidity, and receiver temperature
	if band == 'low_band3':
		amb_rec = np.zeros((len(seconds_data), 4))
	
	else:
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






	
def models_antenna_s11_remove_delay(band, f_MHz, year=2018, day=145, delay_0=0.17, model_type='polynomial', Nfit=10, plot_fit_residuals='no', MC_mag='no', MC_ang='no', sigma_mag=0.0001, sigma_ang_deg=0.1):	


	


	# Paths
	path_data = home_folder + '/EDGES/calibration/antenna_s11/' + band + '/corrected/' + str(year) + '_' + str(day) + '/' 

	
	# Mid-Band
	if band == 'mid_band':	
		if (year == 2018) and (day == 145):
			d = np.genfromtxt(path_data + 'antenna_s11_mid_band_2018_145.txt'); print('Antenna S11: 2018-145')
	
		if (year == 2018) and (day == 147):
			d = np.genfromtxt(path_data + 'antenna_s11_mid_band_2018_147.txt'); print('Antenna S11: 2018-147')
	
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








def balun_and_connector_loss(f, ra, MC=[0,0,0,0,0,0,0,0]):

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



















def antenna_efficiency(band, f):
		
	return 1











def antenna_beam_factor_interpolation(band, lst_hires, fnew):

	"""


	"""



	# Mid-Band
	if band == 'mid_band':

		file_path = home_folder + '/EDGES/calibration/beam_factors/mid_band/'
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













def beam_factor_table(f, N_lst, file_name_hdf5):


	# Produce the beam factor at high resolution
	# ------------------------------------------
	#N_lst = 6000   # number of LST points within 24 hours
	
	
	lst_hires = np.arange(0,24,24/N_lst)
	bf        = antenna_beam_factor_interpolation('mid_band', lst_hires, f)
	
	
	
	# Save
	# ----
	file_path = home_folder + '/EDGES/calibration/beam_factors/mid_band/'
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


























def level2_to_level3(band, year_day_hdf5, flag_folder='test', receiver_cal_file=1, antenna_s11_year=2018, antenna_s11_day=145, antenna_s11_Nfit=13, FLOW=50, FHIGH=130, Nfg=5):
	
	"""

	"""


	fin  = 0
	tc   = 0
	w_2D = 0
	m_2D = 0


	# Load daily data
	# ---------------
	path_data = home_folder + '/EDGES/spectra/level2/' + band + '/'
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
		f_table, lst_table, bf_table = beam_factor_table_read('/data5/raul/EDGES/calibration/beam_factors/mid_band/beam_factor_table_hires.hdf5') 
		#lst_in                 = np.arange(0, 24, 24/144)
		bf = beam_factor_table_evaluate(f_table, lst_table, bf_table, m_2D[:,3])
		
		
			
		# Antenna S11
		# -----------
		s11_ant = models_antenna_s11_remove_delay(band, fin, year=antenna_s11_year, day=antenna_s11_day, delay_0=0.17, model_type='polynomial', Nfit=antenna_s11_Nfit, plot_fit_residuals='no')
		
		
		
		# Balun+Connector Loss
		# --------------------
		Gb, Gc = balun_and_connector_loss(fin, s11_ant)
		G      = Gb*Gc

		
		
		# Receiver calibration quantities
		# -------------------------------
		if receiver_cal_file == 1:
			print('Receiver calibration FILE 1')
			rcv_file = home_folder + '/EDGES/calibration/receiver_calibration/receiver1/2018_01_25C/results/nominal/calibration_files/calibration_file_mid_band_cfit6_wfit14.txt'
		
		elif receiver_cal_file == 2:
			print('Receiver calibration FILE 2')
			rcv_file = home_folder + '/EDGES/calibration/receiver_calibration/receiver1/2018_01_25C/results/nominal/calibration_files/calibration_file_mid_band_cfit10_wfit14.txt'
		
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
		#tc_with_loss_and_beam = ba.calibrated_antenna_temperature(t_2D, s11_ant, s11_LNA, C1, C2, TU, TC, TS)
		tc_with_loss_and_beam = ba.calibrated_antenna_temperature(t_2D, s11_ant, s11_LNA, C1, C2, TU, TC, TS)
		
		
		
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
		rms_all = np.random.rand(lt, 2)
		
		
		
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
			rri   = tti - model_i
			
			
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
			
			
			# Store
			# -----
			t_all[i,:]    = tti
			p_all[i,:]    = par_fg[0]
			r_all[i,:]    = rri
			w_all[i,:]    = wwi
			rms_all[i,0]  = RMS1
			rms_all[i,1]  = RMS2
			
			
			print(year_day_hdf5 + ': Spectrum number: ' + str(i+1) + ': RMS: ' + str(RMS1) + ', ' + str(RMS2))
	



	# Save
	# ----
	save_folder = home_folder + '/EDGES/spectra/level3/' + band + '/' + flag_folder + '/'
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

	print('NUMBER OF TRACES: ' + str(len(index_all)))



	return index_all







def rms_filter():	
	
	# Listing files to be processed
	# -----------------------------
	path_files = data_folder + '/nominal_60_160MHz_fullcal/'
	new_list   = listdir(path_files)
	new_list.sort()
	
	
	# Loading data
	# ------------
	for i in range(8): #range(len(new_list)): 
		print(new_list[i])
		
		f, t, p, r, w, rms, m = level3read(path_files + new_list[i])
		
		#IX = data_selection(m, GHA_or_LST='GHA', TIME_1=GHA1, TIME_2=GHA2, sun_el_max=90, moon_el_max=90, amb_hum_max = 200, min_receiver_temp=0, max_receiver_temp=100)
		
		#tx   = t[IX,:]
		#px   = p[IX,:]
		#rx   = r[IX,:]
		#wx   = w[IX,:]
		#rmsx = rms[IX,:]
		#mx   = m[IX,:]
		
		
		if i == 0:
			p_all   = np.copy(p)
			r_all   = np.copy(r)
			w_all   = np.copy(w)
			rms_all = np.copy(rms)
			m_all   = np.copy(m)
			
		elif i > 0:
			p_all   = np.vstack((p_all, p))
			r_all   = np.vstack((r_all, r))
			w_all   = np.vstack((w_all, w))
			rms_all = np.vstack((rms_all, rms))
			m_all   = np.vstack((m_all, m))
	
	
	
	
	
	
	
	
	
	# Necessary for analysis
	# ----------------------
	LST  = m_all[:,3]
	RMS1 = rms_all[:,0]
	RMS2 = rms_all[:,1]
	
	IN   = np.arange(0,len(LST))
	
	Npar   = 3
	Nsigma = 3
	
	
	
	
	# Analysis for low-frequency half of the spectrum
	# -----------------------------------------------
	for i in range(24):
		LST_x  = LST[(LST>=i) & (LST<=(i+1))]
		RMS_x  = RMS1[(LST>=i) & (LST<=(i+1))]
		IN_x   = IN[(LST>=i) & (LST<=(i+1))]
		
		W       = np.ones(len(LST_x))
		bad_old = -1
		bad     =  0
		
		iteration = 0
		while bad > bad_old:

			iteration = iteration + 1

			print(' ')
			print('------------')
			print('LST: ' + str(i) + '-' + str(i+1) + 'hr')
			print('Iteration: ' + str(iteration))
		
			par   = np.polyfit(LST_x[W>0], RMS_x[W>0], Npar-1)
			model = np.polyval(par, LST_x)
			res   = RMS_x - model
			std   = np.std(res[W>0])
			
			IN_x_bad = IN_x[np.abs(res) > Nsigma*std]
			W[np.abs(res) > Nsigma*std] = 0
			
			bad_old = np.copy(bad)
			bad     = len(IN_x_bad)
			
			print('STD: ' + str(np.round(std,3)) + ' K')
			print('Number of bad points excised: ' + str(bad))



		if i == 0:
			IN1_bad = np.copy(IN_x_bad)

		else:
			IN1_bad = np.append(IN1_bad, IN_x_bad)
	
	
	
	
	
	# Analysis for high-frequency half of the spectrum
	# ------------------------------------------------
	for i in range(24):
		LST_x  = LST[(LST>=i) & (LST<=(i+1))]
		RMS_x  = RMS2[(LST>=i) & (LST<=(i+1))]
		IN_x   = IN[(LST>=i) & (LST<=(i+1))]
		
		W       = np.ones(len(LST_x))
		bad_old = -1
		bad     =  0
		
		iteration = 0
		while bad > bad_old:

			iteration = iteration + 1

			print(' ')
			print('------------')
			print('LST: ' + str(i) + '-' + str(i+1) + 'hr')
			print('Iteration: ' + str(iteration))
		
			par   = np.polyfit(LST_x[W>0], RMS_x[W>0], Npar-1)
			model = np.polyval(par, LST_x)
			res   = RMS_x - model
			std   = np.std(res[W>0])
			
			IN_x_bad = IN_x[np.abs(res) > Nsigma*std]
			W[np.abs(res) > Nsigma*std] = 0
			
			bad_old = np.copy(bad)
			bad     = len(IN_x_bad)
			
			print('STD: ' + str(np.round(std,3)) + ' K')
			print('Number of bad points excised: ' + str(bad))



		if i == 0:
			IN2_bad = np.copy(IN_x_bad)

		else:
			IN2_bad = np.append(IN2_bad, IN_x_bad)	



	# All bad spectra
	# ---------------
	IN_bad = np.union1d(IN1_bad, IN2_bad)




					
			
	return LST, RMS1, RMS2, IN1_bad, IN2_bad, IN_bad, f, r_all, w_all, m_all





























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









def plot_daily_residuals_LST(f, r, list_names, DY=2, FLOW=50, FHIGH=180, XTICKS=np.arange(60, 180+1, 20), XTEXT=160, YLABEL='ylabel', TITLE='hello', save='no', figure_path='/home/raul/Desktop/', figure_name='2018_150_00'):
	
	# Nspectra_column=35,
	#Ncol_real = len(r[:,0])/Nspectra_columns
	#Ncol_int  = int(np.ceil(Ncol_real))

	
	N_spec = len(r[:,0])
		
	plt.close()
	plt.close()	
	plt.figure(figsize=(7,12))
	
	for i in range(len(list_names)):
		print(i)
	
		if i % 2 == 0:
			color = 'r'
		else:
			color = 'b'
			
		plt.plot(f, r[i]-i*DY, color)
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
		plt.savefig(figure_path + figure_name + '.png', bbox_inches='tight')
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























def level2_to_level3_old(band, year_day, save='no', save_folder='', save_flag='', LST_1=0, LST_2=24, sun_el_max=-10, moon_el_max=90, amb_hum_max=90, min_receiver_temp=23.4, max_receiver_temp=27.4, FLOW=50, FHIGH=120, antenna_s11_year=2018, antenna_s11_day=225, antenna_s11_Nfit=10, fgl=1, glt='value', glp=0.5, fal=1, fbcl=1, receiver_temperature=25, receiver_cal_file=1, beam_correction='yes'):

	"""

	"""


	fin  = 0
	tc   = 0
	w_2D = 0
	m_2D = 0

	# Load daily data
	# ---------------	
	fin_X, t_2D_X, w_2D_X, m_2D = data_selection_single_day(band, year_day, LST_1=LST_1, LST_2=LST_2, sun_el_max=sun_el_max, moon_el_max=moon_el_max, amb_hum_max=amb_hum_max, min_receiver_temp=min_receiver_temp, max_receiver_temp=max_receiver_temp)



	
	# Continue if there are data available
	# ------------------------------------
	if np.sum(t_2D_X) > 0:
		
		
		# Cut the frequency range
		# -----------------------
		fin  = fin_X[(fin_X>=FLOW) & (fin_X<=FHIGH)]
		t_2D = t_2D_X[:, (fin_X>=FLOW) & (fin_X<=FHIGH)]
		w_2D = w_2D_X[:, (fin_X>=FLOW) & (fin_X<=FHIGH)]
		
		
		

		# Chromaticity factor, recomputed for every day because the LSTs are different 
		# ----------------------------------------------------------------------------
		if beam_correction != 'no':
			cf = np.zeros((len(m_2D[:,0]), len(fin)))
			for j in range(len(m_2D[:,0])):
				print('Beam correction LST: ' + str(j))
				cf[j,:] = antenna_beam_factor_interpolation(band, np.array([m_2D[j,3]]), fin)

		# Antenna S11
		# -----------
		s11_ant = models_antenna_s11_remove_delay(band, fin, year=antenna_s11_year, day=antenna_s11_day, delay_0=0.17, model_type='polynomial', Nfit=antenna_s11_Nfit, plot_fit_residuals='no')
		
	
	
	
		# Receiver calibration quantities
		# -------------------------------
		print('Receiver calibration')
		rcv_file = home_folder + '/EDGES/calibration/receiver_calibration/receiver1/2018_01_25C/results/nominal/calibration_files/calibration_file_mid_band_cfit6_wfit14.txt'
		#rcv_file = home_folder + '/EDGES/calibration/receiver_calibration/receiver1/2018_01_25C/results/nominal/calibration_files/calibration_file_mid_band_cfit10_wfit14.txt'
		rcv = np.genfromtxt(rcv_file)
		
		fX   = rcv[:,0]
		rcv2 = rcv[(fX>=FLOW) & (fX<=FHIGH),:]
		s11_LNA = rcv2[:,1] + 1j*rcv2[:,2]
		C1 = rcv2[:,3]
		C2 = rcv2[:,4]
		TU = rcv2[:,5]
		TC = rcv2[:,6]
		TS = rcv2[:,7]
		



		# Calibrated antenna temperature with losses and beam chromaticity
		# ----------------------------------------------------------------
		tc_with_loss_and_beam = ba.calibrated_antenna_temperature(t_2D, s11_ant, s11_LNA, C1, C2, TU, TC, TS)




		## Removing loss
		## -------------
		#print('Loss correction')

		## Combined gain (loss), computed only once, at the beginning, same for all days
		## -----------------------------------------------------------------------------
		#cg = combined_gain(band, fin, antenna_s11_day=ant_s11, antenna_s11_Nfit=ant_s11_Nfit, flag_ground_loss=fgl, ground_loss_type=glt, ground_loss_percent=glp, flag_antenna_loss=fal, flag_balun_connector_loss=fbcl)
		ant_eff = antenna_efficiency(band, fin)

		Tambient = 273.15 + 25 #m_2D[i,9]		
		tc_with_beam = (tc_with_loss_and_beam - Tambient*(1-ant_eff))/ant_eff


		# Removing beam chromaticity
		# --------------------------
		if beam_correction == 'no':
			print('NO beam correction')
			tc = np.copy(tc_with_beam)
		else:
			print('Beam correction')
			tc = tc_with_beam/cf

		## Save
		## --------------
		#if save =='yes':
			#path_save = home_folder + '/DATA/EDGES/spectra/level3/' + band + '/' + save_folder + '/'
			#save_file = path_save + year_day + save_flag + '.hdf5'	

			#with h5py.File(save_file, 'w') as hf:
				#hf.create_dataset('frequency',            data = fin)
				#hf.create_dataset('antenna_temperature',  data = tc)
				#hf.create_dataset('weights',              data = w_2D)
				#hf.create_dataset('meta_data',            data = m_2D)



	return fin, tc_with_loss_and_beam, w_2D, m_2D          #fin, tc, w_2D, m_2D














def basic_test(ff, tt, ww, m, FLOW, FHIGH, Nfit):
	
	f = ff[(ff>=FLOW) & (ff<=FHIGH)]
	t = tt[:, (ff>=FLOW) & (ff<=FHIGH)]
	w = ww[:, (ff>=FLOW) & (ff<=FHIGH)]
	
	
	
	avt = np.mean(t, axis=0)
	
	tc1, wc1 = rfi.excision_raw_frequency(f, avt, np.ones(len(f)))
	tc2, wc2 = rfi.cleaning_sweep(f, tc1, wc1, window_width_MHz=4, Npolyterms_block=4, N_choice=20, N_sigma=3.0)
	
	
	p = ba.fit_polynomial_fourier('LINLOG', f, tc2, Nfit, Weights=wc2)
	
	fb, rb, wb = ba.spectral_binning_number_of_samples(f, tc2-p[1], wc2)
	
	
	
	plt.plot(fb[wb>0], rb[wb>0])
	
	
	
	
	
	return 0









