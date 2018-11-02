

import numpy as np
import datetime as dt
import astropy.coordinates as apc
import astropy.units as apu
import astropy.time as apt
import healpy as hp
import h5py
import matplotlib.pyplot as plt

from astropy.io import fits
from os.path import expanduser
home_folder = expanduser("~")
import scipy as sp
from scipy import special

def test(model_type,vrange,elliptypex = 'flat', elliptypey = 'flat',
         bftype = 'linearneg', amp_type = 'flat',abmax=.3,AZ1=0, fwhmxmin=90, fwhmxmax = 90, fwhmymin = 90, fwhmymax = 90,
         save_plot ='no',save_plot_name_prefix = ' ',phi0 = 0,amin = 1, amax = 1,plotname = '90fwhmflat'):
	#AZ1=0

	# Data paths
	#path_root          = home_folder + '/DATA/EDGES/'
	#path_data          = path_root + 'beam_convolution/'

	path_data = '/Users/katie/map_scaling/'

	coord              = fits.open(path_data + 'pixel_coords_map_nested_celestial_res6.fits') #res 6 so that it doesn't take as long
	coord_array        = coord[1].data
	lon                = coord_array['LONGITUDE']
	lat                = coord_array['LATITUDE']
	COORD_object       = apc.SkyCoord(lon, lat, frame='icrs', unit='deg')  # defaults to ICRS frame


	haslam_map = fits.open(path_data + 'lambda_haslam408_dsds.fits')
	haslam512     = (haslam_map[1].data)['temperature']

	haslam = hp.ud_grade(haslam512,nside_out = 64,order_in = 'nested',order_out = 'nested')

	# Spectral index in HEALPix RING Galactic Coordinates, nside=512
	beta_file ='/Users/katie/map_scaling/sky_spectral_index_original_45_408_MHz_maps_galactic_coordinates_nside_512_ring_3Ksubtracted.hdf5'
	with h5py.File(beta_file, 'r') as hf:			
		hf_beta   = hf.get('spectral_index')
		beta_ring = np.array(hf_beta) #this is making the spectral index vary depending on where you are in the sky


	# Convert beta to NESTED format
	beta = hp.reorder(beta_ring, r2n=True) 
	beta = hp.ud_grade(beta,nside_out = 64,order_in = 'nested',order_out = 'nested') #this is reducing the pixel # (words?)    


	# Filling the hole
	beta[lat>68] = np.mean(beta[(lat>60) & (lat<68)]) #filling the beta hole

	EDGES_lat_deg  =  np.ones(2) *(-27,33.9)#np.arange(0,1,1) #np.linspace(90,-90,36) # # #-26.714778 #the farther from the poles you are, the more dramatic the weighted average is
	EDGES_lon_deg  = 116.605528



	# Time equivalent to LST 3.5, at long=116.etc.,    np.array([2014, 1, 1, 12, 59, 10])
	# time equivalent to LST 165 , at lon = 116        np.array([2014, 1, 1, 20, 28, 0])


	#in an ideal world, we'd be able to sweep over all the points in AZ and EL. 
	#but this is not an ideal world.
	#Trump is president, and that computation would take like 10 days
	#so instead we're just gonna pick two points lolol
	# these two points correspond to low-foreground regions

	point1 = [3.5,-26.7] #LST, DEC (Ie lon lat)
	point2 = [11,33.9]
	cubecoord = np.array([point1,point2])
	time = [np.array([2014, 1, 1, 12, 59, 10]),np.array([2014, 1, 1, 20, 28, 0])]

	avg = np.zeros((len(cubecoord),81))
	#beam = np.zeros((len(cubecoord),81,49152))        

	for k in range(len(cubecoord)): #this is sweeping over latitude #or it used to be. now we just have 2 points

		#print(k)
		# EDGES location	

		EDGES_location = apc.EarthLocation(lat=EDGES_lat_deg[k]*apu.deg, lon=EDGES_lon_deg*apu.deg)


		# Reference observation time. At this time, the LST is 0.1666 (00:10 Hrs LST) at the EDGES location 
		#Time_iter    = np.array([2014, 1, 1, 9, 49, 42])    OLD
		Time_iter     = time[k] #CHANGE made time_iter one of the two coordinate point times
		Time_iter_dt = dt.datetime(Time_iter[0], Time_iter[1], Time_iter[2], Time_iter[3], Time_iter[4], Time_iter[5]) 


		# Looping over LST
		#LST = np.zeros(72)
		LST = utc2lst(Time_iter, EDGES_lon_deg)


		##convolution = np.zeros((len(LST), len(beam_all[:,0,0])))
		#for j in range(2): #range(len(LST)): #this means we have a five degree resolution. #SHOULD BE 72- CHANGING IT TO BE FASTER RN

			#print(j)

		##minutes_offset = 19 #this is to account for a sidereal day
		#seconds_offset = 57
		#if k > 0:
			#Time_iter_dt = Time_iter_dt + dt.timedelta(minutes = minutes_offset, seconds = seconds_offset)
			#Time_iter    = np.array([Time_iter_dt.year, Time_iter_dt.month, Time_iter_dt.day, Time_iter_dt.hour, Time_iter_dt.minute, Time_iter_dt.second]) 


			## LST 




		#Transforming Celestial coordinates of Sky to Local coordinates		#ADD THE TIME HERE ___>
		altaz          = COORD_object.transform_to(apc.AltAz(location=EDGES_location, obstime=apt.Time(Time_iter_dt, format='datetime')))
		AZ             = np.asarray(altaz.az)
		EL             = np.asarray(altaz.alt)

			##gaussian = first_gaussian(AZ,EL) 

			### Selecting coordinates and sky data above the horizon
			##AZ_above_horizon         = AZ[EL>=0]
			##EL_above_horizon         = EL[EL>=0]


		beam           = beam_functions(model_type,vrange,AZ,EL,elliptypex = elliptypex, elliptypey = elliptypey,
		                                bftype = bftype, amp_type = amp_type,abmax=abmax,AZ1=AZ1, fwhmxmin=fwhmxmin, fwhmxmax = fwhmxmax, fwhmymin = fwhmymin, fwhmymax = fwhmymax,
		                                save_plot ='no',phi0 = phi0,amin = amin, amax = amax)


			##this is where we compute the weighted averages


		print(beam)
		for i in range(len(vrange)): 

			Tcmb = 2.725
			haslam_scaled   = Tcmb + (haslam-Tcmb)*(vrange[i]/408)**(-beta)    #scaling the haslam map. beta has 
			#different values at different frequencies
			#THIS IS WHERE YOU COMPUTE THE ANTENNA TEMPERATURE AS A FUNCTION OF FREQUENCY AND SAVE IT


			if k == 1:

				bf2 = np.copy(beam)






			num   = np.sum(haslam_scaled*beam) #weighted average calculation. 
			denom = np.sum(beam)

			avg[k,i] = num/denom #how to do this??


	def bfdata(model_type,bftype):
		if model_type == 'beg' or model_type == 'beg_gauss':
			if bftype == 'flat':
				bt = 1
			elif bftype == 'linearpos':
				bt = 2
			elif bftype == 'linearneg':
				bt = 3
			elif bftype == 'squarepos':
				bt = 4
			elif bftype == 'squareneg':
				bt = 5
			elif bftype == 'fourpos':
				bt = 6
			elif bftype == 'fourneg':
				bt = 7   
			else:
				bt = 8   
		else: 
			bt = 0
		return bt


	bt = bfdata(model_type, bftype)    


	def mtdata(model_type,fwhmxmin,fwhmymin):
		if model_type == 'gaussian':
			if fwhmxmin == fwhmymin and fwhmxmax == fwhmymax:
				mt = 1
			else:
				mt = 2
		elif model_type == 'beg' or model_type == 'beg_gauss':
			mt = 3
		return mt
	mt = mtdata(model_type,fwhmxmin,fwhmymin) 

	def bidata(model_type):
		if model_type == 'beg':
			bt = 1
		elif model_type == 'beg_gauss':
			bt = 2
		else:
			bt = 0
		return bt
	bit = bidata(model_type)

	def adata(amp_type):
		if amp_type == 'flat':
			at = 1
		elif amp_type == 'linearpos':
			at = 2
		elif amp_type == 'linearneg':
			at = 3
		elif amp_type == 'squarepos':
			at = 4
		elif amp_type == 'squareneg':
			at = 5
		elif amp_type == 'fourpos':
			at = 6
		elif amp_type == 'fourneg':
			at = 7
		return at

	at = adata(amp_type)        

	def ellipdatax(model_type,fwhmxmin,fwhmxmaxymin,elliptypex):
		if model_type == 'gaussian':
			if elliptypex == 'flat':
				etx = 1
			elif elliptypex == 'linearpos':
				etx = 2
			elif elliptypex == 'linearneg':
				etx = 3
			elif elliptypex == 'squarepos':
				etx = 4
			elif elliptypex == 'squareneg':
				etx = 5
			elif elliptypex == 'fourpos':
				etx = 6
			elif elliptypex == 'fourneg':
				etx = 7   
			else:
				etx = 0

		else:
			etx = 0

		return etx

	etx = ellipdatax(model_type, fwhmxmin, fwhmymin, elliptypex)

	def ellipdatay(model_type,xmin,ymin,elliptypey):
		if model_type == 'gaussian':
			if elliptypey == 'flat':
				ety = 1
			elif elliptypey == 'linearpos':
				ety = 2
			elif elliptypey == 'linearneg':
				ety = 3
			elif elliptypey == 'squarepos':
				ety = 4
			elif elliptypey == 'squareneg':
				ety = 5
			elif elliptypey == 'fourpos':
				ety = 6
			elif elliptypey == 'fourneg':
				ety = 7  
			else:
				ety = 0

		else:
			ety = 0  
		return ety


	ety = ellipdatay(model_type, fwhmxmin, fwhmymin, elliptypey)        




	path_save = '/Users/katie/cubes/circular/'
	save_file = path_save + plotname + '.hdf5'






	with h5py.File(save_file, 'w') as hf:
		hf.create_dataset('Convolution',   data = avg)
		hf.create_dataset('Azimuth',       data = AZ)
		hf.create_dataset('Elevation',     data = EL)
		hf.create_dataset('Beam',          data = bf2)         
		hf.create_dataset('modeltype',     data = mt)
		hf.create_dataset('amplitudetype', data = at)
		hf.create_dataset('ellipse_x',      data = etx)
		hf.create_dataset('ellipse_y',      data = ety)
		hf.create_dataset('bftype_amp',      data = bt)
		hf.create_dataset('bftype',        data = bit)
		hf.create_dataset('fwhm_xmin',      data = fwhmxmin)
		hf.create_dataset('fwhm_xmax',      data = fwhmxmax)
		hf.create_dataset('fwhm_ymin',      data = fwhmymin)
		hf.create_dataset('fwhm_ymax',      data = fwhmymax)     
		hf.create_dataset('amp_min',        data =  amin)
		hf.create_dataset('amp_max',        data =  amax)
		hf.create_dataset('coordinates',     data = cubecoord)
		hf.create_dataset('freqvector',     data = vrange)




	return haslam_scaled,beam, avg, AZ, EL,bit #,gaussian,haslam,haslam_scaled,gaussian #COORD_object.ra, COORD_object.dec, haslam
#
#
#

def reading_convolution_results(path_file):



	with h5py.File(path_file,'r') as hf:
			#print([key for key in hf.keys()])

		cnv = hf.get('Convolution')
		convolution    = np.array(cnv)

		az = hf.get('Azimuth')
		azimuth    = np.array(az)

		el = hf.get('Elevation')
		elevation    = np.array(el)

		be = hf.get('Beam')
		beam    = np.array(be)

		mt = hf.get('modeltype')
		modeltype    = np.array(mt)

		at = hf.get('amplitudetype')
		amptype    = np.array(at)

		ex = hf.get('ellipse_x')
		ellip_x = np.array(ex)

		ey = hf.get('ellipse_y')
		ellip_y = np.array(ey)	

		bfa = hf.get('bftype_amp')
		bif_amp = np.array(bfa)

		bft = hf.get('bftype')
		bif_typ = np.array(bft)

		fxmi = hf.get('fwhm_xmin')
		f_xmin = np.array(fxmi)

		fxma = hf.get('fwhm_xmax')
		f_xmax = np.array(fxma)

		fymi = hf.get('fwhm_ymin')
		f_ymin = np.array(fymi)

		fyma = hf.get('fwhm_ymax')
		f_ymax = np.array(fyma)		

		amin = hf.get('amp_min')
		amp_min = np.array(amin)

		amax = hf.get('amp_max')
		amp_max = np.array(amax)

		co = hf.get('coordinates')
		coord = np.array(co)
		
		vr = hf.get('freqvector')
		vrange = np.array(vr)

		config_array = np.array([modeltype,amptype,ellip_x,ellip_y,bif_amp,bif_typ,f_xmin,f_xmax,f_ymin,f_ymax,amp_min,amp_max])

		return convolution,coord,azimuth,elevation,beam,config_array,vrange
#
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


def ellipse_one(AZ,EL):
	rprime = np.cos(np.deg2rad(0))
	phi = AZ
	phi0=31
	sigmay = 1.7
	sigmax = 1
	a=1
	el0=90
	sigma = 30
	#beam =  np.cos(np.deg2rad(phi-phi0))**2/sigmax**2 + np.sin(np.deg2rad(phi-phi0))**2/sigmay**2 
	exponent            = a*np.exp(-(EL-el0)**2/(2*sigma**2))
	beam = np.sqrt((sigmax**2 * sigmay**2)/(sigmax**2 * np.sin(np.deg2rad(phi-phi0))**2 + sigmay**2*np.cos(np.deg2rad(phi-phi0))**2))
	gaussian = beam*exponent
	return gaussian,beam

def gamma_prac(AZ,EL):
	shape = 5.5
	scale = 5.5
	AZ1=0
	#x = np.arange(0,90,1)
	x  = np.linspace(90,-90,49152)
	gamma = 1/(special.gamma(shape))
	eqn1  = 1/(scale**shape) * (90-x)**(shape-1) * np.exp(-(90-x)/scale)
	gamma_distrib = gamma*eqn1
	bifurcation = (np.cos(np.deg2rad(2*AZ-2*AZ1)) + 1) *gamma_distrib
	return gamma_distrib,bifurcation




def beam_functions(model_type,vrange,AZ,EL,elliptypex = 'fourneg', elliptypey = 'linearneg',
                   bftype = 'linearneg', amp_type = 'linearpos',abmax=.3,AZ1=0, fwhmxmin=30, fwhmxmax = 40, fwhmymin = 20, fwhmymax = 40,
                   save_plot ='no',save_plot_name_prefix = 'Linear Bifurcated Gamma,fwhm=70',phi0 = 0,amin = .9, amax = 1.1):

	#new- need to check if a =1 always. originally in gaussian function:

	#initial parameters of elliptical gaussian:
	#a = 1
	az0 =0
	el0 = 90


	#sigma = fwhm/(2*np.sqrt(2*np.log(2))) #need to define and put fwhm into input parameters
	xmax =  fwhmxmax/(2*np.sqrt(2*np.log(2))) #changing the fwhms' into sigmas 
	xmin =  fwhmxmin/(2*np.sqrt(2*np.log(2)))
	ymax =  fwhmymax/(2*np.sqrt(2*np.log(2)))
	ymin =  fwhmymin/(2*np.sqrt(2*np.log(2)))


	def amplitude(amp_type,vrange,amin,amax):
		if amp_type    == 'flat':
			avar= amin*np.ones(len(vrange))

		elif amp_type  ==  'linearpos':
			avar=  np.linspace(amin,amax,len(vrange))

		elif amp_type  ==   'linearneg':
			avar=  np.linspace(amax,amin,len(vrange))

		elif amp_type  ==    'squarepos':
			avar= (amax-amin)/6400*(vrange-40)**2 + amin

		elif amp_type  ==     'squareneg':
			avar=  (amax-amin)/6400*(vrange-120)**2 + amin

		elif amp_type  ==   'fourpos':
			avar=  (amax-amin)/40960000*(vrange-40)**2 + amin

		elif amp_type  == 'fourneg':
			avar=  (amax-amin)/40960000*(vrange-120)**2 + amin
		return avar


	avar = amplitude(amp_type, vrange, amin,amax)
	print(avar)

	def beam_type(model_type,xmax,xmin,vrange,ymax,ymin,EL):
		#if model_type     == 'gaussian': #this is just a normal beam, with no bifurcation, ellipticity, or any other abormalities
			#bfcnormal = 0
			#ab = np.zeros(len(vrange))
			#sigmax = sigmay = sigma * np.ones(len(vrange))
			#return bfcnormal,ab,sigmax,sigmay

		if model_type   == 'gaussian':
			bfcnormal = 0
			ab = np.zeros(len(vrange))
			def typex(elliptypex,xmax,xmin,vrange):
				if elliptypex == 'flat':
					sigmax = xmin* np.ones(len(vrange))

				elif elliptypex == 'linearneg':
					sigmax = np.linspace(xmax,xmin,len(vrange))

				elif elliptypex == 'linearpos':
					sigmax = np.linspace(xmin,xmax,len(vrange))

				elif elliptypex == 'squarepos':

					sigmax = (xmax-xmin)/(6400)*(vrange-40)**2 + xmin 

				elif elliptypex == 'squareneg':
					sigmax = (xmax-xmin)/(6400)*(vrange-120)**2 + xmin #all of this comes from making an equation where, for example we know that at
					#vrange = 40, f(x)= xmax and at vrange = 12o, f(x) =xmin

				elif elliptypex == 'fourpos':
					sigmax = (xmax-xmin)/40960000 * (vrange-40)**4 +xmin

				elif elliptypex == 'fourneg':
					sigmax = (xmax-xmin)/40960000 * (vrange-120)**4 +xmin

				else:
					sigmax = np.ones(len(vrange))    

				return sigmax
			sigmax = typex(elliptypex, xmax, xmin, vrange)

			def typey(elliptypey,ymax,ymin,vrange):
				if elliptypey == 'flat':
					sigmay = np.ones(len(vrange))* ymin
				elif elliptypey == 'linearneg':
					sigmay = np.linspace(ymax,ymin,len(vrange))

				elif elliptypey == 'linearpos':
					sigmay = np.linspace(ymin,ymax,len(vrange))

				elif elliptypey == 'squarepos':

					sigmay = (ymax-ymin)/(6400)*(vrange-40)**2 + ymin 

				elif elliptypey == 'squareneg':
					sigmay = (ymax-ymin)/(6400)*(vrange-120)**2 + ymin #all of this comes from making an equation where, for example we know that at
					#vrange = 40, f(x)= xmax and at vrange = 12o, f(x) =xmin

				elif elliptypey == 'fourpos':
					sigmay = (ymax-ymin)/40960000 * (vrange-40)**4 +ymin

				elif elliptypey == 'fourneg':
					sigmay = (ymax-ymin)/40960000 * (vrange-120)**4 +ymin                

				else:
					sigmay = np.ones(len(vrange))

				return sigmay

			sigmay = typey(elliptypey, ymax, ymin, vrange)

			return bfcnormal,ab,sigmax,sigmay



		elif model_type   == 'beg' or model_type =='beg_gauss' : #if the model is a bifurcated gaussian with a 
			sigmax = sigmay = xmin *np.ones(len(vrange))

			shape =   7 #5.5
			scale =  3.5 #5.5 #10 

			def abtype(bftype,abmax,vrange):

				if bftype == 'linearneg':
					ab = np.linspace(abmax,0,len(vrange))
				elif bftype == 'linearpos':
					ab = np.linspace(0,abmax,len(vrange))

				elif bftype == 'squarepos':
					ab = abmax/(6400)*(vrange-40)**2

				elif bftype == 'squareneg':
					ab = abmax/(6400)*(vrange-120)**2

				elif bftype == 'cubicpos':
					ab = abmax/512000 * (vrange - 40)**3

				elif bftype == 'cubicneg':
					ab = abmax/512000 * (vrange-120)**3

				elif bftype == 'fourpos':
					ab = abmax/40960000 * (vrange-40)**4

				elif bftype == 'fourneg':
					ab = abmax/40960000 * (vrange-120)**4

				else:
					ab = np.ones(len(vrange))
				return ab
			ab = abtype(bftype, abmax, vrange)


			def modeltype_two(model_type,EL):
				if model_type == 'beg': 

					gamma         = 1/(special.gamma(shape)) #this is the gamma function
					eqn1          = 1/(scale**shape) * (90-EL)**(shape-1) * np.exp(-(90-EL)/scale) 
					gamma_distrib = gamma*eqn1 #this is the gamma distribution. modifies the profile of the beam. Function of scale, theta, and 90-elevation
					bfc           =  (np.cos(np.deg2rad(2*AZ-2*AZ1)) + 1) * gamma_distrib #this is to make the gamma distribution also azimuthally dependent
					bfcnormal     = bfc/np.max(bfc) #normalizing  

					return bfcnormal

				elif model_type == 'beg_gauss': #a bifurcated gaussian, instead of a bifurcated gamma distribution

					el0_g     = 45 #this is the centering elevation value (halfway between 0 and 90)
					fwhm_g      = 30 #fwhm of the gaussian
					sigma_g   = fwhm_g/(2*np.sqrt(2*np.log(2)))

					gauss     = np.exp(-(EL-el0_g)**2/(2*sigma_g**2)) 
					bfc       = (np.cos(np.deg2rad(2*AZ-2*AZ1)) + 1) * gauss
					bfcnormal = bfc/np.max(bfc)
					return bfcnormal


			bfcnormal = modeltype_two(model_type, EL)               
			return  bfcnormal,ab,sigmax,sigmay    


	bfcnormal,ab,sigmax,sigmay = beam_type(model_type, xmax, xmin, vrange, 
	                                       ymax, 
	                                       ymin, 
	                                       EL)
	beam            = np.zeros((len(vrange),len(AZ))) #initializing empty arrays to later make the beam profiles
	bfcprofile      = np.zeros((len(vrange),len(AZ)))
	azimuthal       = np.zeros((len(vrange),len(AZ)))

	#fwhm = 45;make it between 25-40. 30 is a good starting point. centered at 45 in elevation


	for i in range(len(vrange)):


		azimuthal[i]    = np.sqrt((sigmax[i]**2 * sigmay[i]**2)/(sigmax[i]**2 * np.sin(np.deg2rad(AZ-phi0))**2 + sigmay[i]**2*np.cos(np.deg2rad(AZ-phi0))**2)) 
		#print('sos')        
		beam[i]         =  ab[i] * bfcnormal + avar[i] * np.exp(-(EL-el0)**2/(2*azimuthal[i]**2))  #this creates the elliptical gaussian   

		bfcprofile[i]   =  ab[i] * bfcnormal #this is the gamma distribution with different amplitudes in front



	if save_plot== 'yes':
		size_x = 8 #these are the dimensions of the saved plots
		size_y = 7

		path_plot_save = '/Users/katie/map_scaling/beamcartview/' #this is the folder where the plots are saved

		f1 = plt.figure(num=1, figsize=(size_x, size_y))
		for i in range(len(beam)):
			hp.cartview(beam[i],nest = True,coord = 'GC',unit = 'Normalized Gain',min = 0, max = 1.1, title = save_plot_name_prefix + ' '+ str(vrange[i]) + ' MHz');hp.graticule()
			x = str(int(i + 1))
			y = x.zfill(3)
			plt.savefig(path_plot_save + save_plot_name_prefix + '_'  +y + '.png', bbox_inches='tight')
			plt.close()   

	return beam
#when we talk about elevation , peak is 90, minimum goes to zero (gain vs elevation) 
#need to add azimuth dependence



def scaling(haslam, v): #this function scales the haslam map down to different frequencies

	beta = -2.5 #spectral index
	Tcmb = 2.725
	haslam_scaled = Tcmb + (haslam-Tcmb)*(v/408)**beta

	return haslam_scaled #the scaled haslam map

def first_gaussian(az,el,a=1,fwhm=80,el0=90):
	#AZ,EL,haslam = test()

	sigma = fwhm/(2*np.sqrt(2*np.log(2)))
	#gaussian = []
	#for i in rangeLen((el)):
	eqn = a*np.exp(-(el-el0)**2/(2*sigma**2))
	#gaussian.append(eqn)
	return eqn


#def weighted_average(vrange,sigma,az,el,a=1,el0=90):
	#AZ, EL,gaussian,haslam,haslam_scaled,gaussian= test()
	#haslam_scaled = []
	#gaussian = []
	#num = np.zeros(len(vrange))
	#denom = np.zeros(len(vrange))
	#tant = np.zeros(len(vrange))

	for i in range(len(sigma)):
		scales = scaling(haslam,vrange[i])

		haslam_scaled.append(scales)
		#print(i)
		eqn = a*np.exp(-(EL-el0)**2/(2*sigma[i]**2))
		gaussian.append(eqn)
		num[i] = np.sum(haslam_scaled[i]*gaussian[i])
		denom[i] = np.sum(gaussian[i])
		avg = num/denom

	gaussian=np.array(gaussian)   
	haslam_scaled = np.array(haslam_scaled)

	#wavg = np.array(avg)      





	#haslam_scaled = np.array(haslam_scaled)


	return gaussian, haslam_scaled, avg




def prac_ordering(yar,sar):

	def yargh(yar):

		if yar =='bear':
			a = 12
			b = 13
			return a,b
		elif yar == 'mom':
			return 4*3 + 2 , 'hi' 
		if sar =='yo':
			return 'roflcopter'
		else:
			return 'there was nothing' , 69
	def sargh(sar):           
		if sar == 'yo':
			return 'i hate my job'
		else:
			return'someone save me'
	a,b=yargh(yar)
	y=sargh(sar)

	return a,b,y

#elif model_type   ==  'bifurcation':
	#shape =  5.5 #5.5
	#scale =  10 #10
	#AZ1=45
	#a = 1
	#az0 =0
	#phi = AZ
	#sigmay = 20
	#sigmax = 20 
	#AZ0= 0
	##sigma = 50


	##x = np.arange(0,90,1)
	##x  = np.linspace(90,0,49152)
	#x = EL
	#gamma = 1/(special.gamma(shape)) #this is the gamma function
	#eqn1  = 1/(scale**shape) * (90-EL)**(shape-1) * np.exp(-(90-EL)/scale) 
	#gamma_distrib = gamma*eqn1 #this is the gamma distribution. modifies the profile of the beam. Function of scale, theta, and 90-elevation
	#bfc =  (np.cos(np.deg2rad(2*AZ-2*AZ1)) + 1) * gamma_distrib #this is to make the gamma distribution also azimuthally dependent
	#bfcnormal = .5 *bfc/np.max(bfc) #normalizing
	##gdb = gamma_distrib/np.max(gamma_distrib) #normalizing the gamma function

	#azimuthal = np.sqrt((sigmax**2 * sigmay**2)/(sigmax**2 * np.sin(np.deg2rad(AZ-AZ0))**2 + sigmay**2*np.cos(np.deg2rad(AZ-AZ0))**2))                                

	#beam  = bfcnormal + a*np.exp(-(EL-el0)**2/(2*sigma**2))  #this creates the elliptical gaussian 