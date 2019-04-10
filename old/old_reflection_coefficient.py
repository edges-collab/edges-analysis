

import numpy as np
import scipy as sp
from os import listdir





def impedance2gamma(Z,Z0):
	return (Z-Z0)/(Z+Z0)

def gamma2impedance(r,Z0):
	return Z0*(1+r)/(1-r)





def gamma_de_embed(s11,s12s21,s22,rp):
	return (rp - s11)/(s22*(rp-s11) + s12s21)

def gamma_shifted(s11,s12s21,s22,r):
	return s11 + (s12s21*r/(1-s22*r))






def s1p_read(path_filename):

	# identifying the format
	d = open(path_filename, 'r')

	row          = 0
	comment_rows = 0

	while (comment_rows == 0):

		# reading line
		temp_line = d.readline()

		# checking settings line
		if temp_line[0] == '#':
			if ('DB' in temp_line) or ('dB' in temp_line):
				flag = 'DB'

			if 'MA' in temp_line:
				flag = 'MA'

			if 'RI' in temp_line:
				flag = 'RI'

		# counting number of lines commented out
		if ('#' in temp_line) or ('!' in temp_line):
			row = row + 1
		else:
			comment_rows = row	

	d.close()



	# loading data
	d = np.genfromtxt(path_filename, skip_header=comment_rows)
	f = d[:,0]

	if flag == 'DB':
		r = 10**(d[:,1]/20) * (np.cos((np.pi/180)*d[:,2]) + 1j*np.sin((np.pi/180)*d[:,2]))

	if flag == 'MA':
		r = d[:,1] * (np.cos((np.pi/180)*d[:,2]) + 1j*np.sin((np.pi/180)*d[:,2]))

	if flag == 'RI':
		r = d[:,1] + 1j*d[:,2]

	return (r, f)







def de_embed(r1a, r2a, r3a, r1m, r2m, r3m, rp):
# This only works with 1D arrays, where each point in the array is 
# a value at a given frequency

# The output is also a 1D array

	
	s11    = np.zeros(len(r1a)) + 0j   # 0j added to make array complex
	s12s21 = np.zeros(len(r1a)) + 0j
	s22    = np.zeros(len(r1a)) + 0j

	for i in range(len(r1a)):
		b = np.array([r1m[i], r2m[i], r3m[i]])#.reshape(-1,1)
		A = np.array([[1, r1a[i], r1a[i]*r1m[i]],
		              [1, r2a[i], r2a[i]*r2m[i]],
		              [1, r3a[i], r3a[i]*r3m[i]]])
		x = np.linalg.lstsq(A,b)[0]
		s11[i]    = x[0]
		s12s21[i] = x[1]+x[0]*x[2]
		s22[i]    = x[2]	

	r = gamma_de_embed(s11, s12s21, s22, rp)

	return r, s11, s12s21, s22








def fiducial_parameters_85033E(R, md = 1, md_value_ps = 38):   # 38

	# Parameters of open
	open_off_Zo     =  50 
	open_off_delay  =  29.243e-12
	open_off_loss   =  2.2*1e9
	open_C0 	=  49.43e-15 
	open_C1 	= -310.1e-27
	open_C2 	=  23.17e-36
	open_C3 	= -0.1597e-45

	op = np.array([open_off_Zo, open_off_delay, open_off_loss, open_C0, open_C1, open_C2, open_C3])



	# Parameters of short
	short_off_Zo    =  50 
	short_off_delay =  31.785e-12
	short_off_loss  =  2.36*1e9
	short_L0 	=  2.077e-12
	short_L1 	= -108.5e-24
	short_L2 	=  2.171e-33
	short_L3 	= -0.01e-42

	sp = np.array([short_off_Zo, short_off_delay, short_off_loss, short_L0, short_L1, short_L2, short_L3])



	# Parameters of match
	match_off_Zo    = 50

	if md == 0:
		match_off_delay = 0
	elif md == 1:
		match_off_delay = md_value_ps*1e-12   # 38 ps, from Monsalve et al. (2016)
	match_off_loss  = 2.3*1e9
	match_R         = R

	mp = np.array([match_off_Zo, match_off_delay, match_off_loss, match_R])

	return (op, sp, mp)







def standard_open(f, par):
	"""
	frequency in Hz
	"""

	offset_Zo    	= par[0]
	offset_delay 	= par[1]
	offset_loss  	= par[2]
	C0 		= par[3]
	C1 		= par[4]
	C2 		= par[5]
	C3 		= par[6]


	# Termination 
	Ct_open = C0 + C1*f + C2*f**2 + C3*f**3
	Zt_open = 0 - 1j/(2*np.pi*f*Ct_open)
	Rt_open = impedance2gamma(Zt_open,50)


	# Transmission line
	Zc_open  = (offset_Zo + (offset_loss/(2*2*np.pi*f))*np.sqrt(f/1e9)) - 1j*(offset_loss/(2*2*np.pi*f))*np.sqrt(f/1e9)
	temp     = ((offset_loss*offset_delay)/(2*offset_Zo))*np.sqrt(f/1e9)
	gl_open  = temp + 1j*( (2*np.pi*f)*offset_delay + temp )


	# Combined reflection coefficient
	R1      = impedance2gamma(Zc_open,50)
	ex      = np.exp(-2*gl_open)
	Rt      = Rt_open
	Ri_open = ( R1*(1 - ex - R1*Rt) + ex*Rt ) / ( 1 - R1*(ex*R1 + Rt*(1 - ex)) )

	return Ri_open







def standard_short(f, par):
	"""
	frequency in Hz
	"""
	
	offset_Zo    	= par[0]
	offset_delay 	= par[1]
	offset_loss  	= par[2]
	L0 		= par[3]
	L1 		= par[4]
	L2 		= par[5]
	L3 		= par[6]



	# Termination
	Lt_short = L0 + L1*f + L2*f**2 + L3*f**3
	Zt_short = 0 + 1j*2*np.pi*f*Lt_short
	Rt_short = impedance2gamma(Zt_short,50)



	# Transmission line %%%%
	Zc_short = (offset_Zo + (offset_loss/(2*2*np.pi*f))*np.sqrt(f/1e9)) - 1j*(offset_loss/(2*2*np.pi*f))*np.sqrt(f/1e9)
	temp     = ((offset_loss*offset_delay)/(2*offset_Zo))*np.sqrt(f/1e9)
	gl_short = temp + 1j*( (2*np.pi*f)*offset_delay + temp )


	# Combined reflection coefficient %%%%
	R1       = impedance2gamma(Zc_short,50)
	ex       = np.exp(-2*gl_short)
	Rt       = Rt_short
	Ri_short = ( R1*(1 - ex - R1*Rt) + ex*Rt ) / ( 1 - R1*(ex*R1 + Rt*(1 - ex)) )

	return Ri_short







def standard_match(f, par):
	"""
	frequency in Hz
	"""

	offset_Zo    = par[0]
	offset_delay = par[1]
	offset_loss  = par[2]
	Resistance   = par[3]


	# Termination	
	Zt_match = Resistance
	Rt_match = impedance2gamma(Zt_match,50)


	# Transmission line
	Zc_match = (offset_Zo + (offset_loss/(2*2*np.pi*f))*np.sqrt(f/1e9)) - 1j*(offset_loss/(2*2*np.pi*f))*np.sqrt(f/1e9)
	temp     = ((offset_loss*offset_delay)/(2*offset_Zo))*np.sqrt(f/1e9)
	gl_match = temp + 1j*( (2*np.pi*f)*offset_delay + temp )


	# combined reflection coefficient %%%%
	R1       = impedance2gamma(Zc_match,50)
	ex       = np.exp(-2*gl_match)
	Rt       = Rt_match
	Ri_match = ( R1*(1 - ex - R1*Rt) + ex*Rt ) / ( 1 - R1*(ex*R1 + Rt*(1 - ex)) )
	
	
	## Alan's approach
	#Zcab = np.copy(Zc_match)
	#T    = ((Resistance-Zcab)/(Resistance+Zcab)) * ex
	#Z    = Zcab*((1+T)/(1-T))
	#Rx   = (Z-50)/(Z+50)

	
	return Ri_match  #, Rx







def agilent_85033E(f, resistance_of_match, m = 1, md_value_ps = 38):# 8):
	"""
	frequency in Hz
	"""
	
	op, sp, mp = fiducial_parameters_85033E(resistance_of_match, md = m, md_value_ps = md_value_ps)
	o = standard_open(f, op)
	s = standard_short(f, sp)
	m = standard_match(f, mp)

	return (o, s, m)








def input_impedance_transmission_line(Z0, gamma, length, Zload):

	"""
	Z0:     complex characteristic impedance
	gamma:  propagation constant
	length: length of transmission line
	Zload:  impedance of termination
	"""


	Zin = Z0 * (Zload + Z0*np.tanh(gamma*length)) / (Zload*np.tanh(gamma*length) + Z0)
	
	
	return Zin













def terminated_airline_8043S15(f, gamma_DUT):   
	
	"""
	Maury 8043S15 airline
	
	"""
	

	
	
	# ---------- MODEL OF AIRLINE ------------------------------
	# ----------------------------------------------------------
	
	# Angular frequency
	w = 2 * np.pi * f   # f in Hz

	# Airline dimensions
	inch2m = 1/39.370               # inch-to-meters conversion
	xx     = 1.00                   # fraction of inner radius
	yy     = 1.0                    # fraction of outer radius	
	ric    = xx*(0.05984*inch2m)/2  # radius of outer wall of inner conductor in meters
	roc    = yy*(0.13780*inch2m)/2  # radius of inner wall of outer conductor in meters
	length = 0.1499 - 0.000025      # in meters. uncertainty is +/-0.000025 m (https://www.maurymw.com/Precision/3.5mm_beadless.php)

	# Permeability
	u0        = 4*np.pi*10**(-7)   # permeability of free space (same for copper, brass, etc., all nonmagnetic materials)
	ur_air    = 1                  # relative permeability of air
	u_air     = u0 * ur_air

	# Permittivity
	c                = 299792458       # speed of light
	e0               = 1/(u0 * c**2)   # permittivity of free space	
	er_air           = 1.0             #
	ep_air           = e0 * er_air
	tan_delta_air    = 0 
	epp_air          = ep_air    * tan_delta_air	

	
	# Metal conductivity
	sigma_copper          = 5.813 * 10**7    # Pozar 3rd edition  #  5.96  # 5.813
	sigma_gold            = 4.1   * 10**7    # Wikipedia
	sigma_copper_fraction = 0.165            # value obtained in the IEEE Standards paper   # 0.22  
	sigma                 = sigma_copper_fraction * sigma_copper

	# Skin Depth
	skin_depth = np.sqrt(2 / (w * u0 * sigma))  # For copper, the skin depth is 3.3e-6 m at 400 MHz,   2.1e-6 m at 1 GHz. For gold it is 3.9e-6 m at 400 MHz,   2.5e-6 m at 1 GHz  

	# For reference:
	# The gold   flushing of the airline is 1e-5*inch2m = 2.5e-7 (about ten times smaller than the skin depth at 1 GHz ??)
	# The copper flushing of the airline is 1e-5*inch2m = 2.5e-7 (about ten times smaller than the skin depth at 1 GHz ??)
	# The actual airline is built from beryllium copper (not the flushing), https://www.maurymw.com/Precision/3.5mm_beadless.php 

	# Surface resistance
	Rs = 1 / (sigma * skin_depth)

	# The actual "equivalent conductivity" has to be a fit parameter.



	# Transmission Line Parameters
	# ----------------------------

	# Inductance per unit length
	L_inner  = u0 * skin_depth / (4 * np.pi * ric)
	L_dielec = (u_air / (2 * np.pi)) * np.log(roc/ric) 
	L_outer  = u0 * skin_depth / (4 * np.pi * roc)
	L        = L_inner + L_dielec + L_outer

	# Capacitance per unit length	
	C = 2 * np.pi * ep_air / np.log(roc/ric)

	# Resistance per unit length
	R = (Rs / (2 * np.pi * ric))   +   (Rs / (2 * np.pi * roc))

	# Conductance per unit length
	G = 2 * np.pi * w * epp_air / np.log(roc/ric)


	# Propagation constant
	gamma = np.sqrt( (R + 1j*w*L) * (G + 1j*w*C) )

	# Complex Cable Impedance
	Zchar = np.sqrt( (R + 1j*w*L) / (G + 1j*w*C) )	


	# -----------------------------------------------------------
	# -----------------------------------------------------------





	# Option 1 (my approach)
	# ---------------------------
	
	# Impedance of input DUT
	Zref  = 50
	Z_DUT = gamma2impedance(gamma_DUT, Zref)

	# Impedance of terminated airline
	Zin_DUT  = input_impedance_transmission_line(Zchar, gamma, length, Z_DUT)	

	# Reflection of terminated airline
	rin_DUT = impedance2gamma(Zin_DUT,  Zref)
	
	
	
	## Option 2 (Alan's) It produces same result
	## ----------------------------
	#Zx = 50*(1+gamma_DUT)/(1-gamma_DUT)
	#GG = (Zx-Zchar)/(Zx+Zchar)
	#GG = GG*np.exp(-gamma*2*length)
	#Zx = Zchar*(1+GG)/(1-GG)
	#rin_DUT = (Zx-50)/(Zx+50)
	
	
	
	return rin_DUT       # f, gamma_DUT, L, C, R, G, Zchar, gamma, rin_DUT





