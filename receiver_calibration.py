

import numpy as np
import reflection_coefficient as rc

home_folder = '/data5/raul'








def switch_correction_receiver1_2018_01_25C(ant_s11, f_in = np.zeros([0,1]), case = 1):  


	"""

	The characterization of the 4-position switch was done using the male standard of Phil's kit

	"""


	# Loading measurements
	path_folder         = home_folder + '/EDGES/calibration/receiver_calibration/receiver1/2018_01_25C/data/s11/raw/InternalSwitch/'
	resistance_of_match = 50.027 # male from Phil's calkit
	
	if case == 1:
			
		o_in, f = rc.s1p_read(path_folder + 'Open01.s1p')
		s_in, f = rc.s1p_read(path_folder + 'Short01.s1p')
		l_in, f = rc.s1p_read(path_folder + 'Match01.s1p')

		o_ex, f = rc.s1p_read(path_folder + 'ExternalOpen01.s1p')
		s_ex, f = rc.s1p_read(path_folder + 'ExternalShort01.s1p')
		l_ex, f = rc.s1p_read(path_folder + 'ExternalMatch01.s1p')


	if case == 2:

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



