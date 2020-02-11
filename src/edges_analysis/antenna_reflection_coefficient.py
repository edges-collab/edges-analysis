
import numpy as np
from src.edges_analysis import reflection_coefficient as rc
import receiver_calibration as rcv

from os.path import exists
from os import makedirs

home_folder = '/data5/raul'





def corrected(band, year, day, save='no', save_name_flag=''):


	# Mid-Band
	if (band == 'mid_band'):
		if (year == 2018) and (day == 145):
			path_antenna_s11 = home_folder + '/EDGES/calibration/antenna_s11/mid_band/raw/mid-recv1-20180525/'	
			date_all = ['145_08_28_27', '145_08_28_27']
	
	
		# Mid-Band
		if (year == 2018) and (day == 147):
			path_antenna_s11 = home_folder + '/EDGES/calibration/antenna_s11/mid_band/raw/mid-rcv1-20180527/'	
			date_all = ['147_17_03_25', '147_17_03_25']
	
	
		# Mid-Band
		if (year == 2018) and (day == 222):
			path_antenna_s11 = home_folder + '/EDGES/calibration/antenna_s11/mid_band/raw/mid_20180809/'
			date_all = ['222_05_41_09', '222_05_43_53', '222_05_45_58', '222_05_48_06']
	
	
		# Alan's noise source + 6dB attenuator
		if (year == 2018) and (day == 223):
			path_antenna_s11 = home_folder + '/EDGES/calibration/antenna_s11/mid_band/raw/mid_20180812_ans1_6db/'
			date_all = ['223_22_40_15', '223_22_41_22', '223_22_42_44']  # Leaving the first one out: '223_22_38_08'






	# Low-Band 3 (Receiver 1, Low-Band 1 antenna, High-Band ground plane with wings)
	if (band == 'low_band3'):
		if (year == 2018) and (day == 225):  
			path_antenna_s11 = home_folder + '/EDGES/calibration/antenna_s11/low_band3/raw/low3_20180813_rcv1/'
			date_all = ['225_12_17_56', '225_12_19_22', '225_12_20_39', '225_12_21_47']

		if (year == 2018) and (day == 227):  
			path_antenna_s11 = home_folder + '/EDGES/calibration/antenna_s11/low_band3/raw/low3-20180815/'
			date_all = ['227_12_31_03', '227_12_33_12', '227_12_35_27', '227_12_36_34']












	# Load and correct individual measurements
	for i in range(len(date_all)):

		# Four measurements
		o_m, f_m  = rc.s1p_read(path_antenna_s11 + '2018_' + date_all[i] + '_input1.s1p')
		s_m, f_m  = rc.s1p_read(path_antenna_s11 + '2018_' + date_all[i] + '_input2.s1p')
		l_m, f_m  = rc.s1p_read(path_antenna_s11 + '2018_' + date_all[i] + '_input3.s1p')
		a_m, f_m  = rc.s1p_read(path_antenna_s11 + '2018_' + date_all[i] + '_input4.s1p')


		# Standards assumed at the switch
		o_sw =  1 * np.ones(len(f_m))
		s_sw = -1 * np.ones(len(f_m))
		l_sw =  0 * np.ones(len(f_m))


		# Correction at switch
		a_sw_c, x1, x2, x3  = rc.de_embed(o_sw, s_sw, l_sw, o_m, s_m, l_m, a_m)


		# Correction at receiver input
		a = rcv.switch_correction_receiver1_2018_01_25C(a_sw_c, f_in = f_m)

		if i == 0:
			a_all = np.copy(a[0])

		elif i > 0:
			a_all = np.vstack((a_all, a[0]))





	# Average measurement	
	f  = f_m/1e6
	av = np.mean(a_all, axis=0)			


	# Output array
	ra_T = np.array([f, np.real(av), np.imag(av)])
	ra   = ra_T.T






	# Save
	if save == 'yes':
		# Creating folder if necessary
		path_save = home_folder + '/EDGES/calibration/antenna_s11/' + band + '/corrected/' + str(year) + '_' + str(day) + '/'
		if not exists(path_save):
			makedirs(path_save)	
		
		np.savetxt(path_save + 'antenna_s11_' + band + '_' + str(year) + '_' + str(day) + save_name_flag + '.txt', ra, header='freq [MHz]   real(s11)   imag(s11)')

	return ra, a_all




