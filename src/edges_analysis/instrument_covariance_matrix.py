



import numpy as np
import os

from src.edges_analysis import basic as ba, edges as eg

edges_folder       = os.environ['EDGES']





def test(band, FLOW=60, FHIGH=160):

	receiver_cal_file = 1
	antenna_s11_year  = 2018
	antenna_s11_day   = 147
	antenna_s11_Nfit  = 14
	
	


	
	# Receiver calibration quantities
	# -------------------------------
	if receiver_cal_file == 1:
		print('Receiver calibration FILE 1')
		rcv_file = edges_folder + band + '/calibration/receiver_calibration/receiver1/2018_01_25C/results/nominal/calibration_files/calibration_file_mid_band_cfit6_wfit14.txt'
	
	elif receiver_cal_file == 2:
		print('Receiver calibration FILE 2')
		rcv_file = edges_folder + band + '/calibration/receiver_calibration/receiver1/2018_01_25C/results/nominal/calibration_files/calibration_file_mid_band_cfit10_wfit14.txt'
	
	rcv = np.genfromtxt(rcv_file)

	fx      = rcv[:,0]
	rcv2    = rcv[(fx>=FLOW) & (fx<=FHIGH),:]
	
	f       = rcv2[:,0]
	rl      = rcv2[:,1] + 1j*rcv2[:,2]
	C1      = rcv2[:,3]
	C2      = rcv2[:,4]
	TU      = rcv2[:,5]
	TC      = rcv2[:,6]
	TS      = rcv2[:,7]
	
	

	# Antenna S11
	# -----------
	ra = eg.models_antenna_s11_remove_delay(band, f, year=antenna_s11_year, day=antenna_s11_day, delay_0=0.17, model_type='polynomial', Nfit=antenna_s11_Nfit, plot_fit_residuals='no')


		
	# Balun+Connector Loss
	# --------------------
	Gb, Gc   = eg.balun_and_connector_loss(f, ra)
	G        = Gb*Gc	



	# Ambient temperature
	# -------------------
	t_amb = 273.15 + 25
	



	
	
	
	
	
	
	
	# Generating calibrated input data
	f0  = 100 # MHz
	t_b = 1000*(f/f0)**(-2.5)
	
	
	# Uncalibrating input data
	t_lb = G * t_b   +   (1 - G) * t_amb
	t_3p = ba.uncalibrated_antenna_temperature(t_lb, ra, rl, C1, C2, TU, TC, TS)
	





	# Perturbed calibration quantities
	mag_ra = np.abs(ra)
	ang_ra = np.unwrap(np.angle(ra))
	
	
	X_mag_ra = mag_ra + 0.0005
	X_ang_ra = ang_ra + 0 #*(np.pi/180)

	
	X_ra    = ra+0 #X_mag_ra * (np.cos(X_ang_ra) + 1j*np.sin(X_ang_ra))

	X_rl    = rl+0
	X_C1    = C1*(1+0.1)
	X_C2    = C2+0
	X_TU    = TU+0
	X_TC    = TC+0
	X_TS    = TS+0
	X_G     = G+0
	X_t_amb = t_amb+0



	# Calibrated uncalibrated data
	X_t_lb = ba.calibrated_antenna_temperature(t_3p, X_ra, X_rl, X_C1, X_C2, X_TU, X_TC, X_TS)
	X_t_b  = (X_t_lb  -  X_t_amb * (1 - X_G)) / X_G
	
	
	
	
	return f, t_b, X_t_b

