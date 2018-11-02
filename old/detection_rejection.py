


from os.path import expanduser

import numpy as np
import matplotlib.pyplot as plt
import edges as eg

# Determining home folder
home_folder = expanduser("~")





















def high_band_physical_detection_rejection():

	#path = '/DATA/EDGES/results/high_band/model_rejection/20161123/figures/'
	path = '/Desktop/'


	data = np.genfromtxt(home_folder + '/DATA/EDGES/results/high_band/high_band_average_low_resolution.txt')
	f = data[:,0]
	d = data[:,1]
	w = data[:,2]

	fb, rb_all, wb, RMS_all, prob, chi_sq, df, model_EoR_all, sigma_model = eg.model_rejection_anastasia_jordan_hyper(f, d, w, model_type='anastasia', fit_type='EDGES_polynomial', flow=89, fhigh=191, nterm_fg=5)

	plt.close()
	plt.close()
	plt.close()
	plt.close()
	plt.close()
	plt.close()
	plt.close()
	plt.close()


	size_x = 9
	size_y = 7.5
	f1 = plt.figure(num=1, figsize=(size_x, size_y))
	plt.subplot(2,1,1)
	plt.plot(f[w>0],  d[w>0], 'b', linewidth=2)
	plt.xlim([80, 200])
	plt.ylim([0, 1100])
	plt.yticks(np.arange(200,1001,200))
	plt.grid()
	plt.ylabel('brightness temperature [K]')
	plt.text(120, 900, 'PRELIMINARY', color='cyan', fontsize=20, rotation=45)
	plt.xticks(np.arange(80,201,20),[])

	plt.subplot(2,1,2)
	plt.plot(fb[wb>0],  1000*rb_all[0,wb>0], 'b', linewidth=2)
	plt.plot(fb,  1000*sigma_model, 'r', linewidth=2)
	plt.plot(fb, -1000*sigma_model, 'r', linewidth=2)
	plt.xlim([80, 200])
	plt.ylim([-150, 150])
	plt.text(160, 100, 'RMS = 20 mK')
	plt.grid()
	plt.xlabel('frequency [MHz]')
	plt.ylabel('residuals [mK]')
	plt.text(120, 70, 'PRELIMINARY', color='cyan', fontsize=20, rotation=45)

	plt.savefig(home_folder + path + 'nominal_uncertainty.pdf', bbox_inches='tight')
	plt.close()	
	plt.close()	






	size_x = 9
	size_y = 7.5
	f1 = plt.figure(num=1, figsize=(size_x, size_y))
	index = np.arange(0, len(prob))
	ms = model_EoR_all[prob<=5,:]
	rs = rb_all[prob<=5,:]

	ms0 = model_EoR_all[prob>5,:]
	rs0 = rb_all[prob>5,:]	

	plt.subplot(2,1,1)
	plt.plot(fb, 1000*ms[0:5,:].T)
	plt.plot(fb, 1000*ms[5:10,:].T, '--')
	plt.plot(fb, 1000*ms[10::,:].T, '.-')
	plt.xlim([80, 200])
	plt.ylim([-350, 50])
	plt.grid()
	#plt.xlabel('frequency [MHz]')
	plt.ylabel('brightness temperature [mK]')
	plt.legend([str(index[prob<=5][0]), str(index[prob<=5][1]), str(index[prob<=5][2]), str(index[prob<=5][3]), str(index[prob<=5][4]), str(index[prob<=5][5]), str(index[prob<=5][6]), str(index[prob<=5][7]), str(index[prob<=5][8]), str(index[prob<=5][9]), str(index[prob<=5][10]), str(index[prob<=5][11]), str(index[prob<=5][12]), str(index[prob<=5][13])], loc=0, fontsize=8)
	plt.text(140, -150, 'PRELIMINARY', color='red', fontsize=20, rotation=-45)
	plt.title('Currently Rejected at 95%', fontsize=18)
	plt.xticks(np.arange(80,201,20),[])

	plt.subplot(2,1,2)
	plt.plot(fb, 1000*ms0.T)
	plt.xlim([80, 200])
	plt.ylim([-350, 50])
	plt.grid()
	plt.xlabel('frequency [MHz]')
	plt.ylabel('brightness temperature [mK]')
	plt.text(140, -150, 'PRELIMINARY', color='cyan', fontsize=20, rotation=-45)
	plt.title('Currently NOT Rejected', fontsize=18)

	plt.savefig(home_folder + path + 'models_fialkov.pdf', bbox_inches='tight')
	plt.close()	
	plt.close()	








	size_x = 7
	size_y = 6
	f1 = plt.figure(num=1, figsize=(size_x, size_y))
	index = np.arange(0, len(prob))
	ms = model_EoR_all[prob<=5,:]
	rs = rb_all[prob<=5,:]

	ms0 = model_EoR_all[prob>5,:]
	rs0 = rb_all[prob>5,:]	

	plt.plot(fb, 1000*ms[0:5,:].T)
	plt.plot(fb, 1000*ms[5:10,:].T, '--')
	plt.plot(fb, 1000*ms[10::,:].T, '.-')
	plt.xlim([80, 200])
	plt.ylim([-350, 50])
	plt.grid()
	#plt.xlabel('frequency [MHz]')
	plt.ylabel('brightness temperature [mK]')
	plt.legend([str(index[prob<=5][0]), str(index[prob<=5][1]), str(index[prob<=5][2]), str(index[prob<=5][3]), str(index[prob<=5][4]), str(index[prob<=5][5]), str(index[prob<=5][6]), str(index[prob<=5][7]), str(index[prob<=5][8]), str(index[prob<=5][9]), str(index[prob<=5][10]), str(index[prob<=5][11]), str(index[prob<=5][12]), str(index[prob<=5][13])], loc=0, fontsize=8)
	plt.text(140, -150, 'PRELIMINARY', color='red', fontsize=20, rotation=-45)
	plt.title('Currently Rejected at 95%', fontsize=18)
	plt.xticks(np.arange(80,201,20))
	plt.xlabel(r'$\nu$ [MHz]')

	plt.savefig(home_folder + path + 'models_fialkov2.pdf', bbox_inches='tight')
	plt.close()	
	plt.close()	















	size_x = 9
	size_y = 7.5
	f1 = plt.figure(num=1, figsize=(size_x, size_y))
	plt.subplot(2,1,1)
	plt.plot(fb[wb>0], np.abs(1000*rs[0:5,wb>0].T), 'c')
	plt.plot(fb[wb>0], np.abs(1000*rs[5:10,wb>0].T), 'c')
	plt.plot(fb[wb>0], np.abs(1000*rs[10::,wb>0].T), 'c')
	plt.plot(fb[wb>0],  np.abs(1000*rb_all[0,wb>0]), 'k', linewidth=2)
	plt.yticks(np.arange(0,161,20))
	plt.xlim([80, 200])
	plt.ylim([0, 160])
	plt.grid()
	#plt.xlabel('frequency [MHz]')
	plt.ylabel('|residuals| [mK]')
	plt.text(120, 100, 'PRELIMINARY', color='red', fontsize=20, rotation=45)
	plt.title('Currently Rejected at 95%', fontsize=18)
	plt.xticks(np.arange(80,201,20),[])

	plt.subplot(2,1,2)
	plt.plot(fb[wb>0], np.abs(1000*rs0[:,wb>0].T), 'c')
	plt.plot(fb[wb>0],  np.abs(1000*rb_all[0,wb>0]), 'k', linewidth=2)
	plt.yticks(np.arange(0,161,20))
	plt.xlim([80, 200])
	plt.ylim([0, 160])
	plt.grid()
	plt.xlabel('frequency [MHz]')
	plt.ylabel('|residuals| [mK]')	
	plt.text(120, 100, 'PRELIMINARY', color='red', fontsize=20, rotation=45)
	plt.title('Currently NOT Rejected', fontsize=18)

	plt.savefig(home_folder + path + 'models_fialkov_residuals.pdf', bbox_inches='tight')
	plt.close()	
	plt.close()	




	fb, rb_all, wb, RMS_all, prob, chi_sq, df, model_EoR_all, sigma_model = eg.model_rejection_anastasia_jordan_hyper(f, d, w, model_type='jordan', fit_type='EDGES_polynomial', flow=89, fhigh=191, nterm_fg=5)






	index = np.arange(0, len(prob))
	ms1 = model_EoR_all[1:34,:]
	rs1 = rb_all[1:34,:]
	pr1 = prob[1:34]
	in1 = index[1:34]

	ms11 = ms1[pr1<=5,:]
	rs11 = rs1[pr1<=5,:]

	ms10 = ms1[pr1>5,:]
	rs10 = rs1[pr1>5,:]



	ms2 = model_EoR_all[34::,:]
	rs2 = rb_all[34::,:]	
	pr2 = prob[34::]

	ms22 = ms2[pr2<=5,:]
	rs22 = rs2[pr2<=5,:]

	ms20 = ms2[pr2>5,:]
	rs20 = rs2[pr2>5,:]





	size_x = 9
	size_y = 7.5
	f1 = plt.figure(num=1, figsize=(size_x, size_y))
	plt.subplot(2,1,1)
	plt.plot(fb, 1000*ms11.T)
	plt.xlim([80, 200])
	plt.ylim([-350, 50])
	plt.grid()
	#plt.xlabel('frequency [MHz]')
	plt.ylabel('brightness temperature [mK]')
	plt.legend([str(in1[pr1<=5][0]), str(in1[pr1<=5][1]), str(in1[pr1<=5][2]), str(in1[pr1<=5][3]), str(in1[pr1<=5][4]), str(in1[pr1<=5][5]), str(in1[pr1<=5][6]), str(in1[pr1<=5][7]), str(in1[pr1<=5][8]), str(in1[pr1<=5][9]), str(in1[pr1<=5][10]), str(in1[pr1<=5][11]), str(in1[pr1<=5][12]), str(in1[pr1<=5][13]), str(in1[pr1<=5][14]), str(in1[pr1<=5][15]), str(in1[pr1<=5][16])], loc=0, fontsize=8)
	plt.text(140, -150, 'PRELIMINARY', color='red', fontsize=20, rotation=-45)
	plt.title('Currently Rejected at 95%', fontsize=18)
	plt.xticks(np.arange(80,201,20),[])

	plt.subplot(2,1,2)
	plt.plot(fb, 1000*ms10.T)
	plt.xlim([80, 200])
	plt.ylim([-350, 50])
	plt.grid()
	plt.xlabel('frequency [MHz]')
	plt.ylabel('brightness temperature [mK]')
	plt.text(140, -150, 'PRELIMINARY', color='red', fontsize=20, rotation=-45)
	plt.title('Currently NOT Rejected', fontsize=18)

	plt.savefig(home_folder + path + 'models_mirocha_late.pdf', bbox_inches='tight')
	plt.close()	
	plt.close()	






	size_x = 9
	size_y = 7.5
	f1 = plt.figure(num=1, figsize=(size_x, size_y))
	plt.subplot(2,1,1)
	plt.plot(fb[wb>0], np.abs(1000*rs11[:,wb>0].T), 'c')
	plt.plot(fb[wb>0],  np.abs(1000*rb_all[0,wb>0]), 'k', linewidth=2)
	plt.yticks(np.arange(0,161,20))
	plt.xlim([80, 200])
	plt.ylim([0, 160])
	plt.grid()
	#plt.xlabel('frequency [MHz]')
	plt.ylabel('|residuals| [mK]')	
	plt.text(120, 100, 'PRELIMINARY', color='red', fontsize=20, rotation=45)
	plt.title('Currently Rejected at 95%', fontsize=18)
	plt.xticks(np.arange(80,201,20),[])

	plt.subplot(2,1,2)
	plt.plot(fb[wb>0], np.abs(1000*rs10[:,wb>0].T), 'c')
	plt.plot(fb[wb>0],  np.abs(1000*rb_all[0,wb>0]), 'k', linewidth=2)
	plt.yticks(np.arange(0,161,20))
	plt.xlim([80, 200])
	plt.ylim([0, 160])
	plt.grid()
	plt.xlabel('frequency [MHz]')
	plt.ylabel('|residuals| [mK]')		
	plt.text(120, 100, 'PRELIMINARY', color='red', fontsize=20, rotation=45)
	plt.title('Currently NOT Rejected', fontsize=18)

	plt.savefig(home_folder + path + 'models_mirocha_late_residuals.pdf', bbox_inches='tight')
	plt.close()	
	plt.close()	







	size_x = 7
	size_y = 6
	f1 = plt.figure(num=1, figsize=(size_x, size_y))
	plt.plot(fb, 1000*ms22.T)
	plt.xlim([80, 200])
	plt.ylim([-350, 50])
	plt.grid()
	plt.xlabel(r'$\nu$ [MHz]')
	plt.ylabel('brightness temperature [mK]')
	text = plt.text(130, -70, 'PRELIMINARY', color='yellow', fontsize=20, rotation=30)
	text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'), path_effects.Normal()])	
	plt.title('Currently Rejected at 95%', fontsize=18)

	plt.savefig(home_folder + path + 'models_mirocha_cold.pdf', bbox_inches='tight')
	plt.close()	
	plt.close()	



	size_x = 9
	size_y = 7.5
	f1 = plt.figure(num=1, figsize=(size_x, size_y))
	plt.plot(fb[wb>0], np.abs(1000*rs22[:,wb>0].T), 'c')
	plt.plot(fb[wb>0], np.abs(1000*rb_all[0,wb>0]), 'k', linewidth=2)
	plt.yticks(np.arange(0,161,20))
	plt.xlim([80, 200])
	plt.ylim([0, 160])
	plt.grid()
	plt.xlabel('frequency [MHz]')
	plt.ylabel('|residuals| [mK]')	
	plt.text(120, 100, 'PRELIMINARY', color='red', fontsize=20, rotation=45)
	plt.title('Currently Rejected at 95%', fontsize=18)

	plt.savefig(home_folder + path + 'models_mirocha_cold_residuals.pdf', bbox_inches='tight')
	plt.close()	
	plt.close()	


	return 0 #pr1 #ms20, ms22, ms2   #fb, rb_all, wb, RMS_all, prob, chi_sq, df, model_EoR_all, sigma_model























def plots_high_band_gaussian(models='no', residuals='no', rejections='no', rejected_models='no', rejections_many_amplitudes='no', rejections_simulated='no'):


	if models=='yes':

		ff, il, ih = eg.frequency_edges(90,190)
		fe = ff[il:ih+1]


		g01, xHI, z = eg.model_eor(fe, T21=-0.15, model_type='gaussian', zr=9, dz=2)
		g02, xHI, z = eg.model_eor(fe, T21=-0.15, model_type='gaussian', zr=10, dz=2)
		g03, xHI, z = eg.model_eor(fe, T21=-0.15, model_type='gaussian', zr=11, dz=2)
		g04, xHI, z = eg.model_eor(fe, T21=-0.15, model_type='gaussian', zr=12, dz=2)
		g05, xHI, z = eg.model_eor(fe, T21=-0.15, model_type='gaussian', zr=13, dz=2)
		g06, xHI, z = eg.model_eor(fe, T21=-0.15, model_type='gaussian', zr=14, dz=2)
		g07, xHI, z = eg.model_eor(fe, T21=-0.15, model_type='gaussian', zr=15, dz=2)
		g08, xHI, z = eg.model_eor(fe, T21=-0.15, model_type='gaussian', zr=16, dz=2)

		g11, xHI, z = eg.model_eor(fe, T21=-0.15, model_type='gaussian', zr=9, dz=4)
		g12, xHI, z = eg.model_eor(fe, T21=-0.15, model_type='gaussian', zr=10, dz=4)
		g13, xHI, z = eg.model_eor(fe, T21=-0.15, model_type='gaussian', zr=11, dz=4)
		g14, xHI, z = eg.model_eor(fe, T21=-0.15, model_type='gaussian', zr=12, dz=4)
		g15, xHI, z = eg.model_eor(fe, T21=-0.15, model_type='gaussian', zr=13, dz=4)
		g16, xHI, z = eg.model_eor(fe, T21=-0.15, model_type='gaussian', zr=14, dz=4)
		g17, xHI, z = eg.model_eor(fe, T21=-0.15, model_type='gaussian', zr=15, dz=4)
		g18, xHI, z = eg.model_eor(fe, T21=-0.15, model_type='gaussian', zr=16, dz=4)
		g19, xHI, z = eg.model_eor(fe, T21=-0.15, model_type='gaussian', zr=17, dz=4)
		g20, xHI, z = eg.model_eor(fe, T21=-0.15, model_type='gaussian', zr=18, dz=4)

		g21, xHI, z = eg.model_eor(fe, T21=-0.15, model_type='gaussian', zr=9, dz=6)
		g22, xHI, z = eg.model_eor(fe, T21=-0.15, model_type='gaussian', zr=10, dz=6)
		g23, xHI, z = eg.model_eor(fe, T21=-0.15, model_type='gaussian', zr=11, dz=6)
		g24, xHI, z = eg.model_eor(fe, T21=-0.15, model_type='gaussian', zr=12, dz=6)
		g25, xHI, z = eg.model_eor(fe, T21=-0.15, model_type='gaussian', zr=13, dz=6)
		g26, xHI, z = eg.model_eor(fe, T21=-0.15, model_type='gaussian', zr=14, dz=6)
		g27, xHI, z = eg.model_eor(fe, T21=-0.15, model_type='gaussian', zr=15, dz=6)
		g28, xHI, z = eg.model_eor(fe, T21=-0.15, model_type='gaussian', zr=16, dz=6)
		g29, xHI, z = eg.model_eor(fe, T21=-0.15, model_type='gaussian', zr=17, dz=6)
		g30, xHI, z = eg.model_eor(fe, T21=-0.15, model_type='gaussian', zr=18, dz=6)
		g31, xHI, z = eg.model_eor(fe, T21=-0.15, model_type='gaussian', zr=19, dz=6)
		g32, xHI, z = eg.model_eor(fe, T21=-0.15, model_type='gaussian', zr=20, dz=6)



		f1 = plt.figure(1, figsize=(6, 8))

		ax1 = f1.add_subplot(3,1,1)
		ax1.plot(z, 1000*g01)
		ax1.plot(z, 1000*g02)
		ax1.plot(z, 1000*g03)
		ax1.plot(z, 1000*g04)
		ax1.plot(z, 1000*g05)
		ax1.plot(z, 1000*g06)
		ax1.plot(z, 1000*g07)
		ax1.plot(z, 1000*g08)
		plt.xlim([15, 6])
		plt.ylim([-170, 20])
		plt.ylabel('$T_b$ [mK]', fontsize=16)
		ax1.set_xticks(np.arange(6,15,1))
		ax1.set_xticklabels([])
		plt.text(7.8, -120, '$\Delta z=2$', fontsize=20)

		ax2 = ax1.twiny()
		ax2.set_xlabel(r'$\nu$   [MHz]',fontsize=16)
		ax2.set_xticks(np.array((np.abs(eg.frequency2redshift(90)-15),np.abs(eg.frequency2redshift(110)-15),np.abs(eg.frequency2redshift(130)-15),np.abs(eg.frequency2redshift(150)-15),np.abs(eg.frequency2redshift(170)-15),np.abs(eg.frequency2redshift(190)-15),np.abs(6-15))))
		ax2.set_xticklabels(['90','110','130','150','170','190',''])



		plt.subplot(3,1,2)
		plt.plot(z, 1000*g13)
		plt.plot(z, 1000*g14)
		plt.plot(z, 1000*g15)
		plt.plot(z, 1000*g16)
		plt.plot(z, 1000*g17)
		plt.plot(z, 1000*g18)
		plt.plot(z, 1000*g19)
		plt.plot(z, 1000*g20)
		plt.xlim([15, 6])
		plt.ylim([-170, 20])
		plt.ylabel('$T_b$ [mK]', fontsize=16)
		plt.xticks(np.arange(6,15,1), [])
		plt.text(7.8, -120, '$\Delta z=4$', fontsize=20)

		plt.subplot(3,1,3)
		plt.plot(z, 1000*g25)
		plt.plot(z, 1000*g26)
		plt.plot(z, 1000*g27)
		plt.plot(z, 1000*g28)
		plt.plot(z, 1000*g29)
		plt.plot(z, 1000*g30)
		plt.plot(z, 1000*g31)
		plt.plot(z, 1000*g32)
		plt.xlim([15, 6])
		plt.ylim([-170, 20])
		plt.ylabel('$T_b$ [mK]', fontsize=16)
		plt.xlabel('$z$', fontsize=20)
		plt.text(7.8, -120, '$\Delta z=6$', fontsize=20)


		plt.savefig(home_folder + '/Desktop/gaussian_models.pdf', bbox_inches='tight')
		plt.close()
		plt.close()	








	if residuals == 'yes':

		d = np.genfromtxt(home_folder + '/DATA/EDGES/results/high_band/spectra/high_band_average_20161129_nominal.txt')
		f = d[:,0]
		t = d[:,1]
		w = d[:,2]


		fb, rb, wb, rms, XX   = eg.data_analysis_residuals_array(f, t.reshape((1,-1)), w.reshape((1,-1)), flow=89, fhigh=191, model_type='EDGES_polynomial', fnorm=140, nfg=5, binning='no', nsamples=64, rfi_flagging='no')
		#fb, rb2, wb2, rms, XX = eg.data_analysis_residuals_array(f, t.reshape((1,-1)), w.reshape((1,-1)), flow=89, fhigh=191, model_type='Physical_model', fnorm=140, nfg=5, binning='no', nsamples=64, rfi_flagging='no')


		f1 = plt.figure(1, figsize=(10, 6))
		plt.subplot(2,1,1)
		plt.plot(f[w>0], t[w>0], linewidth=2)
		plt.ylim([0, 1100])
		plt.grid()
		plt.ylabel('brightness temperature [K]')
		plt.xticks(np.arange(90,191,20),[])
		plt.text(120, 700, 'PRELIMINARY', color='cyan', fontsize=20, rotation=45)



		plt.subplot(2,1,2)
		plt.plot(fb[wb>0], 1000*rb[wb>0], linewidth=2)
		#plt.plot(fb[wb>0], 1000*rb2[wb>0], 'r')
		plt.xlabel(r'$\nu$   [MHz]', fontsize=16)
		plt.ylim([-200, 200])
		plt.grid()
		plt.ylabel('residuals [mK]')
		plt.xticks(np.arange(90,191,20))

		plt.yticks([-100, 0, 100])
		plt.text(131, 146, 'Residuals to 5-term EDGES polynomial', fontsize=14)
		plt.text(131, 106, 'Weighted RMS: ' + str(int(1000*rms)) + ' mK', fontsize=14)
		plt.text(110, 40, 'PRELIMINARY', color='cyan', fontsize=20, rotation=45)



		plt.savefig(home_folder + '/Desktop/residuals.pdf', bbox_inches='tight')
		plt.close()
		plt.close()




	if rejections == 'yes':

		#d1 = np.genfromtxt(home_folder + '/DATA/EDGES/results/high_band/spectra/high_band_average_20161129_nominal.txt')

		z1, dz1, dr1 = high_band_phenomenological_detection_rejection(case='gaussian', real_simulated='real', datacase=1, T21_K_ref=-0.15, plot='no')

		# Ant S11
		z3, dz3, dr3 = high_band_phenomenological_detection_rejection(case='gaussian', real_simulated='real', datacase=3, T21_K_ref=-0.15, plot='no')
		###z4, dz4, dr4 = high_band_phenomenological_detection_rejection(case='gaussian', real_simulated='real', datacase=4, plot='no')

		# Gnd loss
		z5, dz5, dr5 = high_band_phenomenological_detection_rejection(case='gaussian', real_simulated='real', datacase=5, T21_K_ref=-0.15, plot='no')

		# Rcv + Rcv S11
		z6, dz6, dr6 = high_band_phenomenological_detection_rejection(case='gaussian', real_simulated='real', datacase=6, T21_K_ref=-0.15, plot='no')
		###z7, dz7, dr7 = high_band_phenomenological_detection_rejection(case='gaussian', real_simulated='real', datacase=7, plot='no')
		z8, dz8, dr8 = high_band_phenomenological_detection_rejection(case='gaussian', real_simulated='real', datacase=8, T21_K_ref=-0.15, plot='no')
		z9, dz9, dr9 = high_band_phenomenological_detection_rejection(case='gaussian', real_simulated='real', datacase=9, T21_K_ref=-0.15, plot='no')

		# Beam
		z12, dz12, dr12 = high_band_phenomenological_detection_rejection(case='gaussian', real_simulated='real', datacase=12, T21_K_ref=-0.15, plot='no')
		z13, dz13, dr13 = high_band_phenomenological_detection_rejection(case='gaussian', real_simulated='real', datacase=13, T21_K_ref=-0.15, plot='no')
		z14, dz14, dr14 = high_band_phenomenological_detection_rejection(case='gaussian', real_simulated='real', datacase=14, T21_K_ref=-0.15, plot='no')

		###z19, dz19, dr19 = high_band_phenomenological_detection_rejection(case='gaussian', real_simulated='real', datacase=19, plot='no')
		z20, dz20, dr20 = high_band_phenomenological_detection_rejection(case='gaussian', real_simulated='real', datacase=20, T21_K_ref=-0.15, plot='no')
		z21, dz21, dr21 = high_band_phenomenological_detection_rejection(case='gaussian', real_simulated='real', datacase=21, T21_K_ref=-0.15, plot='no')













		plt.close()

		c_off = 2



		f1  = plt.figure(1, figsize=(7, 6))
		ax1 = f1.add_subplot(1,1,1)
		ax1.plot(z1, dz1, 'b--')
		ax1.plot(z1[dr1>c_off], dr1[dr1>c_off], 'k', linewidth=2)
		ax1.plot(z1, (2.5/2.5)*(z1-6) ,'r', linewidth=2)

		ax1.set_xlabel(r'$z_r$', fontsize=16)
		ax1.set_ylabel(r'$\Delta z$', fontsize=16)
		plt.grid()
		plt.ylim([0, 9])
		plt.xlim([15, 6])
		plt.legend(['from error bars alone','from error bars and estimates'])

		plt.text(14.7, 5.5, 'NOT' + '\n' + 'rejected', fontsize=20)
		plt.text(13, 1.5, 'Rejected at 95%', fontsize=20)
		plt.text(12, 5.5, 'PRELIMINARY', color='cyan', fontsize=20, rotation=45)

		ax2 = ax1.twiny()
		ax2.set_xlabel(r'$\nu$   [MHz]',fontsize=16)
		ax2.set_xticks(np.array((np.abs(eg.frequency2redshift(90)-15),np.abs(eg.frequency2redshift(110)-15),np.abs(eg.frequency2redshift(130)-15),np.abs(eg.frequency2redshift(150)-15),np.abs(eg.frequency2redshift(170)-15),np.abs(eg.frequency2redshift(190)-15),np.abs(6-15))))
		ax2.set_xticklabels(['90','110','130','150','170','190',''])

		plt.savefig(home_folder + '/Desktop/rejections_nominal.pdf', bbox_inches='tight')
		plt.close()
		plt.close()






		f1 = plt.figure(2, figsize=(7, 6))

		ax1 = f1.add_subplot(1,1,1)
		ax1.plot(z1[dr1>c_off], dr1[dr1>c_off], 'k', linewidth=2)
		ax1.plot(z12[dr12>c_off], dr12[dr12>c_off], 'g')
		ax1.plot(z13[dr13>c_off], dr13[dr13>c_off], 'g')
		ax1.plot(z14[dr14>c_off], dr14[dr14>c_off], 'g')   #plt.plot(z19[dr19>c_off], dr19[dr19>c_off], 'g')
		ax1.plot(z20[dr20>c_off], dr20[dr20>c_off], 'g')
		ax1.plot(z21[dr21>c_off], dr21[dr21>c_off], 'g')		

		ax1.plot(z6[dr6>c_off], dr6[dr6>c_off], 'g')  #plt.plot(z7[dr7>c_off], dr7[dr7>c_off], 'g')
		ax1.plot(z8[dr8>c_off], dr8[dr8>c_off], 'g')
		ax1.plot(z9[dr9>c_off], dr9[dr9>c_off], 'g')		

		ax1.plot(z3[dr3>c_off], dr3[dr3>c_off], 'g')  #plt.plot(z4[dr4>c_off], dr4[dr4>c_off], 'r')

		ax1.plot(z5[dr5>c_off], dr5[dr5>c_off], 'g')


		ax1.plot(z1[dr1>c_off], dr1[dr1>c_off], 'k', linewidth=2)



		ax1.plot(z21, (2.5/2.5)*(z21-6) ,'r', linewidth=2)
		plt.text(14.8, 5.5, 'NOT' + '\n' + 'rejected', fontsize=20, backgroundcolor='w')
		plt.text(13, 1.5, 'Rejected at 95%', fontsize=20, backgroundcolor='w')
		plt.text(12, 5.5, 'PRELIMINARY', color='cyan', fontsize=20, rotation=45)

		plt.legend(['nominal','due to uncertainty'])



		ax1.set_xlabel(r'$z_r$', fontsize=16)
		ax1.set_ylabel(r'$\Delta z$', fontsize=16)
		plt.grid()
		plt.ylim([0, 9])
		plt.xlim([15, 6])		




		ax2 = ax1.twiny()
		ax2.set_xlabel(r'$\nu$   [MHz]',fontsize=16)
		ax2.set_xticks(np.array((np.abs(eg.frequency2redshift(90)-15),np.abs(eg.frequency2redshift(110)-15),np.abs(eg.frequency2redshift(130)-15),np.abs(eg.frequency2redshift(150)-15),np.abs(eg.frequency2redshift(170)-15),np.abs(eg.frequency2redshift(190)-15),np.abs(6-15))))
		ax2.set_xticklabels(['90','110','130','150','170','190',''])		


		#plt.legend(['nominal', 'changes in antenna beam', 'changes in receiver calibration', 'changes in antenna S11', 'changes in ground loss'])

		plt.savefig(home_folder + '/Desktop/rejections_others.pdf', bbox_inches='tight')
		plt.close()
		plt.close()











	if rejections_many_amplitudes == 'yes':

		N_ampl = 20
		for i in range(N_ampl):
			T21_K_ref = -0.06 - 0.01*i

			print(i)
			z, dz, dr = high_band_phenomenological_detection_rejection(case='gaussian', real_simulated='real', datacase=1, T21_K_ref=T21_K_ref, plot='no')

			if i == 0:
				dz_all = np.copy(dz)
				dr_all = np.copy(dr)

			elif i > 0:
				dz_all = np.vstack((dz_all, dz))
				dr_all = np.vstack((dr_all, dr))









		# ------------------------------------------------
		plt.close()
		plt.close()
		plt.close()

		f1    = plt.figure(1, figsize=(7, 6))
		ax1   = f1.add_subplot(1,1,1)
		c_off = 0.5
		for i in range(N_ampl):
			ax1.plot(z[dr_all[i,:]>c_off], dr_all[i, dr_all[i,:]>c_off], linewidth=1)  # 'm'

		#ax1.plot(z[dr_all[10,:]>c_off], dr_all[10, dr_all[10,:]>c_off], 'k', linewidth=1)



		ax1.plot(z, (2.5/2.5)*(z-6) ,'r', linewidth=2)
		#plt.text(14.8, 5.5, 'NOT' + '\n' + 'rejected', fontsize=20, backgroundcolor='w')
		#plt.text(13, 1.5, 'Rejected at 95%', fontsize=20, backgroundcolor='w')
		plt.text(12, 5.5, 'PRELIMINARY', color='black', fontsize=25, rotation=45)

		#plt.legend(['nominal','due to uncertainty'])


		ax1.set_xlabel(r'$z_r$', fontsize=16)
		ax1.set_ylabel(r'$\Delta z$', fontsize=16)
		plt.grid()
		plt.ylim([0, 9])
		plt.xlim([15, 6])		


		ax2 = ax1.twiny()
		ax2.set_xlabel(r'$\nu$   [MHz]',fontsize=16)
		ax2.set_xticks(np.array((np.abs(eg.frequency2redshift(90)-15),np.abs(eg.frequency2redshift(110)-15),np.abs(eg.frequency2redshift(130)-15),np.abs(eg.frequency2redshift(150)-15),np.abs(eg.frequency2redshift(170)-15),np.abs(eg.frequency2redshift(190)-15),np.abs(6-15))))
		ax2.set_xticklabels(['90','110','130','150','170','190',''])		


		#plt.legend(['nominal', 'changes in antenna beam', 'changes in receiver calibration', 'changes in antenna S11', 'changes in ground loss'])

		plt.savefig(home_folder + '/Desktop/rejections_many_amplitudes.pdf', bbox_inches='tight')
		plt.close()
		plt.close()






		plt.close()
		plt.close()
		f2 = plt.figure(2, figsize=(7, 7))
		im = plt.imshow(dr_all, interpolation='none', aspect='auto', extent=[6, 15, -200, -100]);im.set_clim([1, 8])
		plt.colorbar()
		plt.xlabel(r'$z_r$', fontsize=20)
		plt.ylabel('Gaussian amplitude [mK]')


















	if rejections_simulated == 'yes':

		# Nominal, EDGES polynomial
		z1, dz1, dr1 = high_band_phenomenological_detection_rejection(model_type='EDGES_polynomial', case='gaussian', real_simulated='simulated', datacase=1, plot='no', sim_f_low=90, sim_f_high=190, time_factor=1, sim_with_gaussian='no')

		# Physical model
		z2, dz2, dr2 = high_band_phenomenological_detection_rejection(model_type='Physical_model', case='gaussian', real_simulated='simulated', datacase=1, plot='no', sim_f_low=90, sim_f_high=190, time_factor=1, sim_with_gaussian='no')

		# Noise reduced by factor 2
		z3, dz3, dr3 = high_band_phenomenological_detection_rejection(model_type='EDGES_polynomial', case='gaussian', real_simulated='simulated', datacase=1, plot='no', sim_f_low=90, sim_f_high=190, time_factor=4, sim_with_gaussian='no')

		# Reduced frequency range
		z4, dz4, dr4 = high_band_phenomenological_detection_rejection(model_type='EDGES_polynomial', case='gaussian', real_simulated='simulated', datacase=1, plot='no', sim_f_low=110, sim_f_high=190, time_factor=1, sim_with_gaussian='no')






		# Sim1
		plt.close()

		c_off = 2


		f1  = plt.figure(1, figsize=(7, 6))
		ax1 = f1.add_subplot(1,1,1)
		ax1.plot(z1, dz1, 'k', linewidth=2)
		#ax1.plot(z2, dz2, 'm', linewidth=2)
		#ax1.plot(z3, dz3, 'g', linewidth=2)
		#ax1.plot(z4, dz4, 'b', linewidth=2)

		#ax1.plot(z1, z1-6 ,'r', linewidth=2)

		ax1.set_xlabel(r'$z_r$', fontsize=16)
		ax1.set_ylabel(r'$\Delta z$', fontsize=16)
		plt.grid()
		plt.ylim([0, 9])
		plt.xlim([15, 6])
		plt.legend(['Nominal, 5-term EDGES polynomial'], fontsize=9)

		#plt.text(14.7, 5.5, 'NOT' + '\n' + 'rejected', fontsize=20)
		plt.text(13.9, 1.2, 'Area for Potential 95% Rejection' + '\n' + r'of Gaussian with $-150$-mK Amplitude', backgroundcolor='w', fontsize=13)
		#plt.text(12, 5.5, 'PRELIMINARY', color='cyan', fontsize=20, rotation=45)

		ax2 = ax1.twiny()
		ax2.set_xlabel(r'$\nu$   [MHz]',fontsize=16)
		ax2.set_xticks(np.array((np.abs(eg.frequency2redshift(90)-15),np.abs(eg.frequency2redshift(110)-15),np.abs(eg.frequency2redshift(130)-15),np.abs(eg.frequency2redshift(150)-15),np.abs(eg.frequency2redshift(170)-15),np.abs(eg.frequency2redshift(190)-15),np.abs(6-15))))
		ax2.set_xticklabels(['90','110','130','150','170','190',''])

		plt.savefig(home_folder + '/Desktop/rejections_simulation_1.pdf', bbox_inches='tight')
		plt.close()
		plt.close()




		# Sim2
		plt.close()

		c_off = 2


		f1  = plt.figure(1, figsize=(7, 6))
		ax1 = f1.add_subplot(1,1,1)
		ax1.plot(z1, dz1, 'k', linewidth=2)
		ax1.plot(z2, dz2, 'm', linewidth=2)
		#ax1.plot(z3, dz3, 'g', linewidth=2)
		#ax1.plot(z4, dz4, 'b', linewidth=2)

		#ax1.plot(z1, z1-6 ,'r', linewidth=2)

		ax1.set_xlabel(r'$z_r$', fontsize=16)
		ax1.set_ylabel(r'$\Delta z$', fontsize=16)
		plt.grid()
		plt.ylim([0, 9])
		plt.xlim([15, 6])
		plt.legend(['Nominal, 5-term EDGES polynomial', '5-term Physical model'], fontsize=9)

		#plt.text(14.7, 5.5, 'NOT' + '\n' + 'rejected', fontsize=20)
		plt.text(13.9, 1.2, 'Area for Potential 95% Rejection' + '\n' + r'of Gaussian with $-150$-mK Amplitude', backgroundcolor='w', fontsize=13)
		#plt.text(12, 5.5, 'PRELIMINARY', color='cyan', fontsize=20, rotation=45)

		ax2 = ax1.twiny()
		ax2.set_xlabel(r'$\nu$   [MHz]',fontsize=16)
		ax2.set_xticks(np.array((np.abs(eg.frequency2redshift(90)-15),np.abs(eg.frequency2redshift(110)-15),np.abs(eg.frequency2redshift(130)-15),np.abs(eg.frequency2redshift(150)-15),np.abs(eg.frequency2redshift(170)-15),np.abs(eg.frequency2redshift(190)-15),np.abs(6-15))))
		ax2.set_xticklabels(['90','110','130','150','170','190',''])

		plt.savefig(home_folder + '/Desktop/rejections_simulation_2.pdf', bbox_inches='tight')
		plt.close()
		plt.close()




		# Sim3
		plt.close()

		c_off = 2


		f1  = plt.figure(1, figsize=(7, 6))
		ax1 = f1.add_subplot(1,1,1)
		ax1.plot(z1, dz1, 'k', linewidth=2)
		ax1.plot(z2, dz2, 'm', linewidth=2)
		ax1.plot(z3, dz3, 'g', linewidth=2)
		#ax1.plot(z4, dz4, 'b', linewidth=2)

		#ax1.plot(z1, z1-6 ,'r', linewidth=2)

		ax1.set_xlabel(r'$z_r$', fontsize=16)
		ax1.set_ylabel(r'$\Delta z$', fontsize=16)
		plt.grid()
		plt.ylim([0, 9])
		plt.xlim([15, 6])
		plt.legend(['Nominal, 5-term EDGES polynomial', '5-term Physical model', '4 x integration time'], fontsize=9)

		#plt.text(14.7, 5.5, 'NOT' + '\n' + 'rejected', fontsize=20)
		plt.text(13.9, 1.2, 'Area for Potential 95% Rejection' + '\n' + r'of Gaussian with $-150$-mK Amplitude', backgroundcolor='w', fontsize=13)
		#plt.text(12, 5.5, 'PRELIMINARY', color='cyan', fontsize=20, rotation=45)

		ax2 = ax1.twiny()
		ax2.set_xlabel(r'$\nu$   [MHz]',fontsize=16)
		ax2.set_xticks(np.array((np.abs(eg.frequency2redshift(90)-15),np.abs(eg.frequency2redshift(110)-15),np.abs(eg.frequency2redshift(130)-15),np.abs(eg.frequency2redshift(150)-15),np.abs(eg.frequency2redshift(170)-15),np.abs(eg.frequency2redshift(190)-15),np.abs(6-15))))
		ax2.set_xticklabels(['90','110','130','150','170','190',''])

		plt.savefig(home_folder + '/Desktop/rejections_simulation_3.pdf', bbox_inches='tight')
		plt.close()
		plt.close()






		# Sim4
		plt.close()

		c_off = 2


		f1  = plt.figure(1, figsize=(7, 6))
		ax1 = f1.add_subplot(1,1,1)
		ax1.plot(z1, dz1, 'k', linewidth=2)
		ax1.plot(z2, dz2, 'm', linewidth=2)
		ax1.plot(z3, dz3, 'g', linewidth=2)
		ax1.plot(z4, dz4, 'b', linewidth=2)

		#ax1.plot(z1, z1-6 ,'r', linewidth=2)

		ax1.set_xlabel(r'$z_r$', fontsize=16)
		ax1.set_ylabel(r'$\Delta z$', fontsize=16)
		plt.grid()
		plt.ylim([0, 9])
		plt.xlim([15, 6])
		plt.legend(['Nominal, 5-term EDGES polynomial', '5-term Physical model', '4 x integration time', 'Reduced bandwidth'], fontsize=9)

		#plt.text(14.7, 5.5, 'NOT' + '\n' + 'rejected', fontsize=20)
		plt.text(13.9, 1.2, 'Area for Potential 95% Rejection' + '\n' + r'of Gaussian with $-150$-mK Amplitude', backgroundcolor='w', fontsize=13)
		#plt.text(12, 5.5, 'PRELIMINARY', color='cyan', fontsize=20, rotation=45)

		ax2 = ax1.twiny()
		ax2.set_xlabel(r'$\nu$   [MHz]',fontsize=16)
		ax2.set_xticks(np.array((np.abs(eg.frequency2redshift(90)-15),np.abs(eg.frequency2redshift(110)-15),np.abs(eg.frequency2redshift(130)-15),np.abs(eg.frequency2redshift(150)-15),np.abs(eg.frequency2redshift(170)-15),np.abs(eg.frequency2redshift(190)-15),np.abs(6-15))))
		ax2.set_xticklabels(['90','110','130','150','170','190',''])

		plt.savefig(home_folder + '/Desktop/rejections_simulation_4.pdf', bbox_inches='tight')
		plt.close()
		plt.close()		






		# Sim5
		plt.close()

		c_off = 2


		f1  = plt.figure(1, figsize=(7, 6))
		ax1 = f1.add_subplot(1,1,1)
		ax1.plot(z1, dz1, 'k', linewidth=2)
		ax1.plot(z2, dz2, 'm', linewidth=2)
		ax1.plot(z3, dz3, 'g', linewidth=2)
		ax1.plot(z4, dz4, 'b', linewidth=2)

		ax1.plot(z1, z1-6 ,'r', linewidth=2)

		ax1.set_xlabel(r'$z_r$', fontsize=16)
		ax1.set_ylabel(r'$\Delta z$', fontsize=16)
		plt.grid()
		plt.ylim([0, 9])
		plt.xlim([15, 6])
		plt.legend(['Nominal, 5-term EDGES polynomial', '5-term Physical model', '4 x integration time', 'Reduced bandwidth', r'Constraint for $T_b\approx0$ at $z=6$'], fontsize=9)

		#plt.text(14.7, 5.5, 'NOT' + '\n' + 'rejected', fontsize=20)
		plt.text(13.9, 1.2, 'Area for Potential 95% Rejection' + '\n' + r'of Gaussian with $-150$-mK Amplitude', backgroundcolor='w', fontsize=13)
		#plt.text(12, 5.5, 'PRELIMINARY', color='cyan', fontsize=20, rotation=45)

		ax2 = ax1.twiny()
		ax2.set_xlabel(r'$\nu$   [MHz]',fontsize=16)
		ax2.set_xticks(np.array((np.abs(eg.frequency2redshift(90)-15),np.abs(eg.frequency2redshift(110)-15),np.abs(eg.frequency2redshift(130)-15),np.abs(eg.frequency2redshift(150)-15),np.abs(eg.frequency2redshift(170)-15),np.abs(eg.frequency2redshift(190)-15),np.abs(6-15))))
		ax2.set_xticklabels(['90','110','130','150','170','190',''])

		plt.savefig(home_folder + '/Desktop/rejections_simulation_5.pdf', bbox_inches='tight')
		plt.close()
		plt.close()		






	if rejected_models == 'yes':


		#g02, xHI, z = eg.model_eor(fe, T21=-0.15, model_type='gaussian', zr=10, dz=2)
		#g03, xHI, z = eg.model_eor(fe, T21=-0.15, model_type='gaussian', zr=11, dz=2)
		#g04, xHI, z = eg.model_eor(fe, T21=-0.15, model_type='gaussian', zr=12, dz=2)
		#g05, xHI, z = eg.model_eor(fe, T21=-0.15, model_type='gaussian', zr=13, dz=2)
		#g06, xHI, z = eg.model_eor(fe, T21=-0.15, model_type='gaussian', zr=14, dz=2)
		#g07, xHI, z = eg.model_eor(fe, T21=-0.15, model_type='gaussian', zr=15, dz=2)
		#g08, xHI, z = eg.model_eor(fe, T21=-0.15, model_type='gaussian', zr=16, dz=2)







		# ---------------------------------------------
		plt.close()
		plt.close()

		ff, il, ih = eg.frequency_edges(90,190)
		fe = ff[il:ih+1]		


		f1  = plt.figure(1, figsize=(8, 8))	
		ax1 = f1.add_subplot(2,1,1)	
		ii = 0
		for i in range(7):



			xzr = 9 + 6*np.random.uniform()
			ddz = (2.5/2.5) * (xzr-6)
			if ddz >= 3:
				ydz = 3
			elif ddz < 3:
				ydz = ddz

			xdz = (ydz/2) + (ydz/2)*np.random.uniform()
			T21 = -0.15 - 0.09*np.random.uniform()


			#ii = np.random.randint(0, 3)

			if ii == 0:
				ls = '-'
				ii = 1
			elif ii == 1:
				ls = '--'
				ii = 0
			#elif ii == 2:
				#ls = '-.'
			#elif ii == 3:
			#	ls = '-.'


			g, xHI, z = eg.model_eor(fe, T21=T21, model_type='gaussian', zr=xzr, dz=xdz)
			ax1.plot(eg.frequency2redshift(fe), 1000*g, linestyle=ls)

		g, xHI, z = eg.model_eor(fe, T21=T21, model_type='gaussian', zr=11, dz=4)
		ax1.plot(eg.frequency2redshift(fe), 1000*g)


		plt.xlim([15, 6])
		#ax1.set_xlabel(r'$z$', fontsize=20)
		ax1.set_ylabel('brightness temperature [mK]', fontsize=14)
		ax1.set_xticklabels([])

		ax2 = ax1.twiny()
		ax2.set_xlabel(r'$\nu$   [MHz]',fontsize=16)
		ax2.set_xticks(np.array((np.abs(eg.frequency2redshift(90)-15),np.abs(eg.frequency2redshift(110)-15),np.abs(eg.frequency2redshift(130)-15),np.abs(eg.frequency2redshift(150)-15),np.abs(eg.frequency2redshift(170)-15),np.abs(eg.frequency2redshift(190)-15),np.abs(6-15))))
		ax2.set_xticklabels(['90','110','130','150','170','190',''])
		plt.ylim([-250, 0])






		plt.subplot(2,1,2)


		ii = 0
		for i in range(8):



			xzr = 10 + 5*np.random.uniform()
			xdz =  4 + 2*np.random.uniform()
			T21 = -0.15 + 0.1*np.random.uniform()


			if ii == 0:
				ls = '-'
				ii = 1
			elif ii == 1:
				ls = '--'
				ii = 0


			g, xHI, z = eg.model_eor(fe, T21=T21, model_type='gaussian', zr=xzr, dz=xdz)
			plt.plot(eg.frequency2redshift(fe), 1000*g, linestyle=ls)

		#g, xHI, z = eg.model_eor(fe, T21=T21, model_type='gaussian', zr=11, dz=4)
		#plt.plot(eg.frequency2redshift(fe), 1000*g)


		plt.xlim([15, 6])
		plt.xlabel(r'$z$', fontsize=20)
		plt.ylabel('brightness temperature [mK]', fontsize=14)

		#ax2 = ax1.twiny()
		#ax2.set_xlabel(r'$\nu$   [MHz]',fontsize=16)
		#ax2.set_xticks(np.array((np.abs(eg.frequency2redshift(90)-15),np.abs(eg.frequency2redshift(110)-15),np.abs(eg.frequency2redshift(130)-15),np.abs(eg.frequency2redshift(150)-15),np.abs(eg.frequency2redshift(170)-15),np.abs(eg.frequency2redshift(190)-15),np.abs(6-15))))
		#ax2.set_xticklabels(['90','110','130','150','170','190',''])
		plt.ylim([-250, 0])








		plt.savefig(home_folder + '/Desktop/sample_of_rejected_models.pdf', bbox_inches='tight')
		plt.close()
		plt.close()






	return z, dz_all, dr_all












def exercise_detection_rejection(figure):


	# Uncertainties for EDGES polynomial with different number of baseline parameters
	if figure == 'figure1':


		T21_K_true = 0.028
		T21_K_ref  = 0.028
		nterms_fg  = 5

		f      = np.arange(90, 190.1, 0.4)
		zlow   = frequency2redshift(190)
		zhigh  = frequency2redshift(90)
		z      = np.arange(zlow, zhigh, 0.05)
		dz     = np.arange(0.02, 6.01, 0.02)



		p_rec  = np.zeros((len(dz), len(z)))
		dp_rec = np.zeros((len(dz), len(z)))

		model_fg = model_evaluate('EDGES_polynomial', [400/(150**(-2.5)), 0.00], f)
		noise    = np.random.normal(0,model_fg/np.sqrt(60*60*5*0.4e6))   # realistic noise similar to real High-Band data


		dz_line = np.zeros((4, len(z)))
		for k in range(4):
			for j in range(len(z)):
				for i in range(len(dz)):

					print((k+2, z[j], dz[i]))
					model_no_noise  = model_evaluate('EDGES_polynomial_plus_tanh', [400/(150**(-2.5)), T21_K_true], f, zr=z[j], dz=dz[i])
					data            = model_no_noise + noise
					p               = fit_polynomial_fourier('EDGES_polynomial_plus_tanh', f, data, k+2, zr=z[j], dz=dz[i])
					p_rec[i,j]      = p[0][k+2]
					dp_rec[i,j]     = np.sqrt(np.diag(p[3]))[k+2]


			# Produce 1\sigma line from 2D uncertainty array
			for j in range(len(z)):
				for i in range(len(dz)-1):
					if (dp_rec[i,j] <= 0.028/2) and (dp_rec[i+1,j] >= 0.028/2):
						dz_line[k,j] = dz[i]



		plt.close()
		plt.close()
		plt.close()

		x0 = 0.09
		y0 = 0.09
		dx = 0.89
		dy = 0.82

		f1 = plt.figure(1)		
		ax = f1.add_axes([1*x0, y0+0*dy, dx, dy])
		ax.plot(z, dz_line[0,:], 'b:',  linewidth=2)
		ax.plot(z, dz_line[1,:], 'b-.', linewidth=2)
		ax.plot(z, dz_line[2,:], 'b--', linewidth=2)
		ax.plot(z, dz_line[3,:], 'b',   linewidth=2)
		ax.legend([r'$\rm{N_{fg}}$ = 2', r'$\rm{N_{fg}}$ = 3', r'$\rm{N_{fg}}$ = 4', r'$\rm{N_{fg}}$ = 5'],loc=0)
		ax.set_xlim([6, 15])
		ax.set_ylim([0, 6])

		plt.gca().invert_xaxis()
		plt.xlabel(r'$z_{\rm{r}}$', fontsize=18)
		plt.ylabel(r'$\Delta z$', fontsize=18)
		plt.grid()


		ax2 = ax.twiny()
		ax2.set_xlabel(r'$\nu$ [MHz]', fontsize=18)

		f2 = np.arange(90, 191, 10)
		z2 = frequency2redshift(f2)

		ax2.set_xticks((z2-6)/(15-6))
		ax2.set_xticklabels(f2)
		plt.gca().invert_xaxis()	






	# Uncertainties for EDGES polynomial and different noise levels	
	if figure == 'figure2':


		T21_K_true = 0.028
		T21_K_ref  = 0.028
		nterms_fg  = 5

		f      = np.arange(90, 190.1, 0.4)
		zlow   = frequency2redshift(190)
		zhigh  = frequency2redshift(90)
		z      = np.arange(zlow, zhigh, 0.05)
		dz     = np.arange(0.02, 4.01, 0.02)



		p_rec  = np.zeros((len(dz), len(z)))
		dp_rec = np.zeros((len(dz), len(z)))

		model_fg = model_evaluate('EDGES_polynomial', [400/(150**(-2.5)), 0.00], f)
		noise    = np.random.normal(0,model_fg/np.sqrt(60*60*5*0.4e6))   # realistic noise similar to real High-Band data


		dz_line = np.zeros((4, len(z)))
		for k in range(4):
			for j in range(len(z)):
				for i in range(len(dz)):

					print((k+2, z[j], dz[i]))
					model_no_noise  = model_evaluate('EDGES_polynomial_plus_tanh', [400/(150**(-2.5)), T21_K_true], f, zr=z[j], dz=dz[i])
					data            = model_no_noise + noise/(k+1)
					p               = fit_polynomial_fourier('EDGES_polynomial_plus_tanh', f, data, nterms_fg, zr=z[j], dz=dz[i])
					p_rec[i,j]      = p[0][nterms_fg]
					dp_rec[i,j]     = np.sqrt(np.diag(p[3]))[nterms_fg]


			# Produce 1\sigma line from 2D uncertainty array
			for j in range(len(z)):
				for i in range(len(dz)-1):
					if (dp_rec[i,j] <= 0.028/2) and (dp_rec[i+1,j] >= 0.028/2):
						dz_line[k,j] = dz[i]



		plt.close()
		plt.close()
		plt.close()

		x0 = 0.09
		y0 = 0.09
		dx = 0.89
		dy = 0.82

		f1 = plt.figure(1)		
		ax = f1.add_axes([1*x0, y0+0*dy, dx, dy])
		ax.plot(z, dz_line[0,:], 'b',   linewidth=2)
		ax.plot(z, dz_line[1,:], 'b--', linewidth=2)
		ax.plot(z, dz_line[2,:], 'b-.', linewidth=2)
		ax.plot(z, dz_line[3,:], 'b:',  linewidth=2)
		ax.legend(['nominal', 'nominal/2', 'nominal/3', 'nominal/4'],loc=0)
		ax.set_xlim([6, 15])
		ax.set_ylim([0, 6])

		plt.gca().invert_xaxis()
		plt.xlabel(r'$z_{\rm{r}}$', fontsize=18)
		plt.ylabel(r'$\Delta z$', fontsize=18)
		plt.grid()

		ax2 = ax.twiny()
		ax2.set_xlabel(r'$\nu$ [MHz]', fontsize=18)

		f2 = np.arange(90, 191, 10)
		z2 = frequency2redshift(f2)

		ax2.set_xticks((z2-6)/(15-6))
		ax2.set_xticklabels(f2)
		plt.gca().invert_xaxis()	













	# Uncertainties for EDGES polynomial and different noise levels, in the range 110-190 MHz	
	if figure == 'figure2b':


		T21_K_true = 0.0
		T21_K_ref  = 0.028
		nterms_fg  = 3

		f      = np.arange(150, 220.1, 0.4)
		zlow   = frequency2redshift(220)
		zhigh  = frequency2redshift(150)
		z      = np.arange(zlow, zhigh, 0.05)
		dz     = np.arange(0.02, 3.01, 0.02)



		p_rec  = np.zeros((len(dz), len(z)))
		dp_rec = np.zeros((len(dz), len(z)))

		model_fg = model_evaluate('EDGES_polynomial', [400/(150**(-2.5)), 0.00], f)
		noise    = np.random.normal(0,model_fg/np.sqrt(60*60*5*0.4e6))   # realistic noise similar to real High-Band data


		dz_line = np.zeros((6, len(z)))
		for k in range(6):
			for j in range(len(z)):
				for i in range(len(dz)):

					print((k+2, z[j], dz[i]))
					model_no_noise  = model_evaluate('EDGES_polynomial_plus_tanh', [400/(150**(-2.5)), T21_K_true], f, zr=z[j], dz=dz[i])
					data            = model_no_noise + noise/(k+1)
					p               = fit_polynomial_fourier('EDGES_polynomial_plus_tanh', f, data, nterms_fg, zr=z[j], dz=dz[i])
					p_rec[i,j]      = p[0][nterms_fg]
					dp_rec[i,j]     = np.sqrt(np.diag(p[3]))[nterms_fg]


			# Produce 1\sigma line from 2D uncertainty array
			for j in range(len(z)):
				for i in range(len(dz)-1):
					if (dp_rec[i,j] <= 0.028/2) and (dp_rec[i+1,j] >= 0.028/2):
						dz_line[k,j] = dz[i]



		plt.close()
		plt.close()
		plt.close()

		x0 = 0.09
		y0 = 0.09
		dx = 0.89
		dy = 0.82

		f1 = plt.figure(1)		
		ax = f1.add_axes([1*x0, y0+0*dy, dx, dy])
		ax.plot(z, dz_line[0,:], 'b',   linewidth=2)
		ax.plot(z, dz_line[1,:], 'b--', linewidth=2)
		ax.plot(z, dz_line[2,:], 'b-.', linewidth=2)
		ax.plot(z, dz_line[3,:], 'b:',  linewidth=2)
		ax.plot(z, dz_line[4,:], 'r',   linewidth=2)
		ax.plot(z, dz_line[5,:], 'r--',  linewidth=2)
		ax.legend(['nominal', 'nominal/2', 'nominal/3', 'nominal/4', 'nominal/5', 'nominal/6'],loc=0)
		ax.set_xlim([6, 15])
		ax.set_ylim([0, 6])

		plt.gca().invert_xaxis()
		plt.xlabel(r'$z_{\rm{r}}$', fontsize=18)
		plt.ylabel(r'$\Delta z$', fontsize=18)
		plt.grid()

		ax2 = ax.twiny()
		ax2.set_xlabel(r'$\nu$ [MHz]', fontsize=18)

		f2 = np.arange(90, 191, 10)
		z2 = frequency2redshift(f2)

		ax2.set_xticks((z2-6)/(15-6))
		ax2.set_xticklabels(f2)
		plt.gca().invert_xaxis()	





















	# Uncertainties for EDGES polynomial in ranges 90-190 and 110-190 MHz
	if figure == 'figure3':

		dz_line_two = []
		for k in range(2):

			T21_K_true = 0.028
			T21_K_ref  = 0.028
			nterms_fg  = 5

			if k == 0:
				f = np.arange(90, 190.1, 0.4)

			elif k == 1:
				f = np.arange(110, 190.1, 0.4)

			zlow   = frequency2redshift(np.max(f))
			zhigh  = frequency2redshift(np.min(f))
			z      = np.arange(zlow, zhigh, 0.05)
			dz     = np.arange(0.02, 2.01, 0.02)

			p_rec  = np.zeros((len(dz), len(z)))
			dp_rec = np.zeros((len(dz), len(z)))

			model_fg = model_evaluate('EDGES_polynomial', [400/(150**(-2.5)), 0.00], f)
			noise    = np.random.normal(0,model_fg/np.sqrt(60*60*5*0.4e6))   # realistic noise similar to real High-Band data


			dz_line = np.zeros(len(z))
			for j in range(len(z)):
				for i in range(len(dz)):

					print((z[j], dz[i]))
					model_no_noise  = model_evaluate('EDGES_polynomial_plus_tanh', [400/(150**(-2.5)), T21_K_true], f, zr=z[j], dz=dz[i])
					data            = model_no_noise + noise
					p               = fit_polynomial_fourier('EDGES_polynomial_plus_tanh', f, data, nterms_fg, zr=z[j], dz=dz[i])
					p_rec[i,j]      = p[0][nterms_fg]
					dp_rec[i,j]     = np.sqrt(np.diag(p[3]))[nterms_fg]


			# Produce 1\sigma line from 2D uncertainty array
			for j in range(len(z)):
				for i in range(len(dz)-1):
					if (dp_rec[i,j] <= 0.028/2) and (dp_rec[i+1,j] >= 0.028/2):
						dz_line[j] = dz[i]


			# Accumulating the two results
			dz_line_two.append(z)
			dz_line_two.append(dz_line)


		plt.close()
		plt.close()
		plt.close()

		x0 = 0.09
		y0 = 0.09
		dx = 0.89
		dy = 0.82

		f1 = plt.figure(1)
		ax = f1.add_axes([1*x0, y0+0*dy, dx, dy])
		ax.plot(dz_line_two[0], dz_line_two[1], 'b', linewidth=2)
		ax.plot(dz_line_two[2], dz_line_two[3], 'r', linewidth=2)

		ax.legend(['5 terms over 90-190 MHz', '5 terms over 110-190 MHz'], loc=0)
		ax.set_xlim([6, 15])
		ax.set_ylim([0, 2])

		plt.xlabel(r'$z_{\rm{r}}$', fontsize=18)
		plt.ylabel(r'$\Delta z$',   fontsize=18)
		plt.grid()
		plt.gca().invert_xaxis()


		ax2 = ax.twiny()
		ax2.set_xlabel(r'$\nu$ [MHz]', fontsize=18)

		f2 = np.arange(90, 191, 10)
		z2 = frequency2redshift(f2)

		ax2.set_xticks((z2-6)/(15-6))
		ax2.set_xticklabels(f2)
		plt.gca().invert_xaxis()













	# Uncertainties for EDGES polynomial and Physical model
	if figure == 'figure4':

		dz_line_two = []
		for k in range(2):

			T21_K_true = 0.028
			T21_K_ref  = 0.028
			nterms_fg  = 5
			f          = np.arange(90, 190.1, 0.4)
			zlow   = frequency2redshift(np.max(f))
			zhigh  = frequency2redshift(np.min(f))
			z      = np.arange(zlow, zhigh, 0.05)
			dz     = np.arange(0.02, 2.01, 0.02)

			p_rec  = np.zeros((len(dz), len(z)))
			dp_rec = np.zeros((len(dz), len(z)))

			model_fg = model_evaluate('EDGES_polynomial', [400/(150**(-2.5)), 0.00], f)
			noise    = np.random.normal(0,model_fg/np.sqrt(60*60*5*0.4e6))   # realistic noise similar to real High-Band data

			dz_line = np.zeros(len(z))
			for j in range(len(z)):
				for i in range(len(dz)):

					print((z[j], dz[i]))
					model_no_noise  = model_evaluate('EDGES_polynomial_plus_tanh', [400/(150**(-2.5)), T21_K_true], f, zr=z[j], dz=dz[i])
					data            = model_no_noise + noise
					if k  == 0:							
						p       = fit_polynomial_fourier('EDGES_polynomial_plus_tanh', f, data, nterms_fg, zr=z[j], dz=dz[i])

					if k  == 1:
						p       = fit_polynomial_fourier('Physical_model_plus_tanh', f, data, nterms_fg, zr=z[j], dz=dz[i])

					p_rec[i,j]      = p[0][nterms_fg]
					dp_rec[i,j]     = np.sqrt(np.diag(p[3]))[nterms_fg]


			# Produce 1\sigma line from 2D uncertainty array
			for j in range(len(z)):
				for i in range(len(dz)-1):
					if (dp_rec[i,j] <= 0.028/2) and (dp_rec[i+1,j] >= 0.028/2):
						dz_line[j] = dz[i]


			# Accumulating the two results
			dz_line_two.append(z)
			dz_line_two.append(dz_line)


		plt.close()
		plt.close()
		plt.close()

		x0 = 0.09
		y0 = 0.09
		dx = 0.89
		dy = 0.82

		f1 = plt.figure(1)
		ax = f1.add_axes([1*x0, y0+0*dy, dx, dy])
		ax.plot(dz_line_two[0], dz_line_two[1], 'b', linewidth=2)
		ax.plot(dz_line_two[2], dz_line_two[3], 'r', linewidth=2)

		ax.legend(['EDGES polynomial, 5 terms over 90-190 MHz', 'Physical model, 5 terms over 90-190 MHz'], loc=0)
		ax.set_xlim([6, 15])
		ax.set_ylim([0, 2])

		plt.xlabel(r'$z_{\rm{r}}$', fontsize=18)
		plt.ylabel(r'$\Delta z$',   fontsize=18)
		plt.grid()
		plt.gca().invert_xaxis()


		ax2 = ax.twiny()
		ax2.set_xlabel(r'$\nu$ [MHz]', fontsize=18)

		f2 = np.arange(90, 191, 10)
		z2 = frequency2redshift(f2)

		ax2.set_xticks((z2-6)/(15-6))
		ax2.set_xticklabels(f2)
		plt.gca().invert_xaxis()










	# Uncertainties for EDGES polynomial with and without weights
	if figure == 'figure5':

		dz_line_two = []
		for k in range(2):

			T21_K_true = 0.028
			T21_K_ref  = 0.028
			nterms_fg  = 5
			f          = np.arange(90, 190.1, 0.4)
			zlow   = frequency2redshift(np.max(f))
			zhigh  = frequency2redshift(np.min(f))
			z      = np.arange(zlow, zhigh, 0.05)
			dz     = np.arange(0.02, 2.01, 0.02)

			p_rec  = np.zeros((len(dz), len(z)))
			dp_rec = np.zeros((len(dz), len(z)))

			model_fg = model_evaluate('EDGES_polynomial', [400/(150**(-2.5)), 0.00], f)
			noise_std = model_fg/np.sqrt(60*60*5*0.4e6)
			noise     = np.random.normal(0, noise_std)   # realistic noise similar to real High-Band data

			dz_line = np.zeros(len(z))
			for j in range(len(z)):
				for i in range(len(dz)):

					print((z[j], dz[i]))
					model_no_noise  = model_evaluate('EDGES_polynomial_plus_tanh', [400/(150**(-2.5)), T21_K_true], f, zr=z[j], dz=dz[i])
					data            = model_no_noise + noise

					if k  == 0:	# Using Inverse-Variance Weights
						p       = fit_polynomial_fourier('EDGES_polynomial_plus_tanh', f, data, nterms_fg, Weights=(1/(noise_std**2)), zr=z[j], dz=dz[i])

					if k  == 1:	# Using Equal Weights
						p       = fit_polynomial_fourier('EDGES_polynomial_plus_tanh', f, data, nterms_fg, zr=z[j], dz=dz[i])


					p_rec[i,j]      = p[0][nterms_fg]
					dp_rec[i,j]     = np.sqrt(np.diag(p[3]))[nterms_fg]


			# Produce 1\sigma line from 2D uncertainty array
			for j in range(len(z)):
				for i in range(len(dz)-1):
					if (dp_rec[i,j] <= 0.028/2) and (dp_rec[i+1,j] >= 0.028/2):
						dz_line[j] = dz[i]


			# Accumulating the two results
			dz_line_two.append(z)
			dz_line_two.append(dz_line)




		plt.close()
		plt.close()
		plt.close()


		f1 = plt.figure(1)
		offx = 0.1
		offy = 0.1
		dx = 0.85
		dy = 0.4
		dxcb = 0.02
		dycb = 0.7*dy

		ax1 = f1.add_axes([offx, 1*dy+offy, dx, dy])
		ax1.plot(frequency2redshift(f), 1000*noise, 'b') #,  linewidth=2)
		ax1.errorbar(frequency2redshift(f), 1000*noise, 1000*noise_std, fmt='.', ecolor='r') #,  linewidth=2)
		ax1.set_xlim([15, 6])
		ax1.set_ylim([-50, 50])
		ax1.grid()
		ax1.set_xticklabels([])
		#plt.gca().invert_xaxis()
		ax1.set_ylabel(r'$\Delta$T [mK]', fontsize=16)

		ax12 = ax1.twiny()
		ax12.set_xlabel(r'$\nu$ [MHz]', fontsize=18)		
		ff2 = np.arange(90, 191, 10)
		z2 = frequency2redshift(ff2)
		ax12.set_xticks((z2-6)/(15-6))
		ax12.set_xticklabels(ff2)
		plt.gca().invert_xaxis()		






		ax2 = f1.add_axes([offx, 0*dy+offy, dx, dy])
		ax2.plot(dz_line_two[0], dz_line_two[1], 'r', linewidth=2)
		ax2.plot(dz_line_two[2], dz_line_two[3], 'b', linewidth=2)

		ax2.legend(['Inverse-Variance Weights', 'Equal Weights'], loc=0)
		ax2.set_xlim([6, 15])
		ax2.set_ylim([0, 2])
		ax2.set_yticks([0, 0.5, 1, 1.5])

		plt.xlabel(r'$z_{\rm{r}}$', fontsize=18)
		plt.ylabel(r'$\Delta z$',   fontsize=18)
		plt.grid()
		plt.gca().invert_xaxis()


		#ax22 = ax2.twiny()
		#ax22.set_xlabel(r'$\nu$ [MHz]', fontsize=18)

		#f2 = np.arange(90, 191, 10)
		#z2 = frequency2redshift(f2)

		#ax22.set_xticks((z2-6)/(15-6))
		#ax22.set_xticklabels(f2)
		#plt.gca().invert_xaxis()









	# Uncertainties for EDGES polynomial and Tanh, with weights, and different noise levels
	if figure == 'figure5a':


		T21_K_true = 0.028
		T21_K_ref  = 0.028
		nterms_fg  = 5
		f          = np.arange(90, 190.1, 0.4)
		zlow   = frequency2redshift(np.max(f))
		zhigh  = frequency2redshift(np.min(f))
		z      = np.arange(zlow, zhigh, 0.01)
		dz     = np.arange(0.01, 6.01, 0.01)

		#z      = np.arange(zlow, zhigh, 0.1)
		#dz     = np.arange(0.01, 4.01, 0.1)                

		p_rec  = np.zeros((len(dz), len(z)))
		dp_rec = np.zeros((len(dz), len(z)))

		model_fg  = model_evaluate('EDGES_polynomial', [300/(150**(-2.5)), 0.00], f)
		noise_std = model_fg/np.sqrt(60*60*5*0.2e6)
		noise     = np.random.normal(0, noise_std)   # realistic noise similar to real High-Band data


		#dz_line_two = []
		dz_line = np.zeros((len(z), 4))
		for k in range(4):

			if k == 0:
				div = 1

			if k == 1:
				div = 2

			if k == 2:
				div = 4

			if k == 3:
				div = 8                        




			for j in range(len(z)):
				for i in range(len(dz)):

					print((z[j], dz[i]))
					model_no_noise  = model_evaluate('EDGES_polynomial_plus_tanh', [300/(150**(-2.5)), T21_K_true], f, zr=z[j], dz=dz[i])
					data            = model_no_noise + noise/div

					# Using Inverse-Variance Weights
					p = fit_polynomial_fourier('EDGES_polynomial_plus_tanh', f, data, nterms_fg, Weights=(1/((noise_std/div)**2)), zr=z[j], dz=dz[i])


					p_rec[i,j]  = p[0][nterms_fg]
					dp_rec[i,j] = np.sqrt(np.diag(p[3]))[nterms_fg]


			# Produce 1\sigma line from 2D uncertainty array
			for j in range(len(z)):
				for i in range(len(dz)-1):
					if (dp_rec[i,j] <= 0.028/2) and (dp_rec[i+1,j] >= 0.028/2):
						dz_line[j,k] = dz[i]


		# Appending z vector
		z_dz = np.hstack((z.reshape(-1,1), dz_line))



		#Saving data
		path_save = home_folder + '/DATA/EDGES/signal_constraints/simulated/'
		np.savetxt(path_save + 'simulated_constraints_tanh_90-190MHz.txt', z_dz)









		#plt.close()
		#plt.close()
		#plt.close()


		#f1 = plt.figure(1)
		#offx = 0.1
		#offy = 0.1
		#dx = 0.85
		#dy = 0.4
		#dxcb = 0.02
		#dycb = 0.7*dy

		#ax1 = f1.add_axes([offx, 1*dy+offy, dx, dy])
		#ax1.plot(frequency2redshift(f), 1000*noise, 'b') #,  linewidth=2)
		#ax1.errorbar(frequency2redshift(f), 1000*noise, 1000*noise_std, fmt='.', ecolor='r') #,  linewidth=2)
		#ax1.set_xlim([15, 6])
		#ax1.set_ylim([-50, 50])
		#ax1.grid()
		#ax1.set_xticklabels([])
		##plt.gca().invert_xaxis()
		#ax1.set_ylabel(r'$\Delta$T [mK]', fontsize=16)

		#ax12 = ax1.twiny()
		#ax12.set_xlabel(r'$\nu$ [MHz]', fontsize=18)		
		#ff2 = np.arange(90, 191, 10)
		#z2 = frequency2redshift(ff2)
		#ax12.set_xticks((z2-6)/(15-6))
		#ax12.set_xticklabels(ff2)
		#plt.gca().invert_xaxis()		






		#ax2 = f1.add_axes([offx, 0*dy+offy, dx, dy])
		#ax2.plot(dz_line_two[0], dz_line_two[1], 'b',   linewidth=2)
		#ax2.plot(dz_line_two[2], dz_line_two[3], 'b--', linewidth=2)
		#ax2.plot(dz_line_two[4], dz_line_two[5], 'b-.', linewidth=2)
		#ax2.plot(dz_line_two[6], dz_line_two[7], 'b:',  linewidth=2)

		#ax2.legend(['mK', 'mK', 'mK', 'mK'], loc=0)
		#ax2.set_xlim([6, 15])
		#ax2.set_ylim([0, 4])
		#ax2.set_yticks([0, 1, 2, 3])

		#plt.xlabel(r'$z_{\rm{r}}$', fontsize=18)
		#plt.ylabel(r'$\Delta z$',   fontsize=18)
		#plt.grid()
		#plt.gca().invert_xaxis()





	# Uncertainties for EDGES polynomial and Gaussian, with weights, and different noise levels
	if figure == 'figure5b':


		T21_K_true = -0.1
		T21_K_ref  = -0.1
		nterms_fg  = 5
		f          = np.arange(90, 190.1, 0.4)
		zlow   = frequency2redshift(np.max(f))
		zhigh  = frequency2redshift(np.min(f))
		z      = np.arange(zlow, zhigh, 0.01)
		dz     = np.arange(0.01, 14.01, 0.01)

		#z      = np.arange(zlow, zhigh, 0.1)
		#dz     = np.arange(0.01, 10.01, 0.1)                

		p_rec  = np.zeros((len(dz), len(z)))
		dp_rec = np.zeros((len(dz), len(z)))

		model_fg  = model_evaluate('EDGES_polynomial', [300/(150**(-2.5)), 0.00], f)
		noise_std = model_fg/np.sqrt(60*60*5*0.2e6)
		noise     = np.random.normal(0, noise_std)   # realistic noise similar to real High-Band data


		#dz_line_two = []
		dz_line = np.zeros((len(z), 4))
		for k in range(4):

			if k == 0:
				div = 1

			if k == 1:
				div = 2

			if k == 2:
				div = 4

			if k == 3:
				div = 8                        




			for j in range(len(z)):
				for i in range(len(dz)):

					print((z[j], dz[i]))
					model_no_noise  = model_evaluate('EDGES_polynomial_plus_gaussian', [300/(150**(-2.5)), T21_K_true], f, zr=z[j], dz=dz[i])
					data            = model_no_noise + noise/div

					# Using Inverse-Variance Weights
					p = fit_polynomial_fourier('EDGES_polynomial_plus_gaussian', f, data, nterms_fg, Weights=(1/((noise_std/div)**2)), zr=z[j], dz=dz[i])


					p_rec[i,j]  = p[0][nterms_fg]
					dp_rec[i,j] = np.sqrt(np.diag(p[3]))[nterms_fg]


			# Produce 1\sigma line from 2D uncertainty array
			for j in range(len(z)):
				for i in range(len(dz)-1):
					if (dp_rec[i,j] <= 0.1/2) and (dp_rec[i+1,j] >= 0.1/2):
						dz_line[j,k] = dz[i]


		# Appending z vector
		z_dz = np.hstack((z.reshape(-1,1), dz_line))



		#Saving data
		path_save = home_folder + '/DATA/EDGES/signal_constraints/simulated/'
		np.savetxt(path_save + 'simulated_constraints_gaussian_90-190MHz.txt', z_dz)









		#plt.close()
		#plt.close()
		#plt.close()


		#f1 = plt.figure(1)
		#offx = 0.1
		#offy = 0.1
		#dx = 0.85
		#dy = 0.4
		#dxcb = 0.02
		#dycb = 0.7*dy

		#ax1 = f1.add_axes([offx, 1*dy+offy, dx, dy])
		#ax1.plot(frequency2redshift(f), 1000*noise, 'b') #,  linewidth=2)
		#ax1.errorbar(frequency2redshift(f), 1000*noise, 1000*noise_std, fmt='.', ecolor='r') #,  linewidth=2)
		#ax1.set_xlim([15, 6])
		#ax1.set_ylim([-50, 50])
		#ax1.grid()
		#ax1.set_xticklabels([])
		##plt.gca().invert_xaxis()
		#ax1.set_ylabel(r'$\Delta$T [mK]', fontsize=16)

		#ax12 = ax1.twiny()
		#ax12.set_xlabel(r'$\nu$ [MHz]', fontsize=18)		
		#ff2 = np.arange(90, 191, 10)
		#z2 = frequency2redshift(ff2)
		#ax12.set_xticks((z2-6)/(15-6))
		#ax12.set_xticklabels(ff2)
		#plt.gca().invert_xaxis()		






		#ax2 = f1.add_axes([offx, 0*dy+offy, dx, dy])
		#ax2.plot(dz_line_two[0], dz_line_two[1], 'b',   linewidth=2)
		#ax2.plot(dz_line_two[2], dz_line_two[3], 'b--', linewidth=2)
		#ax2.plot(dz_line_two[4], dz_line_two[5], 'b-.', linewidth=2)
		#ax2.plot(dz_line_two[6], dz_line_two[7], 'b:',  linewidth=2)

		#ax2.legend(['mK', 'mK', 'mK', 'mK'], loc=0)
		#ax2.set_xlim([6, 15])
		#ax2.set_ylim([0, 4])
		#ax2.set_yticks([0, 1, 2, 3])

		#plt.xlabel(r'$z_{\rm{r}}$', fontsize=18)
		#plt.ylabel(r'$\Delta z$',   fontsize=18)
		#plt.grid()
		#plt.gca().invert_xaxis()













































	# Uncertainties from measured spectra at same LST, different days
	if figure == 'figure6':
		path_file = home_folder + '/DATA/EDGES/spectra/level4/high_band/high_band_v1_blade_s11day_262_recv_temp_full_correction_tambient_300_ground_loss_percent_0.5.hdf5'
		freq, Ta, meta, weights	 = level4read(path_file)

		T21_K_true = 0.028
		T21_K_ref  = 0.028
		nterms_fg  = 5

		f      = np.arange(90, 190.1, 0.4)
		zlow   = frequency2redshift(190)
		zhigh  = frequency2redshift(90)
		z      = np.arange(zlow, zhigh, 0.05)
		dz     = np.arange(0.02, 2.01, 0.02)


		model_fg = model_evaluate('EDGES_polynomial', [400/(150**(-2.5)), 0.00], f)
		noise    = np.random.normal(0, model_fg/np.sqrt(60*60*5*0.4e6))   # realistic noise similar to real High-Band data			


		dz_line = np.zeros((5, len(z)))
		for k in range(5):

			par_fg   = fit_polynomial_fourier('EDGES_polynomial', freq, Ta[10, 20+k*50, :], nterms_fg, Weights = weights[10, 20+k*50, :])

			p_rec  = np.zeros((len(dz), len(z)))
			dp_rec = np.zeros((len(dz), len(z)))

			for j in range(len(z)):
				for i in range(len(dz)):

					print((k, z[j], dz[i]))
					model_no_noise  = model_evaluate('EDGES_polynomial_plus_tanh', np.append(par_fg[0], T21_K_true), f, zr=z[j], dz=dz[i])
					data            = model_no_noise + noise
					p               = fit_polynomial_fourier('EDGES_polynomial_plus_tanh', f, data, nterms_fg, zr=z[j], dz=dz[i])
					p_rec[i,j]      = p[0][nterms_fg]
					dp_rec[i,j]     = np.sqrt(np.diag(p[3]))[nterms_fg]


			# Produce 1\sigma line from 2D uncertainty array
			for j in range(len(z)):
				for i in range(len(dz)-1):
					if (dp_rec[i,j] <= 0.028/2) and (dp_rec[i+1,j] >= 0.028/2):
						dz_line[k,j] = dz[i]



		plt.close()
		plt.close()
		plt.close()

		x0 = 0.09
		y0 = 0.09
		dx = 0.89
		dy = 0.82

		f1 = plt.figure(1)		
		ax = f1.add_axes([1*x0, y0+0*dy, dx, dy])
		ax.plot(z, dz_line[0,:], 'b',   linewidth=2)
		ax.plot(z, dz_line[1,:], 'g--', linewidth=2)
		ax.plot(z, dz_line[2,:], 'r-.', linewidth=2)
		ax.plot(z, dz_line[3,:], 'k:',  linewidth=2)
		ax.plot(z, dz_line[4,:], 'm:',  linewidth=2)


		ax.legend(['day 1','day 2','day 3','day 4','day 5'], loc=0)
		ax.set_xlim([6, 15])
		ax.set_ylim([0, 2])

		plt.gca().invert_xaxis()
		plt.xlabel(r'$z_{\rm{r}}$', fontsize=18)
		plt.ylabel(r'$\Delta z$', fontsize=18)
		plt.grid()

		ax2 = ax.twiny()
		ax2.set_xlabel(r'$\nu$ [MHz]', fontsize=18)

		f2 = np.arange(90, 191, 10)
		z2 = frequency2redshift(f2)

		ax2.set_xticks((z2-6)/(15-6))
		ax2.set_xticklabels(f2)
		plt.gca().invert_xaxis()	









	# Uncertainties from measured spectra at different LSTs
	if figure == 'figure7':
		path_file = home_folder + '/DATA/EDGES/spectra/level4/high_band/high_band_v1_blade_s11day_262_recv_temp_full_correction_tambient_300_ground_loss_percent_0.5.hdf5'
		freq, Ta, meta, weights	 = level4read(path_file)

		T21_K_true = 0.028
		T21_K_ref  = 0.028
		nterms_fg  = 5

		f      = np.arange(90, 190.1, 0.4)
		zlow   = frequency2redshift(190)
		zhigh  = frequency2redshift(90)
		z      = np.arange(zlow, zhigh, 0.05)
		dz     = np.arange(0.02, 2.01, 0.02)

		model_fg = model_evaluate('EDGES_polynomial', [400/(150**(-2.5)), 0.00], f)
		noise    = np.random.normal(0, model_fg/np.sqrt(60*60*5*0.4e6))   # realistic noise similar to real High-Band data			

		dz_line = np.zeros((5, len(z)))
		for k in range(5):

			par_fg   = fit_polynomial_fourier('EDGES_polynomial', freq, Ta[5 + k*15, 100, :], nterms_fg, Weights = weights[5 + k*15, 100, :])

			p_rec  = np.zeros((len(dz), len(z)))
			dp_rec = np.zeros((len(dz), len(z)))

			for j in range(len(z)):
				for i in range(len(dz)):

					print((k, z[j], dz[i]))
					model_no_noise  = model_evaluate('EDGES_polynomial_plus_tanh', np.append(par_fg[0], T21_K_true), f, zr=z[j], dz=dz[i])
					data            = model_no_noise + noise
					p               = fit_polynomial_fourier('EDGES_polynomial_plus_tanh', f, data, nterms_fg, zr=z[j], dz=dz[i])
					p_rec[i,j]      = p[0][nterms_fg]
					dp_rec[i,j]     = np.sqrt(np.diag(p[3]))[nterms_fg]


			# Produce 1\sigma line from 2D uncertainty array
			for j in range(len(z)):
				for i in range(len(dz)-1):
					if (dp_rec[i,j] <= 0.028/2) and (dp_rec[i+1,j] >= 0.028/2):
						dz_line[k,j] = dz[i]



		plt.close()
		plt.close()
		plt.close()

		x0 = 0.09
		y0 = 0.09
		dx = 0.89
		dy = 0.82

		f1 = plt.figure(1)		
		ax = f1.add_axes([1*x0, y0+0*dy, dx, dy])
		ax.plot(z, dz_line[0,:], 'b',   linewidth=2)
		ax.plot(z, dz_line[1,:], 'g--', linewidth=2)
		ax.plot(z, dz_line[2,:], 'r-.', linewidth=2)
		ax.plot(z, dz_line[3,:], 'k:',  linewidth=2)
		ax.plot(z, dz_line[4,:], 'm:',  linewidth=2)


		ax.legend([str(round(5*(24/72))) + ' Hr', str(round(20*(24/72))) + ' Hr', str(round(35*(24/72))) + ' Hr', str(round(50*(24/72))) + ' Hr', str(round(65*(24/72))) + ' Hr'], loc=0)
		ax.set_xlim([6, 15])
		ax.set_ylim([0, 2])

		plt.gca().invert_xaxis()
		plt.xlabel(r'$z_{\rm{r}}$', fontsize=18)
		plt.ylabel(r'$\Delta z$', fontsize=18)
		plt.grid()

		ax2 = ax.twiny()
		ax2.set_xlabel(r'$\nu$ [MHz]', fontsize=18)

		f2 = np.arange(90, 191, 10)
		z2 = frequency2redshift(f2)

		ax2.set_xticks((z2-6)/(15-6))
		ax2.set_xticklabels(f2)
		plt.gca().invert_xaxis()	



















	# Rejection results for Tanh in the range 90-190 MHz
	if figure == 'figure8':


		T21_K_true = 0  #0.028
		T21_K_ref  = 0.028
		nterms_fg  = 5

		f      = np.arange(90, 190.1, 0.4)
		zlow   = frequency2redshift(190)
		zhigh  = frequency2redshift(90)
		z      = np.arange(zlow, zhigh, 0.05)
		dz     = np.arange(0.02, 2.01, 0.02)



		p_rec  = np.zeros((len(dz), len(z)))
		dp_rec = np.zeros((len(dz), len(z)))

		model_fg = model_evaluate('EDGES_polynomial', [400/(150**(-2.5)), 0.00], f)
		noise    = np.random.normal(0,model_fg/np.sqrt(60*60*5*0.4e6))   # realistic noise similar to real High-Band data


		dz_line = np.zeros(len(z))
		dr_line = np.zeros(len(z))

		for j in range(len(z)):
			for i in range(len(dz)):

				print((z[j], dz[i]))
				model_no_noise  = model_evaluate('EDGES_polynomial_plus_tanh', [400/(150**(-2.5)), T21_K_true], f, zr=z[j], dz=dz[i])
				data            = model_no_noise + noise
				p               = fit_polynomial_fourier('EDGES_polynomial_plus_tanh', f, data, nterms_fg, zr=z[j], dz=dz[i])
				p_rec[i,j]      = p[0][nterms_fg]
				dp_rec[i,j]     = np.sqrt(np.diag(p[3]))[nterms_fg]






		# 2D detection/rejection array
		dr = (p_rec - T21_K_ref)/dp_rec	




		# Produce 1\sigma line from 2D uncertainty array
		for j in range(len(z)):
			for i in range(len(dz)-1):
				if (dp_rec[i,j] <= np.abs(T21_K_ref)/2) and (dp_rec[i+1,j] >= np.abs(T21_K_ref)/2):
					dz_line[j] = dz[i]




		# Produce line of 95% rejection
		dr_line = np.copy(dz_line)		
		for j in range(len(z)):

			if T21_K_ref > 0:

				if dr[0,j] >= -2:
					dr_line[j] = dz[0]

				elif dr[0,j]<= -2:
					for i in range(len(dz)-1):			
						if (dp_rec[i,j] <= np.abs(T21_K_ref)/2) and ((dr[i,j] <= -2) and (dr[i+1,j] >= -2)):					
							dr_line[j] = dz[i]


			elif T21_K_ref < 0:

				if dr[0,j] <= 2:
					dr_line[j] = dz[0]

				elif dr[0,j]>= 2:
					for i in range(len(dz)-1):			
						if (dp_rec[i,j] <= np.abs(T21_K_ref)/2) and ((dr[i,j] >= 2) and (dr[i+1,j] <= 2)):					
							dr_line[j] = dz[i]




		# Assign NaNs to region outside the relevant band
		for j in range(len(z)):
			for i in range(len(dz)):
				if dz[i] > dz_line[j]:
					p_rec[i,j]  = np.NaN
					dp_rec[i,j] = np.NaN
					dr[i,j]     = np.NaN

				#if dz[i] > dr_line[j]:
					#dr_2sigma[i,j] = np.NaN



		# Assign NaNs to rejection array
		dr_2sigma = np.copy(dr)
		for j in range(len(z)):
			for i in range(len(dz)):
				if (T21_K_ref > 0) and (dr[i,j] > -2):
					dr_2sigma[i,j] = np.NaN

				if (T21_K_ref < 0) and (dr[i,j] < 2):
					dr_2sigma[i,j] = np.NaN		


















		plt.close()
		plt.close()
		plt.close()

		x0 = 0.09
		y0 = 0.09
		dx = 0.89
		dy = 0.82

		f1 = plt.figure(1)		
		ax = f1.add_axes([1*x0, y0+0*dy, dx, dy])
		ax.plot(z, dz_line, 'b',  linewidth=2)
		#ax.legend([r'$\rm{N_{fg}}$ = 2', r'$\rm{N_{fg}}$ = 3', r'$\rm{N_{fg}}$ = 4', r'$\rm{N_{fg}}$ = 5'],loc=0)
		ax.set_xlim([6, 15])
		ax.set_ylim([0, 2])

		plt.gca().invert_xaxis()
		plt.xlabel(r'$z_{\rm{r}}$', fontsize=18)
		plt.ylabel(r'$\Delta z$', fontsize=18)
		plt.grid()


		ax2 = ax.twiny()
		ax2.set_xlabel(r'$\nu$ [MHz]', fontsize=18)

		f2 = np.arange(90, 191, 10)
		z2 = frequency2redshift(f2)

		ax2.set_xticks((z2-6)/(15-6))
		ax2.set_xticklabels(f2)
		plt.gca().invert_xaxis()	







		f2 = plt.figure(2)
		offx = 0.1
		offy = 0.1
		dx = 0.8
		dy = 0.16
		dxcb = 0.02
		dycb = 0.7*dy

		ax1 = f2.add_axes([offx, 4*dy+offy, dx, dy])
		ax1.plot(frequency2redshift(f), 1000*noise, 'b',  linewidth=2)
		ax1.set_xlim([15, 6])
		ax1.set_ylim([-50, 50])
		ax1.grid()
		ax1.set_xticklabels([])
		#plt.gca().invert_xaxis()
		ax1.set_ylabel(r'$\Delta$T [mK]', fontsize=16)

		ax12 = ax1.twiny()
		ax12.set_xlabel(r'$\nu$ [MHz]', fontsize=18)		
		ff2 = np.arange(90, 191, 10)
		z2 = frequency2redshift(ff2)
		ax12.set_xticks((z2-6)/(15-6))
		ax12.set_xticklabels(ff2)
		plt.gca().invert_xaxis()


		ax2 = f2.add_axes([offx, 3*dy+offy, dx, dy])
		im = ax2.imshow( np.fliplr(1000*dp_rec), interpolation='none', aspect='auto', extent=[zhigh, zlow, 0, 2], origin='lower');im.set_clim(0, 14);
		ax2.set_xlim([15, 6])
		ax2.set_ylim([0, 2])
		cbax = f2.add_axes([1.0*offx+dx, 3*dy+offy+(dy/2)-(dycb/2), dxcb, dycb])
		cb = plt.colorbar(im, cax=cbax, ticks=[0,7,14])
		cb.set_label(r'$1\sigma$ [mK]', rotation=90)
		ax2.set_xticklabels([])
		ax2.set_yticks([0, 0.5, 1, 1.5])
		ax2.set_ylabel(r'$\Delta z$', fontsize=18)


		ax3 = f2.add_axes([offx, 2*dy+offy, dx, dy])
		im = ax3.imshow( np.fliplr(1000*p_rec), interpolation='none', aspect='auto', extent=[zhigh, zlow, 0, 2], origin='lower');im.set_clim(-28, 28);
		ax3.set_xlim([15, 6])
		ax3.set_ylim([0, 2])
		cbax = f2.add_axes([1.0*offx+dx, 2*dy+offy+(dy/2)-(dycb/2), dxcb, dycb])
		cb = plt.colorbar(im, cax=cbax, ticks=[-28, -14, 0, 14, 28])
		cb.set_label(r'estimate [mK]', rotation=90)
		ax3.set_xticklabels([])
		ax3.set_yticks([0, 0.5, 1, 1.5])
		ax3.set_ylabel(r'$\Delta z$', fontsize=18)


		ax4 = f2.add_axes([offx, 1*dy+offy, dx, dy])
		im = ax4.imshow( np.fliplr(dr), interpolation='none', aspect='auto', extent=[zhigh, zlow, 0, 2], origin='lower');im.set_clim(-2, 2);
		ax4.set_xlim([15, 6])
		ax4.set_ylim([0, 2])
		cbax = f2.add_axes([1.0*offx+dx, 1*dy+offy+(dy/2)-(dycb/2), dxcb, dycb])
		cb = plt.colorbar(im, cax=cbax, ticks=[-2,-1,0,1,2])
		cb.set_label(r'error [$\sigma$]', rotation=90)
		ax4.set_xticklabels([])
		ax4.set_yticks([0, 0.5, 1, 1.5])
		ax4.set_ylabel(r'$\Delta z$', fontsize=18)


		ax5 = f2.add_axes([offx, 0*dy+offy, dx, dy])
		im = ax5.imshow( np.fliplr(dr_2sigma), interpolation='none', aspect='auto', extent=[zhigh, zlow, 0, 2], origin='lower');im.set_clim(-2, 2);
		ax5.set_xlim([15, 6])
		ax5.set_ylim([0, 2])
		#cbax = f2.add_axes([1.0*offx+dx, 0*dy+offy+(dy/2)-(dycb/2), dxcb, dycb])
		#cb = plt.colorbar(im, cax=cbax, ticks=[-2,-1,0,1,2])
		#cb.set_label(r'error [$\sigma$]', rotation=90)
		ax5.set_xlabel(r'$z_{\rm{r}}$', fontsize=18)
		ax5.set_ylabel(r'$\Delta z$', fontsize=18)
		ax5.set_yticks([0, 0.5, 1, 1.5])

























	# Rejection results for Tanh in the range 110-190 MHz
	if figure == 'figure9':


		T21_K_true = 0  #0.028
		T21_K_ref  = 0.028
		nterms_fg  = 5

		f      = np.arange(110, 190.1, 0.4)
		zlow   = frequency2redshift(190)
		zhigh  = frequency2redshift(110)
		z      = np.arange(zlow, zhigh, 0.05)
		dz     = np.arange(0.02, 2.01, 0.02)



		p_rec  = np.zeros((len(dz), len(z)))
		dp_rec = np.zeros((len(dz), len(z)))

		model_fg = model_evaluate('EDGES_polynomial', [400/(150**(-2.5)), 0.00], f)
		noise    = np.random.normal(0,model_fg/np.sqrt(60*60*5*0.4e6))   # realistic noise similar to real High-Band data


		dz_line = np.zeros(len(z))
		dr_line = np.zeros(len(z))

		for j in range(len(z)):
			for i in range(len(dz)):

				print((z[j], dz[i]))
				model_no_noise  = model_evaluate('EDGES_polynomial_plus_tanh', [400/(150**(-2.5)), T21_K_true], f, zr=z[j], dz=dz[i])
				data            = model_no_noise + noise
				p               = fit_polynomial_fourier('EDGES_polynomial_plus_tanh', f, data, nterms_fg, zr=z[j], dz=dz[i])
				p_rec[i,j]      = p[0][nterms_fg]
				dp_rec[i,j]     = np.sqrt(np.diag(p[3]))[nterms_fg]






		# 2D detection/rejection array
		dr = (p_rec - T21_K_ref)/dp_rec	




		# Produce 1\sigma line from 2D uncertainty array
		for j in range(len(z)):
			for i in range(len(dz)-1):
				if (dp_rec[i,j] <= np.abs(T21_K_ref)/2) and (dp_rec[i+1,j] >= np.abs(T21_K_ref)/2):
					dz_line[j] = dz[i]




		# Produce line of 95% rejection
		dr_line = np.copy(dz_line)		
		for j in range(len(z)):

			if T21_K_ref > 0:

				if dr[0,j] >= -2:
					dr_line[j] = dz[0]

				elif dr[0,j]<= -2:
					for i in range(len(dz)-1):			
						if (dp_rec[i,j] <= np.abs(T21_K_ref)/2) and ((dr[i,j] <= -2) and (dr[i+1,j] >= -2)):					
							dr_line[j] = dz[i]


			elif T21_K_ref < 0:

				if dr[0,j] <= 2:
					dr_line[j] = dz[0]

				elif dr[0,j]>= 2:
					for i in range(len(dz)-1):			
						if (dp_rec[i,j] <= np.abs(T21_K_ref)/2) and ((dr[i,j] >= 2) and (dr[i+1,j] <= 2)):					
							dr_line[j] = dz[i]




		# Assign NaNs to region outside the relevant band
		for j in range(len(z)):
			for i in range(len(dz)):
				if dz[i] > dz_line[j]:
					p_rec[i,j]  = np.NaN
					dp_rec[i,j] = np.NaN
					dr[i,j]     = np.NaN

				#if dz[i] > dr_line[j]:
					#dr_2sigma[i,j] = np.NaN



		# Assign NaNs to rejection array
		dr_2sigma = np.copy(dr)
		for j in range(len(z)):
			for i in range(len(dz)):
				if (T21_K_ref > 0) and (dr[i,j] > -2):
					dr_2sigma[i,j] = np.NaN

				if (T21_K_ref < 0) and (dr[i,j] < 2):
					dr_2sigma[i,j] = np.NaN		


















		plt.close()
		plt.close()
		plt.close()

		x0 = 0.09
		y0 = 0.09
		dx = 0.89
		dy = 0.82

		f1 = plt.figure(1)		
		ax = f1.add_axes([1*x0, y0+0*dy, dx, dy])
		ax.plot(z, dz_line, 'b',  linewidth=2)
		#ax.legend([r'$\rm{N_{fg}}$ = 2', r'$\rm{N_{fg}}$ = 3', r'$\rm{N_{fg}}$ = 4', r'$\rm{N_{fg}}$ = 5'],loc=0)
		ax.set_xlim([6, 15])
		ax.set_ylim([0, 2])

		plt.gca().invert_xaxis()
		plt.xlabel(r'$z_{\rm{r}}$', fontsize=18)
		plt.ylabel(r'$\Delta z$', fontsize=18)
		plt.grid()


		ax2 = ax.twiny()
		ax2.set_xlabel(r'$\nu$ [MHz]', fontsize=18)

		f2 = np.arange(90, 191, 10)
		z2 = frequency2redshift(f2)

		ax2.set_xticks((z2-6)/(15-6))
		ax2.set_xticklabels(f2)
		plt.gca().invert_xaxis()	







		f2 = plt.figure(2)
		offx = 0.1
		offy = 0.1
		dx = 0.8
		dy = 0.16
		dxcb = 0.02
		dycb = 0.7*dy

		ax1 = f2.add_axes([offx, 4*dy+offy, dx, dy])
		ax1.plot(frequency2redshift(f), 1000*noise, 'b',  linewidth=2)
		ax1.set_xlim([15, 6])
		ax1.set_ylim([-50, 50])
		ax1.grid()
		ax1.set_xticklabels([])
		#plt.gca().invert_xaxis()
		ax1.set_ylabel(r'$\Delta$T [mK]', fontsize=16)

		ax12 = ax1.twiny()
		ax12.set_xlabel(r'$\nu$ [MHz]', fontsize=18)		
		ff2 = np.arange(90, 191, 10)
		z2 = frequency2redshift(ff2)
		ax12.set_xticks((z2-6)/(15-6))
		ax12.set_xticklabels(ff2)
		plt.gca().invert_xaxis()


		ax2 = f2.add_axes([offx, 3*dy+offy, dx, dy])
		im = ax2.imshow( np.fliplr(1000*dp_rec), interpolation='none', aspect='auto', extent=[zhigh, zlow, 0, 2], origin='lower');im.set_clim(0, 14);
		ax2.set_xlim([15, 6])
		ax2.set_ylim([0, 2])
		cbax = f2.add_axes([1.0*offx+dx, 3*dy+offy+(dy/2)-(dycb/2), dxcb, dycb])
		cb = plt.colorbar(im, cax=cbax, ticks=[0,7,14])
		cb.set_label(r'$1\sigma$ [mK]', rotation=90)
		ax2.set_xticklabels([])
		ax2.set_yticks([0, 0.5, 1, 1.5])
		ax2.set_ylabel(r'$\Delta z$', fontsize=18)


		ax3 = f2.add_axes([offx, 2*dy+offy, dx, dy])
		im = ax3.imshow( np.fliplr(1000*p_rec), interpolation='none', aspect='auto', extent=[zhigh, zlow, 0, 2], origin='lower');im.set_clim(-28, 28);
		ax3.set_xlim([15, 6])
		ax3.set_ylim([0, 2])
		cbax = f2.add_axes([1.0*offx+dx, 2*dy+offy+(dy/2)-(dycb/2), dxcb, dycb])
		cb = plt.colorbar(im, cax=cbax, ticks=[-28, -14, 0, 14, 28])
		cb.set_label(r'estimate [mK]', rotation=90)
		ax3.set_xticklabels([])
		ax3.set_yticks([0, 0.5, 1, 1.5])
		ax3.set_ylabel(r'$\Delta z$', fontsize=18)


		ax4 = f2.add_axes([offx, 1*dy+offy, dx, dy])
		im = ax4.imshow( np.fliplr(dr), interpolation='none', aspect='auto', extent=[zhigh, zlow, 0, 2], origin='lower');im.set_clim(-2, 2);
		ax4.set_xlim([15, 6])
		ax4.set_ylim([0, 2])
		cbax = f2.add_axes([1.0*offx+dx, 1*dy+offy+(dy/2)-(dycb/2), dxcb, dycb])
		cb = plt.colorbar(im, cax=cbax, ticks=[-2,-1,0,1,2])
		cb.set_label(r'error [$\sigma$]', rotation=90)
		ax4.set_xticklabels([])
		ax4.set_yticks([0, 0.5, 1, 1.5])
		ax4.set_ylabel(r'$\Delta z$', fontsize=18)


		ax5 = f2.add_axes([offx, 0*dy+offy, dx, dy])
		im = ax5.imshow( np.fliplr(dr_2sigma), interpolation='none', aspect='auto', extent=[zhigh, zlow, 0, 2], origin='lower');im.set_clim(-2, 2);
		ax5.set_xlim([15, 6])
		ax5.set_ylim([0, 2])
		#cbax = f2.add_axes([1.0*offx+dx, 0*dy+offy+(dy/2)-(dycb/2), dxcb, dycb])
		#cb = plt.colorbar(im, cax=cbax, ticks=[-2,-1,0,1,2])
		#cb.set_label(r'error [$\sigma$]', rotation=90)
		ax5.set_xlabel(r'$z_{\rm{r}}$', fontsize=18)
		ax5.set_ylabel(r'$\Delta z$', fontsize=18)
		ax5.set_yticks([0, 0.5, 1, 1.5])





































	# Rejection results for Gaussian in the range 90-190 MHz
	if figure == 'figure10':


		T21_K_true = 0  #0.028
		T21_K_ref  = -0.1
		nterms_fg  = 5

		f      = np.arange(90, 190.1, 0.4)
		zlow   = frequency2redshift(190)
		zhigh  = frequency2redshift(90)
		z      = np.arange(zlow, zhigh, 0.05)
		dz     = np.arange(0.02, 8.01, 0.02)



		p_rec  = np.zeros((len(dz), len(z)))
		dp_rec = np.zeros((len(dz), len(z)))

		model_fg = model_evaluate('EDGES_polynomial', [400/(150**(-2.5)), 0.00], f)
		noise    = np.random.normal(0,model_fg/np.sqrt(60*60*5*0.4e6))   # realistic noise similar to real High-Band data


		dz_line = np.zeros(len(z))
		dr_line = np.zeros(len(z))

		for j in range(len(z)):
			for i in range(len(dz)):

				print((z[j], dz[i]))
				model_no_noise  = model_evaluate('EDGES_polynomial_plus_gaussian', [400/(150**(-2.5)), T21_K_true], f, zr=z[j], dz=dz[i])
				data            = model_no_noise + noise
				p               = fit_polynomial_fourier('EDGES_polynomial_plus_gaussian', f, data, nterms_fg, zr=z[j], dz=dz[i])
				p_rec[i,j]      = p[0][nterms_fg]
				dp_rec[i,j]     = np.sqrt(np.diag(p[3]))[nterms_fg]




		# 2D detection/rejection array
		dr = (p_rec - T21_K_ref)/dp_rec	




		# Produce 1\sigma line from 2D uncertainty array
		for j in range(len(z)):
			for i in range(len(dz)-1):
				if (dp_rec[i,j] <= np.abs(T21_K_ref)/2) and (dp_rec[i+1,j] >= np.abs(T21_K_ref)/2):
					dz_line[j] = dz[i]




		# Produce line of 95% rejection
		dr_line = np.copy(dz_line)		
		for j in range(len(z)):

			if T21_K_ref > 0:

				if dr[0,j] >= -2:
					dr_line[j] = dz[0]

				elif dr[0,j]<= -2:
					for i in range(len(dz)-1):			
						if (dp_rec[i,j] <= np.abs(T21_K_ref)/2) and ((dr[i,j] <= -2) and (dr[i+1,j] >= -2)):					
							dr_line[j] = dz[i]


			elif T21_K_ref < 0:

				if dr[0,j] <= 2:
					dr_line[j] = dz[0]

				elif dr[0,j]>= 2:
					for i in range(len(dz)-1):			
						if (dp_rec[i,j] <= np.abs(T21_K_ref)/2) and ((dr[i,j] >= 2) and (dr[i+1,j] <= 2)):					
							dr_line[j] = dz[i]




		# Assign NaNs to region outside the relevant band
		for j in range(len(z)):
			for i in range(len(dz)):
				if dz[i] > dz_line[j]:
					p_rec[i,j]  = np.NaN
					dp_rec[i,j] = np.NaN
					dr[i,j]     = np.NaN

				#if dz[i] > dr_line[j]:
					#dr_2sigma[i,j] = np.NaN



		# Assign NaNs to rejection array
		dr_2sigma = np.copy(dr)
		for j in range(len(z)):
			for i in range(len(dz)):
				if (T21_K_ref > 0) and (dr[i,j] > -2):
					dr_2sigma[i,j] = np.NaN

				if (T21_K_ref < 0) and (dr[i,j] < 2):
					dr_2sigma[i,j] = np.NaN				





		plt.close()
		plt.close()
		plt.close()

		x0 = 0.09
		y0 = 0.09
		dx = 0.89
		dy = 0.82

		f1 = plt.figure(1)		
		ax = f1.add_axes([1*x0, y0+0*dy, dx, dy])
		ax.plot(z, dz_line, 'b',  linewidth=2)
		#ax.legend([r'$\rm{N_{fg}}$ = 2', r'$\rm{N_{fg}}$ = 3', r'$\rm{N_{fg}}$ = 4', r'$\rm{N_{fg}}$ = 5'],loc=0)
		ax.set_xlim([6, 15])
		ax.set_ylim([0, np.max(dz)])

		plt.gca().invert_xaxis()
		plt.xlabel(r'$z_{\rm{r}}$', fontsize=18)
		plt.ylabel(r'$\Delta z$', fontsize=18)
		plt.grid()


		ax2 = ax.twiny()
		ax2.set_xlabel(r'$\nu$ [MHz]', fontsize=18)

		f2 = np.arange(90, 191, 10)
		z2 = frequency2redshift(f2)

		ax2.set_xticks((z2-6)/(15-6))
		ax2.set_xticklabels(f2)
		plt.gca().invert_xaxis()	







		f2 = plt.figure(2)
		offx = 0.1
		offy = 0.1
		dx = 0.8
		dy = 0.16
		dxcb = 0.02
		dycb = 0.7*dy

		ax1 = f2.add_axes([offx, 4*dy+offy, dx, dy])
		ax1.plot(frequency2redshift(f), 1000*noise, 'b',  linewidth=2)
		ax1.set_xlim([15, 6])
		ax1.set_ylim([-50, 50])
		ax1.grid()
		ax1.set_xticklabels([])
		#plt.gca().invert_xaxis()
		ax1.set_ylabel(r'$\Delta$T [mK]', fontsize=16)

		ax12 = ax1.twiny()
		ax12.set_xlabel(r'$\nu$ [MHz]', fontsize=18)		
		ff2 = np.arange(90, 191, 10)
		z2 = frequency2redshift(ff2)
		ax12.set_xticks((z2-6)/(15-6))
		ax12.set_xticklabels(ff2)
		plt.gca().invert_xaxis()


		ax2 = f2.add_axes([offx, 3*dy+offy, dx, dy])
		im = ax2.imshow( np.fliplr(1000*dp_rec), interpolation='none', aspect='auto', extent=[zhigh, zlow, np.min(dz), np.max(dz)], origin='lower');im.set_clim(0, round(np.abs(1000*T21_K_ref)/2));
		ax2.set_xlim([15, 6])
		ax2.set_ylim([0, np.max(dz)])
		cbax = f2.add_axes([1.0*offx+dx, 3*dy+offy+(dy/2)-(dycb/2), dxcb, dycb])
		cb = plt.colorbar(im, cax=cbax, ticks=[0, round(np.abs(1000*T21_K_ref)/4), round(np.abs(1000*T21_K_ref)/2)])
		cb.set_label(r'$1\sigma$ [mK]', rotation=90)
		ax2.set_xticklabels([])
		ax2.set_yticks(np.arange(0,np.max(dz)-1,2))
		ax2.set_ylabel(r'$\Delta z$', fontsize=18)


		ax3 = f2.add_axes([offx, 2*dy+offy, dx, dy])
		im = ax3.imshow( np.fliplr(1000*p_rec), interpolation='none', aspect='auto', extent=[zhigh, zlow, np.min(dz), np.max(dz)], origin='lower');im.set_clim(-round(np.abs(1000*T21_K_ref)/2), round(np.abs(1000*T21_K_ref)/2));
		ax3.set_xlim([15, 6])
		ax3.set_ylim([0, np.max(dz)])
		cbax = f2.add_axes([1.0*offx+dx, 2*dy+offy+(dy/2)-(dycb/2), dxcb, dycb])
		cb = plt.colorbar(im, cax=cbax, ticks=[-round(np.abs(1000*T21_K_ref)/2), -round(np.abs(1000*T21_K_ref)/4), 0, round(np.abs(1000*T21_K_ref)/4), round(np.abs(1000*T21_K_ref)/2)])
		cb.set_label(r'estimate [mK]', rotation=90)
		ax3.set_xticklabels([])
		ax3.set_yticks(np.arange(0,np.max(dz)-1,2))
		ax3.set_ylabel(r'$\Delta z$', fontsize=18)


		ax4 = f2.add_axes([offx, 1*dy+offy, dx, dy])
		im = ax4.imshow( np.fliplr(dr), interpolation='none', aspect='auto', extent=[zhigh, zlow, np.min(dz), np.max(dz)], origin='lower');im.set_clim(-2, 2);
		ax4.set_xlim([15, 6])
		ax4.set_ylim([0, np.max(dz)])
		cbax = f2.add_axes([1.0*offx+dx, 1*dy+offy+(dy/2)-(dycb/2), dxcb, dycb])
		cb = plt.colorbar(im, cax=cbax, ticks=[-2,-1,0,1,2])
		cb.set_label(r'error [$\sigma$]', rotation=90)
		ax4.set_xticklabels([])
		ax4.set_yticks(np.arange(0,np.max(dz)-1,2))
		ax4.set_ylabel(r'$\Delta z$', fontsize=18)


		ax5 = f2.add_axes([offx, 0*dy+offy, dx, dy])
		im = ax5.imshow( np.fliplr(dr_2sigma), interpolation='none', aspect='auto', extent=[zhigh, zlow, np.min(dz), np.max(dz)], origin='lower');im.set_clim(-2, 2);
		ax5.set_xlim([15, 6])
		ax5.set_ylim([0, np.max(dz)])
		#cbax = f2.add_axes([1.0*offx+dx, 0*dy+offy+(dy/2)-(dycb/2), dxcb, dycb])
		#cb = plt.colorbar(im, cax=cbax, ticks=[-2,-1,0,1,2])
		#cb.set_label(r'error [$\sigma$]', rotation=90)
		ax5.set_xlabel(r'$z_{\rm{r}}$', fontsize=18)
		ax5.set_ylabel(r'$\Delta z$', fontsize=18)
		ax5.set_yticks(np.arange(0,np.max(dz)-1,2))











	# Rejection results for Gaussian in the range 110-190 MHz
	if figure == 'figure11':


		T21_K_true = 0  #0.028
		T21_K_ref  = -0.1
		nterms_fg  = 5

		f      = np.arange(110, 190.1, 0.4)
		zlow   = frequency2redshift(190)
		zhigh  = frequency2redshift(110)
		z      = np.arange(zlow, zhigh, 0.05)
		dz     = np.arange(0.02, 8.01, 0.02)



		p_rec  = np.zeros((len(dz), len(z)))
		dp_rec = np.zeros((len(dz), len(z)))

		model_fg = model_evaluate('EDGES_polynomial', [400/(150**(-2.5)), 0.00], f)
		noise    = np.random.normal(0,model_fg/np.sqrt(60*60*5*0.4e6))   # realistic noise similar to real High-Band data


		dz_line = np.zeros(len(z))
		dr_line = np.zeros(len(z))

		for j in range(len(z)):
			for i in range(len(dz)):

				print((z[j], dz[i]))
				model_no_noise  = model_evaluate('EDGES_polynomial_plus_gaussian', [400/(150**(-2.5)), T21_K_true], f, zr=z[j], dz=dz[i])
				data            = model_no_noise + noise
				print((len(f), len(data)))
				p               = fit_polynomial_fourier('EDGES_polynomial_plus_gaussian', f, data, nterms_fg, zr=z[j], dz=dz[i])
				p_rec[i,j]      = p[0][nterms_fg]
				dp_rec[i,j]     = np.sqrt(np.diag(p[3]))[nterms_fg]




		# 2D detection/rejection array
		dr = (p_rec - T21_K_ref)/dp_rec	




		# Produce 1\sigma line from 2D uncertainty array
		for j in range(len(z)):
			for i in range(len(dz)-1):
				if (dp_rec[i,j] <= np.abs(T21_K_ref)/2) and (dp_rec[i+1,j] >= np.abs(T21_K_ref)/2):
					dz_line[j] = dz[i]




		# Produce line of 95% rejection
		dr_line = np.copy(dz_line)		
		for j in range(len(z)):

			if T21_K_ref > 0:

				if dr[0,j] >= -2:
					dr_line[j] = dz[0]

				elif dr[0,j]<= -2:
					for i in range(len(dz)-1):			
						if (dp_rec[i,j] <= np.abs(T21_K_ref)/2) and ((dr[i,j] <= -2) and (dr[i+1,j] >= -2)):					
							dr_line[j] = dz[i]


			elif T21_K_ref < 0:

				if dr[0,j] <= 2:
					dr_line[j] = dz[0]

				elif dr[0,j]>= 2:
					for i in range(len(dz)-1):			
						if (dp_rec[i,j] <= np.abs(T21_K_ref)/2) and ((dr[i,j] >= 2) and (dr[i+1,j] <= 2)):					
							dr_line[j] = dz[i]




		# Assign NaNs to region outside the relevant band
		for j in range(len(z)):
			for i in range(len(dz)):
				if dz[i] > dz_line[j]:
					p_rec[i,j]  = np.NaN
					dp_rec[i,j] = np.NaN
					dr[i,j]     = np.NaN

				#if dz[i] > dr_line[j]:
					#dr_2sigma[i,j] = np.NaN



		# Assign NaNs to rejection array
		dr_2sigma = np.copy(dr)
		for j in range(len(z)):
			for i in range(len(dz)):
				if (T21_K_ref > 0) and (dr[i,j] > -2):
					dr_2sigma[i,j] = np.NaN

				if (T21_K_ref < 0) and (dr[i,j] < 2):
					dr_2sigma[i,j] = np.NaN				





		plt.close()
		plt.close()
		plt.close()

		x0 = 0.09
		y0 = 0.09
		dx = 0.89
		dy = 0.82

		f1 = plt.figure(1)		
		ax = f1.add_axes([1*x0, y0+0*dy, dx, dy])
		ax.plot(z, dz_line, 'b',  linewidth=2)
		#ax.legend([r'$\rm{N_{fg}}$ = 2', r'$\rm{N_{fg}}$ = 3', r'$\rm{N_{fg}}$ = 4', r'$\rm{N_{fg}}$ = 5'],loc=0)
		ax.set_xlim([6, 15])
		ax.set_ylim([0, np.max(dz)])

		plt.gca().invert_xaxis()
		plt.xlabel(r'$z_{\rm{r}}$', fontsize=18)
		plt.ylabel(r'$\Delta z$', fontsize=18)
		plt.grid()


		ax2 = ax.twiny()
		ax2.set_xlabel(r'$\nu$ [MHz]', fontsize=18)

		f2 = np.arange(90, 191, 10)
		z2 = frequency2redshift(f2)

		ax2.set_xticks((z2-6)/(15-6))
		ax2.set_xticklabels(f2)
		plt.gca().invert_xaxis()	







		f2 = plt.figure(2)
		offx = 0.1
		offy = 0.1
		dx = 0.8
		dy = 0.16
		dxcb = 0.02
		dycb = 0.7*dy

		ax1 = f2.add_axes([offx, 4*dy+offy, dx, dy])
		ax1.plot(frequency2redshift(f), 1000*noise, 'b',  linewidth=2)
		ax1.set_xlim([15, 6])
		ax1.set_ylim([-50, 50])
		ax1.grid()
		ax1.set_xticklabels([])
		#plt.gca().invert_xaxis()
		ax1.set_ylabel(r'$\Delta$T [mK]', fontsize=16)

		ax12 = ax1.twiny()
		ax12.set_xlabel(r'$\nu$ [MHz]', fontsize=18)		
		ff2 = np.arange(90, 191, 10)
		z2 = frequency2redshift(ff2)
		ax12.set_xticks((z2-6)/(15-6))
		ax12.set_xticklabels(ff2)
		plt.gca().invert_xaxis()


		ax2 = f2.add_axes([offx, 3*dy+offy, dx, dy])
		im = ax2.imshow( np.fliplr(1000*dp_rec), interpolation='none', aspect='auto', extent=[zhigh, zlow, np.min(dz), np.max(dz)], origin='lower');im.set_clim(0, round(np.abs(1000*T21_K_ref)/2));
		ax2.set_xlim([15, 6])
		ax2.set_ylim([0, np.max(dz)])
		cbax = f2.add_axes([1.0*offx+dx, 3*dy+offy+(dy/2)-(dycb/2), dxcb, dycb])
		cb = plt.colorbar(im, cax=cbax, ticks=[0, round(np.abs(1000*T21_K_ref)/4), round(np.abs(1000*T21_K_ref)/2)])
		cb.set_label(r'$1\sigma$ [mK]', rotation=90)
		ax2.set_xticklabels([])
		ax2.set_yticks(np.arange(0,np.max(dz)-1,2))
		ax2.set_ylabel(r'$\Delta z$', fontsize=18)


		ax3 = f2.add_axes([offx, 2*dy+offy, dx, dy])
		im = ax3.imshow( np.fliplr(1000*p_rec), interpolation='none', aspect='auto', extent=[zhigh, zlow, np.min(dz), np.max(dz)], origin='lower');im.set_clim(-round(np.abs(1000*T21_K_ref)/2), round(np.abs(1000*T21_K_ref)/2));
		ax3.set_xlim([15, 6])
		ax3.set_ylim([0, np.max(dz)])
		cbax = f2.add_axes([1.0*offx+dx, 2*dy+offy+(dy/2)-(dycb/2), dxcb, dycb])
		cb = plt.colorbar(im, cax=cbax, ticks=[-round(np.abs(1000*T21_K_ref)/2), -round(np.abs(1000*T21_K_ref)/4), 0, round(np.abs(1000*T21_K_ref)/4), round(np.abs(1000*T21_K_ref)/2)])
		cb.set_label(r'estimate [mK]', rotation=90)
		ax3.set_xticklabels([])
		ax3.set_yticks(np.arange(0,np.max(dz)-1,2))
		ax3.set_ylabel(r'$\Delta z$', fontsize=18)


		ax4 = f2.add_axes([offx, 1*dy+offy, dx, dy])
		im = ax4.imshow( np.fliplr(dr), interpolation='none', aspect='auto', extent=[zhigh, zlow, np.min(dz), np.max(dz)], origin='lower');im.set_clim(-2, 2);
		ax4.set_xlim([15, 6])
		ax4.set_ylim([0, np.max(dz)])
		cbax = f2.add_axes([1.0*offx+dx, 1*dy+offy+(dy/2)-(dycb/2), dxcb, dycb])
		cb = plt.colorbar(im, cax=cbax, ticks=[-2,-1,0,1,2])
		cb.set_label(r'error [$\sigma$]', rotation=90)
		ax4.set_xticklabels([])
		ax4.set_yticks(np.arange(0,np.max(dz)-1,2))
		ax4.set_ylabel(r'$\Delta z$', fontsize=18)


		ax5 = f2.add_axes([offx, 0*dy+offy, dx, dy])
		im = ax5.imshow( np.fliplr(dr_2sigma), interpolation='none', aspect='auto', extent=[zhigh, zlow, np.min(dz), np.max(dz)], origin='lower');im.set_clim(-2, 2);
		ax5.set_xlim([15, 6])
		ax5.set_ylim([0, np.max(dz)])
		#cbax = f2.add_axes([1.0*offx+dx, 0*dy+offy+(dy/2)-(dycb/2), dxcb, dycb])
		#cb = plt.colorbar(im, cax=cbax, ticks=[-2,-1,0,1,2])
		#cb.set_label(r'error [$\sigma$]', rotation=90)
		ax5.set_xlabel(r'$z_{\rm{r}}$', fontsize=18)
		ax5.set_ylabel(r'$\Delta z$', fontsize=18)
		ax5.set_yticks(np.arange(0,np.max(dz)-1,2))













	# Rejection results for Anastasia's models in the range 90-190 MHz
	if figure == 'figure12':


		T21_K_true = 0  #0.028
		T21_K_ref  = 1
		nterms_fg  = 5

		f      = np.arange(90, 190.1, 0.4)
		#zlow   = frequency2redshift(190)
		#zhigh  = frequency2redshift(90)
		#z      = np.arange(zlow, zhigh, 0.05)
		#dz     = np.arange(0.02, 8.01, 0.02)





		model_fg = model_evaluate('EDGES_polynomial', [400/(150**(-2.5)), 0.00], f)
		noise    = np.random.normal(0,model_fg/np.sqrt(60*60*5*0.4e6))   # realistic noise similar to real High-Band data


		#dz_line = np.zeros(len(z))
		#dr_line = np.zeros(len(z))

		n_models = 195
		p_rec  = np.zeros(n_models)
		dp_rec = np.zeros(n_models)		

		#anastasia_amplitude = 1
		for j in range(n_models):

			print(j)
			model_no_noise  = model_evaluate('EDGES_polynomial_plus_anastasia', [400/(150**(-2.5)), T21_K_true], f, anastasia_model_number=j)
			data            = model_no_noise + noise
			p               = fit_polynomial_fourier('EDGES_polynomial_plus_anastasia', f, data, nterms_fg, anastasia_model_number=j)
			p_rec[j]        = p[0][nterms_fg]
			dp_rec[j]       = np.sqrt(np.diag(p[3]))[nterms_fg]



		# 2D detection/rejection array
		dr = (p_rec - T21_K_ref)/dp_rec	


		plt.figure(1)
		plt.subplot(2,1,1)
		plt.errorbar(np.arange(0,len(p_rec)), p_rec, dp_rec)
		plt.ylim([-2, 2])
		plt.grid()
		plt.ylabel('estimated amplitude')

		plt.subplot(2,1,2)
		plt.plot(np.arange(0,len(p_rec)), dr)
		plt.ylim([-2, 2])
		plt.grid()
		plt.xlabel('model number')
		plt.ylabel(r'detection / rejection significance [$\sigma$]')









	# Rejection results for Anastasia's models in the range 110-190 MHz
	if figure == 'figure13':


		T21_K_true = 0  #0.028
		#T21_K_ref  = -0.1
		nterms_fg  = 5

		f      = np.arange(110, 190.1, 0.4)
		#zlow   = frequency2redshift(190)
		#zhigh  = frequency2redshift(90)
		#z      = np.arange(zlow, zhigh, 0.05)
		#dz     = np.arange(0.02, 8.01, 0.02)





		model_fg = model_evaluate('EDGES_polynomial', [400/(150**(-2.5)), 0.00], f)
		noise    = np.random.normal(0,model_fg/np.sqrt(60*60*5*0.4e6))   # realistic noise similar to real High-Band data


		#dz_line = np.zeros(len(z))
		#dr_line = np.zeros(len(z))

		n_models = 195
		p_rec  = np.zeros(n_models)
		dp_rec = np.zeros(n_models)		

		anastasia_amplitude = 1
		for j in range(n_models):

			print(j)
			model_no_noise  = model_evaluate('EDGES_polynomial_plus_anastasia', [400/(150**(-2.5)), T21_K_true], f, anastasia_model_number=j)
			data            = model_no_noise + noise
			p               = fit_polynomial_fourier('EDGES_polynomial_plus_anastasia', f, data, nterms_fg, anastasia_model_number=j)
			p_rec[j]        = p[0][nterms_fg]
			dp_rec[j]       = np.sqrt(np.diag(p[3]))[nterms_fg]







	# Rejection results for Jordan's models in the range 90-190 MHz
	if figure == 'figure14':


		T21_K_true = 0  #0.028
		#T21_K_ref  = -0.1
		nterms_fg  = 5

		f      = np.arange(90, 190.1, 0.4)
		#zlow   = frequency2redshift(190)
		#zhigh  = frequency2redshift(90)
		#z      = np.arange(zlow, zhigh, 0.05)
		#dz     = np.arange(0.02, 8.01, 0.02)





		model_fg = model_evaluate('EDGES_polynomial', [400/(150**(-2.5)), 0.00], f)
		noise    = np.random.normal(0, model_fg/np.sqrt(60*60*5*0.4e6))   # realistic noise similar to real High-Band data


		#dz_line = np.zeros(len(z))
		#dr_line = np.zeros(len(z))

		n_models = 391
		p_rec  = np.zeros(n_models)
		dp_rec = np.zeros(n_models)		

		jordan_amplitude = 1
		for j in range(n_models):

			print(j)
			model_no_noise  = model_evaluate('EDGES_polynomial_plus_jordan', [400/(150**(-2.5)), T21_K_true], f, jordan_model_number=j)
			data            = model_no_noise + noise
			p               = fit_polynomial_fourier('EDGES_polynomial_plus_jordan', f, data, nterms_fg, jordan_model_number=j)
			p_rec[j]        = p[0][nterms_fg]
			dp_rec[j]       = np.sqrt(np.diag(p[3]))[nterms_fg]





	return z_dz    #p_rec, dp_rec, f, noise #p_rec, dp_rec, dr, dr_line #p_rec, dp_rec, noise































def exercise_detection_rejection_2():



	T21_K_true = -0.1
	T21_K_ref  = -0.1
	nterms_fg  = 5
	f          = np.arange(90, 190.1, 0.4)



	zlow   = frequency2redshift(np.max(f))
	zhigh  = frequency2redshift(np.min(f))
	z      = np.arange(zlow, zhigh, 0.01)
	dz     = np.arange(0.01, 14.01, 0.01)               

	p_rec  = np.zeros((len(dz), len(z)))
	dp_rec = np.zeros((len(dz), len(z)))

	model_fg  = model_evaluate('EDGES_polynomial', [300/(150**(-2.5)), 0.00], f)
	noise_std = model_fg/np.sqrt(60*60*5*0.2e6)
	noise     = np.random.normal(0, noise_std)   # realistic noise similar to real High-Band data


	#dz_line_two = []
	dz_line = np.zeros((len(z), 4))
	for k in range(4):

		if k == 0:
			div = 1

		if k == 1:
			div = 2

		if k == 2:
			div = 4

		if k == 3:
			div = 8                        




		for j in range(len(z)):
			for i in range(len(dz)):

				print((z[j], dz[i]))
				model_no_noise  = model_evaluate('EDGES_polynomial_plus_gaussian', [300/(150**(-2.5)), T21_K_true], f, zr=z[j], dz=dz[i])
				data            = model_no_noise + noise/div

				# Using Inverse-Variance Weights
				p = fit_polynomial_fourier('EDGES_polynomial_plus_gaussian', f, data, nterms_fg, Weights=(1/((noise_std/div)**2)), zr=z[j], dz=dz[i])


				p_rec[i,j]  = p[0][nterms_fg]
				dp_rec[i,j] = np.sqrt(np.diag(p[3]))[nterms_fg]


		# Produce 1\sigma line from 2D uncertainty array
		for j in range(len(z)):
			for i in range(len(dz)-1):
				if (dp_rec[i,j] <= 0.1/2) and (dp_rec[i+1,j] >= 0.1/2):
					dz_line[j,k] = dz[i]


	# Appending z vector
	z_dz = np.hstack((z.reshape(-1,1), dz_line))






	return 0










