


import numpy as np
import basic as ba
import edges as eg
import matplotlib.pyplot as plt

import os, sys

edges_folder       = os.environ['EDGES_vol2']
print('EDGES Folder: ' + edges_folder)


sys.path.insert(0, "/home/raul/edges/old")
import old_edges as oeg









def comparison_mid_band_to_low2_EW():


	filename = '/home/raul/DATA1/EDGES_vol1/spectra/level3/low_band2_2017/EW_with_shield_nominal/2017_160_00.hdf5'
	flx, tlx, wlx, ml = oeg.level3_read_raw_spectra(filename)


	filename = '/home/raul/DATA2/EDGES_vol2/mid_band/spectra/level3/case2_75MHz/2018_150_00.hdf5'
	fmx, tmx, pm, rmx, wmx, rmsm, tpm, mm  = eg.level3read(filename)

	FLOW  = 60
	FHIGH = 100

	fl = flx[(flx>=FLOW) & (flx<=FHIGH)]
	tl = tlx[:, (flx>=FLOW) & (flx<=FHIGH)]
	wl = wlx[:, (flx>=FLOW) & (flx<=FHIGH)]

	fm = fmx[(fmx>=FLOW) & (fmx<=FHIGH)]
	tm = tmx[:, (fmx>=FLOW) & (fmx<=FHIGH)]
	wm = wmx[:, (fmx>=FLOW) & (fmx<=FHIGH)]



	il_0 = 1597
	il_2 = 1787
	il_4 = 1976
	il_6 = 2166
	il_8 = 77
	il_10 = 267
	il_12 = 457
	il_14 = 647
	il_16 = 837
	il_18 = 1027
	il_20 = 1217
	il_22 = 1407
	
	
	im_0 = 1610
	im_2 = 1793
	im_4 = 1977
	im_6 = 2161
	im_8 = 138
	im_10 = 322
	im_12 = 505
	im_14 = 689
	im_16 = 873
	im_18 = 1057
	im_20 = 1241
	im_22 = 1425
	
	
	il = [il_0, il_2, il_4, il_6, il_8, il_10, il_12, il_14, il_16, il_18, il_20, il_22]
	im = [im_0, im_2, im_4, im_6, im_8, im_10, im_12, im_14, im_16, im_18, im_20, im_22]
	
	
	
	#tl_0  = tl[0,:]
	#wl_0  = wl[0,:]


	#tm63 = tm[63,:]
	#wm63 = wm[63,:]

	#tl0[wm63==0]=0 #np.nan
	#wl0[wm63==0]=0

	#tm63[wm63==0]=0 #np.nan
	#wm63[wm63==0]=0
	
	
	#r          = tm63-tl0
	##fb, rb, wb = ba.spectral_binning_number_of_samples(fl, r, wm63)


	#plt.figure(1)
	#plt.plot(fl, tl0 + 100)
	#plt.plot(fm, tm63)
	#plt.legend(['low-band 2 EW','mid-band'])
	#plt.ylim([500, 3500])

	#plt.figure(2)
	#plt.plot(fl, tm63-tl0)
	
	
	##plt.figure(3)
	##plt.plot(fb, rb)



	plt.figure()
	#gg = [0.5, 0.5, 0.5]
	gg = 'c'
	for i in range(12):
		tli = tl[il[i],:]
		wli = wl[il[i],:]
		
		tmi = tm[im[i],:]
		wmi = wm[im[i],:]
		
		if i%2 == 0:
			plt.plot(fl[wmi>0], (tmi-tli)[wmi>0] - i*600, color=gg)
		else:
			plt.plot(fl[wmi>0], (tmi-tli)[wmi>0] - i*600, color=gg)
			
		plt.plot([50, 150], [-i*600, -i*600], 'k')
		
	plt.xlim([58, 102])
			
		









	return 0 #fl, tl, wl, ml, fm, tm, wm, mm, r, wm63





















