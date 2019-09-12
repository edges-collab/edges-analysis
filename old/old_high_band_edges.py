
import edges as eg
import numpy as np
import matplotlib.pyplot as plt
import h5py
#import emcee as ec
import time


from os import listdir, makedirs, system
from os.path import exists
from os.path import expanduser


# Determining home folder
home_folder = expanduser("~")






def single_file_rfi_check(name):

	# How to process files from Level1 to Level2
	#
	# fe, tt_no_rfi, ww_no_rfi, meta = eg.level1_to_level2_v2('high_band', '2015', '206_00', 'v2_65_195_MHz', 'plot_v2_65_195_MHz')
	#
	
	
	
	
	
	f, t, m, w = eg.level2read_v2('/home/ramo7131/DATA/EDGES/spectra/level2/high_band/v2_65_195_MHz/' + name + '.hdf5')  # 2015_252_00

	#ff = f[f<85]
	#tt = t[:,f<85]
	#ww = w[:,f<85]
	
	ff = f[f>170]
	tt = t[:,f>170]
	ww = w[:,f>170]	


	Nf       = 2
	Nspectra = len(t[:,0])

	rms_all  = np.zeros(Nspectra)

	for i in range(Nspectra):  
		print(str(i+1) + ' ' + str(Nspectra))

		par          = np.polyfit(np.log(ff/20000)[ww[i,:]>0], np.log(tt[i,:])[ww[i,:]>0], Nf-1)
		log_model    = np.polyval(par, np.log(ff/20000))
		model        = np.exp(log_model)
		res          = tt[i,:] - model
		#print(ww.shape)
		
		fb, rb, wb   = eg.spectral_binning_number_of_samples(ff, res, ww[i,:], nsamples=64)
		
		
		if i == 0:
			rb_all  = np.zeros((Nspectra, len(fb)))
		
		rb_all[i,:] = rb
		rms_all[i]   = np.std(   rb[wb>0]   )



	return fb, rb_all, rms_all





def daily_data_filter(name):
	
	rfi_index = 0
	bad_file  = 1
	
	


	# Fourpoint

	if name == '2015_109_11':
		rfi_index = [1103, 1104, 1105, 1106, 1107, 1108]
		bad_file = 0	

	if name == '2015_110_00':
		rfi_index = [470, 480, 511, 512, 513, 520, 1720, 1835, 1869]
		bad_file = 0	


	if name == '2015_111_00':
		rfi_index = [72, 195, 204, 205, 206, 207, 239, 271, 272, 273, 274, 275, 276, 277, 293, 357, 370, 371, 372, 373, 403, 413, 421, 422, 423, 424, 425, 426, 440, 471, 472, 473, 474, 475, 476, 477, 478, 650, 651, 652, 653, 769, 781, 782, 783, 784, 785, 786, 787, 788, 789, 853, 883, 907, 908, 909, 1207, 1478, 1586, 1602, 1607, 1802]  + list(range(108, 128))  + list(range(306, 331)) + list(range(622, 635)) + list(range(807, 833))
		bad_file = 0	


	if name == '2015_112_00':
		rfi_index = [1, 145, 171, 576, 596, 602, 653, 743, 753, 773, 818, 917, 984, 996, 1265, 1526, 1527, 1528, 2058, 2064, 2172, 2173, 2174] + list(range(320, 338)) + list(range(434, 448))
		bad_file = 0	


	#if name == '113_00':
		#rfi_index = []
		#bad_file = 0


	if name == '2015_113_09':
		rfi_index =[1435]  + list(range(0, 250))
		bad_file = 0


	if name == '2015_114_00':
		rfi_index = list(range(0, 960))
		bad_file = 0
		
	if name == '2015_115_00':
		rfi_index = list(range(0, 630)) + [747, 748, 749, 750, 751, 752, 753, 754, 1348, 1721, 2033, 2034, 2035, 2036] + list(range(2265, 2336))
		bad_file = 0		


	if name == '2015_116_00':
		rfi_index = list(range(0, 180)) + [491, 650, 734, 898, 1772, 1901, 1902, 2084] + list(range(522, 568)) + list(range(2237, 2335))
		bad_file = 0


	if name == '2015_117_00':
		rfi_index = list(range(0, 150)) + [258, 292, 293, 302, 303, 304, 305, 306, 307, 883, 884, 906, 933, 937, 1329, 1906, 1916, 1963] + list(range(716, 737))
		bad_file = 0


	if name == '2015_118_00':
		rfi_index = [704, 705, 706, 707, 708, 1291, 1322, 1402, 1437, 1987]
		bad_file = 0


	if name == '2015_119_13':
		rfi_index = [162, 323, 626, 666, 667, 668, 669, 670, 704, 775]
		bad_file = 0
		

	if name == '2015_120_00':
		rfi_index = list(range(380, 880))
		bad_file = 0


	#if name == '120_14':
		#rfi_index = []
		#bad_file = 0


	#if name == '121_00':
		#rfi_index = []
		#bad_file = 0


	if name == '2015_126_00':
		rfi_index = list(range(125, 775)) + [1, 22, 28, 32, 33, 60, 70, 1537, 2146, 2180, 2181, 2182, 2243, 2244, 2245]
		bad_file = 0


	#if name == '127_00':
		#rfi_index = []
		#bad_file = 0


	if name == '2015_128_00':
		rfi_index = list(range(0, 890)) + list(range(2160, 2264))
		bad_file = 0


	if name == '2015_129_00':
		rfi_index = list(range(0, 920)) + list(range(2160, 2264)) + [1519, 1593]
		bad_file = 0


	if name == '2015_130_00':
		rfi_index = list(range(0, 885)) + list(range(2145, 2263))
		bad_file = 0


	if name == '2015_131_00':
		rfi_index = list(range(0, 950)) + [1063, 1064, 1883, 1884, 2100, 2101, 2102, 2103] + list(range(2150, 2262))
		bad_file = 0


	#if name == '132_00':
		#rfi_index = 0
		#bad_file = 0
		
	
	if name == '2015_133_00':
		rfi_index = list(range(0, 870)) + list(range(2210, 2295))
		bad_file = 0		

	
	if name == '2015_134_00':
		rfi_index = list(range(0, 930))
		bad_file = 0
		
	if name == '2015_135_00':
		rfi_index = list(range(0, 980)) + list(range(2245, 2300))
		bad_file = 0
		

	#if name == '136_00':
		#rfi_index = 0
		#bad_file = 0		


	#if name == '137_00':
		#rfi_index = 0
		#bad_file = 0		


	if name == '2015_138_00':
		rfi_index = list(range(0, 250)) + [723, 735, 2275, 2301]
		bad_file = 0


	if name == '2015_139_00':
		rfi_index = list(range(0, 600)) + [719, 720, 721, 722, 723, 894, 1339, 2224]
		bad_file = 0


	if name == '2015_140_00':
		rfi_index = [706, 707, 708, 709, 710]
		bad_file = 0

	if name == '2015_141_00':
		rfi_index = [665, 693, 695, 697, 2200]
		bad_file = 0


	if name == '2015_142_00':
		rfi_index = [681, 841, 978, 1541] + list(range(1150, 1225)) + list(range(1735, 1770)) + list(range(1900, 1920)) + list(range(1960, 1990))
		bad_file = 0


	if name == '2015_143_00':
		rfi_index = [433, 438, 663, 664, 665, 666, 667, 668, 828]
		bad_file = 0


	if name == '2015_144_00':
		rfi_index = []   # pretty good
		bad_file = 0


	if name == '2015_145_00':
		rfi_index = [797, 798, 799, 800, 801, 802, 803, 1021]
		bad_file = 0


	if name == '2015_146_00':
		rfi_index = [575, 579, 785, 2187]   
		bad_file = 0


	if name == '2015_147_00':
		rfi_index = [767, 772, 775] + list(range(1630, 1690))
		bad_file = 0


	if name == '2015_148_00':
		rfi_index = list(range(350, 400)) + list(range(745, 780)) + [1301]
		bad_file = 0

	if name == '2015_149_00':
		rfi_index = list(range(739, 747))
		bad_file = 0

	if name == '2015_150_00':
		rfi_index = [638, 641, 727, 728, 729, 730, 731, 732, 2154, 2187] 
		bad_file = 0

	if name == '2015_151_00':
		rfi_index = [516, 717, 757, 815] 
		bad_file = 0
		
	if name == '2015_152_00':
		rfi_index = [25, 94, 99, 226, 374, 414, 415, 701, 742, 859, 864, 903, 933, 1203, 1240, 1355, 1362, 1446, 1605, 1759, 1765, 2131, 2205] 
		bad_file = 0		
		
	if name == '2015_153_00':
		rfi_index = [298, 393, 394, 495, 497, 614] + list(range(678, 685)) + list(range(1500, 2287))
		bad_file = 0		
	
	if name == '2015_154_00':
		rfi_index = list(range(0, 1010))
		bad_file = 0
		
	if name == '2015_155_00':
		rfi_index = [819, 820, 1021, 1022, 1023, 1344, 1379, 1380, 1381, 1563, 1564, 2032, 2209, 2240] 
		bad_file = 0		
	
	if name == '2015_156_00':
		rfi_index = [51, 52, 53, 72, 806] 
		bad_file = 0


	if name == '2015_162_00':
		rfi_index = list(range(400, 520)) + [719] 
		bad_file = 0


	if name == '2015_163_00':
		rfi_index = [290, 542, 706, 707, 708, 1473, 1474]  + list(range(1870, 1905))
		bad_file = 0
		
	
	if name == '2015_164_00':
		rfi_index = list(range(0, 900))
		bad_file = 0		
		
	
	if name == '2015_165_00':
		rfi_index = list(range(345, 445)) + [679, 841, 842, 883, 906, 907, 943, 944, 1002, 1326, 1327, 1407, 1824]
		bad_file = 0
				
	#if name == '166_00':
		#rfi_index = 0
		#bad_file = 0				

	if name == '2015_167_00':
		rfi_index = list(range(145, 865))
		bad_file = 0

	#if name == '168_00':
		#rfi_index = 0
		#bad_file = 0				
	
	#if name == '169_00':
		#rfi_index = 0
		#bad_file = 0		

	#if name == '170_00':
		#rfi_index = 0
		#bad_file = 0	

	#if name == '171_00':
		#rfi_index = 0
		#bad_file = 0	

	#if name == '172_00':
		#rfi_index = 0
		#bad_file = 0	

	#if name == '173_00':
		#rfi_index = 0
		#bad_file = 0	
		
	#if name == '174_00':
		#rfi_index = 0
		#bad_file = 0	


	if name == '2015_175_00':
		rfi_index = list(range(0, 900)) + [983, 984, 1085, 1215, 1227, 1242, 1261, 1335, 1419, 1543, 1544, 2134] + list(range(2270, 2297))
		bad_file = 0


	#if name == '176_00':
		#rfi_index = 0
		#bad_file = 0	


	if name == '2015_177_00':
		rfi_index = [37, 55, 198, 202, 208, 230, 242, 264, 668, 669, 670, 671, 672, 673, 674, 675, 676, 677, 678, 764, 787, 904, 1656, 1665, 1678, 1686, 1703, 1704, 1774, 1778, 1780, 2140, 2146, 2171, 2274, 2275] + list(range(300, 400))
		bad_file = 0	

	if name == '2015_178_00':
		rfi_index = list(range(0, 500)) + [554, 565, 657, 663, 813, 814, 815, 816, 817, 818, 819, 820, 821, 1046, 1504, 1773, 1832, 2118]
		bad_file = 0


	#if name == '179_00':
		#rfi_index = 0
		#bad_file = 0	


	if name == '2015_180_00':
		rfi_index = list(range(100, 120)) + list(range(339, 400)) + [516, 640, 668, 731, 748, 764, 859, 1324, 1336, 1420, 1426, 1592, 1678, 2281] + list(range(590, 630)) + list(range(782, 798))
		bad_file = 0

	if name == '2015_181_00':
		rfi_index = [84, 182, 232, 247, 286, 293, 382, 407, 618, 776, 777, 962, 1283, 1990, 2093, 2157, 2274] + list(range(324, 355)) + list(range(450, 601)) + list(range(662, 701))
		bad_file = 0

	if name == '2015_182_00':
		rfi_index = [1337, 1467, 1469, 1739, 1872, 2007] + list(range(725, 800)) + list(range(2000, 2298))
		bad_file = 0
		
	if name == '2015_183_00':
		rfi_index = [155, 156, 157, 746, 747, 748, 749, 750, 751, 1303, 2111] + list(range(1422, 1462))
		bad_file = 0		
			
	if name == '2015_184_00':
		rfi_index = [163, 311, 592, 734, 736, 1001, 1781] + list(range(1422, 1462))
		bad_file = 0		
				
	if name == '2015_185_00':
		rfi_index = list(range(717, 727)) + list(range(1000, 1250))
		bad_file = 0		

	if name == '2015_186_00':
		rfi_index = [702, 703, 704, 705, 1973, 2122, 2253, 2291]
		bad_file = 0

	if name == '2015_187_00':
		rfi_index = list(range(0, 950)) + [1383, 1427, 1569, 2273, 2274, 2275, 2276, 2277, 2278]
		bad_file = 0

	if name == '2015_188_00':
		rfi_index = list(range(0, 840)) + [1498, 1499, 1542, 1543, 1640, 1837] + list(range(2200, 2300))
		bad_file = 0

	if name == '2015_189_00':
		rfi_index = list(range(0, 850)) + list(range(2240, 2300))
		bad_file = 0
		
	if name == '2015_190_00':
		rfi_index = list(range(0, 900)) + list(range(2240, 2301))
		bad_file = 0		
		
	if name == '2015_191_00':
		rfi_index = list(range(0, 900)) + list(range(2200, 2299))
		bad_file = 0		
						
	if name == '2015_192_00':
		rfi_index = list(range(0, 875)) + [1182, 1194, 1454, 1463]
		bad_file = 0	

	if name == '2015_192_18':
		rfi_index = list(range(470, 510))
		bad_file = 0	

	if name == '2015_193_00':
		rfi_index = list(range(0, 1055)) + list(range(1655, 2300))
		bad_file = 0


	#if name == '194_00':
		#rfi_index = 0
		#bad_file = 0	


	if name == '2015_195_00':
		rfi_index = list(range(0, 1100))
		bad_file = 0


	if name == '2015_198_00':
		rfi_index = [143, 192, 218, 219, 235, 287, 693, 694, 695, 696, 697, 698, 699, 700, 745, 749, 759, 760, 780, 857, 1025, 1570] + list(range(350, 550))
		bad_file = 0
















	# Blade
	
	if name == '2015_206_00':
		rfi_index = [81, 82, 111, 150, 152, 159, 175, 189, 202, 230, 306, 350, 356, 735, 737, 811, 1119, 1279, 1337, 1406, 1495, 1610, 2005]
		bad_file = 0	
	
	if name == '2015_207_00':
		rfi_index = [496, 718, 723]
		bad_file = 0		

	if name == '2015_208_00':
		rfi_index = [703, 704, 706, 707, 708]
		bad_file = 0		
	
	if name == '2015_209_00':
		rfi_index = [689, 692, 693]
		bad_file = 0		
	
	if name == '2015_210_00':
		rfi_index = []  # nothing, pretty good
		bad_file = 0	

	if name == '2015_210_03':
		rfi_index = [346, 351, 470, 505, 678, 689, 750, 1044, 1087, 1133, 1721]  # nothing, pretty good
		bad_file = 0
	
	if name == '2015_211_00':
		rfi_index = list(range(370, 450)) + [668, 670, 791, 795, 826, 1212]
		bad_file = 0	

	if name == '2015_211_18':
		rfi_index = []  # nothing, pretty good
		bad_file = 0
	
	if name == '2015_212_00':
		rfi_index = list(range(625, 1071))
		bad_file = 0

	if name == '2015_215_08':
		rfi_index = []  # nothing, pretty good
		bad_file = 0

	if name == '2015_216_00':
		rfi_index = list(range(0, 900)) 
		bad_file = 0
		
	if name == '2015_217_00':
		rfi_index = list(range(0, 825)) + [972, 1032, 1226] + list(range(2200, 2269))
		bad_file = 0		
	
	if name == '2015_218_00':
		rfi_index = list(range(0, 970)) + list(range(1170, 1300)) + list(range(1830, 2278))
		bad_file = 0		
		
	if name == '2015_219_00':
		rfi_index = list(range(0, 911)) + [959, 1016, 1137, 1138, 1141, 1188, 1368, 1370, 1477, 1494, 1742, 1893, 1930] + list(range(2210, 2276))
		bad_file = 0
		
	if name == '2015_220_00':
		rfi_index = list(range(0, 950)) + list(range(2200, 2274))
		bad_file = 0		
		
	if name == '2015_221_00':
		rfi_index = list(range(0, 950))
		bad_file = 0
		
	if name == '2015_222_00':
		rfi_index = list(range(0, 900)) + list(range(2235, 2277))
		bad_file = 0		
	
	if name == '2015_223_00':
		rfi_index = list(range(0, 900))
		bad_file = 0

	if name == '2015_224_00':
		rfi_index = list(range(0, 850)) + [2168, 2229, 2242, 2248]
		bad_file = 0

	if name == '2015_225_00':
		rfi_index = list(range(0, 850))
		bad_file = 0
		
	if name == '2015_226_00':
		rfi_index = [318, 355, 367, 1022, 1266] + list(range(510, 900))
		bad_file = 0		
		
	if name == '2015_227_00':
		rfi_index = [762, 763, 764, 765, 766, 767, 768]
		bad_file = 0

	if name == '2015_228_00':
		rfi_index = [43, 518, 743, 751, 752, 758, 1339, 1432, 1470, 1513, 1590, 1650, 1886]
		bad_file = 0

	if name == '2015_229_00':
		rfi_index = list(range(0, 460)) + list(range(560, 580)) + [735, 738]
		bad_file = 0
		
	if name == '2015_230_00':
		rfi_index = [722, 724, 806, 807, 808] + list(range(2241, 2250))
		bad_file = 0
		
	if name == '2015_231_00':
		rfi_index = [257, 593, 706, 707, 708, 709] 
		bad_file = 0
		
	if name == '2015_232_00':
		rfi_index = list(range(0, 1000))
		bad_file = 0				
		
	if name == '2015_233_00':
		rfi_index = list(range(678, 687))
		bad_file = 0
		
	if name == '2015_234_00':
		rfi_index = list(range(0, 1150)) + [2242, 2252, 2253, 2254, 2268]
		bad_file = 0
		
	if name == '2015_235_00':
		rfi_index = list(range(0, 830)) + list(range(2195, 2275))
		bad_file = 0		

	if name == '2015_236_00':
		rfi_index = list(range(0, 950)) + list(range(2220, 2275))
		bad_file = 0			
	
	if name == '2015_237_00':
		rfi_index = list(range(0, 950)) + list(range(2150, 2274))
		bad_file = 0	
		
	if name == '2015_238_00':
		rfi_index = list(range(0, 1200)) + list(range(2150, 2275))
		bad_file = 0

	if name == '2015_239_00':
		rfi_index = list(range(0, 1000)) + list(range(1690, 2274))
		bad_file = 0

	if name == '2015_240_00':
		rfi_index = list(range(0, 950)) + list(range(2160, 2274))
		bad_file = 0
		
	if name == '2015_241_00':
		rfi_index = list(range(0, 910)) + [1300, 1368, 1423, 2263]
		bad_file = 0
		
	if name == '2015_242_00':
		rfi_index = list(range(0, 840))
		bad_file = 0
		
	if name == '2015_243_00':
		rfi_index = [80, 266, 294, 414, 916, 1381]
		bad_file = 0
		
	# if name == '244_07':
		# rfi_index = []
		# bad_file = 0		

	if name == '2015_245_00':
		rfi_index = list(range(0, 900)) + [1803]
		bad_file = 0






		

	# Good data set
	
	
	
	if name == '2015_250_15':
		rfi_index = [392]   # Nothing bad 
		bad_file = 0

	if name == '2015_251_00':
		rfi_index = [176, 193, 330, 343, 441, 485, 529, 532, 541, 542, 564, 565, 579, 587, 610, 745, 746, 747, 748, 749, 750, 751, 1163]
		bad_file = 0
		
	elif name == '2015_252_00':
		rfi_index = [736, 737, 738, 739, 740, 741, 742, 743, 744, 1083, 1252, 2151, 2152, 2153, 2154]
		bad_file = 0
		
	elif name == '2015_253_00':
		rfi_index = [724, 725, 726, 727, 728, 729, 730, 731, 764, 780, 798, 1045, 1117, 1118, 1119, 1120, 1121, 1122, 1646, 2158, 2159, 2160, 2161, 2162]   # Nothing bad
		bad_file = 0
		
	elif name == '2015_254_00':
		rfi_index = [687, 688, 689, 690, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 872, 873, 874, 875, 1124, 1125, 1126, 1127, 1128, 1129, 1130, 1131, 1132, 1133, 1152, 1400, 1478, 1907, 2140, 2183, 2276]   # Nothing bad
		bad_file = 0
		
	elif name == '2015_255_00':
		rfi_index = list(range(0, 901)) + [469, 541, 565, 576, 596, 1015, 1074]
		bad_file = 0
		
	elif name == '2015_256_00':
		rfi_index = [62, 393, 440, 738, 871, 920, 947, 1021] + list(range(0, 1000)) + list(range(2000, 2280))
		bad_file = 0
		
	elif name == '2015_257_00':
		rfi_index = [70, 71, 72, 73, 74, 106, 169, 237, 244, 664, 665, 666, 667 ,668, 669, 670, 671, 672, 673, 674, 675, 676, 825, 826, 827, 828, 829, 830, 831, 832, 833, 834, 835, 836, 1084, 1467, 1638, 1644, 2018, 1987, 2186, 2187, 2188, 2200, 2208, 2233, 2237, 2258, 2257, 2258, 2259, 2260] 
		bad_file = 0

	elif name == '2015_258_00':
		rfi_index = [19, 22, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 98, 112, 113, 187, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 270, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 392, 393, 394, 395, 396, 397, 450, 459, 464, 472, 478, 489, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 591, 592, 593, 594, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 650, 651, 652, 653, 654, 655, 656, 657, 658, 659, 660, 661, 662, 679, 697, 743, 752, 904, 980, 981, 982, 983, 984, 985, 986, 2207, 2277] + list(range(808, 823)) + list(range(2015, 2023))
		bad_file = 0
		
	elif name == '2015_259_00':
		rfi_index = [148, 233, 234, 383, 400, 401, 409, 410, 420, 421, 422, 437, 438, 439, 440, 468, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 495, 496, 525, 526, 527, 528, 529, 549, 550, 578, 593, 604, 627, 637, 640, 642, 643, 644, 645, 680, 753, 795, 796, 797, 798, 799, 800, 801, 802, 803, 804, 805, 1013, 1014, 1015, 1016]
		bad_file = 0
		
	elif name == '2015_260_00':
		rfi_index = [172, 173, 174, 175, 176, 177, 178, 179, 180, 235, 236, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 319, 322, 323, 326, 355, 455, 456, 457, 458, 501, 545, 622, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 649, 664, 665, 666, 667, 717, 840, 987, 988, 989, 990, 991, 992, 993, 994, 995, 996, 1170, 1324, 1340, 1505, 1717, 2161, 2162, 2198] + list(range(779, 793))
		bad_file = 0

	elif name == '2015_261_00':
		rfi_index = list(range(34, 132)) + [164, 238, 239, 240, 241, 242, 243, 244, 304, 870, 994, 995, 996, 997, 998, 999, 1000, 1001, 1002, 1003, 1004, 1048, 1049, 1050, 1051, 1952] + list(range(312, 324)) + list(range(374, 626)) + list(range(762, 776)) + [2223, 2224]
		bad_file = 0
		
	elif name == '2015_262_00':
		rfi_index = [87, 108, 141, 257, 267, 324, 436, 542, 600, 625, 626, 750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 865, 866, 867, 1104, 1105, 1113, 1173, 1223]
		bad_file = 0
		
	elif name == '2015_265_19':
		rfi_index = [5, 9, 10]   # Nothing bad !!
		bad_file = 0
		
	elif name == '2015_266_00':
		rfi_index = list(range(40, 95)) + list(range(125, 136)) + [157, 158, 159, 160, 161] + list(range(180, 202))  + list(range(215, 222)) + [282, 283, 284, 285, 286, 287, 288, 294, 295, 296, 297, 298, 312, 313, 402, 497, 498, 499, 500, 501, 641, 668, 732, 1190] + list(range(692, 703))  + list(range(852, 863)) + list(range(1728, 1735)) + [1883, 1884, 1885, 1886, 1887, 1939, 1968, 1969, 1970, 1976, 1977, 1978, 2020]
		bad_file = 0
		
	elif name == '2015_267_00':
		rfi_index = list(range(325, 447)) + [486, 838, 839, 840, 841] + list(range(675, 715)) + list(range(2195, 2264))
		bad_file = 0
		
	elif name == '2015_268_00':
		rfi_index = list(range(0, 975)) + [2211]
		bad_file = 0
		
	elif name == '2015_269_00':
		rfi_index = list(range(0, 865)) + [1743]
		bad_file = 0
		
	elif name == '2015_270_00':
		rfi_index = [217, 221, 276, 277, 278, 279, 380, 389] + list(range(408, 661)) + [693, 700, 721, 962, 1161, 1188, 1638, 1644, 2265] + list(range(794, 811))
		bad_file = 0
		
	elif name == '2015_271_00':
		rfi_index = list(range(0, 940)) + [997, 998, 999, 1000, 1001, 1002, 1989, 1990, 2164, 2165, 2166, 2167, 2168, 2169] + list(range(2195, 2251))	
		bad_file = 0
		
	elif name == '2015_272_00':
		rfi_index = list(range(0, 920)) + [1290, 1759, 1760, 1761, 1762, 1763, 1764, 1891] + list(range(1980, 2267))
		bad_file = 0
		
	elif name == '2015_273_00':
		rfi_index = list(range(0, 945)) + list(range(2155, 2267))
		bad_file = 0
		
	elif name == '2015_274_00':
		rfi_index = list(range(0, 940)) + [1604, 2100] + list(range(2220, 2264))
		bad_file = 0
		
	elif name == '2015_275_00':
		rfi_index = list(range(0, 801))
		bad_file = 0
		
	elif name == '2015_276_00':
		rfi_index = [317, 716, 717, 718, 719, 720, 721, 722, 723, 1468]
		bad_file = 0

	elif name == '2015_277_00':
		rfi_index = [120, 595, 702, 703, 704, 705, 706, 707, 708, 1122, 1138, 1139, 1451] + list(range(1500, 2050)) + list(range(150, 400))
		bad_file = 0

	elif name == '2015_278_00':
		rfi_index = [428] + [847, 1229, 1279, 1405, 1445, 1471, 1498, 1971] + list(range(2100, 2263)) + list(range(540, 700)) 
		bad_file = 0

	elif name == '2015_279_00':
		rfi_index = [45, 75, 92, 95, 99, 102, 372, 530, 610, 673, 677, 681, 831, 835, 836, 839, 841]
		bad_file = 0

	elif name == '2015_280_00':
		rfi_index = [211, 304, 658]
		bad_file = 0

	elif name == '2015_281_17':
		rfi_index = [479]
		bad_file = 0

	elif name == '2015_282_00':
		rfi_index = [130, 142, 199, 205, 277, 343, 1499] + list(range(625, 651)) + list(range(784, 805))  # nothing bad!!
		bad_file = 0

	elif name == '2015_283_00':
		rfi_index = [36, 261, 405, 424, 425, 474, 475, 476, 477, 478] + list(range(612, 636)) + list(range(773, 791)) + [1864, 1865, 1866, 1867, 1868, 1869, 1870]
		bad_file = 0

	elif name == '2015_284_00':
		rfi_index = [16, 75, 78, 79, 80, 256, 269, 278, 478, 480, 490, 496, 497, 498, 499, 682, 723, 732, 744] + list(range(573, 631)) + list(range(764, 781)) + list(range(1508, 1521))
		bad_file = 0

	elif name == '2015_285_00':
		rfi_index = [218, 262, 376, 377, 384, 392, 417, 426] + list(range(745, 1046)) + list(range(470, 492)) + list(range(590, 603)) + [1132, 1551]
		bad_file = 0

	elif name == '2015_286_00':
		rfi_index = list(range(0, 100)) + [457, 575, 901, 1339, 1360, 1361, 1362, 1363, 1517, 1518, 1519, 1520, 1521, 1613, 2242, 2276] + list(range(741, 753))
		bad_file = 0

	elif name == '2015_287_00':
		rfi_index = list(range(0, 700)) + [815, 832, 933, 934, 935, 1285, 1364, 1365, 1366, 1367, 1368, 1369, 1370, 1371, 1969, 1970]
		bad_file = 0

	elif name == '2015_288_00':
		rfi_index = [41, 60, 474, 475, 476, 477, 535, 536, 537, 572, 601, 602, 603, 604, 605, 606, 661, 673, 678, 679, 680, 681, 684, 691, 698, 711, 712, 713, 714, 715, 716, 717, 718, 775, 800, 1378, 2101, 2118, 2119, 2120, 2150] + list(range(100, 450)) 
		bad_file = 0

	elif name == '2015_289_00':
		rfi_index = [2, 34, 58, 72, 230, 326, 340, 343, 344, 345, 346, 373, 430, 500, 604, 605, 606, 607, 608, 628, 629, 673, 687, 1240, 1380, 1381, 1382, 1383, 1384, 1385, 1482, 1520, 1764, 1781] + list(range(695, 711)) + list(range(2035, 2276))
		bad_file = 0
		
	#elif name == '290_00':
		#rfi_index = [ALL BAD]
		
	elif name == '2015_291_00':
		rfi_index = [54, 73, 74, 75, 100, 127, 132, 142, 143, 151, 254, 290, 309, 310, 348, 357, 364, 366, 389, 408, 417, 455, 611, 631, 700, 702, 793, 803, 857, 858, 859, 860, 896, 948, 949, 950, 951, 952, 953, 979, 1139, 1145, 1390, 1393, 1423, 1424, 1425, 1466, 1510, 1525] + list(range(669, 679))  + list(range(832, 841))
		bad_file = 0
		
	elif name == '2015_292_00':
		rfi_index = [312, 364, 367, 632, 657, 814, 818]  # Nothing bad !!
		bad_file = 0

	elif name == '2015_293_00':
		rfi_index = list(range(0, 90)) + [112, 157, 218, 328, 375, 417, 592, 660, 665, 1227, 1228, 1229, 1230, 1231, 1232, 1233, 1234, 1235] + list(range(793, 806))
		bad_file = 0

	#elif name == '294_00':
		#rfi_index = [pretty much ALL BAD]
		
	elif name == '2015_295_00':
		rfi_index = list(range(0, 850)) + [1248] 
		bad_file = 0
		
	elif name == '2015_296_00':
		rfi_index = [55, 126, 127, 128, 129, 138, 139, 140, 141, 142, 155, 157, 250, 251, 252, 253, 280, 290, 295, 298, 299, 300, 303, 304, 305, 306, 325, 346, 356, 366, 367, 409, 452, 453, 454, 476, 539, 544, 545, 546, 547, 548, 549] + list(range(193, 225)) + list(range(539, 548)) + list(range(583, 635)) + list(range(758, 769)) + [658, 675, 698, 816, 832, 847, 1499, 1500, 1501, 1502, 1503, 1504, 1505, 1506, 1507, 1508, 1509]
		bad_file = 0
		
	elif name == '2015_297_00':
		rfi_index = [42, 156, 157, 181, 182, 186, 187, 188, 189, 190, 191, 192, 193, 194, 223, 224, 225, 226, 227, 228, 413, 414, 805, 1513, 1514, 1515, 1516, 1562] + list(range(585, 596)) + list(range(742, 758)) + list(range(902, 913)) + list(range(2100, 2171))
		bad_file = 0
		
	elif name == '2015_298_00':
		rfi_index = [166, 506, 552, 575, 566, 622, 1104] + list(range(727, 743))
		bad_file = 0
		
	elif name == '2015_299_00':
		rfi_index = list(range(77, 160)) +  list(range(285, 565)) + [69, 235, 708, 719, 755, 806, 927, 1105, 1106, 1246, 1256, 1277, 1328, 1352, 1353, 1354, 1355, 1356, 1357] + list(range(1427, 1821)) + [1839, 1843, 1852, 1887] + list(range(2125, 2273))
		bad_file = 0
		




	# Third data set
	# ----------------------------------------------------------
		
	# elif name == '300_00':
		# rfi_index = []
		# bad_file = 0	

	elif name == '2015_301_00':
		rfi_index = list(range(0, 1000)) + list(range(1780, 2280))
		bad_file = 0	

	# elif name == '302_00':
		# rfi_index = []
		# bad_file = 0		

	# elif name == '303_00':
		# rfi_index = []
		# bad_file = 0

	elif name == '2015_310_18':
		rfi_index = [515]
		bad_file = 0	

	elif name == '2015_311_00':
		rfi_index = list(range(0, 950)) + [1422, 1617, 1618, 2067, 2263, 2264, 2265]
		bad_file = 0		
	
	elif name == '2015_312_00':
		rfi_index = list(range(699, 707)) + [862, 1783]
		bad_file = 0	
	
	elif name == '2015_313_00':
		rfi_index = [572, 685, 691, 851, 2256]
		bad_file = 0
		
	elif name == '2015_314_00':
		rfi_index = list(range(0, 950)) + [2209] 
		bad_file = 0

	elif name == '2015_315_00':
		rfi_index = list(range(0, 830)) + [2239] 
		bad_file = 0


	# elif name == '316_00':
		# rfi_index = []
		# bad_file = 0


	elif name == '2015_317_00':
		rfi_index = [264, 633, 788, 792, 803] 
		bad_file = 0

	elif name == '2015_318_00':
		rfi_index = [465, 1136, 1986] + list(range(645, 1030)) 
		bad_file = 0

	elif name == '2015_319_00':
		rfi_index = [153, 154, 155, 513, 757, 758, 763, 1841] 
		bad_file = 0

	elif name == '2015_320_00':
		rfi_index = list(range(0, 600)) + list(range(743, 752)) + list(range(1340, 2284))
		bad_file = 0

	elif name == '2015_321_00':
		rfi_index = list(range(23, 35)) + list(range(728, 738)) + [1811, 1815]
		bad_file = 0
		
	elif name == '2015_322_00':
		rfi_index = [42, 1792, 1793, 1794, 1795] + list(range(712, 725))
		bad_file = 0		
		
	elif name == '2015_323_00':
		rfi_index = list(range(700, 712)) + [866]
		bad_file = 0	

	elif name == '2015_324_00':
		rfi_index = list(range(689, 694)) + [851, 1854, 1855]
		bad_file = 0

	elif name == '2015_325_00':
		rfi_index = [439, 673, 674, 675, 676, 677, 835]
		bad_file = 0

	# elif name == '326_00':
		# rfi_index = []
		# bad_file = 0
		
	# elif name == '327_00':
		# rfi_index = []
		# bad_file = 0
		
	# elif name == '328_00':
		# rfi_index = []
		# bad_file = 0
		
	elif name == '2015_329_00':
		rfi_index = list(range(0, 1150)) + [1960, 2007, 2233]
		bad_file = 0
		
	elif name == '2015_330_00':
		rfi_index = list(range(0, 1410)) + [2129, 2190, 2276]
		bad_file = 0		
		
	# elif name == '331_00':
		# rfi_index = []
		# bad_file = 0
		
	elif name == '2015_332_00':
		rfi_index = list(range(0, 800))
		bad_file = 0
		
	elif name == '2015_333_00':
		rfi_index = list(range(0, 750)) + [1800, 1801, 1802, 1803, 1804, 2255, 2256, 2257, 2258, 2259]
		bad_file = 0
		
	elif name == '2015_334_00':
		rfi_index = [706, 707, 708, 709, 710, 868, 869, 870, 871, 872, 873, 1788, 1789, 1790, 1791, 1792, 1793, 2199, 2201, 2266, 2268]
		bad_file = 0
		
	# elif name == '335_00':
		# rfi_index = []
		# bad_file = 0
	
	elif name == '2015_336_00':
		rfi_index = list(range(0, 1220)) + [1323, 1627, 1681, 1758, 1759]
		bad_file = 0
		
	elif name == '2015_337_00':
		rfi_index = list(range(650, 1090)) + [2078, 2080, 2101] 
		bad_file = 0
		
	elif name == '2015_338_00':
		rfi_index = [336, 337, 651, 653, 654, 655, 656, 814, 815, 816, 817, 818] + list(range(1889, 1898))
		bad_file = 0
		
	elif name == '2015_339_00':
		rfi_index = [655] + list(range(1874, 1886))
		bad_file = 0
		
	elif name == '2015_340_00':
		rfi_index = [666, 667, 668, 669, 781, 782, 783, 784, 785, 786, 1864, 1865, 1866, 1867, 1868, 1869]
		bad_file = 0		
		
	elif name == '2015_341_00':
		rfi_index = [129, 174, 390, 570, 609, 610, 611, 713, 1537, 2247, 2265, 2266] + list(range(766, 780)) + list(range(1847, 1858))
		bad_file = 0
		
	# elif name == '342_00':
		# rfi_index = []
		# bad_file = 0
		
	elif name == '2015_343_00':
		rfi_index = list(range(0, 850))
		bad_file = 0
			
	elif name == '2015_344_00':
		rfi_index = [94] + list(range(351, 360)) + list(range(724, 732)) + list(range(1803, 1811))
		bad_file = 0
		
	elif name == '2015_345_00':
		rfi_index = [620, 2231, 2232, 2280] + list(range(712, 720)) + list(range(1793, 1800))
		bad_file = 0
		
	elif name == '2015_346_00':
		rfi_index = list(range(0, 800)) + [2272, 2275]
		bad_file = 0
		
	elif name == '2015_347_00':
		rfi_index = list(range(0, 980)) + list(range(1750, 2284))
		bad_file = 0		
		
	# elif name == '348_00':
		# rfi_index = []
		# bad_file = 0
		
	# elif name == '349_00':
		# rfi_index = []
		# bad_file = 0
		
	# elif name == '350_00':
		# rfi_index = []
		# bad_file = 0
		
	# elif name == '351_00':
		# rfi_index = []
		# bad_file = 0

	elif name == '2015_352_00':
		rfi_index = list(range(0, 900)) + [1552, 1553, 2145, 2146, 2147, 2148, 2149, 2150, 2151, 2229, 2230, 2231, 2232, 2233, 2234, 2235, 2236, 2237]
		bad_file = 0
		
	elif name == '2015_353_00':
		rfi_index = list(range(0, 1000)) + list(range(2150, 2283))
		bad_file = 0

	elif name == '2015_354_00':
		rfi_index = list(range(0, 930))
		bad_file = 0
		
	elif name == '2015_355_00':
		rfi_index = list(range(0, 930)) + [1844, 1845, 1846, 1847, 1848, 2165, 2166, 2232, 2233, 2234] + list(range(2276, 2284))
		bad_file = 0
		
	elif name == '2015_356_00':
		rfi_index = list(range(0, 1010)) + [1250, 1251, 1252, 1253, 1265, 1829] + list(range(1796, 1805))
		bad_file = 0
	
	elif name == '2015_357_00':
		rfi_index = list(range(0, 950)) + [1783, 1784, 1785, 1786] + list(range(2225, 2284))
		bad_file = 0
		
	elif name == '2015_358_00':
		rfi_index = list(range(0, 1050)) + [1200, 1768, 1769, 1783, 1784, 1785, 1786, 1871, 1872, 1930] + list(range(2290, 2283))
		bad_file = 0
		
	elif name == '2015_359_00':
		rfi_index = list(range(0, 1050)) + [1143, 1144, 1145] + list(range(2140, 2284))
		bad_file = 0
		
	elif name == '2015_360_00':
		rfi_index = list(range(0, 980)) + list(range(2190, 2283))
		bad_file = 0	
	
	elif name == '2015_361_00':
		rfi_index = list(range(0, 970)) + [1725] + list(range(2230, 2273))
		bad_file = 0
		
	elif name == '2015_362_00':
		rfi_index = list(range(0, 1060)) + [1872, 1873, 1874, 1875, 1876, 1877] + list(range(2050, 2283))
		bad_file = 0
		
	elif name == '2015_363_00':
		rfi_index = list(range(0, 1150)) + [1413, 1418, 1433, 1434, 1451, 1480, 1548, 1608, 1609, 1612, 1629, 1645, 1679, 1680, 1726, 1727] + list(range(1857, 1866)) + list(range(2230, 2283))
		bad_file = 0
		
	elif name == '2015_364_00':
		rfi_index = list(range(0, 860)) + [2217]
		bad_file = 0
		
	elif name == '2015_365_00':
		rfi_index = list(range(0, 850)) + [914, 969, 1247, 1248, 1981, 1982, 1983, 1984, 2165, 2166, 2167, 2225]
		bad_file = 0
		
		
		
		
		
	elif name == '2016_001_00':
		rfi_index = list(range(0, 950)) + list(range(2200, 2286)) + [1029, 1030, 1031, 1817, 1818, 1819, 1820, 1821, 1822, 1823, 1824, 2151, 2152, 2153, 2154, 2155]
		bad_file = 0	
		
	elif name == '2016_002_00':
		rfi_index = list(range(0, 650)) + [688, 724, 725, 726, 727, 728, 729, 982, 1800, 1801, 1802, 1803, 1804, 2184, 2185, 2186]
		bad_file = 0
		
	elif name == '2016_003_00':
		rfi_index = list(range(0, 850))
		bad_file = 0
		
	elif name == '2016_004_00':
		rfi_index = [611, 612, 613, 614, 615, 616, 697, 698, 699, 700, 701]
		bad_file = 0
	
	elif name == '2016_005_00':
		rfi_index = list(range(0, 1000)) + [1226, 2183, 2184] + list(range(1307, 1327)) + list(range(2109, 2129)) + list(range(2260, 2271))
		bad_file = 0	
	
	elif name == '2016_006_00':
		rfi_index = list(range(0, 850)) + list(range(1600, 2286)) + [933]
		bad_file = 0	
	
	elif name == '2016_007_00':
		rfi_index = list(range(0, 1050)) + [1194, 1242, 1252, 1269, 1321, 1334, 1335, 1336, 1357, 1663, 2152, 2153, 2154, 2155, 2156, 2157, 2168] + list(range(1485, 1523)) + list(range(2250, 2285))
		bad_file = 0	
	
	elif name == '2016_008_00':
		rfi_index = list(range(0, 900))
		bad_file = 0		
	
	elif name == '2016_009_00':
		rfi_index = list(range(0, 900)) + [1169, 1170, 1171, 1172, 1401, 1402, 1412, 1436, 1437, 1438, 1439, 1440, 1510, 1519, 1705, 1706]
		bad_file = 0
		
	elif name == '2016_010_00':
		rfi_index = [136, 145, 201, 202, 238, 328, 329, 330, 331, 386, 1315]
		bad_file = 0
	
	elif name == '2016_011_00':
		rfi_index = [188, 586, 705, 2253]
		bad_file = 0

	elif name == '2016_012_00':
		rfi_index = list(range(0, 850)) + [889, 1712, 1826, 1827, 1828, 1829, 1830, 1831]
		bad_file = 0	
	
	elif name == '2016_013_00':
		rfi_index = list(range(0, 200)) + [529, 729, 730]
		bad_file = 0
		
	elif name == '2016_014_00':
		rfi_index = list(range(0, 1280)) + [1798, 1799, 1800, 1801, 1802]
		bad_file = 0		
		
	elif name == '2016_015_00':
		rfi_index = [266, 267, 268, 269, 270, 271, 272, 1783, 1784, 1785, 1786, 1787] + list(range(697, 710))
		bad_file = 0
		
	elif name == '2016_016_00':
		rfi_index = list(range(0, 600)) + [684, 848, 849, 850, 851, 852, 853, 854, 865] + list(range(1086, 1117))
		bad_file = 0
	
	elif name == '2016_017_00':
		rfi_index = list(range(650, 1300)) + list(range(2195, 2205)) + [2282]
		bad_file = 0
		
	elif name == '2016_018_00':
		rfi_index = list(range(0, 790)) + [815, 816, 817, 818, 819, 820, 924, 2203] 
		bad_file = 0
		
	elif name == '2016_019_00':
		rfi_index = list(range(348, 356)) + [636]
		bad_file = 0
		
	# elif name == '2016_020_00':
		# rfi_index = [209]
		# bad_file = 0	
	
	elif name == '2016_028_00':
		rfi_index = list(range(0, 1100)) + [2280]
		bad_file = 0
	
	elif name == '2016_029_00':
		rfi_index = list(range(0, 900)) + list(range(2240, 2289))
		bad_file = 0	
	
	elif name == '2016_030_00':
		rfi_index = list(range(0, 870)) + list(range(1220, 1400)) + list(range(2250, 2288))
		bad_file = 0
	
	elif name == '2016_031_00':
		rfi_index = list(range(0, 1025))
		bad_file = 0
		
	elif name == '2016_032_00':
		rfi_index = list(range(0, 950)) + [986, 987, 1003, 1021, 1066, 1099, 1100, 1115, 1481, 2268, 2269, 2270, 2271, 2272, 2273]
		bad_file = 0
		
	elif name == '2016_033_00':
		rfi_index = list(range(0, 890))
		bad_file = 0
	
	elif name == '2016_034_00':
		rfi_index = list(range(0, 890)) + [2247, 2258]
		bad_file = 0
		
	elif name == '2016_035_00':
		rfi_index = list(range(0, 900)) + [926, 961, 986, 1035, 1542, 1571, 1661] + list(range(2230, 2285))
		bad_file = 0
		
	elif name == '2016_036_00':
		rfi_index = list(range(0, 1000)) + list(range(2100, 2283))
		bad_file = 0
		
	elif name == '2016_037_00':
		rfi_index = list(range(0, 900)) + [949, 2220]
		bad_file = 0
	
	elif name == '2016_038_00':
		rfi_index = list(range(0, 960)) + list(range(2150, 2279))
		bad_file = 0
	
	elif name == '2016_039_00':
		rfi_index = list(range(0, 1050)) + list(range(2150, 2280))
		bad_file = 0
		
	elif name == '2016_040_00':
		rfi_index = list(range(0, 1000)) + list(range(2189, 2197)) + list(range(2272, 2291)) + [2219, 2220] 
		bad_file = 0
		
	elif name == '2016_041_00':
		rfi_index = list(range(0, 950)) + [1004, 1005, 1006, 1007, 1076, 1077, 2220, 2221, 2222, 2223, 2257]
		bad_file = 0
		
	elif name == '2016_042_00':
		rfi_index = list(range(0, 850)) + list(range(2150, 2279))
		bad_file = 0
		
	elif name == '2016_043_00':
		rfi_index = list(range(0, 700)) + [795, 796, 797, 798, 1179, 1180, 1181]
		bad_file = 0	

	elif name == '2016_044_00':
		rfi_index = [621, 678, 777, 778, 779, 780, 781, 782, 783]
		bad_file = 0		
	
	elif name == '2016_045_00':
		rfi_index = list(range(0, 800)) + [2203]
		bad_file = 0		
	
	elif name == '2016_046_00':
		rfi_index = [751, 752, 753, 754, 755, 756, 808, 882, 1765]
		bad_file = 0
	
	elif name == '2016_047_00':
		rfi_index = [650, 651, 652, 653, 654, 731, 732, 733, 734, 735, 736, 737, 738, 784, 785, 786, 787, 1990, 2001, 2075, 2076, 2077]
		bad_file = 0	
	
	elif name == '2016_048_00':
		rfi_index = list(range(0, 700)) + list(range(1500, 2283)) + [721, 722, 723, 724, 725, 726, 727, 897]
		bad_file = 0
		
	elif name == '2016_049_00':
		rfi_index = [170, 866, 867, 868, 869]
		bad_file = 0	

	elif name == '2016_050_00':
		rfi_index = [692, 693, 694, 695, 696, 697, 698, 819, 820, 821, 853, 854, 855, 856]
		bad_file = 0

	elif name == '2016_051_00':
		rfi_index = [678, 679, 680, 681, 682]
		bad_file = 0

	elif name == '2016_052_00':
		rfi_index = [828, 829, 830, 831, 832, 833] + list(range(1680, 1870))
		bad_file = 0

	elif name == '2016_053_00':
		rfi_index = [651, 652, 653, 654 ,655, 656, 797, 813, 814, 815, 816, 817, 828, 950, 985, 986, 987, 995, 1005]
		bad_file = 0
		
	# elif name == '2016_055_07':
		# rfi_index = []
		# bad_file = 0
	
	elif name == '2016_056_00':
		rfi_index = [773, 774, 775, 776, 777, 778, 794, 1746]
		bad_file = 0
	
	elif name == '2016_057_00':
		rfi_index = list(range(0, 1050))
		bad_file  = 0

	elif name == '2016_058_00':
		rfi_index = [578, 579, 580, 581, 582, 583, 584, 585, 586, 1106, 1138, 1152, 1179, 1567, 1568, 1569, 1570]
		bad_file  = 0

	elif name == '2016_059_00':
		rfi_index = [20, 130, 131, 132, 339, 1805, 1819, 1844, 1870, 1880, 1888, 1938, 2193, 2249, 2269] + list(range(700, 1400)) 
		bad_file  = 0

	elif name == '2016_060_00':
		rfi_index = list(range(0, 850))
		bad_file  = 0

	elif name == '2016_061_00':
		rfi_index = list(range(0, 1000)) + [1115, 2198]
		bad_file  = 0

	elif name == '2016_062_00':
		rfi_index = list(range(0, 1350))
		bad_file  = 0
		
	elif name == '2016_063_00':
		rfi_index = [836, 837] + list(range(1280, 2283))
		bad_file  = 0
		
	elif name == '2016_064_00':
		rfi_index = list(range(0, 900))
		bad_file  = 0

	elif name == '2016_065_00':
		rfi_index = list(range(0, 850))
		bad_file  = 0
		
	elif name == '2016_066_00':
		rfi_index = list(range(150, 950)) + list(range(2200, 2283))
		bad_file  = 0	
		
	elif name == '2016_067_00':
		rfi_index = list(range(0, 800))
		bad_file  = 0

	elif name == '2016_068_00':
		rfi_index = list(range(0, 800))
		bad_file  = 0
		
	elif name == '2016_069_00':
		rfi_index = list(range(746, 756))
		bad_file  = 0
		
	elif name == '2016_070_00':
		rfi_index = [726] + list(range(735, 744))
		bad_file  = 0
		
	elif name == '2016_071_00':
		rfi_index = list(range(717, 739))
		bad_file  = 0
		
	elif name == '2016_072_00':
		rfi_index = list(range(701, 712))
		bad_file  = 0		
		
	elif name == '2016_073_00':
		rfi_index = [215, 444, 445, 956, 986, 1027, 1039, 1090, 1921]
		bad_file  = 0		
				
	elif name == '2016_074_00':
		rfi_index = [253, 677, 678, 679, 680, 681, 682, 797, 800, 836, 837, 838, 839] + list(range(2150, 2284))
		bad_file  = 0		

	elif name == '2016_075_00':
		rfi_index = list(range(0, 1320)) + list(range(2070, 2283)) + [1422, 1434]
		bad_file  = 0
		
	elif name == '2016_076_00':
		rfi_index = list(range(0, 1300)) + list(range(2200, 2284))
		bad_file  = 0		
		
	elif name == '2016_077_00':
		rfi_index = list(range(0, 850)) + list(range(2050, 2286))
		bad_file  = 0

	elif name == '2016_078_00':
		rfi_index = list(range(0, 930)) + [2246]
		bad_file  = 0
		
	elif name == '2016_079_00':
		rfi_index = list(range(0, 950)) + [2273]
		bad_file  = 0
		
	elif name == '2016_080_00':
		rfi_index = list(range(0, 950)) + [2251, 2280]
		bad_file  = 0

	elif name == '2016_081_00':
		rfi_index = list(range(0, 850)) + [896, 900, 901, 902, 1241]
		bad_file  = 0
	
	elif name == '2016_082_00':
		rfi_index = list(range(0, 970))
		bad_file  = 0
		
	elif name == '2016_083_00':
		rfi_index = list(range(0, 970)) + list(range(1400, 1480)) + [1681]
		bad_file  = 0

	elif name == '2016_084_00':
		rfi_index = list(range(0, 875)) + list(range(2200, 2285)) + [1157]
		bad_file  = 0
		
	elif name == '2016_085_00':
		rfi_index = list(range(0, 1040)) + list(range(1751, 1765)) + [2260]
		bad_file  = 0

	elif name == '2016_086_00':
		rfi_index = list(range(0, 1250)) + [1419, 1447, 2050]
		bad_file  = 0

	elif name == '2016_087_00':
		rfi_index = list(range(0, 870))
		bad_file  = 0
		
	elif name == '2016_088_00':
		rfi_index = list(range(0, 870))
		bad_file  = 0
		
	elif name == '2016_089_00':
		rfi_index = list(range(0, 820))
		bad_file  = 0

	elif name == '2016_090_00':
		rfi_index = list(range(0, 800)) + [2262, 2270, 2276]
		bad_file  = 0

	elif name == '2016_091_00':
		rfi_index = list(range(0, 870)) + list(range(1470, 2279))
		bad_file  = 0
		
	elif name == '2016_092_00':
		rfi_index = list(range(0, 850)) + list(range(1470, 2279))
		bad_file  = 0

	elif name == '2016_093_00':
		rfi_index = [733]
		bad_file  = 0

	elif name == '2016_094_00':
		rfi_index = [721, 723]
		bad_file  = 0

	elif name == '2016_095_00':
		rfi_index = [703, 705, 821, 849, 851, 853]
		bad_file  = 0
		
	elif name == '2016_096_00':
		rfi_index = [192, 465, 686, 691, 847, 848, 849, 860]
		bad_file  = 0

	elif name == '2016_097_00':
		rfi_index = list(range(420, 850))
		bad_file  = 0
		
	elif name == '2016_098_00':
		rfi_index = list(range(0, 850)) + [906, 1436, 1909, 1940]
		bad_file  = 0
		
	# elif name == '2016_099_00':
		# rfi_index = []
		# bad_file = 0		
		
	# elif name == '2016_100_00':
		# rfi_index = []
		# bad_file = 0

	elif name == '2016_101_00':
		rfi_index = list(range(0, 1000)) + list(range(2000, 2283))
		bad_file  = 0

	elif name == '2016_102_00':
		rfi_index = list(range(0, 1100)) + list(range(2150, 2283))
		bad_file  = 0
		
	elif name == '2016_103_00':
		rfi_index = list(range(0, 960)) + list(range(2160, 2285))
		bad_file  = 0
		

	# elif name == '2016_104_00':
		# rfi_index = []
		# bad_file = 0		
	
	# elif name == '2016_107_00':
		# rfi_index = []
		# bad_file = 0		
		
	elif name == '2016_108_00':
		rfi_index = list(range(0, 900))
		bad_file = 0			

	elif name == '2016_109_00':
		rfi_index = list(range(0, 750)) + [763, 814, 815, 816, 817, 1666, 1667, 1668, 1669]
		bad_file = 0
		
	elif name == '2016_110_00':
		rfi_index = list(range(0, 900))
		bad_file = 0
		
	elif name == '2016_111_00':
		rfi_index = list(range(0, 800))
		bad_file = 0
		
	elif name == '2016_112_00':
		rfi_index = list(range(0, 800)) + [1647, 1648, 2037]
		bad_file = 0
		
	elif name == '2016_113_00':
		rfi_index = [206, 214, 229, 347, 409, 756, 759, 760, 1791, 1792, 1793]
		bad_file = 0
		
	elif name == '2016_114_00':
		rfi_index = [224, 746]
		bad_file = 0
		
	elif name == '2016_115_00':
		rfi_index = list(range(600, 750))
		bad_file = 0
		
	elif name == '2016_116_00':
		rfi_index = [717, 721, 789, 790]
		bad_file = 0
		
	elif name == '2016_117_00':
		rfi_index = [400, 471, 702, 705]
		bad_file = 0
				
	elif name == '2016_118_00':
		rfi_index = []
		bad_file = 0		
		
	elif name == '2016_123_00':
		rfi_index = list(range(0, 860))
		bad_file = 0		
		
	elif name == '2016_124_00':
		rfi_index = list(range(0, 500)) + list(range(2000, 2283)) + [775, 776, 777, 7778, 779, 780]
		bad_file = 0
		
	elif name == '2016_125_00':
		rfi_index = list(range(0, 530)) + list(range(1650, 1935)) + [761, 762, 763, 764]
		bad_file = 0		
		
	# elif name == '2016_126_00':
		# rfi_index = []
		# bad_file = 0
		
	elif name == '2016_127_00':
		rfi_index = [167, 735, 736, 737, 738, 739, 740, 741] + list(range(400, 500))
		bad_file = 0		
		
	elif name == '2016_128_00':
		rfi_index = [272, 1681] + list(range(704, 730)) + list(range(2260, 2296))
		bad_file = 0
		
	elif name == '2016_129_00':
		rfi_index = list(range(0, 750)) + [872]
		bad_file = 0
		
	elif name == '2016_130_00':
		rfi_index = list(range(0, 800)) + list(range(2220, 2295))
		bad_file = 0
				
	elif name == '2016_131_00':
		rfi_index = list(range(0, 800)) + [1260]
		bad_file = 0
		
	elif name == '2016_132_00':
		rfi_index = list(range(0, 700)) + [825, 826, 827, 828, 829]
		bad_file = 0
		
	elif name == '2016_133_00':
		rfi_index = list(range(350, 750))
		bad_file = 0
		
	elif name == '2016_134_00':
		rfi_index = [800, 801]
		bad_file = 0
		
	elif name == '2016_135_00':
		rfi_index = [373, 422, 556, 784, 785, 786, 787]
		bad_file = 0
		
	elif name == '2016_136_00':
		rfi_index = list(range(0, 950))
		bad_file = 0
		
	elif name == '2016_137_00':
		rfi_index = list(range(0, 800)) + [1309, 1310]
		bad_file = 0
		
	elif name == '2016_138_00':
		rfi_index = list(range(735, 747)) + list(range(1717, 1730))
		bad_file = 0		
		
	elif name == '2016_139_00':
		rfi_index = list(range(0, 800))
		bad_file = 0
		
	elif name == '2016_140_00':
		rfi_index = list(range(0, 1050))
		bad_file = 0

	elif name == '2016_141_00':
		rfi_index = [348, 611, 670, 698] + list(range(2250, 2294))
		bad_file = 0

	elif name == '2016_142_00':
		rfi_index = list(range(0, 1000)) + list(range(2200, 2294))
		bad_file = 0
		
	elif name == '2016_143_00':
		rfi_index = list(range(0, 750)) + [831, 832, 833, 834, 835]
		bad_file = 0
		
	elif name == '2016_144_00':
		rfi_index = list(range(0, 860)) + list(range(1700, 2070))
		bad_file = 0
		
	elif name == '2016_145_00':
		rfi_index = list(range(0, 950))
		bad_file = 0
		
	elif name == '2016_146_00':
		rfi_index = list(range(0, 850)) + list(range(2250, 2295))
		bad_file = 0
		
	elif name == '2016_147_00':
		rfi_index = list(range(0, 950)) + list(range(2200, 2291)) + [1578, 1704]
		bad_file = 0
		
	elif name == '2016_148_00':
		rfi_index = list(range(0, 850)) + [2182, 2278, 2279]
		bad_file = 0
		
	elif name == '2016_149_00':
		rfi_index = list(range(0, 550)) + [745, 746, 747, 748, 749, 750, 791, 792, 793]
		bad_file = 0
		
	elif name == '2016_150_00':
		rfi_index = list(range(210, 330)) + list(range(600, 800))
		bad_file = 0
		
	elif name == '2016_151_00':
		rfi_index = list(range(700, 900))
		bad_file = 0
		
	elif name == '2016_152_00':
		rfi_index = [699, 704, 705, 706] + list(range(850, 870))
		bad_file = 0
		
	elif name == '2016_153_00':
		rfi_index = [472, 685, 847, 851, 852, 853, 854] + list(range(850, 870))
		bad_file = 0
		
	elif name == '2016_154_00':
		rfi_index = list(range(320, 520))
		bad_file = 0
		
	# elif name == '2016_155_00':
		# rfi_index = []
		# bad_file = 0
		
	elif name == '2016_156_00':
		rfi_index = list(range(0, 1100))
		bad_file = 0
		
	elif name == '2016_157_00':
		rfi_index = [790]
		bad_file = 0
		
	elif name == '2016_158_00':
		rfi_index = [778, 779, 780, 781, 782] + list(range(814, 824))
		bad_file = 0
		
	elif name == '2016_159_00':
		rfi_index = [609, 610] + list(range(761, 774))
		bad_file = 0
		
	elif name == '2016_160_00':
		rfi_index = list(range(747, 761))
		bad_file = 0		
		
	elif name == '2016_161_00':
		rfi_index = [740, 741, 742, 743, 744, 897, 898, 899, 900, 901, 902]
		bad_file = 0
				
	elif name == '2016_162_00':
		rfi_index = [343, 709, 723, 724, 725]
		bad_file = 0		
		
	elif name == '2016_163_00':
		rfi_index = [706, 1390]
		bad_file = 0			
		
	elif name == '2016_164_00':
		rfi_index = list(range(0, 800))
		bad_file = 0
		
	elif name == '2016_165_00':
		rfi_index = list(range(0, 900)) + list(range(2220, 2293))
		bad_file = 0
		
	elif name == '2016_166_00':
		rfi_index = list(range(0, 860)) + [2285]
		bad_file = 0
		
	elif name == '2016_167_00':
		rfi_index = list(range(0, 820))
		bad_file = 0
		
	elif name == '2016_168_00':
		rfi_index = list(range(120, 820)) + [2029]
		bad_file = 0
		
	elif name == '2016_169_00':
		rfi_index = list(range(110, 190)) + [674, 675, 784, 785, 786, 787, 1927, 1928, 1929, 1930]
		bad_file = 0		
		
	elif name == '2016_170_00':
		rfi_index = [773]
		bad_file = 0
		
	elif name == '2016_171_00':
		rfi_index = [757, 758, 759, 2063]
		bad_file = 0
		
	elif name == '2016_172_00':
		rfi_index = [160, 205, 206, 207, 208, 209, 210, 495] + list(range(739, 749))
		bad_file = 0
		
	elif name == '2016_173_00':
		rfi_index = list(range(0, 950))
		bad_file = 0		
	
	
	# Switch problem starts ??	
	# -------------------------------------
	
	# elif name == '2016_174_00':
		# switch_problem = 'yes'
		# bad_file = 0
		
	# elif name == '2016_175_00':
		# switch_problem = 'yes'
		# bad_file = 0	
		
	# elif name == '2016_176_00':
		# switch_problem = 'yes'
		# bad_file = 0		
		
	# elif name == '2016_177_00':
		# switch_problem = 'yes'
		# bad_file = 0
		
	# elif name == '2016_178_00':
		# switch_problem = 'yes'
		# bad_file = 0
		
	# elif name == '2016_179_00':
		# switch_problem = 'yes'
		# bad_file = 0
		
	# elif name == '2016_180_00':
		# switch_problem = 'yes'
		# bad_file = 0		
		
	# -------------------------------------
	# Switch problem ends ??		
	
	
	
	elif name == '2016_181_00':
		rfi_index = [775, 776, 777, 778, 779, 780] + list(range(739, 749))
		bad_file = 0		
	
	elif name == '2016_182_00':
		rfi_index = list(range(758, 800)) + [1046, 1557, 1653]
		bad_file = 0
		
	elif name == '2016_183_00':
		rfi_index = list(range(1900, 2292)) + list(range(745, 754)) + [1467]
		bad_file = 0
		
	elif name == '2016_184_00':
		rfi_index = list(range(0, 600)) + list(range(732, 739)) + [894]
		bad_file = 0
		
	elif name == '2016_185_00':
		rfi_index = [717]
		bad_file = 0
	
	elif name == '2016_186_00':
		rfi_index = [705, 711, 712, 937]
		bad_file = 0
		
	elif name == '2016_187_00':
		rfi_index = [] # pretty good
		bad_file = 0
		
	elif name == '2016_188_00':
		rfi_index = [679, 837] # pretty good
		bad_file = 0
		
	elif name == '2016_189_00':
		rfi_index = [641, 822, 823, 824, 825, 826, 827, 828] # pretty good
		bad_file = 0		
		
	elif name == '2016_190_00':
		rfi_index = list(range(1450, 2295)) + [591, 717, 793, 797, 798, 808, 809, 810, 811, 812, 813]
		bad_file = 0
		
	elif name == '2016_191_00':
		rfi_index = [793, 800, 801, 2222]
		bad_file = 0
		
	elif name == '2016_192_00':
		rfi_index = list(range(110, 126)) + list(range(778, 788))
		bad_file = 0		
		
	elif name == '2016_193_00':
		rfi_index = [198, 419, 763, 764, 765, 766, 767, 768, 1060, 2236]
		bad_file = 0
		
	elif name == '2016_194_00':
		rfi_index = list(range(170, 950)) + [1646, 1647, 1648, 1954] + list(range(2270, 2295))
		bad_file = 0
		
	elif name == '2016_195_00':
		rfi_index = list(range(0, 950)) + [1997]
		bad_file = 0
		
	elif name == '2016_196_00':
		rfi_index = list(range(0, 970))
		bad_file = 0
		
	elif name == '2016_197_00':
		rfi_index = list(range(0, 200)) + [384, 385, 714]
		bad_file = 0
		
	elif name == '2016_198_00':
		rfi_index = list(range(0, 900)) + list(range(2260, 2290))
		bad_file = 0
		
	elif name == '2016_199_00':
		rfi_index = list(range(0, 950))
		bad_file = 0
		
	elif name == '2016_200_00':
		rfi_index = list(range(0, 950))
		bad_file = 0
		
	elif name == '2016_201_00':
		rfi_index = list(range(0, 900)) + [2175]
		bad_file = 0
		
	elif name == '2016_202_00':
		rfi_index = list(range(0, 950)) + list(range(2222, 2292))
		bad_file = 0
		
	elif name == '2016_203_00':
		rfi_index = list(range(0, 950))
		bad_file = 0
		
	elif name == '2016_204_00':
		rfi_index = list(range(0, 950)) + list(range(2280, 2294))
		bad_file = 0
		
	elif name == '2016_205_00':
		rfi_index = list(range(0, 950)) + [1139]
		bad_file = 0
		
	elif name == '2016_206_00':
		rfi_index = list(range(0, 750))
		bad_file = 0
		
	elif name == '2016_207_00':
		rfi_index = list(range(0, 760)) + [887, 892, 1705]
		bad_file = 0
		
	elif name == '2016_208_00':
		rfi_index = [715, 873, 874, 877, 916]
		bad_file = 0
		
	elif name == '2016_209_00':
		rfi_index = [705, 716]
		bad_file = 0
		
	elif name == '2016_210_00':
		rfi_index = [222]
		bad_file = 0
		
	elif name == '2016_211_00':
		rfi_index = [871, 1063, 1798]
		bad_file = 0

	elif name == '2016_212_00':
		rfi_index = [238, 270, 668, 669, 829]
		bad_file = 0
		
	elif name == '2016_213_00':
		rfi_index = []
		bad_file = 0
		
	elif name == '2016_216_16':
		rfi_index = [414]
		bad_file = 0
		
	elif name == '2016_217_00':
		rfi_index = list(range(752, 764)) + list(range(1370, 1480)) + list(range(1640, 1776)) + list(range(2240, 2298))
		bad_file = 0
		
	elif name == '2016_218_00':
		rfi_index = [739, 742, 743, 744, 745, 746]
		bad_file = 0

	elif name == '2016_219_00':
		rfi_index = []
		bad_file = 0
		
	elif name == '2016_220_00':
		rfi_index = list(range(130, 750)) + [877]
		bad_file = 0
		
	# elif name == '2016_221_00':
		# rfi_index = []
		# bad_file = 0

	elif name == '2016_223_18':
		rfi_index = [303]
		bad_file = 0
		
	elif name == '2016_224_00':
		rfi_index = list(range(0, 850)) + [1578, 1579, 1580]
		bad_file = 0
		
	elif name == '2016_225_00':
		rfi_index = list(range(0, 850))
		bad_file = 0
		
	elif name == '2016_226_00':
		rfi_index = list(range(0, 810))
		bad_file = 0
		
	elif name == '2016_227_00':
		rfi_index = list(range(767, 777))
		bad_file = 0
		
	elif name == '2016_228_00':
		rfi_index = list(range(754, 761))
		bad_file = 0
		
	elif name == '2016_229_00':
		rfi_index = list(range(737, 745)) + [897]
		bad_file = 0
		
	elif name == '2016_230_00':
		rfi_index = [724, 725, 726, 727]
		bad_file = 0		
		
	elif name == '2016_236_09':
		rfi_index = []  # pretty good
		bad_file = 0
		
	elif name == '2016_237_00':
		rfi_index = [790, 791, 792, 793, 794, 795, 796, 797, 1864, 1865, 1866]  # pretty good
		bad_file = 0
	
	elif name == '2016_239_00':
		rfi_index = [453, 455, 554, 762, 763, 764, 765, 766, 767, 768, 769, 770] + list(range(2050, 2150))
		bad_file = 0	
	
	elif name == '2016_240_00':
		rfi_index = list(range(300, 600)) + list(range(745, 775)) + [797, 843, 910, 911, 912, 913, 914, 915, 916, 917, 918, 1668] + list(range(2278, 2290))
		bad_file = 0
		
	elif name == '2016_241_00':
		rfi_index = list(range(0, 930))
		bad_file = 0
		
	elif name == '2016_242_00':
		rfi_index = list(range(0, 900))
		bad_file = 0
		
	elif name == '2016_243_00':
		rfi_index = list(range(400, 575))
		bad_file = 0
		
	# elif name == '2016_244_07':
		# rfi_index = []
		# bad_file = 0
	
	elif name == '2016_245_00':
		rfi_index = list(range(0, 950)) + list(range(2220, 2282))
		bad_file = 0
		
	elif name == '2016_246_00':
		rfi_index = list(range(0, 1000))
		bad_file = 0
		
	elif name == '2016_247_00':
		rfi_index = list(range(0, 850))
		bad_file = 0
		
	elif name == '2016_248_00':
		rfi_index = list(range(350, 450))
		bad_file = 0		
	
	elif name == '2016_248_06':
		rfi_index = list(range(0, 450)) + [1088]
		bad_file = 0
		
	# elif name == '2016_249_00':
		# rfi_index = []
		# bad_file = 0

	# elif name == '2016_250_02':
		# rfi_index = []
		# bad_file = 0
		
	elif name == '2016_251_00':
		rfi_index = list(range(0, 300)) + [368, 370, 473, 477, 612]
		bad_file = 0
		
	elif name == '2016_252_00':
		rfi_index = [139, 140, 206, 714, 715, 716, 717, 718, 897, 902] + list(range(187, 210))
		bad_file = 0
	
	elif name == '2016_253_13':
		rfi_index = [547, 548]
		bad_file = 0
			
	# elif name == '2016_254_00':
		# rfi_index = []
		# bad_file = 0
		
	elif name == '2016_254_09':
		rfi_index = []   # pretty good !
		bad_file = 0
		
	elif name == '2016_255_00':
		rfi_index = [176, 640, 670, 671, 825, 866, 2269]
		bad_file = 0
		
	elif name == '2016_256_00':
		rfi_index = list(range(0, 840))
		bad_file = 0
		
	elif name == '2016_257_00':
		rfi_index = [59, 69, 98, 101, 186, 279, 335, 374, 602, 629, 643, 901, 902, 903, 904, 905, 906, 907, 1433, 1460, 1494] + list(range(801, 810))
		bad_file = 0
		
	elif name == '2016_258_00':
		rfi_index = [202]
		bad_file = 0






	
	return rfi_index, bad_file

























def level2_to_level3(name, save, folder, LST1=0, LST2=6, SUN_EL_MIN=-95, SUN_EL_MAX=95, MOON_EL_MIN=-95, MOON_EL_MAX=95, LOSS=1, BEAM=0, FMIN=65, FMAX=195, antenna_type='fourpoint', model_type='EDGES_polynomial', Nfg=5):
	
	
	"""
	
	April 5, 2018
	
	
	"""
	
	
	
	
	
	


	# window_width_MHz_1 = 4
	# window_width_MHz_2 = 10
	# window_width_MHz_3 = 10
	
	# Npolyterms_block   = 4
	# N_choice           = 20
	# N_sigma            = 2.5











	# # Save folders
	# # Saving average corrected measurement
	if save == 'yes':
		save_folder_spectra = home_folder + '/DATA/EDGES/spectra/level3/high_band_2015/2018_analysis/' + folder + '/'
			
		# Creating folders if necessary
		if not exists(save_folder_spectra):
			makedirs(save_folder_spectra)
		




	# Identify problematic spectra in the file
	rfi_index, bad_file = daily_data_filter(name)
	
	
	# Process data
	if bad_file == 0:
		f, tt, mm, ww = eg.level2read_v2(home_folder + '/DATA/EDGES/spectra/level2/high_band/v2_65_195_MHz/' + name + '.hdf5')  # 2015_252_00
		length = len(tt[:,0])
		index = list(range(length))
		
		# Remove manually raw channels pre-identified with RFI
		tt, ww     = eg.RFI_excision_raw_frequency(f, tt, ww)
		
		# Remove manually raw spectra pre-identified as problematic
		index_clean = list(set(index) - set(rfi_index))
		t = tt[index_clean,:]
		m = mm[index_clean,:]
		w = ww[index_clean,:]
			

		
		
		# Select data based on LST, etc.
		index_temp      = np.arange(len(t[:,0]))
		
		if (LST1 < LST2):
			index_selection = index_temp[(m[:,3] >= LST1) & (m[:,3] <= LST2) & (m[:,6] >= SUN_EL_MIN) & (m[:,6] <= SUN_EL_MAX) & (m[:,8] >= MOON_EL_MIN) & (m[:,8] <= MOON_EL_MAX)]

		if (LST1 > LST2):
			index_selection = index_temp[    ((m[:,3] >= LST1) | (m[:,3] <= LST2))     & (m[:,6] >= SUN_EL_MIN) & (m[:,6] <= SUN_EL_MAX) & (m[:,8] >= MOON_EL_MIN) & (m[:,8] <= MOON_EL_MAX)]
							
		
		
		print(' ')
		print(' ')
		print(' ')
		print('----------- NUMBER of SPECTRA: ' + str(len(index_selection)) + ' ------------')
		print(' ')
		print(' ')
		print(' ')
		
		# Continue analysis if there are data		
		if len(index_selection) > 0:
			

			# Data selection
			ts = t[index_selection,:]
			ms = m[index_selection,:]
			ws = w[index_selection,:]
			


			# Calibration
			# ---------------------------------------------------------
			# 
			# Antenna S11
			if antenna_type == 'fourpoint':
				antenna_s11_day = 157
			
			elif antenna_type == 'blade':
				antenna_s11_day = 262
	
			s11_ant = eg.models_antenna_s11_remove_delay('high_band_2015', antenna_type, f, antenna_s11_day = antenna_s11_day, model_type='polynomial', Nfit=15, MC_mag='no', MC_ang='no', sigma_mag=0.0001, sigma_ang_deg=0.1)
	
	
			# Receiver calibration parameters
			rcv     = np.genfromtxt('/home/ramo7131/DATA/EDGES/calibration/receiver_calibration/high_band1/2015_03_25C/results/65_195_MHz/calibration_files/calibration_file_high_band_2015_65_195_MHz_ct14_wt14.txt')
			s11_LNA = rcv[:,1] + 1j*rcv[:,2]
			sca     = rcv[:,3]
			off     = rcv[:,4]
			TU      = rcv[:,5]
			TC      = rcv[:,6]
			TS      = rcv[:,7]
		
		
			# Calibrated antenna temperature, including loss and beam chromaticity
			ta = eg.calibrated_antenna_temperature(ts, s11_ant, s11_LNA, sca, off, TU, TC, TS)
	
			
			# Loss correction	
			if LOSS == 1:
				
				t_amb = 298.15
				
				flag_ground_loss    = 1
				ground_loss_type    = 'value'
				ground_loss_percent = 0.5
				
				flag_antenna_loss         = 0
				flag_balun_connector_loss = 1
				
				cg = eg.combined_gain('high_band_2015', f, antenna_type, antenna_s11_day = antenna_s11_day, antenna_s11_Nfit=15, flag_ground_loss=flag_ground_loss, ground_loss_type=ground_loss_type, ground_loss_percent=ground_loss_percent, flag_antenna_loss=flag_antenna_loss, flag_balun_connector_loss=flag_balun_connector_loss)
				
				cX = 1*cg
				
				ta = (ta - t_amb*(1-cX))/cX
	
		
		
		
			# Cut to frequency range
			ta  = ta[:, (f>=FMIN) & (f<=FMAX)]
			ws  = ws[:, (f>=FMIN) & (f<=FMAX)]
			f   = f[(f>=FMIN) & (f<=FMAX)]		
			
					
			# Beam chromaticity correction
			if BEAM == 1:
				print('------------- BEAM CHROMATICITY CORRECTION ----------------')
				cf = eg.antenna_beam_factor_interpolation('high_band_2015', ms[:,3], f, case_beam_factor=1, high_band_antenna_type=antenna_type)
				ta = ta/cf
				print('---------------------------DONE ----------------------------')
			
			elif BEAM == 0:
				print(' ')
				print('------------- NO BEAM CORRECTION !!!!!!!!!!!!!!!!!!! ---------------------------')
				print(' ')
					
			#
			# 
			# ---------------------------------------------------------------------------
				
			
			
			
			
			
			
			# RFI cleaning, computing residuals, spectral averaging
			# ----------------------------------------------------------------------------
			#
			
			N_spectra = len(ta[:,0])
			for i in range(N_spectra):
				
				print(' ')
				print(' ')
				print(' ')
				print('-----------------------------------------------------')
				print( str(i+1) + ' of ' + str(N_spectra) )
				
				
				
				# RFI cleaning spectrum
				#tb, wb = RFI_spectrum_cleaning(fb, tb, wb, window_width_MHz=window_width_MHz_1, Npolyterms_block=Npolyterms_block, N_choice=N_choice, N_sigma=N_sigma)

				w_no_RFI = eg.RFI_cleaning_spectrum(f, ta[i,:], ws[i,:], Nsamples=32, Nterms_fg=16, Nterms_std=3, Nsigma=4)
				
				
				
				
					
				# Fit a foreground model to binned spectrum
				N_bin_samples_intermediate = 16  # 97 kHz
				fb, tb, wb = eg.spectral_binning_number_of_samples(f, ta[i,:], w_no_RFI, nsamples=N_bin_samples_intermediate)
				
				if model_type == 'EDGES_polynomial':
					par          = np.polyfit((fb/200)[wb>0], (tb/((fb/200)**(-2.5)))[wb>0], Nfg-1)
					model_factor = np.polyval(par, (f/200))     #   <------ model is evaluated at RAW frequency, NOT at binned frequency
					model        = model_factor * ((f/200)**(-2.5))


				# Computing and storing residuals
				if i == 0:
					r_all   = np.zeros((N_spectra, len(f)))
					p_all   = np.zeros((N_spectra, Nfg))
					w_all   = np.zeros((N_spectra, len(f)))
					rms_all = np.zeros(N_spectra)
										
				rk                = ta[i,:] - model
				rk[ w_no_RFI==0 ] = 0				
				rms_k             = np.sqrt(np.sum((rk[ w_no_RFI>0 ])**2)/len(f[ w_no_RFI>0 ]))
				
				r_all[i,:] = rk
				p_all[i,:] = par
				w_all[i,:] = w_no_RFI
				rms_all[i] = rms_k
			
			
			
			
			
			m_all = np.copy(ms)
				
				
				

			
			# Averaging of N_av raw spectra for intermediate RFI cleaning
			#rnew, pnew, wnew, rms = spectral_averaging_RFI_cleaning(fb, r_all, p_all, w_all, N_av=16, model_type=model_type, window_width_MHz=window_width_MHz_2, Npolyterms_block=Npolyterms_block, N_choice=N_choice, N_sigma=N_sigma)





			# APPLY SOME SORT OF RMS FILTER HERE, to the intermediate averages!!!






			# Final spectrum (returned by the function, but not saved)
			# ----------------------------------------------------------------
			#
			# Total average after cleanning
			# av_par          = np.mean(pnew, axis=0)
			# if model_type == 'EDGES_polynomial':
				# av_model_factor = np.polyval(av_par, fb/200)
				# av_model        = av_model_factor * ((fb/200)**(-2.5))
				# #fx, fgb, wx     = eg.spectral_binning_number_of_samples(fb, av_model, np.ones(len(f)), nsamples=64)
				
			
			# avr, avw    = eg.spectral_averaging(rnew, wnew)
			# avt         = av_model + avr
			# avt[avw==0] = 0
			
	
			# # RFI cleaning final spectrum
			# print(' ')
			# print(' ')
			# print('---------------------------------------')
			# print('Final averaged spectrum')
			# tx, wx = RFI_spectrum_cleaning(fb, avt, avw, window_width_MHz=window_width_MHz_3, Npolyterms_block=Npolyterms_block, N_choice=N_choice, N_sigma=N_sigma)

			
			
			
			
				# # Figures
				# # -----------------------------------------------------------------------------------
				# #
				# # Figures of intermediate residuals (from averages of N_av spectra)
				# # -----------------------------------------------------------------------------------
				# #				
				# N_spec_total = len(rnew[:,0])  # number of total spectra to plot
				# N_spec_fig   = 7  # number of spectra per figure
				# N_figures    = int(np.ceil(N_spec_total/N_spec_fig))
				
				
				# for j in range(N_figures):
					
					# plt.figure(figsize=(7,10), facecolor='y')
					# for i in range(N_spec_fig):
						
						# plt.subplot(N_spec_fig, 1, i+1)
						
						# index_spectrum = j*N_spec_fig + i
						
						# if (index_spectrum + 1) <= N_spec_total:
							# plt.plot(fb[wnew[index_spectrum,:]>0], rnew[index_spectrum,:][wnew[index_spectrum,:]>0])
							# plt.legend(['index: ' + str(index_spectrum) + ',   RMS: ' + str(int(1000*rms[index_spectrum])) + ' mK'], loc=1, fontsize=8)
						
						# plt.yticks([-3, -2, -1, 0, 1, 2, 3])
						# plt.ylim([-3.5, 3.5])
						# plt.ylabel('[K]')
						
						# if i == 0:
							# xmin, xmax = plt.xlim()
						# if i > 0:
							# plt.xlim([xmin, xmax])
	
						# if i < (N_spec_fig-1):
							# plt.xticks([])
						
						# if i == (N_spec_fig-1):
							# plt.xlabel('frequency [MHz]')						
							
						# if i == 0:
							# plt.title(name + ': Figure ' + str(j+1) + ' of ' + str(N_figures))
							
							
					# plt.savefig(save_folder_plots + name + '_' + str(int(j+1)) + '_of_' + str(N_figures) + '.png', bbox_inches='tight', facecolor='y')
					# plt.close()
					# plt.close()
				

		# No data
		# -------------------------------
		else:
			print(' ')
			print('----------- No Data --------------')
			print(' ')
			
			f       = 0
			r_all   = 0
			p_all   = 0
			w_all   = 0
			rms_all = 0
			m_all   = 0			
						
			
	# No data
	# -----------------------------
	else:
		print(' ')
		print('----------- No Data --------------')	
		print(' ')
		
		f       = 0
		r_all   = 0
		p_all   = 0
		w_all   = 0
		rms_all = 0
		m_all   = 0		











	# Saving results 
	# ----------------------------------------------------------
	#
	if save == 'yes':
		
		save_file = save_folder_spectra + name + '.hdf5'
		with h5py.File(save_file, 'w') as hf:
			
			hf.create_dataset('frequency',    data = f)
			hf.create_dataset('residuals',    data = r_all)
			hf.create_dataset('parameters',   data = p_all)
			hf.create_dataset('weights',      data = w_all)
			hf.create_dataset('rms',          data = rms_all)
			hf.create_dataset('metadata',     data = m_all)
		




	return f, r_all, p_all, w_all, rms_all, m_all         

























def batch_level2_to_level3(case, save, folder, antenna_type):
	

	# Checked manually to make sure data quality is high
	if antenna_type == 'fourpoint':
		day = ['2015_109_11', '2015_110_00', '2015_111_00', '2015_113_09', '2015_114_00', '2015_115_00', '2015_116_00', '2015_117_00', '2015_118_00', '2015_119_13', '2015_128_00', '2015_129_00', '2015_130_00', '2015_131_00', '2015_133_00', '2015_134_00', '2015_135_00', '2015_138_00', '2015_139_00', '2015_140_00', '2015_141_00', '2015_142_00', '2015_143_00', '2015_144_00', '2015_145_00', '2015_146_00', '2015_147_00', '2015_148_00', '2015_149_00', '2015_150_00', '2015_151_00', '2015_152_00', '2015_153_00', '2015_154_00', '2015_155_00', '2015_156_00', '2015_162_00', '2015_163_00', '2015_164_00', '2015_165_00', '2015_167_00', '2015_175_00', '2015_177_00', '2015_178_00', '2015_180_00', '2015_181_00', '2015_182_00', '2015_183_00', '2015_184_00', '2015_185_00', '2015_186_00', '2015_187_00', '2015_188_00', '2015_189_00', '2015_190_00', '2015_191_00', '2015_192_00', '2015_192_18', '2015_193_00', '2015_195_00', '2015_198_00']



	# Checked manually to make sure data quality is high
	elif antenna_type == 'blade':
		day = ['2015_206_00', '2015_207_00', '2015_208_00', '2015_209_00', '2015_210_00', '2015_210_03', '2015_211_00', '2015_211_18', '2015_212_00', '2015_215_08', '2015_216_00', '2015_217_00', '2015_218_00', '2015_219_00', '2015_220_00', '2015_221_00', '2015_222_00', '2015_223_00', '2015_224_00', '2015_225_00', '2015_226_00', '2015_227_00', '2015_228_00', '2015_229_00', '2015_230_00', '2015_231_00', '2015_232_00', '2015_233_00', '2015_234_00', '2015_235_00', '2015_236_00', '2015_237_00', '2015_238_00', '2015_239_00', '2015_240_00', '2015_241_00', '2015_242_00', '2015_243_00', '2015_245_00', '2015_250_15', '2015_251_00', '2015_252_00', '2015_253_00', '2015_254_00', '2015_255_00', '2015_256_00', '2015_257_00', '2015_258_00', '2015_259_00', '2015_260_00', '2015_261_00', '2015_262_00', '2015_265_19', '2015_267_00', '2015_268_00', '2015_269_00', '2015_270_00', '2015_271_00', '2015_272_00', '2015_273_00', '2015_274_00', '2015_275_00', '2015_276_00', '2015_277_00', '2015_278_00', '2015_279_00', '2015_280_00', '2015_281_17', '2015_282_00', '2015_283_00', '2015_284_00', '2015_285_00', '2015_286_00', '2015_287_00', '2015_288_00', '2015_291_00', '2015_292_00', '2015_293_00', '2015_295_00', '2015_296_00', '2015_297_00', '2015_298_00', '2015_301_00', '2015_310_18', '2015_311_00', '2015_312_00', '2015_313_00', '2015_314_00', '2015_315_00', '2015_317_00', '2015_318_00', '2015_319_00', '2015_320_00', '2015_321_00', '2015_322_00', '2015_323_00', '2015_324_00', '2015_325_00', '2015_329_00', '2015_330_00', '2015_332_00', '2015_333_00', '2015_334_00', '2015_336_00', '2015_337_00', '2015_338_00', '2015_339_00', '2015_340_00', '2015_341_00', '2015_343_00', '2015_344_00', '2015_345_00', '2015_346_00', '2015_347_00', '2015_352_00', '2015_353_00', '2015_354_00', '2015_355_00', '2015_356_00', '2015_357_00', '2015_358_00', '2015_359_00', '2015_360_00', '2015_361_00', '2015_362_00', '2015_363_00', '2015_364_00', '2015_365_00', '2016_001_00', '2016_002_00', '2016_003_00', '2016_004_00', '2016_005_00', '2016_008_00', '2016_009_00', '2016_010_00', '2016_011_00', '2016_012_00', '2016_013_00', '2016_015_00', '2016_017_00', '2016_018_00', '2016_019_00', '2016_028_00', '2016_029_00', '2016_030_00', '2016_031_00', '2016_032_00', '2016_033_00', '2016_034_00', '2016_035_00', '2016_036_00', '2016_037_00', '2016_038_00', '2016_039_00', '2016_040_00', '2016_041_00', '2016_042_00', '2016_043_00', '2016_044_00', '2016_045_00', '2016_046_00', '2016_047_00', '2016_048_00', '2016_049_00', '2016_050_00', '2016_051_00', '2016_052_00', '2016_053_00',  '2016_060_00', '2016_062_00', '2016_063_00', '2016_064_00', '2016_065_00', '2016_066_00', '2016_067_00', '2016_068_00', '2016_069_00', '2016_070_00', '2016_071_00', '2016_072_00', '2016_073_00', '2016_074_00', '2016_075_00', '2016_076_00', '2016_077_00', '2016_078_00', '2016_079_00', '2016_081_00', '2016_082_00', '2016_083_00', '2016_084_00', '2016_085_00', '2016_086_00', '2016_087_00', '2016_088_00', '2016_090_00', '2016_091_00', '2016_092_00', '2016_093_00', '2016_094_00', '2016_095_00', '2016_096_00', '2016_097_00', '2016_098_00', '2016_101_00', '2016_102_00', '2016_103_00', '2016_108_00', '2016_109_00', '2016_110_00', '2016_111_00', '2016_112_00', '2016_113_00', '2016_114_00', '2016_115_00', '2016_116_00', '2016_117_00', '2016_118_00', '2016_123_00', '2016_124_00', '2016_125_00', '2016_127_00', '2016_128_00', '2016_129_00', '2016_130_00', '2016_131_00', '2016_132_00', '2016_133_00', '2016_134_00', '2016_135_00', '2016_136_00', '2016_137_00', '2016_138_00', '2016_139_00', '2016_140_00', '2016_141_00', '2016_142_00', '2016_143_00', '2016_144_00', '2016_145_00', '2016_146_00', '2016_147_00', '2016_148_00', '2016_149_00', '2016_150_00', '2016_151_00', '2016_152_00', '2016_153_00', '2016_154_00', '2016_156_00', '2016_157_00', '2016_158_00', '2016_159_00', '2016_160_00', '2016_161_00', '2016_162_00', '2016_163_00', '2016_164_00', '2016_165_00', '2016_166_00', '2016_167_00', '2016_168_00', '2016_169_00', '2016_170_00', '2016_171_00', '2016_172_00', '2016_173_00', '2016_181_00', '2016_182_00', '2016_183_00', '2016_184_00', '2016_185_00', '2016_186_00', '2016_187_00', '2016_188_00', '2016_189_00', '2016_190_00', '2016_191_00', '2016_192_00', '2016_193_00', '2016_194_00', '2016_195_00', '2016_196_00', '2016_197_00', '2016_198_00', '2016_199_00', '2016_200_00', '2016_201_00', '2016_202_00', '2016_203_00', '2016_204_00', '2016_205_00', '2016_206_00', '2016_207_00', '2016_208_00', '2016_209_00', '2016_210_00', '2016_211_00', '2016_212_00', '2016_213_00', '2016_216_16', '2016_217_00', '2016_218_00', '2016_219_00', '2016_220_00', '2016_223_18', '2016_224_00', '2016_225_00', '2016_226_00', '2016_227_00', '2016_228_00', '2016_229_00', '2016_230_00', '2016_236_09', '2016_237_00', '2016_239_00', '2016_240_00', '2016_241_00', '2016_242_00', '2016_243_00', '2016_245_00', '2016_246_00', '2016_247_00', '2016_248_00', '2016_248_06', '2016_251_00', '2016_252_00', '2016_253_13', '2016_254_09', '2016_255_00', '2016_256_00', '2016_257_00', '2016_258_00']
		
		
		
	start_time = time.time()	
	length_day = len(day)
		
	count = 0
	for i in range(length_day):
		
		print(' ')
		print('........................ DAY:  ' + day[i] + ' .........................')
		print(' ')
		
		
		# Case 100
		# ------------------------------------------------------
		if case == 100:					
			f, r, p, w, rms, m = level2_to_level3(day[i], save, folder, LST1=23.76, LST2=11.76, SUN_EL_MIN=-95, SUN_EL_MAX=0, MOON_EL_MIN=-95, MOON_EL_MAX=0, LOSS=1, BEAM=0, FMIN=70, FMAX=100, antenna_type='blade', model_type ='EDGES_polynomial', Nfg=5)

		
		# Case 101
		# ------------------------------------------------------
		if case == 101:		
			f, r, p, w, rms, m = level2_to_level3(day[i], save, folder, LST1=0, LST2=24, SUN_EL_MIN=-95, SUN_EL_MAX=0, MOON_EL_MIN=-95, MOON_EL_MAX=0, LOSS=1, BEAM=1, FMIN=80, FMAX=195, antenna_type='blade', model_type='EDGES_polynomial', Nfg=10)



	end_time = time.time()
	print("Elapsed time was %g seconds" % (end_time - start_time))

	
	return 0



































def level3read(file_path):

	'''
	
	# file_path = path_to + / + filename.hdf5
	
	'''
	

	with h5py.File(file_path,'r') as hf:
		
		X   = hf.get('frequency')
		f   = np.array(X)
		
		X   = hf.get('residuals')
		r   = np.array(X)
		
		X   = hf.get('parameters')
		p   = np.array(X) 
		
		X   = hf.get('weights')
		w   = np.array(X)
		
		X   = hf.get('rms')
		rms = np.array(X)

		X   = hf.get('metadata')
		m   = np.array(X)
			
	return f, r, p, w, rms, m




















def filter_minimum_temperature(f, r_all, w_all, p_all, year_all, day_all):
	
	
	fg_all = np.zeros((len(r_all[:,0]), len(r_all[0,:])))
	
	for i in range(len(r_all[:,0])):
		av_model_factor = np.polyval(p_all[i,:], f/200);    
		av_model        = av_model_factor * ((f/200)**(-2.5))
		
		fg_all[i,:] = av_model
	
	# Keep only data where the foreground model is above 0K at all frequencies
	index = np.arange(len(fg_all[:,0]))
	index.astype(int)
	index_good = index[np.min(fg_all, axis=1) > 0]
	ig = index_good.tolist()
	
	print(len(index))
	print(len(ig))
		
	r     = r_all[index_good, :]
	w     = w_all[index_good, :]
	p     = p_all[index_good, :]
	year  = year_all[index_good]
	day   = day_all[index_good]
	
	
	
	# # Average the good data
	# avr, avw   = eg.spectral_averaging(r, w)
	# fb, rb, wb = eg.spectral_binning_number_of_samples(f, avr, avw, nsamples=64) 

	# avp             = np.mean(p, axis=0)
	# av_model_factor = np.polyval(avp, fb/200)
	# fgb             = av_model_factor * ((fb/200)**(-2.5))
	
	# tb = rb + fgb
	
	# fx=fb[(fb>=fLOW) & (fb<=fHIGH)]
	# tx=tb[(fb>=fLOW) & (fb<=fHIGH)]
	# wx=wb[(fb>=fLOW) & (fb<=fHIGH)]
	
		
	return f, r, w, p, year, day






def batch_level3_to_level4(folder, LST1, LST2, fLOW, fHIGH, save, save_file_name):
	

	file_path   = home_folder + '/DATA/EDGES/spectra/level3/high_band_2015/2018_analysis/' + folder + '/'
	save_folder = home_folder + '/DATA/EDGES/spectra/level4/high_band_2015/2018_analysis/'
		

	# Listing files to be processed
	full_list = listdir(file_path)
	full_list.sort()

	
	count = 0
	for i in range(len(full_list)):
		
		file_name = full_list[i]
		print(file_name)
		
		f, r, p, w, rms, m = level3read(folder, file_name[0:-5])   #'2015_315_00')
		
		test = np.array([f])
		#print(test.size)
		
		
		if test.size > 1:
			
			LST   = m[:,3]
			index = np.arange(len(LST))
			
			if LST1 < LST2:
				index2 = index[  (LST >= LST1) & (LST <= LST2) ]

			if LST1 > LST2:
				index2 = index[  (LST >= LST1) | (LST <= LST2) ]
			
			
			
			if len(index2) > 0:
				
				print('Number of raw traces: ' + str(len(index2)))
				
				# Selection of spectra
				px   = p[index2, :]
				rx   = r[index2, :]
				wx   = w[index2, :]
				
								
				# Cut to frequency range
				fc   =   f[(f>=fLOW) & (f<=fHIGH)]
				rx2  =  rx[:, (f>=fLOW) & (f<=fHIGH)]
				wx2  =  wx[:, (f>=fLOW) & (f<=fHIGH)]			
				
				avp              = np.mean(px, axis=0)
				av_model_factor  = np.polyval(avp, fc/200)
				av_model         = av_model_factor * ((fc/200)**(-2.5))
				
				
				# Daily average antenna temperature	
				avr, avw         = eg.spectral_averaging(rx2, wx2)
				
				
				if count == 0:
					print(avr.size)
					r_all = np.copy(avr)
					w_all = np.copy(avw)
					p_all = np.copy(avp)
					year_all = np.array([int(file_name[0:4])])
					day_all  = np.array([int(file_name[5:8])])
					count = 1
					
					
				elif count > 0:
					r_all = np.vstack((r_all, avr))
					w_all = np.vstack((w_all, avw))
					p_all = np.vstack((p_all, avp))
					year_all = np.append(year_all, np.array([int(file_name[0:4])]))
					day_all  = np.append(day_all,  np.array([int(file_name[5:8])]))
	





	# Only keep files whose minimum temperature (not residuals, but total temperature, foregrounds included)
	fx, rx, wx, px, yx, dx = filter_minimum_temperature(fc, r_all, w_all, p_all, year_all, day_all)
	
	
	
	if save == 'yes':
		save_file = save_folder + save_file_name + '.hdf5'
		with h5py.File(save_file, 'w') as hf:
			
			hf.create_dataset('frequency',    data = fx)
			hf.create_dataset('residuals',    data = rx)
			hf.create_dataset('parameters',   data = px)
			hf.create_dataset('weights',      data = wx)
			hf.create_dataset('year',         data = yx)
			hf.create_dataset('day',          data = dx)
				


	return fx, rx, wx, px, yx, dx








def level4read(filename):

	# Loading results
	file_path = home_folder + '/DATA/EDGES/spectra/level4/high_band_2015/2018_analysis/' + filename + '.hdf5'
	with h5py.File(file_path,'r') as hf:
		
		X   = hf.get('frequency')
		f   = np.array(X)
		
		X   = hf.get('residuals')
		r   = np.array(X)
		
		X   = hf.get('parameters')
		p   = np.array(X) 
		
		X   = hf.get('weights')
		w   = np.array(X)
		
		X    = hf.get('year')
		year = np.array(X)

		X    = hf.get('day')
		day  = np.array(X)
			
	return f, r, p, w, year, day













def daily_RFI_filter(case, r_all):
	
	index  = np.arange(len(r_all[:,0]))
	
	if case == 101:
		remove = [4, 56, 58, 62, 63, 71, 81, 84, 85, 86, 88, 92, 96, 98, 99, 102, 119, 120, 132, 143, 150, 151, 152, 154, 161, 165, 189]
	
	index_clean = np.delete(index, remove)
	

	return index_clean










def plot_binned_residuals(case, f, r_all, w_all, p_all):
	
	plt.close()
	plt.close()
	plt.close()
	
	
	index_clean = daily_RFI_filter(case, r_all)
	
	
	r = r_all[index_clean,:]
	w = w_all[index_clean,:]
	p = p_all[index_clean,:]
	
	
	for i in range(10): #(len(r[:,0])):
		
		print(i)
		
		rx, wx = RFI_spectrum_cleaning(f, r[i,:], w[i,:], window_width_MHz=10, Npolyterms_block=4, N_choice=20, N_sigma=2.5)
		
		fb, rb, wb = eg.spectral_binning_number_of_samples(f, rx, wx, nsamples=64)
		plt.plot(fb, rb-i*1)
	
	
	
	return 0
































def spectral_averaging_RFI_cleaning(f, r, p, w, N_av=16, model_type='EDGES_polynomial', RFI_cleaning='no', window_width_MHz=4, Npolyterms_block=4, N_choice=20, N_sigma=2.5):
	

	# Number of original spectra
	N_spectra_raw = len(r[:,0])
	
	# Number of output spectra
	N_spectra_new = int(np.floor(N_spectra_raw / N_av))
	
	
	
	
	# Produce and clean every output spectra
	for i in range(N_spectra_new):
		
		
		# Produce average spectrum from N_av original spectra  
		# ---------------------------------------------------------------------------------------
		# 
		if i < (N_spectra_new-1):
			
			# Indices of spectra to average (except last spectrum)
			index_low  = i*N_av
			index_high = (i+1)*N_av - 1
			
			# Average residuals and model parameters
			avr, avw   = eg.spectral_averaging(r[index_low:(index_high+1), :], w[index_low:(index_high+1), :])		
			avp        = np.mean(p[index_low:(index_high+1), :], axis=0)			
			
				
		elif i == (N_spectra_new-1):
			
			# Indices of last spectrum to average
			index_low  = i*N_av
			index_high = N_spectra_raw - 1
						
			# Average residuals and model parameters
			avr, avw   = eg.spectral_averaging(r[index_low::, :], w[index_low::, :])		
			avp        = np.mean(p[index_low::, :], axis=0)			

		print(' ')
		print(' ')
		print('-----------------------------------------------------')
		print( str(i+1) + ' of ' + str(N_spectra_new) )
		print('Indices of average spectrum: ' + str(index_low) + ' ' + str(index_high))
		
	

		# Average foreground model
		if model_type == 'EDGES_polynomial':
			avfg_factor = np.polyval(avp, f/200)
			avfg        = avfg_factor * ((f/200)**(-2.5))
			
				
		# Total temperature
		avt = avfg + avr
		#
		# --------------------------------------------------------------------------------------
		
		
		if RFI_cleaning == 'yes':
			
			# RFI cleaning
			avt, avw = RFI_spectrum_cleaning(f, avt, avw, window_width_MHz=window_width_MHz, Npolyterms_block=Npolyterms_block, N_choice=N_choice, N_sigma=N_sigma)
		
		
		# Fit a foreground model to cleaned averaged spectrum
		Nfg = len(p[0,:])
		if model_type == 'EDGES_polynomial':
			par          = np.polyfit((f/200)[avw>0], (avt/((f/200)**(-2.5)))[avw>0], Nfg-1)
			model_factor = np.polyval(par, (f/200))
			model        = model_factor * ((f/200)**(-2.5))
			
					
		# Computing and storing residuals
		if i == 0:
			rnew = np.zeros((N_spectra_new, len(f)))
			pnew = np.zeros((N_spectra_new, Nfg))
			wnew = np.zeros((N_spectra_new, len(f)))
			rms  = np.zeros((N_spectra_new))
			
			
		res          = avt - model
		res[avw==0]  = 0
		
		rnew[i,:] = res			
		pnew[i,:] = par
		wnew[i,:] = avw
		rms[i]    = np.std(res[avw>0])


	return rnew, pnew, wnew, rms

















































def RFI_spectrum_cleaning(f, avt, avw, window_width_MHz=10, Npolyterms_block=4, N_choice=20, N_sigma=2.5):
	
	
	# RFI cleaning spectrum
	# --------------------------------------------------------------------------------------
	#
	# Reverse temperature and weights, to sweep the spectrum from high- to low-frequencies.
	# This produces a cleaner spectrum
	rev_avt = np.flip(avt, axis=0)
	rev_avw = np.flip(avw, axis=0)

	# Initialize flags
	len_wb_zero      = 1
	len_wb_zero_last = 0
	count = 0
	
	# Clean spectrum until no more channels are removed
	while len_wb_zero > len_wb_zero_last:
		if count > 0:
			len_wb_zero_last = np.copy(len_wb_zero)
		
		rev_avt, rev_avw  = eg.RFI_cleaning_sweep(f, rev_avt, rev_avw, window_width_MHz=window_width_MHz, Npolyterms_block=Npolyterms_block, N_choice=N_choice, N_sigma=N_sigma)
		len_wb_zero       = len(rev_avw[rev_avw==0])
		count             = count + 1
		print('RFI cleaning iteration: ' + str(count) + '. Number of bad channels: ' + str(len_wb_zero))
	
	# Reverse back to original order
	avt = np.flip(rev_avt, axis=0)
	avw = np.flip(rev_avw, axis=0)
	#
	# ---------------------------------------------------------------------------------------
	
			
	return avt, avw







































































def MCMC_simulated_data():
	
	v          = np.arange(80, 120.1, 0.390)
	noise      = 0.015*np.random.randn(len(v))
	par_100MHz = [1000, -300, 4000, -30, 100]
	tfg        = eg.model_evaluate('Physical_model', par_100MHz, v/100)
	t21        = eg.model_eor_flattened_gaussian(v, model_type=1, T21=-0.5, vr=78, dv=22, tau0=5)
	
	
	t = tfg + t21 + noise
	
	return v, tfg, t21, noise, t








def MCMC_real_data(fLOW, fHIGH):
	
	fx, rx, px, wx, yx, dx = level4read('GHA_7-15hr_80-195MHz')
	index_clean            = daily_RFI_filter(101, rx)
	
	ryy = rx[index_clean,:]
	wyy = wx[index_clean,:]
	pyy = px[index_clean,:]
	yyy = yx[index_clean]
	dyy = dx[index_clean]
	
	index_new      = np.arange(len(index_clean))
	index_year_day = index_new#[(yyy==2015) & (dyy>=250) & (dyy<=299)]
	
	print(len(index_year_day))
	

	ry = ryy[index_year_day, :]
	wy = wyy[index_year_day,:]
	p  = pyy[index_year_day,:]
	y  = yyy[index_year_day]
	d  = dyy[index_year_day]	
	
	
	
	
	
	f = fx[(fx >= fLOW) & (fx <= fHIGH)]
	r = ry[:,(fx >= fLOW) & (fx <= fHIGH)]
	w = wy[:,(fx >= fLOW) & (fx <= fHIGH)]
	
	
	# Average residuals and model parameters
	avr, avw    = eg.spectral_averaging(r, w)
	fb, rb, wb  = eg.spectral_binning_number_of_samples(f, avr, avw, nsamples=64)
	
	
	avp         = np.mean(p, axis=0)
	avfg_factor = np.polyval(avp, fb/200)
	avfg        = avfg_factor * ((fb/200)**(-2.5))
	
	# Total temperature
	tb = avfg + rb
	tb[wb==0] = 0
	
	# Reference residuals
	par = eg.fit_polynomial_fourier('EDGES_polynomial', fb, tb, 7, Weights=wb)
	#par = eg.fit_polynomial_fourier('Physical_model', fb, tb, 5, Weights=wb)
	rb2 = (tb-par[1])
	
	
	
	
	return fb, tb, wb, rb2






def MCMC_absorption_feature(vb, db, nb, wb, save='no', save_filename='x', flow=80, fhigh=120, model_fg='EDGES_polynomial', Nfg=3, model_21cm='flattened_gaussian', T21_range=[-5, 5], vr_range=[75, 100], dv_range=[1, 40], tau0_range=[0, 40], zr_range=[7, 15], dz_range=[0.05, 2], Nchain=10000, Nthreads=6, rejected_fraction=0.3): 

	'''

	March 22, 2018

	## Example of running the code
	## 3 terms 81-129 MHz   20 mK noise
	## samples, par, v, t21, res, rms = eg.MCMC(fb, tb, 0.02*np.ones(len(fb)), wb, flow=80, fhigh=129, model_fg='EDGES_polynomial', Nfg=3, Nchain=6000, Nthreads=6)

	## Example of plotting 21-cm models
	## plt.plot(v, tcosmo_all[np.random.randint(len(tcosmo_all[:,0]), size=500) ,:].T, color='c', alpha=0.1)
	## plt.plot(v, tcosmo_all[-1,:], color='k')


	'''


	# Setting Up MCMC
	# -----------------------------------------------------
	# Cutting data within desired range	
	v = vb[(vb>=flow) & (vb<=fhigh)]
	d = db[(vb>=flow) & (vb<=fhigh)]
	n = nb[(vb>=flow) & (vb<=fhigh)]
	w = wb[(vb>=flow) & (vb<=fhigh)]


	# Noise profile
	# sigma_noise = 0.03* np.ones(len(v))  #(60  - 0.5*v)/1000  # in K   (~ 30 mK at 60 MHz, and 10 mK at 100 MHz)


	# MCMC
	#if model_21cm == 'gaussian_flattened_1':
		#N21=3
		
	if model_21cm == 'flattened_gaussian':
		N21 = 4
	elif model_21cm == 'tanh_eor':
		N21 = 3		
	Ndim     = Nfg + N21
	Nwalkers = 2*Ndim + 2
	
	
	
	if model_fg == 'EDGES_polynomial':
		vn = 100
	elif model_fg == 'Physical_model':
		vn = 1000
		



	# Random start point of the chain
	p0 = (np.random.uniform(size=Ndim*Nwalkers).reshape((Nwalkers, Ndim)) - 0.5) / 10         # noise STD: ([0,1] - 0.5) / 10  =  [-0.05, 0.05]

	# Appropriate offset to the random starting point of some parameters
	# First are foreground model coefficients
	x        = eg.fit_polynomial_fourier(model_fg, v/vn, d, Nfg, Weights=w)
	poly_par = x[0]
	print(poly_par)
	print(vn)
	for i in range(Nfg):
		p0[:,i] = p0[:,i] + poly_par[i]

	

	# Next are phenomenological models
	if model_21cm == 'flattened_gaussian':
		p0[:,Nfg+0] = p0[:,Nfg+0] - 0.5
		p0[:,Nfg+1] = p0[:,Nfg+1] + 78
		p0[:,Nfg+2] = p0[:,Nfg+2] + 20
		p0[:,Nfg+3] = p0[:,Nfg+3] + 7
		
	elif model_21cm == 'tanh_eor':
		p0[:,Nfg+0] = p0[:,Nfg+0] + 0.02 # amplitude mK
		p0[:,Nfg+1] = p0[:,Nfg+1] + np.mean(zr_range)   # central redshift
		p0[:,Nfg+2] = p0[:,Nfg+2] + np.mean(dz_range)  # redshift width

	



	# Computing results
	# ------------------------------------------------------
	print('Computing MCMC ...')
	sampler = ec.EnsembleSampler(Nwalkers, Ndim, data_analysis_MCMC_log_likelihood, threads=Nthreads, args=(v[w>0], d[w>0], n[w>0]), kwargs={'model_fg':model_fg, 'Nfg':Nfg, 'vn':vn, 'model_21cm':model_21cm, 'T21_range':T21_range, 'vr_range':vr_range, 'dv_range':dv_range, 'tau0_range':tau0_range, 'zr_range':zr_range, 'dz_range':dz_range})
	sampler.run_mcmc(p0, Nchain)













	# Analyzing results
	# ------------------------------------------------------

	# All samples
	#samples = sampler.chain.reshape((-1, Ndim))


	# Only accepted samples
	samples_cut = sampler.chain[:, (np.int(rejected_fraction*Nchain)):, :].reshape((-1, Ndim))




	## For flattened gaussian Invert sign of a21 and multiply by 1000
	#if model_21cm == 'gaussian_flattened_1':
		#samples_cut[:,Nfg] = -1000*samples_cut[:,Nfg]


	## For tanh_EoR, just multiply by 1000
	#elif (model_21cm == 'tanh_EoR') and (N21 == 3):
		#samples_cut[:,Nfg] = 1000*samples_cut[:,Nfg]	




	## Corner plot for 21-cm parameters
	#if model_21cm == 'gaussian_flattened_1':	
		#if N21 == 3:
			#labels=[r'$a_{21}$ [mK]', r'$\nu_r$ [MHz]', r'$\Delta \nu$ [MHz]']
		#elif N21 == 4:
			#labels=[r'$a_{21}$ [mK]', r'$\nu_r$ [MHz]', r'$\Delta \nu$ [MHz]', r'$\tau$']
		#elif N21 == 5:
			#labels=[r'$a_{21}$ [mK]', r'$\nu_r$ [MHz]', r'$\Delta \nu$ [MHz]', r'$\tau$', r'$\chi$']


	#elif model_21cm == 'tanh_EoR':
		#if N21 == 2:
			#labels=[r'$z_r$', r'$\Delta z$']

		#elif N21 == 3:
			#labels=[r'$a_{21}$ [mK]', r'$z_r$', r'$\Delta z$']




	#fig = corner.corner(samples_cut[:,Nfg::], labels=labels, bins=50, label_kwargs={'fontsize':20})
	##fig.set_size_inches(14,10)
	##path_plot_save = home_folder + '/Desktop/'
	##plt.savefig(path_plot_save + 'test.png', bbox_inches='tight')
	##plt.close()





	## , r'$\tau$', r'$\chi$'



	## Best fit parameters and 95% C.L.
	#best_fit_parameters = np.zeros((Ndim,7))
	#for i in range(len(samples_cut[0,:])):
		#per = np.percentile(samples_cut[:,i], [50, 50-(68.27/2), 50+(68.27/2), 50-(95.45/2), 50+(95.45/2), 50-(99.73/2), 50+(99.73/2)], axis=0)
		#print('best fit and 68% limits: ' + str(per[0]) + '+' + str(per[2]-per[0]) + str(per[1]-per[0]))
		#best_fit_parameters[i,:] = np.copy(per)







	## Storing 21-cm models, residuals, and rms
	## ------------------------------------------------------
	##
	## Initializing arrays
	#Tcosmo_all     = np.zeros((len(samples_cut[:,0])+1, len(v)))   # Array of 21cm models
	#residuals_all  = np.zeros((len(samples_cut[:,0])+1, len(v)))   # Array of residuals
	#rms_all        = np.zeros(len(samples_cut[:,0])+1)             # Array of RMS 

	## First, processing all the samples
	#if model_21cm == 'gaussian_flattened_1':
		#for i in range(len(samples_cut[:,0])):
			#Tfg     = model_evaluate(model_fg, samples_cut[i,0:Nfg], v/vn)				
			#if N21 > 3:
				#tau0 = samples_cut[i, Nfg+3]
				#if N21 > 4:
					#tilt = samples_cut[i, Nfg+4]				
			#Tcosmo  = model_eor_flattened_gaussian(v, model_type=int(model_21cm[-1]), T21=-samples_cut[i, Nfg]/1000, vr=samples_cut[i, Nfg+1], dv=samples_cut[i, Nfg+2], tau0=tau0, tilt=tilt)

			#Tfull   = Tfg + Tcosmo
			#rms     = np.sqrt(np.sum(((d-Tfull)[w>0])**2)/len(v[w>0]))

			#Tcosmo_all[i,:]     = Tcosmo
			#residuals_all[i,:]  = d - Tfull
			#rms_all[i]          = rms			




	#if model_21cm == 'tanh_EoR':
		#for i in range(len(samples_cut[:,0])):
			#Tfg     = model_evaluate(model_fg, samples_cut[i,0:Nfg], v/vn)

			#if N21 == 2:
				#Tcosmo, x1, x2 = model_eor(v, T21=a21_eor, model_type='tanh', zr=samples_cut[i, Nfg], dz=samples_cut[i, Nfg+1])

			#elif N21 == 3:
				#Tcosmo, x1, x2 = model_eor(v, T21=samples_cut[i, Nfg]/1000, model_type='tanh', zr=samples_cut[i, Nfg+1], dz=samples_cut[i, Nfg+2])

			#Tfull   = Tfg + Tcosmo
			#rms     = np.sqrt(np.sum(((d-Tfull)[w>0])**2)/len(v[w>0]))

			#Tcosmo_all[i,:]     = Tcosmo
			#residuals_all[i,:]  = d - Tfull
			#rms_all[i]          = rms			




	## Now, for the best fit model
	#Tfg     = model_evaluate(model_fg, best_fit_parameters[0:Nfg,1], v/vn)

	#if model_21cm == 'gaussian_flattened_1':		
		#if N21 > 3:
			#tau0 = best_fit_parameters[Nfg+3,0]
			#if N21 > 4:
				#tilt = best_fit_parameters[Nfg+4,0]
		#Tcosmo  = model_eor_flattened_gaussian(v, model_type=int(model_21cm[-1]), T21=-best_fit_parameters[Nfg,0]/1000, vr=best_fit_parameters[Nfg+1,0], dv=best_fit_parameters[Nfg+2,0], tau0=tau0, tilt=tilt)


	#if model_21cm == 'tanh_EoR':
		#if N21 == 2:
			#Tcosmo, x1, x2 = model_eor(v, T21=a21_eor, model_type='tanh', zr=best_fit_parameters[Nfg,0], dz=best_fit_parameters[Nfg+1,0])

		#elif N21 == 3:
			#Tcosmo, x1, x2 = model_eor(v, T21=best_fit_parameters[Nfg,0]/1000, model_type='tanh', zr=best_fit_parameters[Nfg+1,0], dz=best_fit_parameters[Nfg+2,0])


	#Tfull   = Tfg + Tcosmo
	#rms     = np.sqrt(np.sum(((d-Tfull)[w>0])**2)/len(v[w>0]))

	#Tcosmo_all[-1,:]     = Tcosmo
	#residuals_all[-1,:]  = d - Tfull
	#rms_all[-1]          = rms



	# Saving data
	if save == 'yes':

		save_file = home_folder + '/DATA/EDGES/results/high_band/products/MCMC_results/' + save_filename + '.hdf5'
		with h5py.File(save_file, 'w') as hf:

			hf.create_dataset('f',    data = v)
			hf.create_dataset('t',    data = d)
			hf.create_dataset('n',    data = n)
			hf.create_dataset('w',    data = w)
			hf.create_dataset('Nfg',  data = Nfg)
			hf.create_dataset('N21',  data = N21)
			hf.create_dataset('fn',   data = vn)

			hf.create_dataset('samples',               data = samples_cut)
			hf.create_dataset('best_fit_parameters',   data = best_fit_parameters)
			hf.create_dataset('models_21cm',           data = Tcosmo_all)
			hf.create_dataset('residuals',             data = residuals_all)
			hf.create_dataset('rms',                   data = rms_all)



	#return v, d, n, w, Nfg, N21, vn, samples_cut, best_fit_parameters, Tcosmo_all, residuals_all, rms_all


	return samples_cut

















































def data_analysis_MCMC_log_likelihood(theta, v, d, sigma_noise, model_fg='EDGES_polynomial', Nfg=3, vn=100, model_21cm='flattened_gaussian', T21_range=[-1, 0], vr_range=[75, 100], dv_range=[10, 40], tau0_range=[0, 20], zr_range=[7, 15], dz_range=[0.05, 2]):

	# Evaluating model foregrounds
	if Nfg == 0:
		Tfg = 0
		log_priors_fg = 0

	elif Nfg > 0:
		Tfg = eg.model_evaluate(model_fg, theta[0:Nfg], v/vn)
		log_priors_fg = data_analysis_MCMC_priors_foreground(theta[0:Nfg])
	
	# Evaluating 21-cm model
	if model_21cm == 'flattened_gaussian':
		Tcosmo        = eg.model_eor_flattened_gaussian(v, model_type=1, T21=theta[Nfg], vr=theta[Nfg+1], dv=theta[Nfg+2], tau0=theta[Nfg+3]) 
		log_priors_21 = data_analysis_MCMC_priors_gaussian_flattened(theta[Nfg::], T21_range=T21_range, vr_range=vr_range, dv_range=dv_range, tau0_range=tau0_range)
		
	elif model_21cm == 'tanh_eor':
		Tcosmo, x, y  = eg.model_eor(v, model_type='tanh', T21=theta[Nfg], zr=theta[Nfg+1], dz=theta[Nfg+2])      # 0.027
		log_priors_21 = data_analysis_MCMC_priors_tanh_eor(theta[Nfg::], T21_range=T21_range, zr_range=zr_range, dz_range=dz_range)



	# Full model
	Tfull = Tfg + Tcosmo


	# Log-likelihood
	log_likelihood =  -(1/2)  *  np.sum(  ((d-Tfull)/sigma_noise)**2  )



	return log_priors_21 + log_priors_fg + log_likelihood








def data_analysis_MCMC_priors_foreground(theta):

	# a0, a1, a2, a3 = theta
	len_theta = len(theta)

	# Counting the parameters with allowable values
	flag = 0
	for i in range(len_theta):
		if (-1e7 <= theta[i] <= 1e7):
			flag = flag + 1

	# Assign likelihood
	if flag == len_theta:
		out = 0
	else:
		out = -1e10

	return out




def data_analysis_MCMC_priors_gaussian_flattened(theta, T21_range=[-1, 0], vr_range=[75, 100], dv_range=[10, 40], tau0_range=[0, 20]):

	T21, vr, dv, tau0 = theta

	if (T21_range[0] <= T21 <= T21_range[1]) and (vr_range[0] <= vr <= vr_range[1]) and (dv_range[0] <= dv <= dv_range[1]) and (tau0_range[0] <= tau0 <= tau0_range[1]):
		return 0

	return -1e10




def data_analysis_MCMC_priors_tanh_eor(theta, T21_range=[0, 0.05], zr_range=[7, 15], dz_range=[0, 1]):
	
	T21, zr, dz = theta
	
	if (T21_range[0] <= T21 <= T21_range[1]) and (zr_range[0] <= zr <= zr_range[1]) and (dz_range[0] <= dz <= dz_range[1]):
		return 0

	return -1e10





def test():
	
	
	
	print('hola')
	
	return 0











def plots_high_band_blade_nominal():
	
	
	
	fb, days, tb_fg_all_NOT_USED, rb_all, wb_all, tb_fg, rb, wb, f, par_all, r_all, w_all, m_all = load_spectra('high_band_blade_nominal_low_foregrounds')
	

	tb = tb_fg + rb 
	
	fLOW  = 83
	fHIGH = 113
	vv    = fb[(fb>=fLOW) & (fb<=fHIGH)]
	tt    = tb[(fb>=fLOW) & (fb<=fHIGH)]
	ww    = wb[(fb>=fLOW) & (fb<=fHIGH)]
	
	
	#model_21 = eg.model_eor_flattened_gaussian(vv, model_type=1, T21=-0.5, vr=78, dv=20, tau0=7, tilt=0)
	par = eg.fit_polynomial_fourier('EDGES_polynomial', vv, tt, 5, Weights=ww)
	print(np.std(tt-par[1]))
	
	
	plt.close()
	plt.figure(1)
	
	plt.subplot(2,3,2)
	plt.plot(vv, tt-par[1])
	plt.xlim([80.1, 114.9])
	plt.ylim([-0.2, 0.2])
	plt.title(r'6 $\leq$ GHA $\leq$ 18 hr')


	model_21 = eg.model_eor_flattened_gaussian(vv, model_type=1, T21=-0.5, vr=78, dv=20, tau0=7, tilt=0)
	par      = eg.fit_polynomial_fourier('EDGES_polynomial', vv, tt-model_21, 5, Weights=ww)
	print(np.std(tt-model_21-par[1]))
	
	plt.subplot(2,3,5)
	plt.plot(vv, tt-model_21-par[1])
	plt.xlim([80.1, 114.9])
	plt.ylim([-0.2, 0.2])
	plt.xlabel('frequency [MHz]')
	
	
	
	# ------------------------------------------------------------------------------------------------
	
	
	index_all = np.arange(len(r_all[:,0]))
	index_accepted   = index_all[(m_all[:,3]>=0.26) & (m_all[:,3]<=6.26)]
	
	pk = par_all[index_accepted,:]
	rk = r_all[index_accepted,:]
	wk = w_all[index_accepted,:]
	mk = m_all[index_accepted,:]
	
	
	
	# Spectral averaging
	rav, wav   = eg.spectral_averaging(rk, wk)

	# Spectral binning
	fb, rb, wb = eg.spectral_binning_number_of_samples(f, rav, wav, nsamples=64)
	
	# Average model parameters
	pb         = np.mean(pk, axis=0)

	# Evaluating foreground model at binned frequencies
	av_model_factor = np.polyval(pb, f/200)
	av_model        = av_model_factor * ((f/200)**(-2.5))
	fx, tb_fg, wx   = eg.spectral_binning_number_of_samples(f, av_model, np.ones(len(f)), nsamples=64)
	
	# Binned total temperature
	tb = tb_fg  +  rb	
		
	


	fLOW  = 83
	fHIGH = 113
	vv    = fb[(fb>=fLOW) & (fb<=fHIGH)]
	tt    = tb[(fb>=fLOW) & (fb<=fHIGH)]
	ww    = wb[(fb>=fLOW) & (fb<=fHIGH)]
	
	
	#model_21 = eg.model_eor_flattened_gaussian(vv, model_type=1, T21=-0.5, vr=78, dv=20, tau0=7, tilt=0)
	par = eg.fit_polynomial_fourier('EDGES_polynomial', vv, tt, 5, Weights=ww)
	print(np.std(tt-par[1]))
	
	

	plt.subplot(2,3,1)
	plt.plot(vv, tt-par[1])
	plt.ylim([-0.2, 0.2])
	plt.ylabel('temperature [K]')
	plt.title(r'6.5 $\leq$ GHA $\leq$ 12.5 hr')


	model_21 = eg.model_eor_flattened_gaussian(vv, model_type=1, T21=-0.5, vr=78, dv=20, tau0=7, tilt=0)
	par      = eg.fit_polynomial_fourier('EDGES_polynomial', vv, tt-model_21, 5, Weights=ww)
	print(np.std(tt-model_21-par[1]))
	
	plt.subplot(2,3,4)
	plt.plot(vv, tt-model_21-par[1])
	plt.ylim([-0.2, 0.2])
	plt.xlabel('frequency [MHz]')
	plt.ylabel('temperature [K]')












	# ---------------------------------------------------------------------------------------------------------------
	
	fb, days, tb_fg_all_NOT_USED, rb_all, wb_all, tb_fg, rb, wb, f, par_all, r_all, w_all, m_all = load_spectra('high_band_blade_nominal_high_foregrounds')
	
	

	tb = tb_fg + rb 
	
	fLOW  = 83
	fHIGH = 103
	vv    = fb[(fb>=fLOW) & (fb<=fHIGH)]
	tt    = tb[(fb>=fLOW) & (fb<=fHIGH)]
	ww    = wb[(fb>=fLOW) & (fb<=fHIGH)]
	
	
	par = eg.fit_polynomial_fourier('EDGES_polynomial', vv, tt, 4, Weights=ww)
	print(np.std(tt-par[1]))
	
		
	plt.subplot(2,3,3)
	plt.plot(vv, tt-par[1])
	plt.xlim([80.1, 114.9])
	plt.ylim([-0.2, 0.2])
	plt.title(r'18 $\leq$ GHA $\leq$ 6 hr')


	model_21 = eg.model_eor_flattened_gaussian(vv, model_type=1, T21=-0.8, vr=80, dv=18, tau0=7, tilt=0)
	par      = eg.fit_polynomial_fourier('EDGES_polynomial', vv, tt-model_21, 4, Weights=ww)
	print(np.std(tt-model_21-par[1]))
	
	plt.subplot(2,3,6)
	plt.plot(vv, tt-model_21-par[1])
	plt.xlim([80.1, 114.9])
	plt.ylim([-0.2, 0.2])
	plt.xlabel('frequency [MHz]')
	
	

	
	
	


	return 0




















def plots_high_band_blade_first_dataset():
	
	
	
	fb, days, tb_fg_all_NOT_USED, rb_all, wb_all, tb_fg, rb, wb, f, par_all, r_all, w_all, m_all = load_spectra('high_band_blade_first_dataset_low_foregrounds')
	

	tb = tb_fg + rb 
	
	fLOW  = 83
	fHIGH = 103
	vv    = fb[(fb>=fLOW) & (fb<=fHIGH)]
	tt    = tb[(fb>=fLOW) & (fb<=fHIGH)]
	ww    = wb[(fb>=fLOW) & (fb<=fHIGH)]
	
	
	#model_21 = eg.model_eor_flattened_gaussian(vv, model_type=1, T21=-0.5, vr=78, dv=20, tau0=7, tilt=0)
	par = eg.fit_polynomial_fourier('EDGES_polynomial', vv, tt, 4, Weights=ww)
	print(np.std(tt-par[1]))
	
	
	plt.close()
	plt.figure(1)
	
	plt.subplot(2,3,2)
	plt.plot(vv, tt-par[1])
	plt.xlim([80.1, 114.9])
	plt.ylim([-0.2, 0.2])
	plt.title(r'6 $\leq$ GHA $\leq$ 18 hr')


	model_21 = eg.model_eor_flattened_gaussian(vv, model_type=1, T21=-0.5, vr=78, dv=20, tau0=7, tilt=0)
	par      = eg.fit_polynomial_fourier('EDGES_polynomial', vv, tt-model_21, 5, Weights=ww)
	print(np.std(tt-model_21-par[1]))
	
	plt.subplot(2,3,5)
	plt.plot(vv, tt-model_21-par[1])
	plt.xlim([80.1, 114.9])
	plt.ylim([-0.2, 0.2])
	plt.xlabel('frequency [MHz]')








	# ---------------------------------------------------------------------------------------------------------------
	
	fb, days, tb_fg_all_NOT_USED, rb_all, wb_all, tb_fg, rb, wb, f, par_all, r_all, w_all, m_all = load_spectra('high_band_blade_first_dataset_high_foregrounds')
	
	

	tb = tb_fg + rb 
	
	fLOW  = 83
	fHIGH = 103
	vv    = fb[(fb>=fLOW) & (fb<=fHIGH)]
	tt    = tb[(fb>=fLOW) & (fb<=fHIGH)]
	ww    = wb[(fb>=fLOW) & (fb<=fHIGH)]
	
	
	par = eg.fit_polynomial_fourier('EDGES_polynomial', vv, tt, 4, Weights=ww)
	print(np.std(tt-par[1]))
	
		
	plt.subplot(2,3,3)
	plt.plot(vv, tt-par[1])
	plt.xlim([80.1, 114.9])
	plt.ylim([-0.2, 0.2])
	plt.title(r'18 $\leq$ GHA $\leq$ 6 hr')


	model_21 = eg.model_eor_flattened_gaussian(vv, model_type=1, T21=-0.8, vr=80, dv=18, tau0=7, tilt=0)
	par      = eg.fit_polynomial_fourier('EDGES_polynomial', vv, tt-model_21, 4, Weights=ww)
	print(np.std(tt-model_21-par[1]))
	
	plt.subplot(2,3,6)
	plt.plot(vv, tt-model_21-par[1])
	plt.xlim([80.1, 114.9])
	plt.ylim([-0.2, 0.2])
	plt.xlabel('frequency [MHz]')
	
	














	return fb, tb, wb

















def low_band_cross_check(file_name = 'extended_gp_yes_bc_s11_2017_93_50_120MHz', foregrounds='low'):
	
	
	# o = eg.data_analysis_low_band_spectra_average('low_band_2015', 'extended_gp_yes_bc_s11_2016_243_50_120_MHz_all_data', 200, GHA1=-1, GHA2=24, GHA_center_array=np.arange(0,23,2), flow=50, fhigh=120, Nfg=7, SUN_LOWER_LIM=-90, SUN_HIGHER_LIM=90, MOON_LOWER_LIM=-90, MOON_HIGHER_LIM=90)
	
	
	
	#o = eg.level4_read('/home/ramo7131/DATA/EDGES/spectra/level4/low_band_2015/extended_gp_yes_bc_s11_2015_342_switch2015_50_120MHz_moon_down.hdf5')
	
	o = eg.level4_read('/home/ramo7131/DATA/EDGES/spectra/level4/low_band_2015/' + file_name + '.hdf5')
	

	f   = o[0]
	t   = o[1]
	r   = o[2]
	w   = o[3]
	p   = o[4]
	m   = o[5]
	rms = o[6]
	
	
	if foregrounds == 'low':
	
		LST1 = 6
		LST2 = 18
		
		index = np.arange(len(t[:,0]))
		index_accepted = index[(m[:,3]>=LST1) & (m[:,3]<=LST2) & (rms>=5.5) & (rms<=9)]
		
		
		tl = t[index_accepted,:]
		rl = r[index_accepted,:]
		wl = w[index_accepted,:]
		pl = p[index_accepted,:]
		ml = m[index_accepted,:]
		
		
	
		for i in range(len(rl[:,0])):
			rx = rl[i,9600::]
			wx = wl[i,9600::]
			std_x = np.std(rx[wx>0])
			#print(std_x)
			if i == 0:
				std_l = np.copy(std_x)
				
				
			elif i > 0:
				std_l = np.append(std_l, np.copy(std_x))
			
		tk = tl[std_l<=2.7,:]
		rk = rl[std_l<=2.7,:]
		wk = wl[std_l<=2.7,:]
		pk = pl[std_l<=2.7,:]
		mk = ml[std_l<=2.7,:]
		



		
	elif foregrounds == 'high':
		
		LST1 = 6
		LST2 = 18
		
		index = np.arange(len(t[:,0]))
		index_accepted = index[(m[:,3]<=LST1) | (m[:,3]>=LST2)]   #  & (rms>=5.5) & (rms<=9)
		
		
		th = t[index_accepted,:]
		rh = r[index_accepted,:]
		wh = w[index_accepted,:]
		ph = p[index_accepted,:]
		mh = m[index_accepted,:]
		
		
	
		for i in range(len(rh[:,0])):
			rx = rh[i,9600::]
			wx = wh[i,9600::]
			std_x = np.std(rx[wx>0])
			#print(std_x)
			if i == 0:
				std_h = np.copy(std_x)
				
				
			elif i > 0:
				std_h = np.append(std_h, np.copy(std_x))
			
		tk = th[std_h<=5.05,:]
		rk = rh[std_h<=5.05,:]
		wk = wh[std_h<=5.05,:]
		pk = ph[std_h<=5.05,:]
		mk = mh[std_h<=5.05,:]		
		
	



	
	# Spectral averaging
	rav, wav   = eg.spectral_averaging(rk, wk)

	# Spectral binning
	fb, rb, wb = eg.spectral_binning_number_of_samples(f, rav, wav, nsamples=64)
	
	# Average model parameters
	pb         = np.mean(pk, axis=0)
	
	# Evaluating foreground model at binned frequencies
	tb_fg      = eg.model_evaluate('EDGES_polynomial', pb, fb/200)
	
	# Binned total temperature
	tb         = tb_fg  +  rb	
	
	
	





	# # Modeling and plots
	# if foregrounds == 'low':	
		# fLOW  = 82  #
		# fHIGH = 116 # 115 #
		
		
	# elif foregrounds == 'high':
		# fLOW  = 82.5  #
		# fHIGH = 109.8 # 115 #		
	
	
	# vv = fb[(fb>=fLOW) & (fb<=fHIGH)]
	# tt = tb[(fb>=fLOW) & (fb<=fHIGH)]
	# ww = wb[(fb>=fLOW) & (fb<=fHIGH)]
	
	
	# par = eg.fit_polynomial_fourier('EDGES_polynomial', vv, tt, 5, Weights=ww)
	
	# plt.figure(1)
	# plt.plot(vv, tt-par[1] + 0.1)
	# plt.ylim([-0.2, 0.2])



	# T21 = eg.model_eor_flattened_gaussian(vv, model_type=1, T21=-0.40, vr=77, dv=17, tau0=7, tilt=0)
	# par = eg.fit_polynomial_fourier('EDGES_polynomial', vv, tt-T21, 5, Weights=ww)
	
	# plt.figure(1)
	# plt.plot(vv, tt-T21-par[1] - 0.1)
	# plt.ylim([-0.2, 0.2])
	# plt.xlabel('frequency [MHz]')
	
	# plt.ylabel('temperature [K]')
	
	# plt.savefig(home_folder + '/Desktop/' + file_name + '.png')
	# plt.close()






	
	
	return fb, tb, rb, wb









