
import numpy as np
import matplotlib.pyplot as plt

import edges as eg
import basic as ba
import rfi as rfi

import calibration as cal

from os import listdir


import os, sys
edges_folder       = os.environ['EDGES']
print('EDGES Folder: ' + edges_folder)




def batch_low_band_level1_to_level2(set_number):


	# Original 10x10 m^2 ground plane
	if set_number == 1:

		# DON'T PROCESS
		#o = eg.level1_to_level2('low_band', '2015', '282_00')
		#o = eg.level1_to_level2('low_band', '2015', '283_00')
		#o = eg.level1_to_level2('low_band', '2015', '284_00')
		#o = eg.level1_to_level2('low_band', '2015', '285_00')
		#o = eg.level1_to_level2('low_band', '2015', '285_14')


		o = eg.level1_to_level2('low_band', '2015', '286_02')
		o = eg.level1_to_level2('low_band', '2015', '287_00')
		o = eg.level1_to_level2('low_band', '2015', '288_00')
		o = eg.level1_to_level2('low_band', '2015', '289_00')

		o = eg.level1_to_level2('low_band', '2015', '291_00')
		o = eg.level1_to_level2('low_band', '2015', '292_00')
		o = eg.level1_to_level2('low_band', '2015', '293_00')
		o = eg.level1_to_level2('low_band', '2015', '294_00')
		o = eg.level1_to_level2('low_band', '2015', '295_00')
		o = eg.level1_to_level2('low_band', '2015', '296_00')
		o = eg.level1_to_level2('low_band', '2015', '297_00')
		o = eg.level1_to_level2('low_band', '2015', '298_00')
		o = eg.level1_to_level2('low_band', '2015', '299_00')

		o = eg.level1_to_level2('low_band', '2015', '300_00')
		o = eg.level1_to_level2('low_band', '2015', '301_00')
		o = eg.level1_to_level2('low_band', '2015', '302_00')
		o = eg.level1_to_level2('low_band', '2015', '303_00')

		o = eg.level1_to_level2('low_band', '2015', '310_18')
		o = eg.level1_to_level2('low_band', '2015', '311_00')
		o = eg.level1_to_level2('low_band', '2015', '312_00')
		o = eg.level1_to_level2('low_band', '2015', '313_00')
		o = eg.level1_to_level2('low_band', '2015', '314_00')
		o = eg.level1_to_level2('low_band', '2015', '315_00')
		o = eg.level1_to_level2('low_band', '2015', '316_00')
		o = eg.level1_to_level2('low_band', '2015', '317_00')
		o = eg.level1_to_level2('low_band', '2015', '318_00')
		o = eg.level1_to_level2('low_band', '2015', '319_00')

		o = eg.level1_to_level2('low_band', '2015', '320_00')
		o = eg.level1_to_level2('low_band', '2015', '321_00')
		o = eg.level1_to_level2('low_band', '2015', '322_00')
		o = eg.level1_to_level2('low_band', '2015', '323_00')
		o = eg.level1_to_level2('low_band', '2015', '324_00')
		o = eg.level1_to_level2('low_band', '2015', '325_00')
		o = eg.level1_to_level2('low_band', '2015', '326_00')
		o = eg.level1_to_level2('low_band', '2015', '327_00')
		o = eg.level1_to_level2('low_band', '2015', '328_00')
		o = eg.level1_to_level2('low_band', '2015', '329_00')

		o = eg.level1_to_level2('low_band', '2015', '330_00')
		o = eg.level1_to_level2('low_band', '2015', '331_00')
		o = eg.level1_to_level2('low_band', '2015', '332_00')
		o = eg.level1_to_level2('low_band', '2015', '333_00')
		o = eg.level1_to_level2('low_band', '2015', '334_00')
		o = eg.level1_to_level2('low_band', '2015', '335_00')
		o = eg.level1_to_level2('low_band', '2015', '336_00')
		o = eg.level1_to_level2('low_band', '2015', '337_00')
		o = eg.level1_to_level2('low_band', '2015', '338_00')
		o = eg.level1_to_level2('low_band', '2015', '339_00')

		o = eg.level1_to_level2('low_band', '2015', '340_00')
		o = eg.level1_to_level2('low_band', '2015', '341_00')
		o = eg.level1_to_level2('low_band', '2015', '342_00')
		o = eg.level1_to_level2('low_band', '2015', '343_14')
		o = eg.level1_to_level2('low_band', '2015', '344_00')
		o = eg.level1_to_level2('low_band', '2015', '344_21')
		o = eg.level1_to_level2('low_band', '2015', '345_00')
		o = eg.level1_to_level2('low_band', '2015', '346_00')

		o = eg.level1_to_level2('low_band', '2015', '347_00')
		o = eg.level1_to_level2('low_band', '2015', '348_00')
		o = eg.level1_to_level2('low_band', '2015', '349_00')

		o = eg.level1_to_level2('low_band', '2015', '350_00')
		o = eg.level1_to_level2('low_band', '2015', '351_00')
		o = eg.level1_to_level2('low_band', '2015', '352_00')
		o = eg.level1_to_level2('low_band', '2015', '353_00')
		o = eg.level1_to_level2('low_band', '2015', '354_00')

		o = eg.level1_to_level2('low_band', '2015', '362_00')
		o = eg.level1_to_level2('low_band', '2015', '363_00')
		o = eg.level1_to_level2('low_band', '2015', '364_00')
		o = eg.level1_to_level2('low_band', '2015', '365_00')

		o = eg.level1_to_level2('low_band', '2016', '001_00')
		o = eg.level1_to_level2('low_band', '2016', '002_00')
		o = eg.level1_to_level2('low_band', '2016', '003_00')
		o = eg.level1_to_level2('low_band', '2016', '004_00')
		o = eg.level1_to_level2('low_band', '2016', '005_00')
		o = eg.level1_to_level2('low_band', '2016', '006_00')
		o = eg.level1_to_level2('low_band', '2016', '007_00')
		o = eg.level1_to_level2('low_band', '2016', '008_00')
		o = eg.level1_to_level2('low_band', '2016', '009_00')

		o = eg.level1_to_level2('low_band', '2016', '010_00')
		o = eg.level1_to_level2('low_band', '2016', '011_00')
		o = eg.level1_to_level2('low_band', '2016', '012_00')
		o = eg.level1_to_level2('low_band', '2016', '013_00')
		o = eg.level1_to_level2('low_band', '2016', '014_00')
		o = eg.level1_to_level2('low_band', '2016', '015_00')
		o = eg.level1_to_level2('low_band', '2016', '016_00')
		o = eg.level1_to_level2('low_band', '2016', '017_00')
		o = eg.level1_to_level2('low_band', '2016', '018_00')
		o = eg.level1_to_level2('low_band', '2016', '019_00')

		o = eg.level1_to_level2('low_band', '2016', '020_00')
		o = eg.level1_to_level2('low_band', '2016', '028_00')
		o = eg.level1_to_level2('low_band', '2016', '029_00')

		o = eg.level1_to_level2('low_band', '2016', '030_00')
		o = eg.level1_to_level2('low_band', '2016', '031_00')
		o = eg.level1_to_level2('low_band', '2016', '032_00')
		o = eg.level1_to_level2('low_band', '2016', '033_00')
		o = eg.level1_to_level2('low_band', '2016', '034_00')
		o = eg.level1_to_level2('low_band', '2016', '035_00')
		o = eg.level1_to_level2('low_band', '2016', '036_00')
		o = eg.level1_to_level2('low_band', '2016', '037_00')
		o = eg.level1_to_level2('low_band', '2016', '038_00')
		o = eg.level1_to_level2('low_band', '2016', '039_00')

		o = eg.level1_to_level2('low_band', '2016', '040_00')
		o = eg.level1_to_level2('low_band', '2016', '041_00')
		o = eg.level1_to_level2('low_band', '2016', '042_00')
		o = eg.level1_to_level2('low_band', '2016', '043_00')
		o = eg.level1_to_level2('low_band', '2016', '044_00')
		o = eg.level1_to_level2('low_band', '2016', '045_00')
		o = eg.level1_to_level2('low_band', '2016', '046_00')
		o = eg.level1_to_level2('low_band', '2016', '047_00')
		o = eg.level1_to_level2('low_band', '2016', '048_00')
		o = eg.level1_to_level2('low_band', '2016', '049_00')

		o = eg.level1_to_level2('low_band', '2016', '050_00')
		o = eg.level1_to_level2('low_band', '2016', '051_00')
		o = eg.level1_to_level2('low_band', '2016', '052_00')
		o = eg.level1_to_level2('low_band', '2016', '053_00')
		o = eg.level1_to_level2('low_band', '2016', '055_21')
		o = eg.level1_to_level2('low_band', '2016', '056_00')
		o = eg.level1_to_level2('low_band', '2016', '057_00')
		o = eg.level1_to_level2('low_band', '2016', '058_00')
		o = eg.level1_to_level2('low_band', '2016', '059_00')

		o = eg.level1_to_level2('low_band', '2016', '060_00')
		o = eg.level1_to_level2('low_band', '2016', '061_00')
		o = eg.level1_to_level2('low_band', '2016', '062_00')
		o = eg.level1_to_level2('low_band', '2016', '063_00')
		o = eg.level1_to_level2('low_band', '2016', '064_00')
		o = eg.level1_to_level2('low_band', '2016', '065_00')
		o = eg.level1_to_level2('low_band', '2016', '066_00')
		o = eg.level1_to_level2('low_band', '2016', '067_00')
		o = eg.level1_to_level2('low_band', '2016', '068_00')
		o = eg.level1_to_level2('low_band', '2016', '069_00')

		o = eg.level1_to_level2('low_band', '2016', '070_00')
		o = eg.level1_to_level2('low_band', '2016', '071_00')
		o = eg.level1_to_level2('low_band', '2016', '072_00')
		o = eg.level1_to_level2('low_band', '2016', '073_00')
		o = eg.level1_to_level2('low_band', '2016', '074_00')
		o = eg.level1_to_level2('low_band', '2016', '075_00')
		o = eg.level1_to_level2('low_band', '2016', '076_00')
		o = eg.level1_to_level2('low_band', '2016', '077_00')
		o = eg.level1_to_level2('low_band', '2016', '078_00')
		o = eg.level1_to_level2('low_band', '2016', '079_00')

		o = eg.level1_to_level2('low_band', '2016', '080_00')
		o = eg.level1_to_level2('low_band', '2016', '081_00')
		o = eg.level1_to_level2('low_band', '2016', '082_00')

		o = eg.level1_to_level2('low_band', '2016', '083_00')
		o = eg.level1_to_level2('low_band', '2016', '084_00')
		o = eg.level1_to_level2('low_band', '2016', '085_00')
		o = eg.level1_to_level2('low_band', '2016', '086_00')
		o = eg.level1_to_level2('low_band', '2016', '087_00')
		o = eg.level1_to_level2('low_band', '2016', '088_00')
		o = eg.level1_to_level2('low_band', '2016', '089_00')

		o = eg.level1_to_level2('low_band', '2016', '090_00')
		o = eg.level1_to_level2('low_band', '2016', '091_00')
		o = eg.level1_to_level2('low_band', '2016', '092_00')
		o = eg.level1_to_level2('low_band', '2016', '093_00')
		o = eg.level1_to_level2('low_band', '2016', '094_00')
		o = eg.level1_to_level2('low_band', '2016', '095_00')
		o = eg.level1_to_level2('low_band', '2016', '096_00')
		o = eg.level1_to_level2('low_band', '2016', '097_00')
		o = eg.level1_to_level2('low_band', '2016', '098_00')
		o = eg.level1_to_level2('low_band', '2016', '099_00')
		o = eg.level1_to_level2('low_band', '2016', '100_00')

		o = eg.level1_to_level2('low_band', '2016', '101_00')
		o = eg.level1_to_level2('low_band', '2016', '102_00')
		o = eg.level1_to_level2('low_band', '2016', '103_00')
		o = eg.level1_to_level2('low_band', '2016', '104_00')
		#o = eg.level1_to_level2('low_band', '2016', '105_00')
		o = eg.level1_to_level2('low_band', '2016', '106_13')
		o = eg.level1_to_level2('low_band', '2016', '107_00')
		o = eg.level1_to_level2('low_band', '2016', '108_00')
		o = eg.level1_to_level2('low_band', '2016', '109_00')
		o = eg.level1_to_level2('low_band', '2016', '110_00')

		o = eg.level1_to_level2('low_band', '2016', '111_00')
		o = eg.level1_to_level2('low_band', '2016', '112_00')
		o = eg.level1_to_level2('low_band', '2016', '113_00')
		o = eg.level1_to_level2('low_band', '2016', '114_00')
		o = eg.level1_to_level2('low_band', '2016', '115_00')
		o = eg.level1_to_level2('low_band', '2016', '116_00')
		o = eg.level1_to_level2('low_band', '2016', '117_00')
		o = eg.level1_to_level2('low_band', '2016', '118_00')		

		o = eg.level1_to_level2('low_band', '2016', '122_16')
		o = eg.level1_to_level2('low_band', '2016', '123_00')
		o = eg.level1_to_level2('low_band', '2016', '124_00')
		o = eg.level1_to_level2('low_band', '2016', '125_00')
		o = eg.level1_to_level2('low_band', '2016', '126_00')
		o = eg.level1_to_level2('low_band', '2016', '127_00')		

		o = eg.level1_to_level2('low_band', '2016', '128_00')
		o = eg.level1_to_level2('low_band', '2016', '129_00')

		o = eg.level1_to_level2('low_band', '2016', '130_00')
		o = eg.level1_to_level2('low_band', '2016', '131_00')
		o = eg.level1_to_level2('low_band', '2016', '132_00')
		o = eg.level1_to_level2('low_band', '2016', '133_00')
		o = eg.level1_to_level2('low_band', '2016', '134_00')
		o = eg.level1_to_level2('low_band', '2016', '135_00')
		o = eg.level1_to_level2('low_band', '2016', '136_00')
		o = eg.level1_to_level2('low_band', '2016', '137_00')
		o = eg.level1_to_level2('low_band', '2016', '138_00')
		o = eg.level1_to_level2('low_band', '2016', '139_00')

		o = eg.level1_to_level2('low_band', '2016', '140_00')
		o = eg.level1_to_level2('low_band', '2016', '141_00')
		o = eg.level1_to_level2('low_band', '2016', '142_00')
		o = eg.level1_to_level2('low_band', '2016', '143_00')
		o = eg.level1_to_level2('low_band', '2016', '144_00')
		o = eg.level1_to_level2('low_band', '2016', '145_00')
		o = eg.level1_to_level2('low_band', '2016', '146_00')
		o = eg.level1_to_level2('low_band', '2016', '147_00')
		o = eg.level1_to_level2('low_band', '2016', '148_00')
		o = eg.level1_to_level2('low_band', '2016', '149_00')

		o = eg.level1_to_level2('low_band', '2016', '150_00')
		o = eg.level1_to_level2('low_band', '2016', '151_00')
		o = eg.level1_to_level2('low_band', '2016', '152_00')
		o = eg.level1_to_level2('low_band', '2016', '153_00')
		o = eg.level1_to_level2('low_band', '2016', '154_00')
		o = eg.level1_to_level2('low_band', '2016', '155_00')
		o = eg.level1_to_level2('low_band', '2016', '156_00')
		o = eg.level1_to_level2('low_band', '2016', '157_00')
		o = eg.level1_to_level2('low_band', '2016', '158_00')
		o = eg.level1_to_level2('low_band', '2016', '159_00')

		o = eg.level1_to_level2('low_band', '2016', '160_00')
		#o = eg.level1_to_level2('low_band', '2016', '165_14')
		#o = eg.level1_to_level2('low_band', '2016', '166_00')
		#o = eg.level1_to_level2('low_band', '2016', '166_14')
		o = eg.level1_to_level2('low_band', '2016', '167_00')
		o = eg.level1_to_level2('low_band', '2016', '168_00')
		o = eg.level1_to_level2('low_band', '2016', '169_00')

		o = eg.level1_to_level2('low_band', '2016', '170_00')
		o = eg.level1_to_level2('low_band', '2016', '171_00')
		o = eg.level1_to_level2('low_band', '2016', '172_00')
		o = eg.level1_to_level2('low_band', '2016', '173_00')

		o = eg.level1_to_level2('low_band', '2016', '180_15')
		o = eg.level1_to_level2('low_band', '2016', '181_00')
		o = eg.level1_to_level2('low_band', '2016', '182_00')
		o = eg.level1_to_level2('low_band', '2016', '183_00')
		o = eg.level1_to_level2('low_band', '2016', '184_00')
		o = eg.level1_to_level2('low_band', '2016', '185_00')
		o = eg.level1_to_level2('low_band', '2016', '186_00')
		o = eg.level1_to_level2('low_band', '2016', '187_00')
		o = eg.level1_to_level2('low_band', '2016', '188_00')
		o = eg.level1_to_level2('low_band', '2016', '189_00')

		o = eg.level1_to_level2('low_band', '2016', '190_00')
		o = eg.level1_to_level2('low_band', '2016', '191_00')
		o = eg.level1_to_level2('low_band', '2016', '192_00')
		o = eg.level1_to_level2('low_band', '2016', '193_00')
		o = eg.level1_to_level2('low_band', '2016', '194_00')
		o = eg.level1_to_level2('low_band', '2016', '195_00')
		o = eg.level1_to_level2('low_band', '2016', '196_00')
		o = eg.level1_to_level2('low_band', '2016', '197_00')
		o = eg.level1_to_level2('low_band', '2016', '198_00')
		o = eg.level1_to_level2('low_band', '2016', '199_00')		

		o = eg.level1_to_level2('low_band', '2016', '200_00')
		o = eg.level1_to_level2('low_band', '2016', '201_00')
		o = eg.level1_to_level2('low_band', '2016', '202_00')
		o = eg.level1_to_level2('low_band', '2016', '203_00')
		o = eg.level1_to_level2('low_band', '2016', '204_00')
		o = eg.level1_to_level2('low_band', '2016', '210_14')
		o = eg.level1_to_level2('low_band', '2016', '211_00')
		o = eg.level1_to_level2('low_band', '2016', '212_00')

		o = eg.level1_to_level2('low_band', '2016', '217_00')
		o = eg.level1_to_level2('low_band', '2016', '218_00')
		o = eg.level1_to_level2('low_band', '2016', '219_00')
		o = eg.level1_to_level2('low_band', '2016', '220_00')

		o = eg.level1_to_level2('low_band', '2016', '226_19')
		o = eg.level1_to_level2('low_band', '2016', '227_00')
		o = eg.level1_to_level2('low_band', '2016', '228_00')
		o = eg.level1_to_level2('low_band', '2016', '229_00')

		o = eg.level1_to_level2('low_band', '2016', '230_00')

		o = eg.level1_to_level2('low_band', '2016', '238_00')

		o = eg.level1_to_level2('low_band', '2016', '246_07')
		o = eg.level1_to_level2('low_band', '2016', '247_00')
		o = eg.level1_to_level2('low_band', '2016', '248_00')
		o = eg.level1_to_level2('low_band', '2016', '249_00')

		o = eg.level1_to_level2('low_band', '2016', '250_02')
		o = eg.level1_to_level2('low_band', '2016', '251_00')
		o = eg.level1_to_level2('low_band', '2016', '252_00')
		o = eg.level1_to_level2('low_band', '2016', '253_13')
		o = eg.level1_to_level2('low_band', '2016', '254_00')









	if set_number == 2:

		# Low Band with NEW GOOD SWITCH and EXTENDED GROUND PLANE
		#-------------------------------------------------------		
		o = eg.level1_to_level2('low_band', '2016', '258_13')
		o = eg.level1_to_level2('low_band', '2016', '259_00')

		o = eg.level1_to_level2('low_band', '2016', '260_00')
		o = eg.level1_to_level2('low_band', '2016', '261_00')
		o = eg.level1_to_level2('low_band', '2016', '262_00')
		o = eg.level1_to_level2('low_band', '2016', '263_00')
		o = eg.level1_to_level2('low_band', '2016', '264_00')
		o = eg.level1_to_level2('low_band', '2016', '265_00')
		o = eg.level1_to_level2('low_band', '2016', '266_00')
		o = eg.level1_to_level2('low_band', '2016', '267_00')
		o = eg.level1_to_level2('low_band', '2016', '268_00')
		o = eg.level1_to_level2('low_band', '2016', '269_00')

		o = eg.level1_to_level2('low_band', '2016', '270_00')
		o = eg.level1_to_level2('low_band', '2016', '271_00')
		o = eg.level1_to_level2('low_band', '2016', '273_15')
		o = eg.level1_to_level2('low_band', '2016', '274_00')
		o = eg.level1_to_level2('low_band', '2016', '275_00')
		o = eg.level1_to_level2('low_band', '2016', '276_00')
		o = eg.level1_to_level2('low_band', '2016', '277_00')
		o = eg.level1_to_level2('low_band', '2016', '278_00')
		o = eg.level1_to_level2('low_band', '2016', '279_00')

		o = eg.level1_to_level2('low_band', '2016', '280_00')
		o = eg.level1_to_level2('low_band', '2016', '281_00')
		o = eg.level1_to_level2('low_band', '2016', '282_00')
		o = eg.level1_to_level2('low_band', '2016', '283_00')
		o = eg.level1_to_level2('low_band', '2016', '284_00')
		o = eg.level1_to_level2('low_band', '2016', '285_00')
		o = eg.level1_to_level2('low_band', '2016', '286_00')
		o = eg.level1_to_level2('low_band', '2016', '287_00')
		o = eg.level1_to_level2('low_band', '2016', '288_00')
		o = eg.level1_to_level2('low_band', '2016', '289_00')

		o = eg.level1_to_level2('low_band', '2016', '290_00')
		o = eg.level1_to_level2('low_band', '2016', '291_00')
		o = eg.level1_to_level2('low_band', '2016', '292_00')
		o = eg.level1_to_level2('low_band', '2016', '293_00')
		o = eg.level1_to_level2('low_band', '2016', '294_00')
		o = eg.level1_to_level2('low_band', '2016', '295_00')
		o = eg.level1_to_level2('low_band', '2016', '296_00')
		o = eg.level1_to_level2('low_band', '2016', '297_00')
		o = eg.level1_to_level2('low_band', '2016', '298_00')
		o = eg.level1_to_level2('low_band', '2016', '299_00')

		o = eg.level1_to_level2('low_band', '2016', '302_14')
		o = eg.level1_to_level2('low_band', '2016', '303_00')
		o = eg.level1_to_level2('low_band', '2016', '304_00')
		o = eg.level1_to_level2('low_band', '2016', '305_00')

		o = eg.level1_to_level2('low_band', '2016', '314_15')
		o = eg.level1_to_level2('low_band', '2016', '315_00')
		o = eg.level1_to_level2('low_band', '2016', '316_00')
		o = eg.level1_to_level2('low_band', '2016', '317_00')
		o = eg.level1_to_level2('low_band', '2016', '318_00')
		o = eg.level1_to_level2('low_band', '2016', '319_00')

		o = eg.level1_to_level2('low_band', '2016', '320_00')
		o = eg.level1_to_level2('low_band', '2016', '321_00')
		o = eg.level1_to_level2('low_band', '2016', '322_00')
		o = eg.level1_to_level2('low_band', '2016', '323_00')
		o = eg.level1_to_level2('low_band', '2016', '324_00')
		o = eg.level1_to_level2('low_band', '2016', '325_00')
		o = eg.level1_to_level2('low_band', '2016', '326_00')
		o = eg.level1_to_level2('low_band', '2016', '327_00')
		o = eg.level1_to_level2('low_band', '2016', '328_00')
		o = eg.level1_to_level2('low_band', '2016', '329_00')

		o = eg.level1_to_level2('low_band', '2016', '330_00')
		o = eg.level1_to_level2('low_band', '2016', '331_00')
		o = eg.level1_to_level2('low_band', '2016', '332_00')
		o = eg.level1_to_level2('low_band', '2016', '333_00')
		o = eg.level1_to_level2('low_band', '2016', '334_00')
		o = eg.level1_to_level2('low_band', '2016', '335_00')
		o = eg.level1_to_level2('low_band', '2016', '336_00')
		o = eg.level1_to_level2('low_band', '2016', '337_00')
		o = eg.level1_to_level2('low_band', '2016', '338_00')
		o = eg.level1_to_level2('low_band', '2016', '339_00')

		o = eg.level1_to_level2('low_band', '2016', '340_00')
		o = eg.level1_to_level2('low_band', '2016', '341_00')
		o = eg.level1_to_level2('low_band', '2016', '342_00')
		o = eg.level1_to_level2('low_band', '2016', '343_00')
		o = eg.level1_to_level2('low_band', '2016', '344_00')
		o = eg.level1_to_level2('low_band', '2016', '345_00')
		o = eg.level1_to_level2('low_band', '2016', '346_00')
		o = eg.level1_to_level2('low_band', '2016', '347_00')
		o = eg.level1_to_level2('low_band', '2016', '348_00')
		o = eg.level1_to_level2('low_band', '2016', '349_00')

		o = eg.level1_to_level2('low_band', '2016', '350_00')
		o = eg.level1_to_level2('low_band', '2016', '351_00')
		o = eg.level1_to_level2('low_band', '2016', '352_00')
		o = eg.level1_to_level2('low_band', '2016', '353_00')
		o = eg.level1_to_level2('low_band', '2016', '354_00')
		o = eg.level1_to_level2('low_band', '2016', '355_00')
		o = eg.level1_to_level2('low_band', '2016', '356_00')
		o = eg.level1_to_level2('low_band', '2016', '356_06')
		o = eg.level1_to_level2('low_band', '2016', '357_00')
		o = eg.level1_to_level2('low_band', '2016', '357_07')
		o = eg.level1_to_level2('low_band', '2016', '358_00')
		o = eg.level1_to_level2('low_band', '2016', '359_00')

		o = eg.level1_to_level2('low_band', '2016', '360_00')
		o = eg.level1_to_level2('low_band', '2016', '361_00')
		o = eg.level1_to_level2('low_band', '2016', '362_00')
		o = eg.level1_to_level2('low_band', '2016', '363_00')
		o = eg.level1_to_level2('low_band', '2016', '364_00')
		o = eg.level1_to_level2('low_band', '2016', '365_00')
		o = eg.level1_to_level2('low_band', '2016', '366_00')

		o = eg.level1_to_level2('low_band', '2017', '001_15')
		o = eg.level1_to_level2('low_band', '2017', '002_00')
		o = eg.level1_to_level2('low_band', '2017', '003_00')
		o = eg.level1_to_level2('low_band', '2017', '005_00')
		o = eg.level1_to_level2('low_band', '2017', '006_00')
		o = eg.level1_to_level2('low_band', '2017', '007_00')
		o = eg.level1_to_level2('low_band', '2017', '008_00')
		o = eg.level1_to_level2('low_band', '2017', '009_00')

		o = eg.level1_to_level2('low_band', '2017', '010_00')
		o = eg.level1_to_level2('low_band', '2017', '011_07')
		o = eg.level1_to_level2('low_band', '2017', '012_00')
		o = eg.level1_to_level2('low_band', '2017', '013_00')
		o = eg.level1_to_level2('low_band', '2017', '014_00')
		o = eg.level1_to_level2('low_band', '2017', '015_00')
		o = eg.level1_to_level2('low_band', '2017', '016_00')
		o = eg.level1_to_level2('low_band', '2017', '017_00')
		o = eg.level1_to_level2('low_band', '2017', '018_00')
		o = eg.level1_to_level2('low_band', '2017', '019_00')

		o = eg.level1_to_level2('low_band', '2017', '023_00')

		o = eg.level1_to_level2('low_band', '2017', '077_07')
		o = eg.level1_to_level2('low_band', '2017', '078_00')
		o = eg.level1_to_level2('low_band', '2017', '079_00')

		o = eg.level1_to_level2('low_band', '2017', '080_00')
		o = eg.level1_to_level2('low_band', '2017', '081_00')
		o = eg.level1_to_level2('low_band', '2017', '081_12')
		o = eg.level1_to_level2('low_band', '2017', '082_00')
		o = eg.level1_to_level2('low_band', '2017', '082_08')
		o = eg.level1_to_level2('low_band', '2017', '083_00')
		o = eg.level1_to_level2('low_band', '2017', '084_00')
		o = eg.level1_to_level2('low_band', '2017', '085_00')
		o = eg.level1_to_level2('low_band', '2017', '086_00')
		o = eg.level1_to_level2('low_band', '2017', '087_00')
		o = eg.level1_to_level2('low_band', '2017', '087_21')
		o = eg.level1_to_level2('low_band', '2017', '088_00')
		o = eg.level1_to_level2('low_band', '2017', '089_00')

		o = eg.level1_to_level2('low_band', '2017', '090_00')
		o = eg.level1_to_level2('low_band', '2017', '091_00')

		o = eg.level1_to_level2('low_band', '2017', '092_00')
		o = eg.level1_to_level2('low_band', '2017', '093_00')
		o = eg.level1_to_level2('low_band', '2017', '093_17')
		o = eg.level1_to_level2('low_band', '2017', '094_00')
		o = eg.level1_to_level2('low_band', '2017', '095_00')
		o = eg.level1_to_level2('low_band', '2017', '095_15')

		o = eg.level1_to_level2('low_band', '2017', '153_12')
		o = eg.level1_to_level2('low_band', '2017', '154_00')
		o = eg.level1_to_level2('low_band', '2017', '155_00')
		o = eg.level1_to_level2('low_band', '2017', '156_00')
	
		o = eg.level1_to_level2('low_band', '2017', '157_00')
		o = eg.level1_to_level2('low_band', '2017', '158_03')
		o = eg.level1_to_level2('low_band', '2017', '159_00')
		o = eg.level1_to_level2('low_band', '2017', '160_00')
		o = eg.level1_to_level2('low_band', '2017', '161_00')
	
		o = eg.level1_to_level2('low_band', '2017', '162_00')
		o = eg.level1_to_level2('low_band', '2017', '163_00')
		o = eg.level1_to_level2('low_band', '2017', '164_00')
		o = eg.level1_to_level2('low_band', '2017', '165_00')
		o = eg.level1_to_level2('low_band', '2017', '166_00')
		o = eg.level1_to_level2('low_band', '2017', '167_00')

		o = eg.level1_to_level2('low_band', '2017', '168_00')
		o = eg.level1_to_level2('low_band', '2017', '169_00')
		o = eg.level1_to_level2('low_band', '2017', '170_00')
		o = eg.level1_to_level2('low_band', '2017', '171_00')



	return 1





























def batch_low_band2_level1_to_level2(set_number):


	# NS with shield
	if set_number == 1:
			
		o = eg.level1_to_level2('low_band2', '2017', '082_03', low2_flag='')
		o = eg.level1_to_level2('low_band2', '2017', '082_08', low2_flag='')
		o = eg.level1_to_level2('low_band2', '2017', '083_00', low2_flag='')
		o = eg.level1_to_level2('low_band2', '2017', '084_00')
		o = eg.level1_to_level2('low_band2', '2017', '085_00')
		o = eg.level1_to_level2('low_band2', '2017', '086_00')
		o = eg.level1_to_level2('low_band2', '2017', '086_14')
		o = eg.level1_to_level2('low_band2', '2017', '087_00')
		o = eg.level1_to_level2('low_band2', '2017', '087_21')
		o = eg.level1_to_level2('low_band2', '2017', '088_00')
		o = eg.level1_to_level2('low_band2', '2017', '089_00')
		o = eg.level1_to_level2('low_band2', '2017', '090_00')
		o = eg.level1_to_level2('low_band2', '2017', '091_00')
		o = eg.level1_to_level2('low_band2', '2017', '092_00')
		o = eg.level1_to_level2('low_band2', '2017', '093_00')
		o = eg.level1_to_level2('low_band2', '2017', '093_17')
		o = eg.level1_to_level2('low_band2', '2017', '094_00')
		o = eg.level1_to_level2('low_band2', '2017', '095_00')
		o = eg.level1_to_level2('low_band2', '2017', '096_00')
		o = eg.level1_to_level2('low_band2', '2017', '097_00')
		o = eg.level1_to_level2('low_band2', '2017', '098_00')
		o = eg.level1_to_level2('low_band2', '2017', '099_00')	
		o = eg.level1_to_level2('low_band2', '2017', '100_00')
		o = eg.level1_to_level2('low_band2', '2017', '101_00')
		o = eg.level1_to_level2('low_band2', '2017', '102_00')
		o = eg.level1_to_level2('low_band2', '2017', '102_15')
		o = eg.level1_to_level2('low_band2', '2017', '103_00')
		o = eg.level1_to_level2('low_band2', '2017', '103_15')
		o = eg.level1_to_level2('low_band2', '2017', '104_00')
		o = eg.level1_to_level2('low_band2', '2017', '105_00')
		o = eg.level1_to_level2('low_band2', '2017', '106_00')
		o = eg.level1_to_level2('low_band2', '2017', '107_00')
		o = eg.level1_to_level2('low_band2', '2017', '108_00')
		o = eg.level1_to_level2('low_band2', '2017', '109_00')
		o = eg.level1_to_level2('low_band2', '2017', '110_00')
		o = eg.level1_to_level2('low_band2', '2017', '111_00')
		o = eg.level1_to_level2('low_band2', '2017', '112_00')
		o = eg.level1_to_level2('low_band2', '2017', '113_00')
		o = eg.level1_to_level2('low_band2', '2017', '114_00')
		o = eg.level1_to_level2('low_band2', '2017', '115_00')
		o = eg.level1_to_level2('low_band2', '2017', '116_00')
		o = eg.level1_to_level2('low_band2', '2017', '117_00')
		o = eg.level1_to_level2('low_band2', '2017', '117_16')
		o = eg.level1_to_level2('low_band2', '2017', '118_00')
		o = eg.level1_to_level2('low_band2', '2017', '119_00')
		o = eg.level1_to_level2('low_band2', '2017', '120_00')
		o = eg.level1_to_level2('low_band2', '2017', '121_00')
		o = eg.level1_to_level2('low_band2', '2017', '122_00')
		o = eg.level1_to_level2('low_band2', '2017', '123_00')
		o = eg.level1_to_level2('low_band2', '2017', '124_00')
		o = eg.level1_to_level2('low_band2', '2017', '125_00')
		o = eg.level1_to_level2('low_band2', '2017', '126_00')
		o = eg.level1_to_level2('low_band2', '2017', '127_00')
		o = eg.level1_to_level2('low_band2', '2017', '128_00')
		o = eg.level1_to_level2('low_band2', '2017', '129_00')
		o = eg.level1_to_level2('low_band2', '2017', '130_00')
		o = eg.level1_to_level2('low_band2', '2017', '131_00')
		o = eg.level1_to_level2('low_band2', '2017', '132_00')
		o = eg.level1_to_level2('low_band2', '2017', '133_00')
		o = eg.level1_to_level2('low_band2', '2017', '134_00')
		o = eg.level1_to_level2('low_band2', '2017', '135_00')
		o = eg.level1_to_level2('low_band2', '2017', '136_00')
		o = eg.level1_to_level2('low_band2', '2017', '137_00')
		o = eg.level1_to_level2('low_band2', '2017', '138_00')
		o = eg.level1_to_level2('low_band2', '2017', '139_00')
		o = eg.level1_to_level2('low_band2', '2017', '140_00')
		o = eg.level1_to_level2('low_band2', '2017', '141_00')
		o = eg.level1_to_level2('low_band2', '2017', '142_00')		




	# Rotation of antenna to EW
	if set_number == 2:
		
		o = eg.level1_to_level2('low_band2', '2017', '154_00')
		o = eg.level1_to_level2('low_band2', '2017', '155_00')
		o = eg.level1_to_level2('low_band2', '2017', '156_00')
		o = eg.level1_to_level2('low_band2', '2017', '157_01')
		o = eg.level1_to_level2('low_band2', '2017', '158_03')
		o = eg.level1_to_level2('low_band2', '2017', '159_00')
		o = eg.level1_to_level2('low_band2', '2017', '160_00')
		o = eg.level1_to_level2('low_band2', '2017', '161_00')
		o = eg.level1_to_level2('low_band2', '2017', '162_00')
		o = eg.level1_to_level2('low_band2', '2017', '163_00')
		o = eg.level1_to_level2('low_band2', '2017', '164_00')
		o = eg.level1_to_level2('low_band2', '2017', '165_00')
		o = eg.level1_to_level2('low_band2', '2017', '166_00')
		o = eg.level1_to_level2('low_band2', '2017', '167_00')
		o = eg.level1_to_level2('low_band2', '2017', '168_00')
		o = eg.level1_to_level2('low_band2', '2017', '169_00')
		o = eg.level1_to_level2('low_band2', '2017', '170_00')
		o = eg.level1_to_level2('low_band2', '2017', '171_00')






	# Removing the Balun Shield
	if set_number == 3:
		o = eg.level1_to_level2('low_band2', '2017', '181_00')
		o = eg.level1_to_level2('low_band2', '2017', '182_00')
		o = eg.level1_to_level2('low_band2', '2017', '183_00')
		o = eg.level1_to_level2('low_band2', '2017', '184_00')
		o = eg.level1_to_level2('low_band2', '2017', '184_17')
		o = eg.level1_to_level2('low_band2', '2017', '185_00')
		o = eg.level1_to_level2('low_band2', '2017', '186_00')
		o = eg.level1_to_level2('low_band2', '2017', '187_00')
		o = eg.level1_to_level2('low_band2', '2017', '188_00')
		o = eg.level1_to_level2('low_band2', '2017', '189_00')
		o = eg.level1_to_level2('low_band2', '2017', '190_00')
		o = eg.level1_to_level2('low_band2', '2017', '191_00')
		o = eg.level1_to_level2('low_band2', '2017', '192_00')
		o = eg.level1_to_level2('low_band2', '2017', '193_00')
		o = eg.level1_to_level2('low_band2', '2017', '194_00')
		o = eg.level1_to_level2('low_band2', '2017', '195_00')
		o = eg.level1_to_level2('low_band2', '2017', '196_00')
		o = eg.level1_to_level2('low_band2', '2017', '197_00')
		o = eg.level1_to_level2('low_band2', '2017', '198_00')
		o = eg.level1_to_level2('low_band2', '2017', '199_00')
		o = eg.level1_to_level2('low_band2', '2017', '200_00')
		o = eg.level1_to_level2('low_band2', '2017', '201_00')
		o = eg.level1_to_level2('low_band2', '2017', '202_00')
		o = eg.level1_to_level2('low_band2', '2017', '203_00')
		o = eg.level1_to_level2('low_band2', '2017', '204_00')
		o = eg.level1_to_level2('low_band2', '2017', '205_00')
		o = eg.level1_to_level2('low_band2', '2017', '206_00')
		o = eg.level1_to_level2('low_band2', '2017', '207_00')
		o = eg.level1_to_level2('low_band2', '2017', '208_00')
		o = eg.level1_to_level2('low_band2', '2017', '209_00')
		o = eg.level1_to_level2('low_band2', '2017', '210_00')
		o = eg.level1_to_level2('low_band2', '2017', '211_00')
		o = eg.level1_to_level2('low_band2', '2017', '212_00')
		o = eg.level1_to_level2('low_band2', '2017', '213_00')
		o = eg.level1_to_level2('low_band2', '2017', '214_00')
		o = eg.level1_to_level2('low_band2', '2017', '215_00')
		o = eg.level1_to_level2('low_band2', '2017', '216_00')
		o = eg.level1_to_level2('low_band2', '2017', '217_00')
		o = eg.level1_to_level2('low_band2', '2017', '218_16')
		o = eg.level1_to_level2('low_band2', '2017', '219_00')
		o = eg.level1_to_level2('low_band2', '2017', '220_00')
		o = eg.level1_to_level2('low_band2', '2017', '221_00')
		o = eg.level1_to_level2('low_band2', '2017', '222_00')
		o = eg.level1_to_level2('low_band2', '2017', '223_00')
		o = eg.level1_to_level2('low_band2', '2017', '224_00')
		o = eg.level1_to_level2('low_band2', '2017', '225_00')
		o = eg.level1_to_level2('low_band2', '2017', '226_00')
		o = eg.level1_to_level2('low_band2', '2017', '227_00')
		o = eg.level1_to_level2('low_band2', '2017', '228_00')
		o = eg.level1_to_level2('low_band2', '2017', '229_00')
		o = eg.level1_to_level2('low_band2', '2017', '230_00')
		o = eg.level1_to_level2('low_band2', '2017', '231_00')
		o = eg.level1_to_level2('low_band2', '2017', '232_00')
		o = eg.level1_to_level2('low_band2', '2017', '233_00')
		o = eg.level1_to_level2('low_band2', '2017', '234_00')
		o = eg.level1_to_level2('low_band2', '2017', '235_00')
		o = eg.level1_to_level2('low_band2', '2017', '236_00')
		o = eg.level1_to_level2('low_band2', '2017', '237_00')
		o = eg.level1_to_level2('low_band2', '2017', '238_00')
		o = eg.level1_to_level2('low_band2', '2017', '239_00')
		o = eg.level1_to_level2('low_band2', '2017', '240_00')
		o = eg.level1_to_level2('low_band2', '2017', '241_00')
		o = eg.level1_to_level2('low_band2', '2017', '242_00')
		o = eg.level1_to_level2('low_band2', '2017', '243_00')
		o = eg.level1_to_level2('low_band2', '2017', '244_00')
		o = eg.level1_to_level2('low_band2', '2017', '245_00')
		o = eg.level1_to_level2('low_band2', '2017', '246_00')
		o = eg.level1_to_level2('low_band2', '2017', '247_00')
		o = eg.level1_to_level2('low_band2', '2017', '248_00')
		o = eg.level1_to_level2('low_band2', '2017', '249_00')
		o = eg.level1_to_level2('low_band2', '2017', '250_00')
		o = eg.level1_to_level2('low_band2', '2017', '251_00')
		o = eg.level1_to_level2('low_band2', '2017', '252_00')
		o = eg.level1_to_level2('low_band2', '2017', '253_00')
		o = eg.level1_to_level2('low_band2', '2017', '254_00')
		o = eg.level1_to_level2('low_band2', '2017', '255_00')
		o = eg.level1_to_level2('low_band2', '2017', '256_00')
		o = eg.level1_to_level2('low_band2', '2017', '257_00')
		o = eg.level1_to_level2('low_band2', '2017', '258_00')
		o = eg.level1_to_level2('low_band2', '2017', '259_00')
		o = eg.level1_to_level2('low_band2', '2017', '260_00')
		o = eg.level1_to_level2('low_band2', '2017', '261_00')
		o = eg.level1_to_level2('low_band2', '2017', '262_00')
		o = eg.level1_to_level2('low_band2', '2017', '263_00')
		o = eg.level1_to_level2('low_band2', '2017', '264_00')
		o = eg.level1_to_level2('low_band2', '2017', '265_00')
		o = eg.level1_to_level2('low_band2', '2017', '266_00')
		o = eg.level1_to_level2('low_band2', '2017', '267_00')
		o = eg.level1_to_level2('low_band2', '2017', '268_00')
		o = eg.level1_to_level2('low_band2', '2017', '269_00')
		o = eg.level1_to_level2('low_band2', '2017', '270_00')
		o = eg.level1_to_level2('low_band2', '2017', '271_00')
		o = eg.level1_to_level2('low_band2', '2017', '272_00')
		o = eg.level1_to_level2('low_band2', '2017', '273_00')
		o = eg.level1_to_level2('low_band2', '2017', '274_00')
		o = eg.level1_to_level2('low_band2', '2017', '275_00')
		o = eg.level1_to_level2('low_band2', '2017', '276_00')
		o = eg.level1_to_level2('low_band2', '2017', '277_00')
		o = eg.level1_to_level2('low_band2', '2017', '278_00')
		o = eg.level1_to_level2('low_band2', '2017', '279_00')
		o = eg.level1_to_level2('low_band2', '2017', '280_00')
		o = eg.level1_to_level2('low_band2', '2017', '281_00')
		o = eg.level1_to_level2('low_band2', '2017', '282_00')
		o = eg.level1_to_level2('low_band2', '2017', '283_00')
		o = eg.level1_to_level2('low_band2', '2017', '284_00')
		o = eg.level1_to_level2('low_band2', '2017', '285_00')
		o = eg.level1_to_level2('low_band2', '2017', '286_00')
		o = eg.level1_to_level2('low_band2', '2017', '287_00')
		o = eg.level1_to_level2('low_band2', '2017', '288_00')
		o = eg.level1_to_level2('low_band2', '2017', '289_00')
		o = eg.level1_to_level2('low_band2', '2017', '290_00')
		o = eg.level1_to_level2('low_band2', '2017', '291_00')
		o = eg.level1_to_level2('low_band2', '2017', '291_21')
		o = eg.level1_to_level2('low_band2', '2017', '292_00')
		o = eg.level1_to_level2('low_band2', '2017', '293_00')
		o = eg.level1_to_level2('low_band2', '2017', '294_00')
		o = eg.level1_to_level2('low_band2', '2017', '295_00')
		o = eg.level1_to_level2('low_band2', '2017', '296_00')
		o = eg.level1_to_level2('low_band2', '2017', '297_00')
		o = eg.level1_to_level2('low_band2', '2017', '298_00')	
		o = eg.level1_to_level2('low_band2', '2017', '300_18')
		o = eg.level1_to_level2('low_band2', '2017', '301_00')
		o = eg.level1_to_level2('low_band2', '2017', '302_00')
		o = eg.level1_to_level2('low_band2', '2017', '303_00')
		o = eg.level1_to_level2('low_band2', '2017', '310_04')
		o = eg.level1_to_level2('low_band2', '2017', '311_00')
		o = eg.level1_to_level2('low_band2', '2017', '312_00')
		o = eg.level1_to_level2('low_band2', '2017', '313_00')
		o = eg.level1_to_level2('low_band2', '2017', '314_19')
		o = eg.level1_to_level2('low_band2', '2017', '315_00')
		o = eg.level1_to_level2('low_band2', '2017', '316_00')
		o = eg.level1_to_level2('low_band2', '2017', '317_00')
		# o = eg.level1_to_level2('low_band2', '2017', '318_00')
		# o = eg.level1_to_level2('low_band2', '2017', '331_00')
		o = eg.level1_to_level2('low_band2', '2017', '332_04')
		o = eg.level1_to_level2('low_band2', '2017', '333_00')
		o = eg.level1_to_level2('low_band2', '2017', '334_00')
		o = eg.level1_to_level2('low_band2', '2017', '335_00')
		o = eg.level1_to_level2('low_band2', '2017', '336_00')
		o = eg.level1_to_level2('low_band2', '2017', '337_20')
		o = eg.level1_to_level2('low_band2', '2017', '338_00')
		o = eg.level1_to_level2('low_band2', '2017', '339_03')
		o = eg.level1_to_level2('low_band2', '2017', '340_00')
		o = eg.level1_to_level2('low_band2', '2017', '341_00')
		o = eg.level1_to_level2('low_band2', '2017', '342_00')
		o = eg.level1_to_level2('low_band2', '2017', '343_00')
		o = eg.level1_to_level2('low_band2', '2017', '344_00')
		o = eg.level1_to_level2('low_band2', '2017', '345_00')
		o = eg.level1_to_level2('low_band2', '2017', '346_00')
		o = eg.level1_to_level2('low_band2', '2017', '347_00')
		o = eg.level1_to_level2('low_band2', '2017', '348_00')
		o = eg.level1_to_level2('low_band2', '2017', '349_00')
		o = eg.level1_to_level2('low_band2', '2017', '350_00')
		o = eg.level1_to_level2('low_band2', '2017', '351_00')
		o = eg.level1_to_level2('low_band2', '2017', '352_00')
		o = eg.level1_to_level2('low_band2', '2017', '353_00')
		o = eg.level1_to_level2('low_band2', '2017', '354_00')
		o = eg.level1_to_level2('low_band2', '2017', '355_00')
		o = eg.level1_to_level2('low_band2', '2017', '356_00')
		o = eg.level1_to_level2('low_band2', '2017', '357_00')
		o = eg.level1_to_level2('low_band2', '2017', '358_00')
		o = eg.level1_to_level2('low_band2', '2017', '359_00')	
		o = eg.level1_to_level2('low_band2', '2017', '360_00')
		o = eg.level1_to_level2('low_band2', '2017', '361_00')
		o = eg.level1_to_level2('low_band2', '2017', '362_00')
		o = eg.level1_to_level2('low_band2', '2017', '363_00')
		o = eg.level1_to_level2('low_band2', '2017', '364_00')
		o = eg.level1_to_level2('low_band2', '2017', '365_00')
		

	if set_number == 4:
		o = eg.level1_to_level2('low_band2', '2018', '001_00')
		o = eg.level1_to_level2('low_band2', '2018', '002_00')
		o = eg.level1_to_level2('low_band2', '2018', '003_00')
		o = eg.level1_to_level2('low_band2', '2018', '004_00')
		o = eg.level1_to_level2('low_band2', '2018', '005_00')
		o = eg.level1_to_level2('low_band2', '2018', '006_00')
		o = eg.level1_to_level2('low_band2', '2018', '007_00')
		o = eg.level1_to_level2('low_band2', '2018', '008_00')
		o = eg.level1_to_level2('low_band2', '2018', '009_00')
		o = eg.level1_to_level2('low_band2', '2018', '010_00')
		o = eg.level1_to_level2('low_band2', '2018', '011_00')
		o = eg.level1_to_level2('low_band2', '2018', '012_00')
		o = eg.level1_to_level2('low_band2', '2018', '013_00')
		o = eg.level1_to_level2('low_band2', '2018', '014_00')
		o = eg.level1_to_level2('low_band2', '2018', '015_00')
		o = eg.level1_to_level2('low_band2', '2018', '016_00')
		o = eg.level1_to_level2('low_band2', '2018', '017_00')
		o = eg.level1_to_level2('low_band2', '2018', '018_00')
		o = eg.level1_to_level2('low_band2', '2018', '019_00')
		o = eg.level1_to_level2('low_band2', '2018', '020_00')
		o = eg.level1_to_level2('low_band2', '2018', '021_00')
		o = eg.level1_to_level2('low_band2', '2018', '022_00')
		o = eg.level1_to_level2('low_band2', '2018', '023_00')
		o = eg.level1_to_level2('low_band2', '2018', '024_00')
		o = eg.level1_to_level2('low_band2', '2018', '025_00')
		o = eg.level1_to_level2('low_band2', '2018', '026_00')
		o = eg.level1_to_level2('low_band2', '2018', '027_00')
		o = eg.level1_to_level2('low_band2', '2018', '028_00')
		o = eg.level1_to_level2('low_band2', '2018', '029_00')
		o = eg.level1_to_level2('low_band2', '2018', '030_00')
		o = eg.level1_to_level2('low_band2', '2018', '031_00')
		o = eg.level1_to_level2('low_band2', '2018', '032_00')
		o = eg.level1_to_level2('low_band2', '2018', '033_00')
		o = eg.level1_to_level2('low_band2', '2018', '034_00')
		o = eg.level1_to_level2('low_band2', '2018', '035_00')
		o = eg.level1_to_level2('low_band2', '2018', '036_00')
		o = eg.level1_to_level2('low_band2', '2018', '037_00')
		o = eg.level1_to_level2('low_band2', '2018', '038_00')
		o = eg.level1_to_level2('low_band2', '2018', '040_00')
		o = eg.level1_to_level2('low_band2', '2018', '041_00')
		o = eg.level1_to_level2('low_band2', '2018', '042_00')
		o = eg.level1_to_level2('low_band2', '2018', '043_00')
		o = eg.level1_to_level2('low_band2', '2018', '044_00')
		o = eg.level1_to_level2('low_band2', '2018', '045_00')
		o = eg.level1_to_level2('low_band2', '2018', '046_00')
		o = eg.level1_to_level2('low_band2', '2018', '047_00')
		o = eg.level1_to_level2('low_band2', '2018', '048_00')
		o = eg.level1_to_level2('low_band2', '2018', '049_00')
		o = eg.level1_to_level2('low_band2', '2018', '050_00')
		o = eg.level1_to_level2('low_band2', '2018', '051_00')
		o = eg.level1_to_level2('low_band2', '2018', '052_00')
		o = eg.level1_to_level2('low_band2', '2018', '053_00')
		o = eg.level1_to_level2('low_band2', '2018', '054_00')
		o = eg.level1_to_level2('low_band2', '2018', '060_17')
		o = eg.level1_to_level2('low_band2', '2018', '061_00')
		o = eg.level1_to_level2('low_band2', '2018', '062_14')
		o = eg.level1_to_level2('low_band2', '2018', '063_00')
		o = eg.level1_to_level2('low_band2', '2018', '064_00')
		o = eg.level1_to_level2('low_band2', '2018', '065_00')
		o = eg.level1_to_level2('low_band2', '2018', '066_00')
		o = eg.level1_to_level2('low_band2', '2018', '067_00')
		o = eg.level1_to_level2('low_band2', '2018', '068_00')
		o = eg.level1_to_level2('low_band2', '2018', '073_19')
		o = eg.level1_to_level2('low_band2', '2018', '074_00')
		o = eg.level1_to_level2('low_band2', '2018', '079_04')
		o = eg.level1_to_level2('low_band2', '2018', '080_00')
		o = eg.level1_to_level2('low_band2', '2018', '084_00')
		o = eg.level1_to_level2('low_band2', '2018', '085_00')
		o = eg.level1_to_level2('low_band2', '2018', '085_06')
		o = eg.level1_to_level2('low_band2', '2018', '086_00')
		o = eg.level1_to_level2('low_band2', '2018', '087_00')
		o = eg.level1_to_level2('low_band2', '2018', '088_00')
		o = eg.level1_to_level2('low_band2', '2018', '089_00')
		o = eg.level1_to_level2('low_band2', '2018', '095_04')
		o = eg.level1_to_level2('low_band2', '2018', '096_00')
		o = eg.level1_to_level2('low_band2', '2018', '097_00')
		o = eg.level1_to_level2('low_band2', '2018', '097_15')
		o = eg.level1_to_level2('low_band2', '2018', '098_00')
		o = eg.level1_to_level2('low_band2', '2018', '099_00')
		o = eg.level1_to_level2('low_band2', '2018', '100_00')
		o = eg.level1_to_level2('low_band2', '2018', '101_00')
		o = eg.level1_to_level2('low_band2', '2018', '102_00')
		o = eg.level1_to_level2('low_band2', '2018', '103_00')
		o = eg.level1_to_level2('low_band2', '2018', '104_00')
		o = eg.level1_to_level2('low_band2', '2018', '105_00')
		o = eg.level1_to_level2('low_band2', '2018', '106_00')
		o = eg.level1_to_level2('low_band2', '2018', '107_00')
		o = eg.level1_to_level2('low_band2', '2018', '108_00')
		o = eg.level1_to_level2('low_band2', '2018', '109_00')
		o = eg.level1_to_level2('low_band2', '2018', '110_00')
		o = eg.level1_to_level2('low_band2', '2018', '111_00')
		o = eg.level1_to_level2('low_band2', '2018', '112_00')
		o = eg.level1_to_level2('low_band2', '2018', '113_00')
		o = eg.level1_to_level2('low_band2', '2018', '114_00')
		o = eg.level1_to_level2('low_band2', '2018', '115_00')
		o = eg.level1_to_level2('low_band2', '2018', '116_00')
		o = eg.level1_to_level2('low_band2', '2018', '117_00')
		o = eg.level1_to_level2('low_band2', '2018', '118_00')
		o = eg.level1_to_level2('low_band2', '2018', '119_00')
		o = eg.level1_to_level2('low_band2', '2018', '120_00')
		o = eg.level1_to_level2('low_band2', '2018', '121_00')
		o = eg.level1_to_level2('low_band2', '2018', '122_00')
		o = eg.level1_to_level2('low_band2', '2018', '123_00')
		o = eg.level1_to_level2('low_band2', '2018', '124_00')
		o = eg.level1_to_level2('low_band2', '2018', '125_00')
		o = eg.level1_to_level2('low_band2', '2018', '126_00')
		o = eg.level1_to_level2('low_band2', '2018', '127_00')
		o = eg.level1_to_level2('low_band2', '2018', '128_00')
		o = eg.level1_to_level2('low_band2', '2018', '129_00')
		o = eg.level1_to_level2('low_band2', '2018', '130_00')
		o = eg.level1_to_level2('low_band2', '2018', '131_00')
		o = eg.level1_to_level2('low_band2', '2018', '132_00')
		o = eg.level1_to_level2('low_band2', '2018', '133_00')
		o = eg.level1_to_level2('low_band2', '2018', '134_00')
		o = eg.level1_to_level2('low_band2', '2018', '135_00')
		o = eg.level1_to_level2('low_band2', '2018', '136_00')
		o = eg.level1_to_level2('low_band2', '2018', '137_00')
		o = eg.level1_to_level2('low_band2', '2018', '138_00')
		o = eg.level1_to_level2('low_band2', '2018', '139_00')
		o = eg.level1_to_level2('low_band2', '2018', '140_00')
		o = eg.level1_to_level2('low_band2', '2018', '141_00')
		o = eg.level1_to_level2('low_band2', '2018', '142_00')
		o = eg.level1_to_level2('low_band2', '2018', '143_00')
		o = eg.level1_to_level2('low_band2', '2018', '144_00')
		o = eg.level1_to_level2('low_band2', '2018', '145_00')
		o = eg.level1_to_level2('low_band2', '2018', '146_00')
		o = eg.level1_to_level2('low_band2', '2018', '147_00')
		o = eg.level1_to_level2('low_band2', '2018', '147_17')
		o = eg.level1_to_level2('low_band2', '2018', '148_00')
		o = eg.level1_to_level2('low_band2', '2018', '149_00')
		o = eg.level1_to_level2('low_band2', '2018', '150_00')
		o = eg.level1_to_level2('low_band2', '2018', '151_00')
		o = eg.level1_to_level2('low_band2', '2018', '152_00')
		o = eg.level1_to_level2('low_band2', '2018', '152_19')
		o = eg.level1_to_level2('low_band2', '2018', '153_00')
		o = eg.level1_to_level2('low_band2', '2018', '154_00')
		o = eg.level1_to_level2('low_band2', '2018', '155_00')
		o = eg.level1_to_level2('low_band2', '2018', '156_00')
		o = eg.level1_to_level2('low_band2', '2018', '157_00')
		o = eg.level1_to_level2('low_band2', '2018', '158_00')
		o = eg.level1_to_level2('low_band2', '2018', '159_00')
		o = eg.level1_to_level2('low_band2', '2018', '160_00')
		o = eg.level1_to_level2('low_band2', '2018', '161_00')
		o = eg.level1_to_level2('low_band2', '2018', '162_00')
		o = eg.level1_to_level2('low_band2', '2018', '163_00')
		o = eg.level1_to_level2('low_band2', '2018', '164_00')
		o = eg.level1_to_level2('low_band2', '2018', '165_00')
		o = eg.level1_to_level2('low_band2', '2018', '166_00')
		o = eg.level1_to_level2('low_band2', '2018', '167_00')
		o = eg.level1_to_level2('low_band2', '2018', '168_00')
		o = eg.level1_to_level2('low_band2', '2018', '169_00')
		o = eg.level1_to_level2('low_band2', '2018', '170_00')
		o = eg.level1_to_level2('low_band2', '2018', '171_00')
		o = eg.level1_to_level2('low_band2', '2018', '181')
		o = eg.level1_to_level2('low_band2', '2018', '182')
		o = eg.level1_to_level2('low_band2', '2018', '183')
		o = eg.level1_to_level2('low_band2', '2018', '184')
		o = eg.level1_to_level2('low_band2', '2018', '185')
		o = eg.level1_to_level2('low_band2', '2018', '186')
		o = eg.level1_to_level2('low_band2', '2018', '187')
		o = eg.level1_to_level2('low_band2', '2018', '188')
		o = eg.level1_to_level2('low_band2', '2018', '189')		
		o = eg.level1_to_level2('low_band2', '2018', '190')
		o = eg.level1_to_level2('low_band2', '2018', '191')
		o = eg.level1_to_level2('low_band2', '2018', '192')
		o = eg.level1_to_level2('low_band2', '2018', '193')
		o = eg.level1_to_level2('low_band2', '2018', '194')
		o = eg.level1_to_level2('low_band2', '2018', '195')
		o = eg.level1_to_level2('low_band2', '2018', '196')
		o = eg.level1_to_level2('low_band2', '2018', '197')
		o = eg.level1_to_level2('low_band2', '2018', '198')
		o = eg.level1_to_level2('low_band2', '2018', '199')
		o = eg.level1_to_level2('low_band2', '2018', '200')
		o = eg.level1_to_level2('low_band2', '2018', '201')
		o = eg.level1_to_level2('low_band2', '2018', '202')
		o = eg.level1_to_level2('low_band2', '2018', '203')
		o = eg.level1_to_level2('low_band2', '2018', '204')
		o = eg.level1_to_level2('low_band2', '2018', '205')
		o = eg.level1_to_level2('low_band2', '2018', '206')
		o = eg.level1_to_level2('low_band2', '2018', '207')
		o = eg.level1_to_level2('low_band2', '2018', '208')
		o = eg.level1_to_level2('low_band2', '2018', '209')
		o = eg.level1_to_level2('low_band2', '2018', '210')
		o = eg.level1_to_level2('low_band2', '2018', '211')
		o = eg.level1_to_level2('low_band2', '2018', '212')
		o = eg.level1_to_level2('low_band2', '2018', '213')
		o = eg.level1_to_level2('low_band2', '2018', '214')
		o = eg.level1_to_level2('low_band2', '2018', '215')
		o = eg.level1_to_level2('low_band2', '2018', '216')
		o = eg.level1_to_level2('low_band2', '2018', '217')
		o = eg.level1_to_level2('low_band2', '2018', '218')		
		

	return 1







def batch_mid_band_level1_to_level2():

	# Listing files to be processed
	path_files = home_folder + '/EDGES/spectra/level1/mid_band/300_350/'
	new_list   = listdir(path_files)
	new_list.sort()
	
	for i in range(26,len(new_list)):
		
		
	
		day = new_list[i][12:18]
		#print(i)
		print(day)
		
		if (int(day[0:3]) <= 170) or (int(day[0:3]) >= 174):  # files in this range have problems
			
			o   = eg.level1_to_level2('mid_band', '2018', day)
					
	return 1







def batch_low_band3_level1_to_level2():

	# Listing files to be processed
	path_files = home_folder + '/EDGES/spectra/level1/low_band3/300_350/'
	new_list   = listdir(path_files)
	new_list.sort()
	
	for i in range(len(new_list)):      #range(26, len(new_list)):
	
		year = new_list[i][7:11]
		day  = new_list[i][12:15]
		
		
		if (int(year) == 2018) and (int(day) == 225):  # files in this range have problems
			print(year + ' ' + day + ': bad file')
		
		else:
			print(year + ' ' + day)
			o = eg.level1_to_level2('low_band3', year, day)
					
	return 1
























def batch_mid_band_level2_to_level3(case):


	# Case selection
	if case == 0:
		flag_folder       = 'case0'
		receiver_cal_file = 1
		antenna_s11_day   = 147
		antenna_s11_Nfit  = 14
		beam_correction   = 0
		balun_correction  = 0
		FLOW  = 60
		FHIGH = 160
		Nfg   = 7
	
	
	if case == 1:
		flag_folder       = 'case1'
		receiver_cal_file = 1
		antenna_s11_day   = 147
		antenna_s11_Nfit  = 14
		beam_correction   = 0
		balun_correction  = 1
		FLOW  = 60
		FHIGH = 160
		Nfg   = 7
	
	
	if case == 2:
		flag_folder       = 'case2'
		receiver_cal_file = 1
		antenna_s11_day   = 147
		antenna_s11_Nfit  = 14
		beam_correction   = 1
		balun_correction  = 1
		FLOW  = 60
		FHIGH = 160
		Nfg   = 7

	
	
	
	



	# Listing files to be processed
	path_files = edges_folder + 'mid_band/spectra/level2/'
	old_list   = listdir(path_files)
	old_list.sort()
	
	bad_files = ['2018_153_00.hdf5', '2018_154_00.hdf5', '2018_155_00.hdf5', '2018_156_00.hdf5', '2018_158_00.hdf5', '2018_168_00.hdf5', '2018_183_00.hdf5', '2018_194_00.hdf5', '2018_202_00.hdf5', '2018_203_00.hdf5', '2018_206_00.hdf5', '2018_207_00.hdf5', '2018_213_00.hdf5', '2018_214_00.hdf5', '2018_221_00.hdf5', '2018_222_00.hdf5']
	
	new_list = []
	for i in range(len(old_list)):
		if old_list[i] not in bad_files:
			new_list.append(old_list[i])
	
		

	# Processing files
	#for i in range(len(new_list)):
	for i in range(10, len(new_list)):
		print(i)
		print(new_list[i])
	
		o = eg.level2_to_level3('mid_band', new_list[i], flag_folder=flag_folder, receiver_cal_file=receiver_cal_file, antenna_s11_year=2018, antenna_s11_day=antenna_s11_day, antenna_s11_Nfit=antenna_s11_Nfit, beam_correction=beam_correction, balun_correction=balun_correction, FLOW=FLOW, FHIGH=FHIGH, Nfg=Nfg)
		

	return new_list #0










def batch_low_band3_level2_to_level3(case):


	# Case selection
	#if case == 0:
		#flag_folder       = 'case0'
		#receiver_cal_file = 1
		#antenna_s11_day   = 147
		#antenna_s11_Nfit  = 14
		#beam_correction   = 0
		#balun_correction  = 0
		#FLOW  = 60
		#FHIGH = 160
		#Nfg   = 7
	
	
	#if case == 1:
		#flag_folder       = 'case1'
		#receiver_cal_file = 1
		#antenna_s11_day   = 147
		#antenna_s11_Nfit  = 14
		#beam_correction   = 0
		#balun_correction  = 1
		#FLOW  = 60
		#FHIGH = 160
		#Nfg   = 7
	
	
	if case == 2:
		flag_folder       = 'case2'
		receiver_cal_file = 1
		antenna_s11_day   = 227
		antenna_s11_Nfit  = 14
		beam_correction   = 1
		balun_correction  = 1
		FLOW  = 50
		FHIGH = 120
		Nfg   = 7

	

	# Listing files to be processed
	path_files = edges_folder + 'low_band3/spectra/level2/'
	new_list   = listdir(path_files)
	new_list.sort()
		

	# Processing files
	#for i in range(len(new_list)):
	#for i in range(10, len(new_list)):
	#for i in range(0, 13):
	for i in range(len(new_list)):
		
		print(i)
		print(new_list[i])
	
		o = eg.level2_to_level3('low_band3', new_list[i], flag_folder=flag_folder, receiver_cal_file=receiver_cal_file, antenna_s11_year=2018, antenna_s11_day=antenna_s11_day, antenna_s11_Nfit=antenna_s11_Nfit, beam_correction=beam_correction, balun_correction=balun_correction, FLOW=FLOW, FHIGH=FHIGH, Nfg=Nfg)
		

	return new_list #0






































def temporary_two_calibrations():
	
	x = batch_mid_band_level2_to_level3(0)
	x = batch_mid_band_level2_to_level3(1)
	
	return 0








































def batch_plot_daily_residuals_LST():
	
	# Listing files to be processed
	path_files = home_folder + '/EDGES/spectra/level3/mid_band/nominal_60_160MHz_fullcal/'
	new_list   = listdir(path_files)
	new_list.sort()
	
	
	
	# Global settings
	# ----------------------------------
	
	# Computation of residuals
	LST_boundaries = np.arange(0,25,1)
	FLOW        =  60
	FHIGH       = 150
	SUN_EL_max  = -20
	MOON_EL_max =  90
	
	# Plotting
	LST_centers = list(np.arange(0.5,24))
	LST_text    = ['LST=' + str(LST_centers[i]) + ' hr' for i in range(len(LST_centers))]
	DY          =   4
	FLOW_plot   =  35
	FHIGH_plot  = 155
	XTICKS      = np.arange(60, 156, 20)
	XTEXT       =  38
	YLABEL      =  '4 K per division'
	
	
	
	
	
	
	# 3 foreground terms
	Nfg = 5
	figure_path = '/data5/raul/EDGES/results/plots/20181031/midband_residuals_nighttime_5terms/'
	
	for i in range(len(new_list)):   # [len(new_list)-1]:   
		print(new_list[i])
		
		# Computing residuals
		fb, rb, wb = eg.daily_residuals_LST(new_list[i], LST_boundaries=LST_boundaries, FLOW=FLOW, FHIGH=FHIGH, Nfg=Nfg, SUN_EL_max=SUN_EL_max, MOON_EL_max=MOON_EL_max)
		
		# Plotting
		x = eg.plot_daily_residuals_LST(fb, rb, LST_text, DY=DY, FLOW=FLOW_plot, FHIGH=FHIGH_plot, XTICKS=XTICKS, XTEXT=XTEXT, YLABEL=YLABEL, TITLE=str(Nfg) + ' TERMS:  ' + new_list[i][0:-5], save='yes', figure_path=figure_path, figure_name=new_list[i][0:-5])
	
	
	
	return 0










def plot_residuals_GHA_1hr_bin(f, r, w):
	
	
	# Settings
	# ----------------------------------

	GHA_edges   = list(np.arange(0,25))
	GHA_text    = ['GHA=' + str(GHA_edges[i]) + '-' + str(GHA_edges[i+1]) + ' hr' for i in range(len(GHA_edges)-1)]
	DY          = 0.4
	FLOW_plot   = 40
	FHIGH_plot  = 125
	XTICKS      = np.arange(60, 121, 20)
	XTEXT       = 42
	YLABEL      = str(DY) + ' K per division'
	TITLE       = '5 LINLOG terms, 60-120 MHz'
	figure_path = '/home/raul/Desktop/'
	figure_name = 'linlog_5terms_60-120MHz'
	
	
	# Plotting
	x = eg.plot_residuals(f, r, w, GHA_text, DY=DY, FLOW=FLOW_plot, FHIGH=FHIGH_plot, XTICKS=XTICKS, XTEXT=XTEXT, YLABEL=YLABEL, TITLE=TITLE, save='yes', figure_path=figure_path, figure_name=figure_name)
	
	
	
	return 0






def plot_residuals_GHA_Xhr_bin(f, r, w):
	
	
	# Settings
	# ----------------------------------

	LST_centers = list(np.arange(0.5,24))
	LST_text    = ['GHA=0-5 hr', 'GHA=5-11 hr', 'GHA=11-18 hr', 'GHA=18-24 hr']
	DY          =  0.5
	FLOW_plot   =  35
	FHIGH_plot  = 165
	XTICKS      = np.arange(60, 161, 20)
	XTEXT       =  38
	YLABEL      =  str(DY) + ' K per division'
	TITLE       = '4 LINLOG terms, 62-120 MHz'
	figure_path = '/home/raul/Desktop/'
	figure_name = 'CASE2_linlog_4terms_62-120MHz'
	FIG_SX      = 8
	FIG_SY      = 7
	
	
	
	# Plotting
	x = eg.plot_residuals(f, r, w, LST_text, FIG_SX=FIG_SX, FIG_SY=FIG_SY, DY=DY, FLOW=FLOW_plot, FHIGH=FHIGH_plot, XTICKS=XTICKS, XTEXT=XTEXT, YLABEL=YLABEL, TITLE=TITLE, save='yes', figure_path=figure_path, figure_name=figure_name)
	
	
	
	return 0












def plots_for_midband_verification_paper(antenna_reflection_loss='no', beam_factor='no'):
	
	if antenna_reflection_loss == 'yes':
		
		plt.close()
		plt.close()
		plt.close()
		
		
		
		
		# Plot 
		# ---------------------------------------
		f1  = plt.figure(num=1, figsize=(5.5, 6))
	
		# Generating reflection coefficient and loss
		f = np.arange(60,160.1,0.1)
		
		s11_ant = eg.models_antenna_s11_remove_delay('mid_band', f, year=2018, day=147, delay_0=0.17, model_type='polynomial', Nfit=14, plot_fit_residuals='no')
		
		Gb, Gc = eg.balun_and_connector_loss(f, s11_ant)
		
		
		
		# Nominal S11
		# -----------
		ax  = f1.add_axes([0.11, 0.1+0.4, 0.73, 0.4])
		h1  = ax.plot(f, 20*np.log10(np.abs(s11_ant)), 'b', linewidth=2, label='magnitude')
		ax2 = ax.twinx()
		h2  = ax2.plot(f, (180/np.pi)*np.unwrap(np.angle(s11_ant)), 'r--', linewidth=2, label='phase')
	
		h      = h1 + h2
		labels = [l.get_label() for l in h]
		ax.legend(h, labels, loc=0, fontsize=12)
	
		ax.set_xlim([60, 160])
		ax.set_ylim([-18, 2])
		ax.set_yticks(np.arange(-16,1,4))
		ax.set_xticks([60, 80, 100, 120, 140, 160])
		ax.set_xticklabels([])
	
		ax2.set_ylim([-800-100, 0+100])
		ax2.set_yticks(np.arange(-800,1,200))
	
		ax.grid()
		ax.set_ylabel('magnitude [dB]', fontsize=14)
		ax2.set_ylabel('phase [degrees]', fontsize=14)
		ax.text(63, -3, '(a)', fontsize=18)		
	
	
	
		# Losses
		# ------
		ax = f1.add_axes([0.11, 0.1, 0.73, 0.4])	
		h1  = ax.plot(f, 100*(1-Gb), 'g', linewidth=2, label='balun loss')
		h2  = ax.plot(f, 100*(1-Gc), 'r', linewidth=2, label='connector loss')
		h3  = ax.plot(f, 100*(1-Gb*Gc), 'k', linewidth=2, label='balun + connector loss')
	
		h      = h1 + h2 + h3
		labels = [l.get_label() for l in h]
		ax.legend(h, labels, loc=0, fontsize=12)
	
		ax.set_xlim([60, 160])
		ax.set_ylim([-0.1, 1.1])
		ax.set_xticks([60, 80, 100, 120, 140, 160])
		#ax.set_xticklabels([])
	
		ax.grid()
		ax.set_ylabel('loss [%]', fontsize=14)
		ax.set_xlabel('frequency [MHz]', fontsize=14)
		ax.text(63, 0.87, '(b)', fontsize=18)
	
	
	plt.savefig('/data5/raul/EDGES/results/plots/20181022/antenna_reflection_loss.pdf', bbox_inches='tight')
	plt.close()
	plt.close()	
	
	
	
	
	
	
	
	if beam_factor == 'yes':
	
		plt.close()
		plt.close()
		plt.close()
		
		plt.figure(figsize=(11, 4))
		
		# ---------------------------------------
		plt.subplot(1,2,1)
		
			
		f, lst_table, bf_table = eg.beam_factor_table_read('/data5/raul/EDGES/calibration/beam_factors/mid_band/beam_factor_table_hires.hdf5') 
		
		lst_in = np.arange(0, 24, 24/144)
		bf     = eg.beam_factor_table_evaluate(f, lst_table, bf_table, lst_in)
		
		plt.imshow(bf, interpolation='none', aspect='auto', vmin=0.95, vmax=1.05, extent=[60, 160, 24, 0])
		plt.yticks(np.arange(0,25,4))
		plt.grid()
		cb = plt.colorbar() #(orientation='horizontal')
		#cb.ax.set_ylabel(r'correction factor, $C$ (no units)')
		plt.xlabel('frequency [MHz]')
		plt.ylabel('LST [hr]')
		
		
		# ---------------------------------------
		plt.subplot(1,2,2)
		
		lst_in = np.arange(0, 24, 4)
		bf     = eg.beam_factor_table_evaluate(f, lst_table, bf_table, lst_in)
		
		plt.plot(f, bf.T)
		plt.legend(['0 hr','4 hr','8 hr','12 hr','16 hr','20 hr'], loc=3)
		plt.xlabel('frequency MHz]')
		plt.ylabel(r'correction factor, $C$')
		
	
	
	
	
		plt.savefig('/data5/raul/EDGES/results/plots/20181022/beam_factor.pdf', bbox_inches='tight')
		plt.close()
		plt.close()


	
		
	return 0










def plot_residuals_simulated_antenna_temperature(model, title):
	
	if model == 1:
		t   = np.genfromtxt(edges_folder + 'mid_band/calibration/beam_factors/raw/mid_band_50-200MHz_90deg_alan1_haslam_2.5_2.62_reffreq_100MHz_tant.txt')
		lst = np.genfromtxt(edges_folder + 'mid_band/calibration/beam_factors/raw/mid_band_50-200MHz_90deg_alan1_haslam_2.5_2.62_reffreq_100MHz_LST.txt')
		f   = np.genfromtxt(edges_folder + 'mid_band/calibration/beam_factors/raw/mid_band_50-200MHz_90deg_alan1_haslam_2.5_2.62_reffreq_100MHz_freq.txt')
		
	if model == 2:
		t   = np.genfromtxt(edges_folder + 'mid_band/calibration/beam_factors/raw/mid_band_50-200MHz_90deg_alan1_haslam_2.5_2.5_reffreq_80MHz_tant.txt')
		lst = np.genfromtxt(edges_folder + 'mid_band/calibration/beam_factors/raw/mid_band_50-200MHz_90deg_alan1_haslam_2.5_2.5_reffreq_80MHz_LST.txt')
		f   = np.genfromtxt(edges_folder + 'mid_band/calibration/beam_factors/raw/mid_band_50-200MHz_90deg_alan1_haslam_2.5_2.5_reffreq_80MHz_freq.txt')	
	
	
	
	
	# Converting LST to GHA
	gha          = lst - 17.76
	gha[gha < 0] = gha[gha < 0] + 24
	
	IX  = np.argsort(gha)
	gha = gha[IX]
	t   = t[IX,:]
	

	t1 = np.zeros((24, len(f)))
	for i in range(24):
		tb   = t[(gha>=i) & (gha<=i+1),:]
		print(gha[(gha>=i) & (gha<=i+1)])
		avtb = np.mean(tb, axis=0)
		t1[i,:] = avtb
		
	
	w   = np.ones((len(t1[:,0]), len(t[0,:])))
	fx, rx, wx = eg.spectra_to_residuals(f, t1, w, 61, 159, 5)
	index      = np.arange(0,24,1)
	
	ar     = np.arange(0,25,1)
	str_ar = ['GHA=' + str(ar[i]) + '-' + str(ar[i+1]) + ' hr' for i in range(len(ar)-1)]	
	
	
	plt.figure()
	o = eg.plot_residuals(fx, rx[index,:], wx[index,:], str_ar, FIG_SX=7, FIG_SY=12, DY=1.5, FLOW=30, FHIGH=165, XTICKS=np.arange(60, 160+1, 20), XTEXT=32, YLABEL='1.5 K per division', TITLE=title, save='yes', figure_name='simulation', figure_format='pdf')

	
	return 0








def plot_season_average_residuals(case, Nfg=3, DDY=1.5, TITLE='No Beam Correction, Residuals to 5 LINLOG terms, 61-159 MHz', figure_name='no_beam_correction'):
	
	if case == '1hr_1':
		delta_HR = 1
		fb = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case2/case2_frequency.txt')
		ty = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case2/case2_1hr_temperature.txt')
		wy = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case2/case2_1hr_weights.txt')
			
		
	
		ff, rr, ww = eg.spectra_to_residuals(fb, ty, wy, 61, 159, Nfg) 
		
		ar     = np.arange(0, 25, delta_HR)
		str_ar = ['GHA=' + str(ar[i]) + '-' + str(ar[i+1]) + ' hr' for i in range(len(ar)-1)]
		
		o = eg.plot_residuals(ff, rr, ww, str_ar, FIG_SX=7, FIG_SY=12, DY=DDY, FLOW=30, FHIGH=165, XTICKS=np.arange(60, 160+1, 20), XTEXT=32, YLABEL=str(DDY) + ' K per division', TITLE=TITLE, save='yes', figure_name=figure_name, figure_format='pdf')
		
		
		
		
	if case == '1hr_2':
		delta_HR = 1
		fb = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case2/case2_frequency.txt')
		ty = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case2/case2_1hr_temperature.txt')
		wy = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case2/case2_1hr_weights.txt')
			
		
	
		ff, rr, ww = eg.spectra_to_residuals(fb, ty, wy, 61, 136, Nfg) 
		
		ar     = np.arange(0, 25, delta_HR)
		str_ar = ['GHA=' + str(ar[i]) + '-' + str(ar[i+1]) + ' hr' for i in range(len(ar)-1)]
		
		o = eg.plot_residuals(ff, rr, ww, str_ar, FIG_SX=6, FIG_SY=12, DY=DDY, FLOW=30, FHIGH=145, XTICKS=np.arange(60, 140+1, 20), XTEXT=32, YLABEL=str(DDY) + ' K per division', TITLE=TITLE, save='yes', figure_name=figure_name, figure_format='pdf')		




	if case == '1hr_3':
		delta_HR = 1
		fb = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case2/case2_frequency.txt')
		ty = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case2/case2_1hr_temperature.txt')
		wy = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case2/case2_1hr_weights.txt')
			
		
	
		ff, rr, ww = eg.spectra_to_residuals(fb, ty, wy, 110, 159, Nfg) 
		
		ar     = np.arange(0, 25, delta_HR)
		str_ar = ['GHA=' + str(ar[i]) + '-' + str(ar[i+1]) + ' hr' for i in range(len(ar)-1)]
		
		o = eg.plot_residuals(ff, rr, ww, str_ar, FIG_SX=7, FIG_SY=12, DY=DDY, FLOW=70, FHIGH=165, XTICKS=np.arange(100, 160+1, 20), XTEXT=72, YLABEL=str(DDY) + ' K per division', TITLE=TITLE, save='yes', figure_name=figure_name, figure_format='pdf')
		
		





		
		
		
	if case == '2hr_1':
		delta_HR = 2
		fb = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case2/case2_frequency.txt')
		ty = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case2/case2_2hr_temperature.txt')
		wy = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case2/case2_2hr_weights.txt')
			
		
	
		ff, rr, ww = eg.spectra_to_residuals(fb, ty, wy, 61, 159, Nfg) 
		
		ar     = np.arange(0, 25, delta_HR)
		str_ar = ['GHA=' + str(ar[i]) + '-' + str(ar[i+1]) + ' hr' for i in range(len(ar)-1)]
		
		o = eg.plot_residuals(ff, rr, ww, str_ar, FIG_SX=7, FIG_SY=12, DY=DDY, FLOW=30, FHIGH=165, XTICKS=np.arange(60, 160+1, 20), XTEXT=32, YLABEL=str(DDY) + ' K per division', TITLE=TITLE, save='yes', figure_name=figure_name, figure_format='pdf')
	
		

	
	if case == '2hr_2':
		delta_HR = 2
		fb = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case2/case2_frequency.txt')
		ty = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case2/case2_2hr_temperature.txt')
		wy = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case2/case2_2hr_weights.txt')
			
		
	
		ff, rr, ww = eg.spectra_to_residuals(fb, ty, wy, 61, 136, Nfg) 
		
		ar     = np.arange(0, 25, delta_HR)
		str_ar = ['GHA=' + str(ar[i]) + '-' + str(ar[i+1]) + ' hr' for i in range(len(ar)-1)]
		
		o = eg.plot_residuals(ff, rr, ww, str_ar, FIG_SX=6, FIG_SY=12, DY=DDY, FLOW=30, FHIGH=145, XTICKS=np.arange(60, 140+1, 20), XTEXT=32, YLABEL=str(DDY) + ' K per division', TITLE=TITLE, save='yes', figure_name=figure_name, figure_format='pdf')				


	if case == '3hr_1':
		delta_HR = 3
		fb = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case2/case2_frequency.txt')
		ty = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case2/case2_3hr_temperature.txt')
		wy = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case2/case2_3hr_weights.txt')
			
		
	
		ff, rr, ww = eg.spectra_to_residuals(fb, ty, wy, 61, 159, Nfg) 
		
		ar     = np.arange(0, 25, delta_HR)
		str_ar = ['GHA=' + str(ar[i]) + '-' + str(ar[i+1]) + ' hr' for i in range(len(ar)-1)]
		
		o = eg.plot_residuals(ff, rr, ww, str_ar, FIG_SX=7, FIG_SY=12, DY=DDY, FLOW=30, FHIGH=165, XTICKS=np.arange(60, 160+1, 20), XTEXT=32, YLABEL=str(DDY) + ' K per division', TITLE=TITLE, save='yes', figure_name=figure_name, figure_format='pdf')
		
		
	if case == '3hr_2':
		delta_HR = 3
		fb = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case2/case2_frequency.txt')
		ty = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case2/case2_3hr_temperature.txt')
		wy = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case2/case2_3hr_weights.txt')
			
		
	
		ff, rr, ww = eg.spectra_to_residuals(fb, ty, wy, 61, 136, Nfg) 
		
		ar     = np.arange(0, 25, delta_HR)
		str_ar = ['GHA=' + str(ar[i]) + '-' + str(ar[i+1]) + ' hr' for i in range(len(ar)-1)]
		
		o = eg.plot_residuals(ff, rr, ww, str_ar, FIG_SX=6, FIG_SY=12, DY=DDY, FLOW=30, FHIGH=145, XTICKS=np.arange(60, 140+1, 20), XTEXT=32, YLABEL=str(DDY) + ' K per division', TITLE=TITLE, save='yes', figure_name=figure_name, figure_format='pdf')


	if case == '4hr_1':
		delta_HR = 4
		fb = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case2/case2_frequency.txt')
		ar = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case2/case2_4hr_gha_edges.txt')
		ty = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case2/case2_4hr_temperature.txt')
		wy = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case2/case2_4hr_weights.txt')
			
		
	
		ff, rr, ww = eg.spectra_to_residuals(fb, ty, wy, 61, 159, Nfg) 
		
		#ar     = np.arange(0, 25, delta_HR)
		str_ar = ['GHA=' + str(int(ar[i])) + '-' + str(int(ar[i+1])) + ' hr' for i in range(len(ar)-1)]
		
		o = eg.plot_residuals(ff, rr, ww, str_ar, FIG_SX=7, FIG_SY=12, DY=DDY, FLOW=30, FHIGH=165, XTICKS=np.arange(60, 160+1, 20), XTEXT=32, YLABEL=str(DDY) + ' K per division', TITLE=TITLE, save='yes', figure_name=figure_name, figure_format='pdf')


	if case == '4hr_2':
		delta_HR = 4
		fb = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case2/case2_frequency.txt')
		ar = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case2/case2_4hr_gha_edges.txt')
		ty = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case2/case2_4hr_temperature.txt')
		wy = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case2/case2_4hr_weights.txt')
			
		
	
		ff, rr, ww = eg.spectra_to_residuals(fb, ty, wy, 61, 136, Nfg) 
		
		#ar     = np.arange(0, 25, delta_HR)
		str_ar = ['GHA=' + str(int(ar[i])) + '-' + str(int(ar[i+1])) + ' hr' for i in range(len(ar)-1)]
		
		o = eg.plot_residuals(ff, rr, ww, str_ar, FIG_SX=6, FIG_SY=12, DY=DDY, FLOW=30, FHIGH=145, XTICKS=np.arange(60, 140+1, 20), XTEXT=32, YLABEL=str(DDY) + ' K per division', TITLE=TITLE, save='yes', figure_name=figure_name, figure_format='pdf')



	if case == '6hr_1':
		delta_HR = 6
		fb = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case2/case2_frequency.txt')
		ar = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case2/case2_6hr_gha_edges.txt')
		ty = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case2/case2_6hr_temperature.txt')
		wy = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case2/case2_6hr_weights.txt')
			
		
	
		ff, rr, ww = eg.spectra_to_residuals(fb, ty, wy, 61, 159, Nfg) 
		
		#ar     = np.arange(0, 25, delta_HR)
		str_ar = ['GHA=' + str(int(ar[i])) + '-' + str(int(ar[i+1])) + ' hr' for i in range(len(ar)-1)]
		
		o = eg.plot_residuals(ff, rr, ww, str_ar, FIG_SX=7, FIG_SY=12, DY=DDY, FLOW=30, FHIGH=165, XTICKS=np.arange(60, 160+1, 20), XTEXT=32, YLABEL=str(DDY) + ' K per division', TITLE=TITLE, save='yes', figure_name=figure_name, figure_format='pdf')




	if case == '6hr_2':
		delta_HR = 6
		fb = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case2/case2_frequency.txt')
		ar = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case2/case2_6hr_gha_edges.txt')
		ty = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case2/case2_6hr_temperature.txt')
		wy = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case2/case2_6hr_weights.txt')
			
		
	
		ff, rr, ww = eg.spectra_to_residuals(fb, ty, wy, 61, 136, Nfg) 
		
		#ar     = np.arange(0, 25, delta_HR)
		str_ar = ['GHA=' + str(int(ar[i])) + '-' + str(int(ar[i+1])) + ' hr' for i in range(len(ar)-1)]
		
		o = eg.plot_residuals(ff, rr, ww, str_ar, FIG_SX=6, FIG_SY=12, DY=DDY, FLOW=30, FHIGH=145, XTICKS=np.arange(60, 140+1, 20), XTEXT=32, YLABEL=str(DDY) + ' K per division', TITLE=TITLE, save='yes', figure_name=figure_name, figure_format='pdf')




	if case == '8hr_1':
		delta_HR = 8
		fb = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case2/case2_frequency.txt')
		ar = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case2/case2_8hr_gha_edges.txt')
		ty = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case2/case2_8hr_temperature.txt')
		wy = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case2/case2_8hr_weights.txt')
			
		
	
		ff, rr, ww = eg.spectra_to_residuals(fb, ty, wy, 61, 159, Nfg) 
		
		#ar     = np.arange(0, 25, delta_HR)
		str_ar = ['GHA=' + str(int(ar[i])) + '-' + str(int(ar[i+1])) + ' hr' for i in range(len(ar)-1)]
		
		o = eg.plot_residuals(ff, rr, ww, str_ar, FIG_SX=7, FIG_SY=12, DY=DDY, FLOW=30, FHIGH=165, XTICKS=np.arange(60, 160+1, 20), XTEXT=32, YLABEL=str(DDY) + ' K per division', TITLE=TITLE, save='yes', figure_name=figure_name, figure_format='pdf')



	if case == '8hr_2':
		delta_HR = 8
		fb = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case2/case2_frequency.txt')
		ar = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case2/case2_8hr_gha_edges.txt')
		ty = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case2/case2_8hr_temperature.txt')
		wy = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case2/case2_8hr_weights.txt')
			
		
	
		ff, rr, ww = eg.spectra_to_residuals(fb, ty, wy, 61, 136, Nfg) 
		
		#ar     = np.arange(0, 25, delta_HR)
		str_ar = ['GHA=' + str(int(ar[i])) + '-' + str(int(ar[i+1])) + ' hr' for i in range(len(ar)-1)]
		
		o = eg.plot_residuals(ff, rr, ww, str_ar, FIG_SX=6, FIG_SY=12, DY=DDY, FLOW=30, FHIGH=145, XTICKS=np.arange(60, 140+1, 20), XTEXT=32, YLABEL=str(DDY) + ' K per division', TITLE=TITLE, save='yes', figure_name=figure_name, figure_format='pdf')



	if case == '12hr_1':
		delta_HR = 12
		fb = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case2/case2_frequency.txt')
		ar = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case2/case2_12hr_gha_edges.txt')
		ty = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case2/case2_12hr_temperature.txt')
		wy = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case2/case2_12hr_weights.txt')
			
		
	
		ff, rr, ww = eg.spectra_to_residuals(fb, ty, wy, 61, 159, Nfg) 
		
		#ar     = np.arange(0, 25, delta_HR)
		str_ar = ['GHA=' + str(int(ar[i])) + '-' + str(int(ar[i+1])) + ' hr' for i in range(len(ar)-1)]
		
		o = eg.plot_residuals(ff, rr, ww, str_ar, FIG_SX=7, FIG_SY=12, DY=DDY, FLOW=30, FHIGH=165, XTICKS=np.arange(60, 160+1, 20), XTEXT=32, YLABEL=str(DDY) + ' K per division', TITLE=TITLE, save='yes', figure_name=figure_name, figure_format='pdf')
		
		
	if case == '12hr_2':
		delta_HR = 12
		fb = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case2/case2_frequency.txt')
		ar = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case2/case2_12hr_gha_edges.txt')
		ty = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case2/case2_12hr_temperature.txt')
		wy = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case2/case2_12hr_weights.txt')
			
		
	
		ff, rr, ww = eg.spectra_to_residuals(fb, ty, wy, 61, 136, Nfg) 
		
		#ar     = np.arange(0, 25, delta_HR)
		str_ar = ['GHA=' + str(int(ar[i])) + '-' + str(int(ar[i+1])) + ' hr' for i in range(len(ar)-1)]
		
		o = eg.plot_residuals(ff, rr, ww, str_ar, FIG_SX=6, FIG_SY=12, DY=DDY, FLOW=30, FHIGH=145, XTICKS=np.arange(60, 140+1, 20), XTEXT=32, YLABEL=str(DDY) + ' K per division', TITLE=TITLE, save='yes', figure_name=figure_name, figure_format='pdf')



	if case == '12hr_3':
		delta_HR = 12
		fb = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case2/case2_frequency.txt')
		ar = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case2/case2_12hr_gha_edges.txt')
		ty = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case2/case2_12hr_temperature.txt')
		wy = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case2/case2_12hr_weights.txt')
			
		
	
		ff, rr, ww = eg.spectra_to_residuals(fb, ty, wy, 61, 105, Nfg) 
		
		#ar     = np.arange(0, 25, delta_HR)
		str_ar = ['GHA=' + str(int(ar[i])) + '-' + str(int(ar[i+1])) + ' hr' for i in range(len(ar)-1)]
		
		o = eg.plot_residuals(ff, rr, ww, str_ar, FIG_SX=6, FIG_SY=12, DY=DDY, FLOW=30, FHIGH=115, XTICKS=np.arange(60, 110+1, 10), XTEXT=32, YLABEL=str(DDY) + ' K per division', TITLE=TITLE, save='yes', figure_name=figure_name, figure_format='pdf')



	if case == '10-15hr_1':
		delta_HR = 12
		fb = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case2/case2_frequency.txt')
		ar = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case2/case2_10-15hr_gha_edges.txt')
		ty = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case2/case2_10-15hr_temperature.txt')
		wy = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case2/case2_10-15hr_weights.txt')
			
		
	
		ff, rr, ww = eg.spectra_to_residuals(fb, ty, wy, 61, 159, Nfg) 
		
		#ar     = np.arange(0, 25, delta_HR)
		str_ar = ['GHA=' + str(int(ar[i])) + '-' + str(int(ar[i+1])) + ' hr' for i in range(len(ar)-1)]
		
		o = eg.plot_residuals(ff, rr, ww, str_ar, FIG_SX=7, FIG_SY=12, DY=DDY, FLOW=30, FHIGH=165, XTICKS=np.arange(60, 160+1, 20), XTEXT=32, YLABEL=str(DDY) + ' K per division', TITLE=TITLE, save='yes', figure_name=figure_name, figure_format='pdf')



	return 0





































def beam_correction_check(FLOW, FHIGH, SP, Nfg):
	
	#tt  = np.genfromtxt(edges_folder + 'mid_band/calibration/beam_factors/raw/mid_band_50-200MHz_90deg_alan1_haslam_2.5_2.5_reffreq_80MHz_tant.txt')
	#bb  = np.genfromtxt(edges_folder + 'mid_band/calibration/beam_factors/raw/mid_band_50-200MHz_90deg_alan1_haslam_2.5_2.5_reffreq_80MHz_data.txt')
	#ff  = np.genfromtxt(edges_folder + 'mid_band/calibration/beam_factors/raw/mid_band_50-200MHz_90deg_alan1_haslam_2.5_2.5_reffreq_80MHz_freq.txt')
	#lst = np.genfromtxt(edges_folder + 'mid_band/calibration/beam_factors/raw/mid_band_50-200MHz_90deg_alan1_haslam_2.5_2.5_reffreq_80MHz_LST.txt')
	
	tt  = np.genfromtxt(edges_folder + 'mid_band/calibration/beam_factors/raw/mid_band_50-200MHz_90deg_alan1_haslam_2.5_2.62_reffreq_100MHz_tant.txt')
	bb  = np.genfromtxt(edges_folder + 'mid_band/calibration/beam_factors/raw/mid_band_50-200MHz_90deg_alan1_haslam_2.5_2.62_reffreq_100MHz_data.txt')
	ff  = np.genfromtxt(edges_folder + 'mid_band/calibration/beam_factors/raw/mid_band_50-200MHz_90deg_alan1_haslam_2.5_2.62_reffreq_100MHz_freq.txt')
	lst = np.genfromtxt(edges_folder + 'mid_band/calibration/beam_factors/raw/mid_band_50-200MHz_90deg_alan1_haslam_2.5_2.62_reffreq_100MHz_LST.txt')	
	
	
	
	t = tt[:,(ff>=FLOW) & (ff<=FHIGH)]
	b = bb[:,(ff>=FLOW) & (ff<=FHIGH)]
	f = ff[(ff>=FLOW) & (ff<=FHIGH)]
	
	
	plt.figure(1);plt.imshow(b, interpolation='none', aspect='auto', vmin=0.99, vmax=1.01); plt.colorbar()
	
	
	
	
	#tc = t/b
	
	#p1 = ba.fit_polynomial_fourier('LINLOG', f/80, t[SP,:], Nfg)
	
	#p2 = ba.fit_polynomial_fourier('LINLOG', f/80, tc[SP,:], Nfg)
	
	#plt.plot(f, t[SP,:] - p1[1])
	#plt.plot(f, tc[SP,:] - p2[1])
	
	
	
	
	
	f_t, lst_t, bf_t = cmb.beam_factor_table_read(edges_folder + 'mid_band/calibration/beam_factors/table/table_hires_mid_band_50-200MHz_90deg_alan1_haslam_2.5_2.62_reffreq_100MHz.hdf5')
	#f_t, lst_t, bf_t = cmb.beam_factor_table_read(edges_folder + 'mid_band/calibration/beam_factors/table/table_hires_mid_band_50-200MHz_90deg_alan1_haslam_2.5_2.5_reffreq_80MHz.hdf5')
	
	plt.figure(2);plt.imshow(bf_t, interpolation='none', aspect='auto', vmin=0.99, vmax=1.01); plt.colorbar()
	
	
	
	
	lin = np.arange(0,24,1)
	bf  = cmb.beam_factor_table_evaluate(f_t, lst_t, bf_t, lin)
	
	plt.figure(3);plt.imshow(bf, interpolation='none', aspect='auto', vmin=0.99, vmax=1.01); plt.colorbar()
	
	
	
	
	
	
		
	return f, f_t






def plot_mid_band_for_talk():
	
	f = np.genfromtxt('/home/raul/DATA/EDGES/mid_band/spectra/level5/case2/case2_2hr_average_2.5_frequency.txt')
	t = np.genfromtxt('/home/raul/DATA/EDGES/mid_band/spectra/level5/case2/case2_2hr_average_2.5_temperature.txt')
	w = np.genfromtxt('/home/raul/DATA/EDGES/mid_band/spectra/level5/case2/case2_2hr_average_2.5_weights.txt')
	
	
	ff1, rr1, ww1 = eg.spectra_to_residuals(f, t, w, 61, 159, 3)
	ff2, rr2, ww2 = eg.spectra_to_residuals(f, t, w, 61, 159, 4)
	ff3, rr3, ww3 = eg.spectra_to_residuals(f, t, w, 61, 159, 5)
	
	X=7; 
	
	
	#h1, = plt.plot(ff2[ww2[X,:]>0], rr2[X,ww2[X,:]>0])
	#h2, = plt.plot(ff1[ww1[X,:]>0], rr1[X,ww1[X,:]>0])
	#h3, = plt.plot(ff2[ww2[X,:]>0], rr2[X,ww2[X,:]>0], 'b')
	#h4, = plt.plot([78, 78],[-5, 5], 'g--', linewidth=2)
	#plt.ylim([-3,2])
	
	#plt.ylabel('brightness temperature [K]')
	#plt.xlabel('frequency [MHz]') 
	
	
	#plt.legend([h2, h3, h4], ['3 foreground terms','4 foreground terms', '78 MHz'])
	
	
	
	
	
	
	fig, ax = plt.subplots()
	z = ba.frequency2redshift(ff1)
		
	h1, = ax.plot(ff2[ww2[X,:]>0], rr2[X,ww2[X,:]>0])
	h2, = ax.plot(ff1[ww1[X,:]>0], rr1[X,ww1[X,:]>0])
	h3, = ax.plot(ff2[ww2[X,:]>0], rr2[X,ww2[X,:]>0], 'k', linewidth=1.5)
	h4, = ax.plot(ff3[ww3[X,:]>0], rr3[X,ww3[X,:]>0], 'r--', linewidth=0.7)
	h5, = ax.plot([78, 78],[-5, 5], 'g--', linewidth=2)
	
	xlow  = 55
	xhigh = 165
	ax.set_xlim([xlow, xhigh])
	
	ax2 = ax.twiny()
	
	# Fix these lines showing the redshift
	#ax2.set_xticks(np.array((60-xlow, 80-xlow, 100-xlow, 120-xlow, 140-xlow, 160-xlow))/len(ff1))
	ax2.set_xticks(np.array([(59.2-xlow), (78.9-xlow), (101.5-xlow), (129.1-xlow), (157.8-xlow)])/(165-55))
	ax2.set_xticklabels([23, 17, 13, 10, 8])
	
	ax.set_yticks([-2, -1, 0, 1, 2])
	
	
	
	
	
	ax.set_ylim([-2.5,2])
	ax.set_ylabel('brightness temperature [K]', fontsize=12)
	ax2.set_xlabel('redshift', fontsize=12)
	ax.set_xlabel('frequency [MHz]', fontsize=12) 
	
	plt.legend([h2, h3, h4, h5], ['3 foreground terms','4 foreground terms', '5 foreground terms', '78 MHz'], loc=4)
	
	
	return 0






def plot_low_band3_for_talk():
	
	f = np.genfromtxt('/home/raul/DATA/EDGES/low_band3/spectra/level5/case2/case2_preliminary_frequency.txt')
	t = np.genfromtxt('/home/raul/DATA/EDGES/low_band3/spectra/level5/case2/case2_preliminary_temperature.txt')
	w = np.genfromtxt('/home/raul/DATA/EDGES/low_band3/spectra/level5/case2/case2_preliminary_weights.txt')
	
	
	model_type1 = 'LINLOG'
	model_type2 = 'EDGES_polynomial'
	ff1, rr1, ww1 = eg.spectra_to_residuals(f, t, w, 61, 101, 3, model_type=model_type1)
	ff2, rr2, ww2 = eg.spectra_to_residuals(f, t, w, 62, 102, 4, model_type=model_type1)
	ff3, rr3, ww3 = eg.spectra_to_residuals(f, t, w, 62, 102, 5, model_type=model_type2)
	
	X=1; 
	
	
	#h1, = plt.plot(ff2[ww2[X,:]>0], rr2[X,ww2[X,:]>0])
	#h2, = plt.plot(ff1[ww1[X,:]>0], rr1[X,ww1[X,:]>0])
	#h3, = plt.plot(ff2[ww2[X,:]>0], rr2[X,ww2[X,:]>0], 'b')
	#h4, = plt.plot([78, 78],[-5, 5], 'g--', linewidth=2)
	#plt.ylim([-3,2])
	
	#plt.ylabel('brightness temperature [K]')
	#plt.xlabel('frequency [MHz]') 
	
	
	#plt.legend([h2, h3, h4], ['3 foreground terms','4 foreground terms', '78 MHz'])
	
	
	
	
	
	
	fig, ax = plt.subplots()
	z = ba.frequency2redshift(ff1)
		
	#h1, = ax.plot(ff2[ww2[X,:]>0], rr2[X,ww2[X,:]>0])
	#h2, = ax.plot(ff1[ww1[X,:]>0], rr1[X,ww1[X,:]>0])
	h3, = ax.plot(ff2[ww2[X,:]>0], rr2[X,ww2[X,:]>0], 'b')
	h4, = ax.plot(ff3[ww3[X,:]>0], rr3[X,ww3[X,:]>0]-0.5, 'r')
	h5, = ax.plot([78, 78],[-5, 5], 'g--', linewidth=2)
	
	xlow  = 58
	xhigh = 110
	ax.set_xlim([xlow, xhigh])
	
	ax2 = ax.twiny()
	
	# Fix these lines showing the redshift
	#ax2.set_xticks(np.array((60-xlow, 80-xlow, 100-xlow, 120-xlow, 140-xlow, 160-xlow))/len(ff1))
	ax2.set_xticks(np.array([(59.2-xlow), (67.6-xlow), 78.9-xlow, 88.77-xlow, 101.5-xlow])/(xhigh-xlow)) # , 129.1-xlow, 157.8-xlow
	ax2.set_xticklabels([23, 20, 17, 15, 13])  # , 10, 8
	
	ax.set_yticks([-1.0, -0.5, 0, 0.5])
	
	
	
	
	
	ax.set_ylim([-1.1, 0.5])
	ax.set_ylabel('brightness temperature [K]', fontsize=12)
	ax2.set_xlabel('redshift', fontsize=12)
	ax.set_xlabel('frequency [MHz]', fontsize=12) 
	
	plt.legend([h3, h4, h5], ['4 foreground terms', '5 foreground terms', '78 MHz'], loc=4)
	
	
	return 0








			
	