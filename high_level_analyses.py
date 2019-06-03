
import numpy as np
import matplotlib.pyplot as plt

import edges as eg
import basic as ba
import rfi as rfi

import calibration as cal
import calibration_receiver1 as cr1

import reflection_coefficient as rc

import data_models as dm
import getdist_plots as gp


from os import listdir


import os, sys
edges_folder       = os.environ['EDGES_vol2']
print('EDGES Folder: ' + edges_folder)

sys.path.insert(0, "/home/raul/edges/old")

import old_edges as oeg



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
























def batch_mid_band_level2_to_level3(case, first_day, last_day):


	# Case selection

	if case == 0:
		flag_folder       = 'case0'
		
		receiver_cal_file = 4   # 8 terms over 50-150 MHz
		
		antenna_s11_day   = 147
		antenna_s11_case  = 6   # average between cases 3 and 6
		antenna_s11_Nfit  = 13  # 13 terms over 55-120 MHz
		
		balun_correction  = 1
		ground_correction = 1
		beam_correction   = 1
		bf_case           = 0   # alan0 beam (30x30m ground plane), haslam map with gaussian lat-function for spectral index
		
		FLOW  = 55
		FHIGH = 120
		Nfg   = 5


	if case == 1:
		flag_folder       = 'case1'
		
		receiver_cal_file = 1   # cterms=7, wterms=7 terms over 50-150 MHz
		
		antenna_s11_day   = 147
		antenna_s11_case  = 3   # taken 2+ minutes after turning on the switch
		antenna_s11_Nfit  = 13  # 13 terms over 55-120 MHz
		
		balun_correction  = 1
		ground_correction = 1
		beam_correction   = 1
		bf_case           = 0   # alan0 beam (30x30m ground plane), haslam map with gaussian lat-function for spectral index
		
		FLOW  = 55
		FHIGH = 120
		Nfg   = 5

	
	if case == 2:
		flag_folder       = 'case2'
		
		receiver_cal_file = 2   # cterms=7, wterms=8 terms over 50-150 MHz
		
		antenna_s11_day   = 147
		antenna_s11_case  = 3   # taken 2+ minutes after turning on the switch
		antenna_s11_Nfit  = 13  # 13 terms over 55-120 MHz
		
		balun_correction  = 1
		ground_correction = 1
		beam_correction   = 1
		bf_case           = 0   # alan0 beam (30x30m ground plane), haslam map with gaussian lat-function for spectral index
		
		FLOW  = 55
		FHIGH = 120
		Nfg   = 5
		
		
		
		
	# ----------------------------------------------------------------------------------------------------
	# Same data as case=2 (nominal), but after RFI cleaning of the integrated spectra
	if case == 21:
		flag_folder       = 'case21'
		
		receiver_cal_file = 21   # cterms=7, wterms=7 terms over 50-150 MHz
		
		antenna_s11_day   = 147
		antenna_s11_case  = 3   # taken 2+ minutes after turning on the switch
		antenna_s11_Nfit  = 13  # 13 terms over 55-120 MHz
		
		balun_correction  = 1
		ground_correction = 1
		beam_correction   = 1
		bf_case           = 0   # alan0 beam (30x30m ground plane), haslam map with gaussian lat-function for spectral index
		
		FLOW  = 55
		FHIGH = 120
		Nfg   = 5
			
	
		

	# Same data as case=2 (nominal), but after RFI cleaning of the integrated spectra
	if case == 22:
		flag_folder       = 'case22'
		
		receiver_cal_file = 22   # cterms=7, wterms=8 terms over 50-150 MHz
		
		antenna_s11_day   = 147
		antenna_s11_case  = 3   # taken 2+ minutes after turning on the switch
		antenna_s11_Nfit  = 13  # 13 terms over 55-120 MHz
		
		balun_correction  = 1
		ground_correction = 1
		beam_correction   = 1
		bf_case           = 0   # alan0 beam (30x30m ground plane), haslam map with gaussian lat-function for spectral index
		
		FLOW  = 55
		FHIGH = 120
		Nfg   = 5	



	# Same data as case=2 (nominal), but after RFI cleaning of the integrated spectra
	if case == 23:
		flag_folder       = 'case23'
		
		receiver_cal_file = 23   # cterms=7, wterms=9 terms over 50-150 MHz
		
		antenna_s11_day   = 147
		antenna_s11_case  = 3   # taken 2+ minutes after turning on the switch
		antenna_s11_Nfit  = 13  # 13 terms over 55-120 MHz
		
		balun_correction  = 1
		ground_correction = 1
		beam_correction   = 1
		bf_case           = 0   # alan0 beam (30x30m ground plane), haslam map with gaussian lat-function for spectral index
		
		FLOW  = 55
		FHIGH = 120
		Nfg   = 5
		
		
		
	# Same data as case=2 (nominal), but after RFI cleaning of the integrated spectra
	if case == 24:
		flag_folder       = 'case24'
		
		receiver_cal_file = 24   # cterms=7, wterms=10 terms over 50-150 MHz
		
		antenna_s11_day   = 147
		antenna_s11_case  = 3   # taken 2+ minutes after turning on the switch
		antenna_s11_Nfit  = 13  # 13 terms over 55-120 MHz
		
		balun_correction  = 1
		ground_correction = 1
		beam_correction   = 1
		bf_case           = 0   # alan0 beam (30x30m ground plane), haslam map with gaussian lat-function for spectral index
		
		FLOW  = 55
		FHIGH = 120
		Nfg   = 5		



	# Same data as case=2 (nominal), but after RFI cleaning of the integrated spectra
	if case == 25:
		flag_folder       = 'case25'
		
		receiver_cal_file = 25   # cterms=8, wterms=8 terms over 50-150 MHz
		
		antenna_s11_day   = 147
		antenna_s11_case  = 3   # taken 2+ minutes after turning on the switch
		antenna_s11_Nfit  = 13  # 13 terms over 55-120 MHz
		
		balun_correction  = 1
		ground_correction = 1
		beam_correction   = 1
		bf_case           = 0   # alan0 beam (30x30m ground plane), haslam map with gaussian lat-function for spectral index
		
		FLOW  = 55
		FHIGH = 120
		Nfg   = 5



	# Same data as case=2 (nominal), but after RFI cleaning of the integrated spectra
	if case == 26:
		flag_folder       = 'case26'
		
		receiver_cal_file = 26   # cterms=8, wterms=11 terms over 50-150 MHz
		
		antenna_s11_day   = 147
		antenna_s11_case  = 3   # taken 2+ minutes after turning on the switch
		antenna_s11_Nfit  = 13  # 13 terms over 55-120 MHz
		
		balun_correction  = 1
		ground_correction = 1
		beam_correction   = 1
		bf_case           = 0   # alan0 beam (30x30m ground plane), haslam map with gaussian lat-function for spectral index
		
		FLOW  = 55
		FHIGH = 120
		Nfg   = 5
		


	# Same data as case=2 (nominal), but after RFI cleaning of the integrated spectra
	if case == 40:
		flag_folder       = 'case40'
		
		receiver_cal_file = 40   # cterms=4, wterms=6 terms over 60-85 MHz
		
		antenna_s11_day   = 147
		antenna_s11_case  = 3   # taken 2+ minutes after turning on the switch
		antenna_s11_Nfit  = 11  # 11 terms over 60-85 MHz
		
		balun_correction  = 1
		ground_correction = 1
		beam_correction   = 1
		bf_case           = 0   # alan0 beam (30x30m ground plane), haslam map with gaussian lat-function for spectral index
		
		FLOW  = 60
		FHIGH = 85
		Nfg   = 3
		




	# -------------------------------------------------------------------------------------------------------------







	
		


	if case == 3:
		flag_folder       = 'case3'
		
		receiver_cal_file = 3   # cterms=7, wterms=15 terms over 50-150 MHz
		
		antenna_s11_day   = 147
		antenna_s11_case  = 3   # taken 2+ minutes after turning on the switch
		antenna_s11_Nfit  = 13  # 13 terms over 55-120 MHz
		
		balun_correction  = 1
		ground_correction = 1
		beam_correction   = 1
		bf_case           = 0   # alan0 beam (30x30m ground plane), haslam map with gaussian lat-function for spectral index
		
		FLOW  = 55
		FHIGH = 120
		Nfg   = 5




	if case == 4:
		flag_folder       = 'case4'
		
		receiver_cal_file = 7   # Receiver calibration using data with no RFI, cterms=9, wterms=9 terms over 50-150 MHz
		
		antenna_s11_day   = 147
		antenna_s11_case  = 3   # taken 2+ minutes after turning on the switch
		antenna_s11_Nfit  = 13  # 13 terms over 55-120 MHz
		
		balun_correction  = 1
		ground_correction = 1
		beam_correction   = 1
		bf_case           = 0   # alan0 beam (30x30m ground plane), haslam map with gaussian lat-function for spectral index
		
		FLOW  = 55
		FHIGH = 120
		Nfg   = 5




	if case == 5:
		flag_folder       = 'case5'
		
		receiver_cal_file = 8   # Receiver calibration using data with no RFI, cterms=8, wterms=8 terms over 50-150 MHz
		
		antenna_s11_day   = 147
		antenna_s11_case  = 3   # taken 2+ minutes after turning on the switch
		antenna_s11_Nfit  = 13  # 13 terms over 55-120 MHz
		
		balun_correction  = 1
		ground_correction = 1
		beam_correction   = 1
		bf_case           = 0   # alan0 beam (30x30m ground plane), haslam map with gaussian lat-function for spectral index
		
		FLOW  = 55
		FHIGH = 120
		Nfg   = 5






	# RFI cleaned lab data, 60-90 MHz
	if case == 20:
		flag_folder       = 'case20'
		
		receiver_cal_file = 20   # Receiver calibration using data with no RFI, cterms=8, wterms=8 terms over 50-150 MHz
		
		antenna_s11_day   = 147
		antenna_s11_case  = 3   # taken 2+ minutes after turning on the switch
		antenna_s11_Nfit  = 11  # 13 terms over 55-120 MHz
		
		balun_correction  = 1
		ground_correction = 1
		beam_correction   = 1
		bf_case           = 0   # alan0 beam (30x30m ground plane), haslam map with gaussian lat-function for spectral index
		
		FLOW  = 60
		FHIGH = 90
		Nfg   = 4




	# RFI cleaned lab data, 60-120 MHz
	if case == 30:
		flag_folder       = 'case30'
		
		receiver_cal_file = 30   # cterms=7, wterms=6 terms over 60-120 MHz
		
		antenna_s11_day   = 147
		antenna_s11_case  = 3   # taken 2+ minutes after turning on the switch
		antenna_s11_Nfit  = 13  # 13 terms over 60-120 MHz
		
		balun_correction  = 1
		ground_correction = 1
		beam_correction   = 1
		bf_case           = 0   # alan0 beam (30x30m ground plane), haslam map with gaussian lat-function for spectral index
		
		FLOW  = 60
		FHIGH = 120
		Nfg   = 5




	# RFI cleaned lab data, 60-120 MHz
	if case == 31:
		flag_folder       = 'case31'
		
		receiver_cal_file = 31   # cterms=7, wterms=8 terms over 60-120 MHz
		
		antenna_s11_day   = 147
		antenna_s11_case  = 3   # taken 2+ minutes after turning on the switch
		antenna_s11_Nfit  = 13  # 13 terms over 60-120 MHz
		
		balun_correction  = 1
		ground_correction = 1
		beam_correction   = 1
		bf_case           = 0   # alan0 beam (30x30m ground plane), haslam map with gaussian lat-function for spectral index
		
		FLOW  = 60
		FHIGH = 120
		Nfg   = 5




	# RFI cleaned lab data, 60-120 MHz
	if case == 32:
		flag_folder       = 'case32'
		
		receiver_cal_file = 32   # cterms=7, wterms=9 terms over 60-120 MHz
		
		antenna_s11_day   = 147
		antenna_s11_case  = 3   # taken 2+ minutes after turning on the switch
		antenna_s11_Nfit  = 13  # 13 terms over 60-120 MHz
		
		balun_correction  = 1
		ground_correction = 1
		beam_correction   = 1
		bf_case           = 0   # alan0 beam (30x30m ground plane), haslam map with gaussian lat-function for spectral index
		
		FLOW  = 60
		FHIGH = 120
		Nfg   = 5







	# RFI cleaned lab data, 60-120 MHz
	if case == 33:
		flag_folder       = 'case33'
		
		receiver_cal_file = 33   # cterms=6, wterms=4 terms over 60-120 MHz
		
		antenna_s11_day   = 147
		antenna_s11_case  = 3   # taken 2+ minutes after turning on the switch
		antenna_s11_Nfit  = 13  # 13 terms over 60-120 MHz
		
		balun_correction  = 1
		ground_correction = 1
		beam_correction   = 1
		bf_case           = 0   # alan0 beam (30x30m ground plane), haslam map with gaussian lat-function for spectral index
		
		FLOW  = 60
		FHIGH = 120
		Nfg   = 5






	# RFI cleaned lab data, 60-120 MHz
	if case == 34:
		flag_folder       = 'case34'
		
		receiver_cal_file = 34   # cterms=6, wterms=8 terms over 60-120 MHz
		
		antenna_s11_day   = 147
		antenna_s11_case  = 3   # taken 2+ minutes after turning on the switch
		antenna_s11_Nfit  = 13  # 13 terms over 60-120 MHz
		
		balun_correction  = 1
		ground_correction = 1
		beam_correction   = 1
		bf_case           = 0   # alan0 beam (30x30m ground plane), haslam map with gaussian lat-function for spectral index
		
		FLOW  = 60
		FHIGH = 120
		Nfg   = 5




	# RFI cleaned lab data, 60-120 MHz
	if case == 35:
		flag_folder       = 'case35'
		
		receiver_cal_file = 35   # cterms=6, wterms=9 terms over 60-120 MHz
		
		antenna_s11_day   = 147
		antenna_s11_case  = 3   # taken 2+ minutes after turning on the switch
		antenna_s11_Nfit  = 13  # 13 terms over 60-120 MHz
		
		balun_correction  = 1
		ground_correction = 1
		beam_correction   = 1
		bf_case           = 0   # alan0 beam (30x30m ground plane), haslam map with gaussian lat-function for spectral index
		
		FLOW  = 60
		FHIGH = 120
		Nfg   = 5






	# RFI cleaned lab data, 60-120 MHz
	if case == 36:
		flag_folder       = 'case36'
		
		receiver_cal_file = 35   # cterms=6, wterms=9 terms over 60-120 MHz
		
		antenna_s11_day   = 147
		antenna_s11_case  = 3   # taken 2+ minutes after turning on the switch
		antenna_s11_Nfit  = 12  # 12 terms over 60-120 MHz
		
		balun_correction  = 1
		ground_correction = 1
		beam_correction   = 1
		bf_case           = 0   # alan0 beam (30x30m ground plane), haslam map with gaussian lat-function for spectral index
		
		FLOW  = 60
		FHIGH = 120
		Nfg   = 5



	# RFI cleaned lab data, 60-120 MHz
	if case == 37:
		flag_folder       = 'case37'
		
		receiver_cal_file = 35   # cterms=6, wterms=9 terms over 60-120 MHz
		
		antenna_s11_day   = 147
		antenna_s11_case  = 3   # taken 2+ minutes after turning on the switch
		antenna_s11_Nfit  = 11  # 12 terms over 60-120 MHz
		
		balun_correction  = 1
		ground_correction = 1
		beam_correction   = 1
		bf_case           = 0   # alan0 beam (30x30m ground plane), haslam map with gaussian lat-function for spectral index
		
		FLOW  = 60
		FHIGH = 120
		Nfg   = 5




	# RFI cleaned lab data, 60-120 MHz
	if case == 38:
		flag_folder       = 'case38'
		
		receiver_cal_file = 35   # cterms=6, wterms=9 terms over 60-120 MHz
		
		antenna_s11_day   = 147
		antenna_s11_case  = 3   # taken 2+ minutes after turning on the switch
		antenna_s11_Nfit  = 10  # 12 terms over 60-120 MHz
		
		balun_correction  = 1
		ground_correction = 1
		beam_correction   = 1
		bf_case           = 0   # alan0 beam (30x30m ground plane), haslam map with gaussian lat-function for spectral index
		
		FLOW  = 60
		FHIGH = 120
		Nfg   = 5





	
	# RFI cleaned lab data, 60-120 MHz
	if case == 39:
		flag_folder       = 'case39'
		
		receiver_cal_file = 39   # cterms=5, wterms=9 terms over 60-120 MHz
		
		antenna_s11_day   = 147
		antenna_s11_case  = 3   # taken 2+ minutes after turning on the switch
		antenna_s11_Nfit  = 11  # 12 terms over 60-120 MHz
		
		balun_correction  = 1
		ground_correction = 1
		beam_correction   = 1
		bf_case           = 0   # alan0 beam (30x30m ground plane), haslam map with gaussian lat-function for spectral index
		
		FLOW  = 60
		FHIGH = 120
		Nfg   = 5














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
	for i in range(len(new_list)):  # range(4): #
		#print(i)
		
		day = int(new_list[i][5:8])
		
		if (day >= first_day) & (day <= last_day):
			print(day)
			
			o = eg.level2_to_level3('mid_band', new_list[i], flag_folder=flag_folder, receiver_cal_file=receiver_cal_file, antenna_s11_year=2018, antenna_s11_day=antenna_s11_day, antenna_s11_case=antenna_s11_case, antenna_s11_Nfit=antenna_s11_Nfit, balun_correction=balun_correction, ground_correction=ground_correction, beam_correction=beam_correction, FLOW=FLOW, FHIGH=FHIGH, Nfg=Nfg)
		
	return 0  # new_list #










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







def plot_MC_receiver():
	
	plt.figure(1)
	
	
	f, r1, r2, t1, t2, t3, t4, m1, m2, m3, m4 = cal.MC_error_propagation()
	
	plt.subplot(4,2,1)
	plt.plot(f, t1-m1)
	plt.plot(f, t2-m2)
	plt.title('Perturbation 1, Low Foreground')
	plt.xticks(np.arange(60,121,10), labels=[])
	plt.ylabel('T [K]')
	plt.legend(['Mid-Band Antenna S11', 'Low-Band 3 Antenna S11'])

	plt.subplot(4,2,2)
	plt.plot(f, t3-m3)
	plt.plot(f, t4-m4)
	plt.title('Perturbation 1, High Foreground')
	plt.xticks(np.arange(60,121,10), labels=[])
	
	
	
	f, r1, r2, t1, t2, t3, t4, m1, m2, m3, m4 = cal.MC_error_propagation()
	
	plt.subplot(4,2,3)
	plt.plot(f, t1-m1)
	plt.plot(f, t2-m2)
	plt.title('Perturbation 2, Low Foreground')
	plt.xticks(np.arange(60,121,10), labels=[])
	plt.ylabel('T [K]')

	plt.subplot(4,2,4)
	plt.plot(f, t3-m3)
	plt.plot(f, t4-m4)	
	plt.title('Perturbation 2, High Foreground')
	plt.xticks(np.arange(60,121,10), labels=[])
	

	f, r1, r2, t1, t2, t3, t4, m1, m2, m3, m4 = cal.MC_error_propagation()
	
	plt.subplot(4,2,5)
	plt.plot(f, t1-m1)
	plt.plot(f, t2-m2)
	plt.title('Perturbation 3, Low Foreground')
	plt.xticks(np.arange(60,121,10), labels=[])
	plt.ylabel('T [K]')

	plt.subplot(4,2,6)
	plt.plot(f, t3-m3)
	plt.plot(f, t4-m4)
	plt.title('Perturbation 3, High Foreground')
	plt.xticks(np.arange(60,121,10), labels=[])
	
	
	
	f, r1, r2, t1, t2, t3, t4, m1, m2, m3, m4 = cal.MC_error_propagation()
	
	plt.subplot(4,2,7)
	plt.plot(f, t1-m1)
	plt.plot(f, t2-m2)
	plt.title('Perturbation 4, Low Foreground')
	plt.xlabel('frequency [MHz]')
	plt.ylabel('T [K]')

	plt.subplot(4,2,8)
	plt.plot(f, t3-m3)
	plt.plot(f, t4-m4)
	plt.title('Perturbation 4, High Foreground')
	plt.xlabel('frequency [MHz]')
	
	
	
	





	plt.figure(2)
	plt.subplot(1,2,1)
	plt.plot(f, 20*np.log10(np.abs(r1)))
	plt.plot(f, 20*np.log10(np.abs(r2)))
	plt.xlabel('frequency [MHz]')
	plt.ylabel('magnitude [dB]')
	
	plt.legend(['Mid-Band Antenna S11', 'Low-Band 3 Antenna S11'])
	#plt.title('antenna reflection coefficient')
	
	
		
	plt.subplot(1,2,2)
	plt.plot(f, (180/np.pi)*np.unwrap(np.angle(r1)))
	plt.plot(f, (180/np.pi)*np.unwrap(np.angle(r2)))
	plt.xlabel('frequency [MHz]')
	plt.ylabel('phase [deg]')
	
	#plt.legend(['Mid-Band Antenna S11', 'Low-Band 3 Antenna S11'])
	#plt.title('antenna reflection coefficient')	
	
	


	return 0
			
	
	
	
	
	
def receiver1_switch_crosscheck():
	
	path15 = '/home/raul/DATA/EDGES_old/calibration/receiver_calibration/low_band1/2015_08_25C/data/s11/raw/20150903/switch25degC/'
	
	
	# Measurements at 25degC 
	o_sw_m15, fd  = rc.s1p_read(path15 + 'open.S1P')
	s_sw_m15, fd  = rc.s1p_read(path15 + 'short.S1P')
	l_sw_m15, fd  = rc.s1p_read(path15 + 'load.S1P')

	o_sw_in15, fd = rc.s1p_read(path15 + 'open_input.S1P')
	s_sw_in15, fd = rc.s1p_read(path15 + 'short_input.S1P')
	l_sw_in15, fd = rc.s1p_read(path15 + 'load_input.S1P')
	
	
	
	# Standards assumed at the switch
	o_sw =  1 * np.ones(len(fd))
	s_sw = -1 * np.ones(len(fd))
	l_sw =  0 * np.ones(len(fd))	



	# Correction at the switch -- 25degC
	om15, xx1, xx2, xx3 = rc.de_embed(o_sw, s_sw, l_sw, o_sw_m15, s_sw_m15, l_sw_m15, o_sw_in15)
	sm15, xx1, xx2, xx3 = rc.de_embed(o_sw, s_sw, l_sw, o_sw_m15, s_sw_m15, l_sw_m15, s_sw_in15)
	lm15, xx1, xx2, xx3 = rc.de_embed(o_sw, s_sw, l_sw, o_sw_m15, s_sw_m15, l_sw_m15, l_sw_in15)
	
	
	
	
	
	# Loading measurements
	path18     = '/home/raul/DATA/EDGES/mid_band/calibration/receiver_calibration/receiver1/2018_01_25C/data/s11/raw/InternalSwitch/'
	
	o_sw_m18, f = rc.s1p_read(path18 + 'Open01.s1p')
	s_sw_m18, f = rc.s1p_read(path18 + 'Short01.s1p')
	l_sw_m18, f = rc.s1p_read(path18 + 'Match01.s1p')

	o_sw_in18, f = rc.s1p_read(path18 + 'ExternalOpen01.s1p')
	s_sw_in18, f = rc.s1p_read(path18 + 'ExternalShort01.s1p')
	l_sw_in18, f = rc.s1p_read(path18 + 'ExternalMatch01.s1p')		

	
	# Standards assumed at the switch
	o_sw =  1 * np.ones(len(f))
	s_sw = -1 * np.ones(len(f))
	l_sw =  0 * np.ones(len(f))	


	# Correction at the switch
	om18, xx1, xx2, xx3 = rc.de_embed(o_sw, s_sw, l_sw, o_sw_m18, s_sw_m18, l_sw_m18, o_sw_in18)
	sm18, xx1, xx2, xx3 = rc.de_embed(o_sw, s_sw, l_sw, o_sw_m18, s_sw_m18, l_sw_m18, s_sw_in18)
	lm18, xx1, xx2, xx3 = rc.de_embed(o_sw, s_sw, l_sw, o_sw_m18, s_sw_m18, l_sw_m18, l_sw_in18)


	
	
	
	
	
	# Plot
	
	plt.figure(1)
	plt.subplot(2,3,1)
	plt.plot(fd/1e6, 20*np.log10(np.abs(om15)),'k')
	plt.plot(f/1e6, 20*np.log10(np.abs(om18)), 'r--')
	plt.plot(fd/1e6, 20*np.log10(np.abs(om15)),'k')
	plt.ylabel('magnitude [dB]')
	plt.title('Open Standard at the Receiver Input\n(Measured from the Switch)')
	
	plt.subplot(2,3,4)
	plt.plot(fd/1e6, (180/np.pi)*np.unwrap(np.angle(om15)),'k')
	plt.plot(f/1e6, (180/np.pi)*np.unwrap(np.angle(om18)), 'r--')
	plt.plot(fd/1e6, (180/np.pi)*np.unwrap(np.angle(om15)),'k')
	plt.ylabel('phase [deg]')
	plt.xlabel('frequency [MHz]')
	
	


	plt.subplot(2,3,2)
	plt.plot(fd/1e6, 20*np.log10(np.abs(sm15)),'k')
	plt.plot(f/1e6, 20*np.log10(np.abs(sm18)), 'r--')
	plt.plot(fd/1e6, 20*np.log10(np.abs(sm15)),'k')
	plt.title('Short Standard at the Receiver Input\n(Measured from the Switch)')
	
	plt.subplot(2,3,5)
	plt.plot(fd/1e6, (180/np.pi)*np.unwrap(np.angle(sm15)),'k')
	plt.plot(f/1e6, (180/np.pi)*np.unwrap(np.angle(sm18)), 'r--')
	plt.plot(fd/1e6, (180/np.pi)*np.unwrap(np.angle(sm15)),'k')
	plt.xlabel('frequency [MHz]')
	
	
	

	plt.subplot(2,3,3)
	plt.plot(fd/1e6, 20*np.log10(np.abs(lm15)),'k')
	plt.plot(f/1e6, 20*np.log10(np.abs(lm18)), 'r--')
	plt.plot(fd/1e6, 20*np.log10(np.abs(lm15)),'k')
	plt.title('50-ohm Load Standard at the Receiver Input\n(Measured from the Switch)')
	plt.legend(['September 2015','February 2018'])
	
	plt.subplot(2,3,6)
	plt.plot(fd/1e6, (180/np.pi)*np.unwrap(np.angle(lm15)),'k')
	plt.plot(f/1e6, (180/np.pi)*np.unwrap(np.angle(lm18)), 'r--')
	plt.plot(fd/1e6, (180/np.pi)*np.unwrap(np.angle(lm15)),'k')
	plt.xlabel('frequency [MHz]')
	
	
	
	
	
	plt.figure(2)
	z15 = rc.gamma2impedance(lm15,50)
	z18 = rc.gamma2impedance(lm18,50)
	
	plt.subplot(1,2,1)
	plt.plot(fd/1e6, np.real(z15), 'k')
	plt.plot(f/1e6, np.real(z18), 'r--')
	plt.plot(fd/1e6, np.real(z15), 'k')
	
	plt.ylabel(r'real(Z$_{50}$) [ohm]')
	plt.xlabel('frequency [MHz]')
	
	
	plt.subplot(1,2,2)
	plt.plot(fd/1e6, np.imag(z15), 'k')
	plt.plot(f/1e6, np.imag(z18), 'r--')
	plt.plot(fd/1e6, np.imag(z15), 'k')
	
	plt.ylabel(r'imag(Z$_{50}$) [ohm]')
	plt.xlabel('frequency [MHz]')
	plt.legend(['September 2015','February 2018'])
	
	

	
		
	return fd, om15, sm15, lm15, f, om18, sm18, lm18
	
	
	
	








def VNA_comparison():
	
	path_folder1  = edges_folder + 'others/vna_comparison/keysight_e5061a/'
	path_folder2  = edges_folder + 'others/vna_comparison/copper_mountain_r60/'
	path_folder3  = edges_folder + 'others/vna_comparison/tektronix_ttr506a/'
	path_folder4  = edges_folder + 'others/vna_comparison/copper_mountain_tr1300/'

	o_K, f       = rc.s1p_read(path_folder1 + 'AGILENT_E5061A_OPEN.s1p')
	s_K, f       = rc.s1p_read(path_folder1 + 'AGILENT_E5061A_SHORT.s1p')
	m_K, f       = rc.s1p_read(path_folder1 + 'AGILENT_E5061A_MATCH.s1p')
	at3_K, f     = rc.s1p_read(path_folder1 + 'AGILENT_E5061A_3dB_ATTENUATOR.s1p')
	at6_K, f     = rc.s1p_read(path_folder1 + 'AGILENT_E5061A_6dB_ATTENUATOR.s1p')
	at10_K, f    = rc.s1p_read(path_folder1 + 'AGILENT_E5061A_10dB_ATTENUATOR.s1p')	
	at15_K, f    = rc.s1p_read(path_folder1 + 'AGILENT_E5061A_15dB_ATTENUATOR.s1p')	

	o_R, f       = rc.s1p_read(path_folder2 + 'OPEN.s1p')
	s_R, f       = rc.s1p_read(path_folder2 + 'SHORT.s1p')
	m_R, f       = rc.s1p_read(path_folder2 + 'MATCH.s1p')
	at3_R, f     = rc.s1p_read(path_folder2 + '3dB_ATTENUATOR.s1p')
	at6_R, f     = rc.s1p_read(path_folder2 + '6dB_ATTENUATOR.s1p')
	at10_R, f    = rc.s1p_read(path_folder2 + '10dB_ATTENUATOR.s1p')	
	at15_R, f    = rc.s1p_read(path_folder2 + '15dB_ATTENUATOR.s1p')	

	o_T, f       = rc.s1p_read(path_folder3 + 'uncalibrated_Open02.s1p')
	s_T, f       = rc.s1p_read(path_folder3 + 'uncalibrated_Short02.s1p')
	m_T, f       = rc.s1p_read(path_folder3 + 'uncalibrated_Match02.s1p')
	at3_T, f     = rc.s1p_read(path_folder3 + 'uncalibrated_3dB_Measurment2.s1p')
	at6_T, f     = rc.s1p_read(path_folder3 + 'uncalibrated_6dB_Measurment2.s1p')
	at10_T, f    = rc.s1p_read(path_folder3 + 'uncalibrated_10dB_Measurment2.s1p')	
	at15_T, f    = rc.s1p_read(path_folder3 + 'uncalibrated_15dB_Measurment2.s1p')	

	o_C, f       = rc.s1p_read(path_folder4 + 'Open_Measurment_01.s1p')
	s_C, f       = rc.s1p_read(path_folder4 + 'Short_Measurment_01.s1p')
	m_C, f       = rc.s1p_read(path_folder4 + 'Match_Measurment_01.s1p')
	at3_C, f     = rc.s1p_read(path_folder4 + '3dB_Measurment_01.s1p')
	at6_C, f     = rc.s1p_read(path_folder4 + '6dB_Measurment_01.s1p')
	at10_C, f    = rc.s1p_read(path_folder4 + '10dB_Measurment_01.s1p')
	at15_C, f    = rc.s1p_read(path_folder4 + '15dB_Measurment_01.s1p')









		
	# Standard values assumed
	#o_a =  1 * np.ones(len(f))
	#s_a = -1 * np.ones(len(f))
	#m_a =  0 * np.ones(len(f))	


	xx  = rc.agilent_85033E(f, 50, m = 1, md_value_ps = 38)
	o_a = xx[0]
	s_a = xx[1]
	m_a = xx[2]





	# Correction 
	at3_Rc, xx1, xx2, xx3  = rc.de_embed(o_a, s_a, m_a, o_R, s_R, m_R, at3_R)
	at6_Rc, xx1, xx2, xx3  = rc.de_embed(o_a, s_a, m_a, o_R, s_R, m_R, at6_R)
	at10_Rc, xx1, xx2, xx3 = rc.de_embed(o_a, s_a, m_a, o_R, s_R, m_R, at10_R)
	at15_Rc, xx1, xx2, xx3 = rc.de_embed(o_a, s_a, m_a, o_R, s_R, m_R, at15_R)

	at3_Tc, xx1, xx2, xx3  = rc.de_embed(o_a, s_a, m_a, o_T, s_T, m_T, at3_T)
	at6_Tc, xx1, xx2, xx3  = rc.de_embed(o_a, s_a, m_a, o_T, s_T, m_T, at6_T)
	at10_Tc, xx1, xx2, xx3 = rc.de_embed(o_a, s_a, m_a, o_T, s_T, m_T, at10_T)
	at15_Tc, xx1, xx2, xx3 = rc.de_embed(o_a, s_a, m_a, o_T, s_T, m_T, at15_T)	
	
	at3_Kc, xx1, xx2, xx3  = rc.de_embed(o_a, s_a, m_a, o_K, s_K, m_K, at3_K)
	at6_Kc, xx1, xx2, xx3  = rc.de_embed(o_a, s_a, m_a, o_K, s_K, m_K, at6_K)
	at10_Kc, xx1, xx2, xx3 = rc.de_embed(o_a, s_a, m_a, o_K, s_K, m_K, at10_K)
	at15_Kc, xx1, xx2, xx3 = rc.de_embed(o_a, s_a, m_a, o_K, s_K, m_K, at15_K)
	
	at3_Cc, xx1, xx2, xx3  = rc.de_embed(o_a, s_a, m_a, o_C, s_C, m_C, at3_C)
	at6_Cc, xx1, xx2, xx3  = rc.de_embed(o_a, s_a, m_a, o_C, s_C, m_C, at6_C)
	at10_Cc, xx1, xx2, xx3 = rc.de_embed(o_a, s_a, m_a, o_C, s_C, m_C, at10_C)
	at15_Cc, xx1, xx2, xx3 = rc.de_embed(o_a, s_a, m_a, o_C, s_C, m_C, at15_C)




	# Plot
	
	plt.figure(1)
	
	plt.subplot(4,2,1)
	plt.plot(f/1e6, 20*np.log10(np.abs(at3_Kc)))
	plt.plot(f/1e6, 20*np.log10(np.abs(at3_Rc)))
	plt.plot(f/1e6, 20*np.log10(np.abs(at3_Tc)))
	plt.plot(f/1e6, 20*np.log10(np.abs(at3_Cc)))		
	plt.ylabel('3-dB Attn [dB]')
	plt.title('MAGNITUDE')
	plt.legend(['Keysight E5061A', 'CM R60', 'Tektronix', 'CM TR1300'])


	plt.subplot(4,2,2)
	plt.plot(f/1e6, (180/np.pi)*np.unwrap(np.angle(at3_Rc)) - (180/np.pi)*np.unwrap(np.angle(at3_Kc)))
	plt.plot(f/1e6, (180/np.pi)*np.unwrap(np.angle(at3_Rc)) - (180/np.pi)*np.unwrap(np.angle(at3_Kc)))
	plt.plot(f/1e6, (180/np.pi)*np.unwrap(np.angle(at3_Tc)) - (180/np.pi)*np.unwrap(np.angle(at3_Kc)))
	plt.plot(f/1e6, (180/np.pi)*np.unwrap(np.angle(at3_Cc)) - (180/np.pi)*np.unwrap(np.angle(at3_Kc)))
	plt.ylabel('3-dB Attn [degrees]')
	plt.title(r'$\Delta$ PHASE')





	plt.subplot(4,2,3)
	plt.plot(f/1e6, 20*np.log10(np.abs(at6_Kc)))
	plt.plot(f/1e6, 20*np.log10(np.abs(at6_Rc)))
	plt.plot(f/1e6, 20*np.log10(np.abs(at6_Tc)))
	plt.plot(f/1e6, 20*np.log10(np.abs(at6_Cc)))
	plt.ylabel('6-dB Attn [dB]')




	plt.subplot(4,2,4)
	plt.plot(f/1e6, (180/np.pi)*np.unwrap(np.angle(at6_Rc)) - (180/np.pi)*np.unwrap(np.angle(at6_Kc)))
	plt.plot(f/1e6, (180/np.pi)*np.unwrap(np.angle(at6_Rc)) - (180/np.pi)*np.unwrap(np.angle(at6_Kc)))
	plt.plot(f/1e6, (180/np.pi)*np.unwrap(np.angle(at6_Tc)) - (180/np.pi)*np.unwrap(np.angle(at6_Kc)))
	plt.plot(f/1e6, (180/np.pi)*np.unwrap(np.angle(at6_Cc)) - (180/np.pi)*np.unwrap(np.angle(at6_Kc)))
	plt.ylabel('6-dB Attn [degrees]')



	plt.subplot(4,2,5)
	plt.plot(f/1e6, 20*np.log10(np.abs(at10_Kc)))
	plt.plot(f/1e6, 20*np.log10(np.abs(at10_Rc)))
	plt.plot(f/1e6, 20*np.log10(np.abs(at10_Tc)))
	plt.plot(f/1e6, 20*np.log10(np.abs(at10_Cc)))
	plt.ylabel('10-dB Attn [dB]')

	plt.subplot(4,2,6)
	plt.plot(f/1e6, (180/np.pi)*np.unwrap(np.angle(at10_Rc)) - (180/np.pi)*np.unwrap(np.angle(at10_Kc)))
	plt.plot(f/1e6, (180/np.pi)*np.unwrap(np.angle(at10_Rc)) - (180/np.pi)*np.unwrap(np.angle(at10_Kc)))
	plt.plot(f/1e6, (180/np.pi)*np.unwrap(np.angle(at10_Tc)) - (180/np.pi)*np.unwrap(np.angle(at10_Kc)))
	plt.plot(f/1e6, (180/np.pi)*np.unwrap(np.angle(at10_Cc)) - (180/np.pi)*np.unwrap(np.angle(at10_Kc)))
	plt.ylabel('10-dB Attn [degrees]')
	
	

	plt.subplot(4,2,7)
	plt.plot(f/1e6, 20*np.log10(np.abs(at15_Kc)))
	plt.plot(f/1e6, 20*np.log10(np.abs(at15_Rc)))
	plt.plot(f/1e6, 20*np.log10(np.abs(at15_Tc)))
	plt.plot(f/1e6, 20*np.log10(np.abs(at15_Cc)))
	plt.xlabel('frequency [MHz]')
	plt.ylabel('15-dB Attn [dB]')

	plt.subplot(4,2,8)
	plt.plot(f/1e6, (180/np.pi)*np.unwrap(np.angle(at15_Rc)) - (180/np.pi)*np.unwrap(np.angle(at15_Kc)))
	plt.plot(f/1e6, (180/np.pi)*np.unwrap(np.angle(at15_Rc)) - (180/np.pi)*np.unwrap(np.angle(at15_Kc)))
	plt.plot(f/1e6, (180/np.pi)*np.unwrap(np.angle(at15_Tc)) - (180/np.pi)*np.unwrap(np.angle(at15_Kc)))
	plt.plot(f/1e6, (180/np.pi)*np.unwrap(np.angle(at15_Cc)) - (180/np.pi)*np.unwrap(np.angle(at15_Kc)))
	plt.xlabel('frequency [MHz]')
	plt.ylabel('15-dB Attn [degrees]')

	return f, at3_K, at6_K, at10_K, at15_K, at3_Kc, at6_Kc, at10_Kc, at15_Kc, at3_R, at6_R, at10_R, at15_R, at3_Rc, at6_Rc, at10_Rc, at15_Rc

	













def VNA_comparison2():
	
	path_folder1  = edges_folder + 'others/vna_comparison/again/ks_e5061a/'
	path_folder2  = edges_folder + 'others/vna_comparison/again/cm_tr1300/'


	o_K1, f       = rc.s1p_read(path_folder1 + 'Open_Measurment_01.s1p')
	s_K1, f       = rc.s1p_read(path_folder1 + 'Short_Measurment_01.s1p')
	m_K1, f       = rc.s1p_read(path_folder1 + 'Match_Measurment_01.s1p')
	at3_K1, f     = rc.s1p_read(path_folder1 + '3dB_Mearsurment_01.s1p')
	at6_K1, f     = rc.s1p_read(path_folder1 + '6dB_Mearsurment_01.s1p')
	at10_K1, f    = rc.s1p_read(path_folder1 + '10dB_Mearsurment_01.s1p')	
	at15_K1, f    = rc.s1p_read(path_folder1 + '15dB_Mearsurment_01.s1p')

	o_K2, f       = rc.s1p_read(path_folder1 + 'Open_Measurment_02.s1p')
	s_K2, f       = rc.s1p_read(path_folder1 + 'Short_Measurment_02.s1p')
	m_K2, f       = rc.s1p_read(path_folder1 + 'Match_Measurment_02.s1p')
	at3_K2, f     = rc.s1p_read(path_folder1 + '3dB_Mearsurment_02.s1p')
	at6_K2, f     = rc.s1p_read(path_folder1 + '6dB_Mearsurment_02.s1p')
	at10_K2, f    = rc.s1p_read(path_folder1 + '10dB_Mearsurment_02.s1p')	
	at15_K2, f    = rc.s1p_read(path_folder1 + '15dB_Mearsurment_02.s1p')	



	o_C1, f       = rc.s1p_read(path_folder2 + 'Open_Measurment_01.s1p')
	s_C1, f       = rc.s1p_read(path_folder2 + 'Short_Measurment_01.s1p')
	m_C1, f       = rc.s1p_read(path_folder2 + 'Match_Measurment_01.s1p')
	at3_C1, f     = rc.s1p_read(path_folder2 + '3dB_Measurment_01.s1p')
	at6_C1, f     = rc.s1p_read(path_folder2 + '6dB_Measurment_01.s1p')
	at10_C1, f    = rc.s1p_read(path_folder2 + '10dB_Measurment_01.s1p')
	at15_C1, f    = rc.s1p_read(path_folder2 + '15dB_Measurment_01.s1p')

	o_C2, f       = rc.s1p_read(path_folder2 + 'Open_Measurment_02.s1p')
	s_C2, f       = rc.s1p_read(path_folder2 + 'Short_Measurment_02.s1p')
	m_C2, f       = rc.s1p_read(path_folder2 + 'Match_Measurment_02.s1p')
	at3_C2, f     = rc.s1p_read(path_folder2 + '3dB_Measurment_02.s1p')
	at6_C2, f     = rc.s1p_read(path_folder2 + '6dB_Measurment_02.s1p')
	at10_C2, f    = rc.s1p_read(path_folder2 + '10dB_Measurment_02.s1p')
	at15_C2, f    = rc.s1p_read(path_folder2 + '15dB_Measurment_02.s1p')

	

	xx  = rc.agilent_85033E(f, 50, m = 1, md_value_ps = 38)
	o_a = xx[0]
	s_a = xx[1]
	m_a = xx[2]





	# Correction	
	#at3_K1c, xx1, xx2, xx3  = rc.de_embed(o_a, s_a, m_a, o_K1, s_K1, m_K1, at3_K1)
	#at6_K1c, xx1, xx2, xx3  = rc.de_embed(o_a, s_a, m_a, o_K1, s_K1, m_K1, at6_K1)
	#at10_K1c, xx1, xx2, xx3 = rc.de_embed(o_a, s_a, m_a, o_K1, s_K1, m_K1, at10_K1)
	#at15_K1c, xx1, xx2, xx3 = rc.de_embed(o_a, s_a, m_a, o_K1, s_K1, m_K1, at15_K1)
	
	#at3_C1c, xx1, xx2, xx3  = rc.de_embed(o_a, s_a, m_a, o_C1, s_C1, m_C1, at3_C1)
	#at6_C1c, xx1, xx2, xx3  = rc.de_embed(o_a, s_a, m_a, o_C1, s_C1, m_C1, at6_C1)
	#at10_C1c, xx1, xx2, xx3 = rc.de_embed(o_a, s_a, m_a, o_C1, s_C1, m_C1, at10_C1)
	#at15_C1c, xx1, xx2, xx3 = rc.de_embed(o_a, s_a, m_a, o_C1, s_C1, m_C1, at15_C1)	
	
	
	at3_Kc, xx1, xx2, xx3  = rc.de_embed(o_a, s_a, m_a, o_K2, s_K2, m_K2, at3_K2)
	at6_Kc, xx1, xx2, xx3  = rc.de_embed(o_a, s_a, m_a, o_K2, s_K2, m_K2, at6_K2)
	at10_Kc, xx1, xx2, xx3 = rc.de_embed(o_a, s_a, m_a, o_K2, s_K2, m_K2, at10_K2)
	at15_Kc, xx1, xx2, xx3 = rc.de_embed(o_a, s_a, m_a, o_K2, s_K2, m_K2, at15_K2)

	at3_Cc, xx1, xx2, xx3  = rc.de_embed(o_a, s_a, m_a, o_C2, s_C2, m_C2, at3_C2)
	at6_Cc, xx1, xx2, xx3  = rc.de_embed(o_a, s_a, m_a, o_C2, s_C2, m_C2, at6_C2)
	at10_Cc, xx1, xx2, xx3 = rc.de_embed(o_a, s_a, m_a, o_C2, s_C2, m_C2, at10_C2)
	at15_Cc, xx1, xx2, xx3 = rc.de_embed(o_a, s_a, m_a, o_C2, s_C2, m_C2, at15_C2)



	# Plot
	
	plt.figure(1, figsize=(15, 10))
	
	plt.subplot(4,2,1)
	plt.plot(f/1e6, 20*np.log10(np.abs(at3_Kc)),'k')
	plt.plot(f/1e6, 20*np.log10(np.abs(at3_Cc)),'r')
	
	
	
	plt.ylabel('3-dB Attn [dB]')
	plt.title('MAGNITUDE')
	plt.legend(['Keysight E5061A', 'CM TR1300'])

	plt.subplot(4,2,2)
	plt.plot(f/1e6, (180/np.pi)*np.unwrap(np.angle(at3_Cc)) - (180/np.pi)*np.unwrap(np.angle(at3_Kc)),'r')
	plt.ylabel('3-dB Attn [degrees]')
	plt.title(r'$\Delta$ PHASE')




	plt.subplot(4,2,3)
	plt.plot(f/1e6, 20*np.log10(np.abs(at6_Kc)),'k')
	plt.plot(f/1e6, 20*np.log10(np.abs(at6_Cc)),'r')
	plt.ylabel('6-dB Attn [dB]')

	plt.subplot(4,2,4)
	plt.plot(f/1e6, (180/np.pi)*np.unwrap(np.angle(at6_Cc)) - (180/np.pi)*np.unwrap(np.angle(at6_Kc)),'r')
	plt.ylabel('6-dB Attn [degrees]')




	plt.subplot(4,2,5)
	plt.plot(f/1e6, 20*np.log10(np.abs(at10_Kc)),'k')
	plt.plot(f/1e6, 20*np.log10(np.abs(at10_Cc)),'r')
	plt.ylabel('10-dB Attn [dB]')

	plt.subplot(4,2,6)
	plt.plot(f/1e6, (180/np.pi)*np.unwrap(np.angle(at10_Cc)) - (180/np.pi)*np.unwrap(np.angle(at10_Kc)),'r')
	plt.ylabel('10-dB Attn [degrees]')
	
	


	plt.subplot(4,2,7)
	plt.plot(f/1e6, 20*np.log10(np.abs(at15_Kc)),'k')
	plt.plot(f/1e6, 20*np.log10(np.abs(at15_Cc)),'r')
	plt.xlabel('frequency [MHz]')
	plt.ylabel('15-dB Attn [dB]')

	plt.subplot(4,2,8)
	plt.plot(f/1e6, (180/np.pi)*np.unwrap(np.angle(at15_Cc)) - (180/np.pi)*np.unwrap(np.angle(at15_Kc)),'r')
	plt.xlabel('frequency [MHz]')
	plt.ylabel('15-dB Attn [degrees]')


	plt.savefig(edges_folder + '/results/plots/20190415/vna_comparison.pdf', bbox_inches='tight')
	plt.close()	
	plt.close()
	plt.close()
	plt.close()



	return 0

	











def VNA_comparison3():
	
	path_folder1  = edges_folder + 'others/vna_comparison/fieldfox_N9923A/agilent_E5061A_male/'
	path_folder2  = edges_folder + 'others/vna_comparison/fieldfox_N9923A/agilent_E5061A_female/'
	path_folder3  = edges_folder + 'others/vna_comparison/fieldfox_N9923A/fieldfox_N9923A_male/'
	path_folder4  = edges_folder + 'others/vna_comparison/fieldfox_N9923A/fieldfox_N9923A_female/'

	REP = '02'


	A_o, f       = rc.s1p_read(path_folder1 + 'Open'+ REP +'.s1p')
	A_s, f       = rc.s1p_read(path_folder1 + 'Short'+ REP +'.s1p')
	A_m, f       = rc.s1p_read(path_folder1 + 'Match'+ REP +'.s1p')
	A_at3, f     = rc.s1p_read(path_folder1 + '3dB_'+ REP +'.s1p')
	A_at6, f     = rc.s1p_read(path_folder1 + '6dB_'+ REP +'.s1p')
	A_at10, f    = rc.s1p_read(path_folder1 + '10dB_'+ REP +'.s1p')	
	A_at15, f    = rc.s1p_read(path_folder1 + '15dB_'+ REP +'.s1p')	
	
	B_o, f       = rc.s1p_read(path_folder2 + 'Open_'+ REP +'.s1p')
	B_s, f       = rc.s1p_read(path_folder2 + 'Short_'+ REP +'.s1p')
	B_m, f       = rc.s1p_read(path_folder2 + 'Match_'+ REP +'.s1p')
	B_at3, f     = rc.s1p_read(path_folder2 + '3dB_'+ REP +'.s1p')
	B_at6, f     = rc.s1p_read(path_folder2 + '6dB_02.s1p')
	B_at10, f    = rc.s1p_read(path_folder2 + '10dB_'+ REP +'.s1p')	
	B_at15, f    = rc.s1p_read(path_folder2 + '15dB_'+ REP +'.s1p')
	
	C_o, f       = rc.s1p_read(path_folder3 + 'OPEN'+ REP +'.s1p')
	C_s, f       = rc.s1p_read(path_folder3 + 'SHORT'+ REP +'.s1p')
	C_m, f       = rc.s1p_read(path_folder3 + 'MATCH'+ REP +'.s1p')
	C_at3, f     = rc.s1p_read(path_folder3 + '3DB_'+ REP +'.s1p')
	C_at6, f     = rc.s1p_read(path_folder3 + '6DB_'+ REP +'.s1p')
	C_at10, f    = rc.s1p_read(path_folder3 + '10DB_'+ REP +'.s1p')	
	C_at15, f    = rc.s1p_read(path_folder3 + '15DB_'+ REP +'.s1p')	
	
	D_o, f       = rc.s1p_read(path_folder4 + 'OPEN'+ REP +'.s1p')
	D_s, f       = rc.s1p_read(path_folder4 + 'SHORT'+ REP +'.s1p')
	D_m, f       = rc.s1p_read(path_folder4 + 'MATCH'+ REP +'.s1p')
	D_at3, f     = rc.s1p_read(path_folder4 + '3DB_'+ REP +'.s1p')
	D_at6, f     = rc.s1p_read(path_folder4 + '6DB_02.s1p')
	D_at10, f    = rc.s1p_read(path_folder4 + '10DB_'+ REP +'.s1p')	
	D_at15, f    = rc.s1p_read(path_folder4 + '15DB_'+ REP +'.s1p')









		
	# Standard values assumed
	#o_a =  1 * np.ones(len(f))
	#s_a = -1 * np.ones(len(f))
	#m_a =  0 * np.ones(len(f))	


	xx  = rc.agilent_85033E(f, 50, m = 1, md_value_ps = 38)
	o_a = xx[0]
	s_a = xx[1]
	m_a = xx[2]





	# Correction 
	A_at3c, xx1, xx2, xx3  = rc.de_embed(o_a, s_a, m_a, A_o, A_s, A_m, A_at3)
	A_at6c, xx1, xx2, xx3  = rc.de_embed(o_a, s_a, m_a, A_o, A_s, A_m, A_at6)
	A_at10c, xx1, xx2, xx3 = rc.de_embed(o_a, s_a, m_a, A_o, A_s, A_m, A_at10)
	A_at15c, xx1, xx2, xx3 = rc.de_embed(o_a, s_a, m_a, A_o, A_s, A_m, A_at15)

	B_at3c, xx1, xx2, xx3  = rc.de_embed(o_a, s_a, m_a, B_o, B_s, B_m, B_at3)
	B_at6c, xx1, xx2, xx3  = rc.de_embed(o_a, s_a, m_a, B_o, B_s, B_m, B_at6)
	B_at10c, xx1, xx2, xx3 = rc.de_embed(o_a, s_a, m_a, B_o, B_s, B_m, B_at10)
	B_at15c, xx1, xx2, xx3 = rc.de_embed(o_a, s_a, m_a, B_o, B_s, B_m, B_at15)	
	
	C_at3c, xx1, xx2, xx3  = rc.de_embed(o_a, s_a, m_a, C_o, C_s, C_m, C_at3)
	C_at6c, xx1, xx2, xx3  = rc.de_embed(o_a, s_a, m_a, C_o, C_s, C_m, C_at6)
	C_at10c, xx1, xx2, xx3 = rc.de_embed(o_a, s_a, m_a, C_o, C_s, C_m, C_at10)
	C_at15c, xx1, xx2, xx3 = rc.de_embed(o_a, s_a, m_a, C_o, C_s, C_m, C_at15)
	
	D_at3c, xx1, xx2, xx3  = rc.de_embed(o_a, s_a, m_a, D_o, D_s, D_m, D_at3)
	D_at6c, xx1, xx2, xx3  = rc.de_embed(o_a, s_a, m_a, D_o, D_s, D_m, D_at6)
	D_at10c, xx1, xx2, xx3 = rc.de_embed(o_a, s_a, m_a, D_o, D_s, D_m, D_at10)
	D_at15c, xx1, xx2, xx3 = rc.de_embed(o_a, s_a, m_a, D_o, D_s, D_m, D_at15)



	# Plot
	
	plt.figure(1)
	
	plt.subplot(4,2,1)
	plt.plot(f/1e6, 20*np.log10(np.abs(A_at3c)),'b')
	plt.plot(f/1e6, 20*np.log10(np.abs(C_at3c)),'b--')
	plt.plot(f/1e6, 20*np.log10(np.abs(B_at3c)),'r')
	plt.plot(f/1e6, 20*np.log10(np.abs(D_at3c)),'r--')		
	plt.ylabel('3-dB Attn [dB]')
	plt.title('MAGNITUDE')


	plt.subplot(4,2,2)
	plt.plot(f/1e6, (180/np.pi)*np.unwrap(np.angle(C_at3c)) - (180/np.pi)*np.unwrap(np.angle(A_at3c)), 'b--')
	plt.plot(f/1e6, (180/np.pi)*np.unwrap(np.angle(D_at3c)) - (180/np.pi)*np.unwrap(np.angle(B_at3c)), 'r--')
	plt.ylabel('3-dB Attn [degrees]')
	plt.title(r'$\Delta$ PHASE')





	plt.subplot(4,2,3)
	plt.plot(f/1e6, 20*np.log10(np.abs(A_at6c)),'b')
	plt.plot(f/1e6, 20*np.log10(np.abs(C_at6c)),'b--')
	plt.plot(f/1e6, 20*np.log10(np.abs(B_at6c)),'r')
	plt.plot(f/1e6, 20*np.log10(np.abs(D_at6c)),'r--')
	plt.ylabel('6-dB Attn [dB]')




	plt.subplot(4,2,4)
	plt.plot(f/1e6, (180/np.pi)*np.unwrap(np.angle(C_at6c)) - (180/np.pi)*np.unwrap(np.angle(A_at6c)), 'b--')
	plt.plot(f/1e6, (180/np.pi)*np.unwrap(np.angle(D_at6c)) - (180/np.pi)*np.unwrap(np.angle(B_at6c)), 'r--')
	plt.ylabel('6-dB Attn [degrees]')



	plt.subplot(4,2,5)
	plt.plot(f/1e6, 20*np.log10(np.abs(A_at10c)),'b')
	plt.plot(f/1e6, 20*np.log10(np.abs(C_at10c)),'b--')
	plt.plot(f/1e6, 20*np.log10(np.abs(B_at10c)),'r')
	plt.plot(f/1e6, 20*np.log10(np.abs(D_at10c)),'r--')
	plt.ylabel('10-dB Attn [dB]')

	plt.subplot(4,2,6)
	plt.plot(f/1e6, (180/np.pi)*np.unwrap(np.angle(C_at10c)) - (180/np.pi)*np.unwrap(np.angle(A_at10c)), 'b--')
	plt.plot(f/1e6, (180/np.pi)*np.unwrap(np.angle(D_at10c)) - (180/np.pi)*np.unwrap(np.angle(B_at10c)), 'r--')
	plt.ylabel('10-dB Attn [degrees]')
	
	

	plt.subplot(4,2,7)
	plt.plot(f/1e6, 20*np.log10(np.abs(A_at15c)),'b')
	plt.plot(f/1e6, 20*np.log10(np.abs(C_at15c)),'b--')
	plt.plot(f/1e6, 20*np.log10(np.abs(B_at15c)),'r')
	plt.plot(f/1e6, 20*np.log10(np.abs(D_at15c)),'r--')
	plt.xlabel('frequency [MHz]')
	plt.ylabel('15-dB Attn [dB]')
	plt.legend(['Male E5061A', 'Male N9923A', 'Female E5061A', 'Female N9923A'])

	plt.subplot(4,2,8)
	plt.plot(f/1e6, (180/np.pi)*np.unwrap(np.angle(C_at15c)) - (180/np.pi)*np.unwrap(np.angle(A_at15c)), 'b--')
	plt.plot(f/1e6, (180/np.pi)*np.unwrap(np.angle(D_at15c)) - (180/np.pi)*np.unwrap(np.angle(B_at15c)), 'r--')
	plt.xlabel('frequency [MHz]')
	plt.ylabel('15-dB Attn [degrees]')

	return 0 #f, at3_K, at6_K, at10_K, at15_K, at3_Kc, at6_Kc, at10_Kc, at15_Kc, at3_R, at6_R, at10_R, at15_R, at3_Rc, at6_Rc, at10_Rc, at15_Rc





























def plots_midband_paper(plot_number):
	
	
	# Receiver calibration parameters
	if plot_number==1:
	

		# Paths
		path_plot_save = edges_folder + 'results/plots/20190407/'


		# Calibration parameters
		rcv_file = edges_folder + 'mid_band/calibration/receiver_calibration/receiver1/2018_01_25C/results/nominal/calibration_files/calibration_file_receiver1_50_150MHz_cterms7_wterms7.txt'

		rcv = np.genfromtxt(rcv_file)
	
		FLOW  = 50
		FHIGH = 150
		
		fX      = rcv[:,0]
		rcv2    = rcv[(fX>=FLOW) & (fX<=FHIGH),:]
		
		fe      = rcv2[:,0]
		rl      = rcv2[:,1] + 1j*rcv2[:,2]
		sca     = rcv2[:,3]
		off     = rcv2[:,4]
		TU      = rcv2[:,5]
		TC      = rcv2[:,6]
		TS      = rcv2[:,7]




		# Plot

		size_x = 5
		size_y = 6.5 #10.5
		x0 = 0.15
		y0 = 0.09
		dx = 0.7
		dy = 0.3


		f1  = plt.figure(num=1, figsize=(size_x, size_y))		


		ax     = f1.add_axes([x0, y0 + 2*dy, dx, dy])	
		h1     = ax.plot(fe, 20*np.log10(np.abs(rl)), 'b', linewidth=2, label='$|\Gamma_{\mathrm{rec}}|$')
		ax2    = ax.twinx()
		h2     = ax2.plot(fe, (180/np.pi)*np.unwrap(np.angle(rl)), 'r--', linewidth=2, label=r'$\angle\/\Gamma_{\mathrm{rec}}$')
		h      = h1 + h2
		labels = [l.get_label() for l in h]
		ax.legend(h, labels, loc=2, fontsize=13)

		ax.set_ylim([-41, -25])
		ax.set_xticklabels('')
		ax.set_yticks(np.arange(-39,-26,3))
		ax.set_ylabel('$|\Gamma_{\mathrm{rec}}|$ [dB]', fontsize=16)
		ax.text(42, -39.6, '(a)', fontsize=20)

		ax2.set_ylim([70, 130])
		ax2.set_xticklabels('')
		ax2.set_yticks(np.arange(80,121,10))		
		ax2.set_ylabel(r'$\angle\/\Gamma_{\mathrm{rec}}$ [ $^\mathrm{o}$]', fontsize=16)

		ax.set_xlim([40, 160])
		ax.tick_params(axis='x', direction='in')
		ax.set_xticks(np.arange(50, 151, 20))
		
		
		
		



		ax     = f1.add_axes([x0, y0 + 1*dy, dx, dy])
		h1     = ax.plot(fe, sca,'b',linewidth=2, label='$C_1$')
		ax2    = ax.twinx()
		h2     = ax2.plot(fe, off,'r--',linewidth=2, label='$C_2$')
		h      = h1 + h2
		labels = [l.get_label() for l in h]
		ax.legend(h, labels, loc=0, fontsize=13)

		ax.set_ylim([3, 5.5])
		ax.set_xticklabels('')
		ax.set_yticks(np.arange(3.5,5.1,0.5))
		ax.set_ylabel('$C_1$', fontsize=16)
		ax.text(42, 3.25, '(b)', fontsize=20)

		ax2.set_ylim([-2.4, -1.8])
		ax2.set_xticklabels('')
		ax2.set_yticks(np.arange(-2.3, -1.85, 0.1))
		ax2.set_ylabel('$C_2$ [K]', fontsize=16)

		ax.set_xlim([40, 160])
		ax.tick_params(axis='x', direction='in')
		ax.set_xticks(np.arange(50, 151, 20))
		
		
		
		



		ax     = f1.add_axes([x0, y0 + 0*dy, dx, dy])
		h1     = ax.plot(fe, TU,'b', linewidth=2, label='$T_{\mathrm{unc}}$')
		ax2    = ax.twinx()
		h2     = ax2.plot(fe, TC,'r--', linewidth=2, label='$T_{\mathrm{cos}}$')
		h3     = ax2.plot(fe, TS,'g--', linewidth=2, label='$T_{\mathrm{sin}}$')		

		h      = h1 + h2 + h3
		labels = [l.get_label() for l in h]
		ax.legend(h, labels, loc=0, fontsize=13, ncol=3)

		ax.set_ylim([178, 190])
		ax.set_yticks(np.arange(180, 189, 2))
		ax.set_ylabel('$T_{\mathrm{unc}}$ [K]', fontsize=16)
		ax.set_xlabel('$\\nu$ [MHz]', fontsize=16)
		ax.text(42, 179, '(c)', fontsize=20)

		ax2.set_ylim([-60, 40])
		ax2.set_yticks(np.arange(-40, 21, 20))
		ax2.set_ylabel('$T_{\mathrm{cos}}, T_{\mathrm{sin}}$ [K]', fontsize=16)
		
		ax.set_xlim([40, 160])
		ax.set_xticks(np.arange(50, 151, 20))


		plt.savefig(path_plot_save + 'receiver_calibration.pdf', bbox_inches='tight')
		plt.close()	
		plt.close()
		plt.close()
		plt.close()










	
	# Antenna calibration parameters
	if plot_number==2:
	

		# Paths
		path_plot_save = edges_folder + 'results/plots/20190422/'




		# Plot

		size_x = 3.5
		size_y = 9.0 #10.5
		x0 = 0.15
		y0 = 0.09
		dx = 0.8
		dy = 0.18


		f1  = plt.figure(num=1, figsize=(size_x, size_y))		










		# Frequency
		f, il, ih = ba.frequency_edges(50, 150)
		fe = f[il:ih+1]	


		# Antenna S11
		# -----------
		ra = cal.models_antenna_s11_remove_delay('mid_band', fe, year=2018, day=147, case=5, delay_0=0.17, model_type='polynomial', Nfit=15, plot_fit_residuals='no')

		xlb1  = np.genfromtxt('/run/media/raul/SSD_4TB/EDGES_vol1/calibration/antenna_s11/low_band1/s11/corrected/2016_243/S11_blade_low_band_2016_243.txt')
		flb1  = xlb1[:,0]/1e6
		ralb1 = xlb1[:,1] + 1j*xlb1[:,2]
		
		xlb2  = np.genfromtxt('/run/media/raul/SSD_4TB/EDGES_vol1/calibration/antenna_s11/low_band2/s11/corrected/2017-06-29-low2-noshield_average/S11_blade_low_band_2017_180_NO_SHIELD.txt')		
		flb2  = xlb2[:,0]/1e6
		ralb2 = xlb2[:,1] + 1j*xlb2[:,2]		
		
		

		ax     = f1.add_axes([x0, y0 + 3*dy, dx, dy])	
		h      = ax.plot(fe, 20*np.log10(np.abs(ra)), 'b', linewidth=1.5, label='$|\Gamma_{\mathrm{ant}}|$')
		h      = ax.plot(flb1[flb1<=110], 20*np.log10(np.abs(ralb1[flb1<=110])), 'r', linewidth=1.5, label='$|\Gamma_{\mathrm{ant}}|$')
		h      = ax.plot(flb2[flb2<=110], 20*np.log10(np.abs(ralb2[flb2<=110])), 'g', linewidth=1.5, label='$|\Gamma_{\mathrm{ant}}|$')
		
		
		
		
		#ax2    = ax.twinx()
		#h2     = ax2.plot(fe, (180/np.pi)*np.unwrap(np.angle(ra)), 'r--', linewidth=2, label=r'$\angle\/\Gamma_{\mathrm{ant}}$')
		#h      = h1 + h2
		#labels = [l.get_label() for l in h]
		ax.legend(['mid-band','low-band 1','low-band 2'], fontsize=9)

		#ax.set_ylim([-41, -25])
		ax.set_xticklabels('')
		#
		ax.set_ylabel('$|\Gamma_{\mathrm{ant}}|$ [dB]') #, fontsize=15)
		
		#
		#		
		#ax2.set_ylabel(r'$\angle\/\Gamma_{\mathrm{ant}}$ [ $^\mathrm{o}$]', fontsize=16)

		ax.set_xlim([45, 155])
		ax.set_ylim([-16, -4])
		ax.set_yticks(np.arange(-14,-5,2))
		#ax.set_xticklabels('')
		ax.tick_params(axis='x', direction='in')
		ax.set_xticks(np.arange(50, 151, 20))
		
		ax.text(144, -15.3, '(a)', fontsize=14)
		













		ax     = f1.add_axes([x0, y0 + 2*dy, dx, dy])	
		h      = ax.plot(fe, (180/np.pi)*np.unwrap(np.angle(ra)), 'b', linewidth=1.5, label=r'$\angle\/\Gamma_{\mathrm{ant}}$')
		h      = ax.plot(flb1[flb1<=110], (180/np.pi)*np.unwrap(np.angle(ralb1[flb1<=110])), 'r', linewidth=1.5, label=r'$\angle\/\Gamma_{\mathrm{ant}}$')
		h      = ax.plot(flb2[flb2<=110], (180/np.pi)*np.unwrap(np.angle(ralb2[flb2<=110])), 'g', linewidth=1.5, label=r'$\angle\/\Gamma_{\mathrm{ant}}$')
	
		
		#ax2    = ax.twinx()
		#h2     = ax2.plot(fe, (180/np.pi)*np.unwrap(np.angle(rl)), 'r--', linewidth=2, label=r'$\angle\/\Gamma_{\mathrm{rec}}$')
		#h      = h1 + h2
		#labels = [l.get_label() for l in h]
		#ax.legend(h, labels, loc=2, fontsize=13)

		#ax.set_ylim([-41, -25])
		#ax.set_xticklabels('')
		#ax.set_yticks(np.arange(-39,-26,3))
		#ax.set_ylabel('$|\Gamma_{\mathrm{rec}}|$ [dB]', fontsize=16)
		ax.set_ylabel(r'$\angle\/\Gamma_{\mathrm{ant}}$ [ $^\mathrm{o}$]') #, fontsize=15)
		#ax.text(42, -39.6, '(a)', fontsize=20)

		#ax2.set_ylim([70, 130])
		#ax2.set_xticklabels('')
		#ax2.set_yticks(np.arange(80,121,10))		
		#ax2.set_ylabel(r'$\angle\/\Gamma_{\mathrm{rec}}$ [ $^\mathrm{o}$]', fontsize=16)

		ax.set_xlim([45, 155])	
		ax.tick_params(axis='x', direction='in')
		ax.set_xticks(np.arange(50, 151, 20))
		ax.set_ylim([-800, 400])
		ax.set_yticks(np.arange(-600,201,200))			
		
		ax.text(144, -730, '(b)', fontsize=14)













		Gb, Gc = cal.balun_and_connector_loss('mid_band', fe, ra)
		Gbc    = Gb*Gc




		Gblb, Gclb = oeg.balun_and_connector_loss('low_band_2015', flb1, ralb1)
		Gbclb   = Gblb*Gclb
		
		Gblb2, Gclb2 = oeg.balun_and_connector_loss('low_band2_2017', flb2, ralb2)
		Gbclb2   = Gblb2*Gclb2





		ax     = f1.add_axes([x0, y0 + 1*dy, dx, dy])	
		h      = ax.plot(fe, (1-Gbc)*100, 'b', linewidth=1.5, label='antenna loss [%]')
		h      = ax.plot(flb1[flb1<=110], (1-Gbclb)[flb1<=110]*100, 'r', linewidth=1.5, label='antenna loss [%]')
		h      = ax.plot(flb1[flb1<=110], (1-Gbclb2)[flb1<=110]*100, 'g', linewidth=1.5, label='antenna loss [%]')
		#ax2    = ax.twinx()
		#h2     = ax2.plot(fe, (180/np.pi)*np.unwrap(np.angle(rl)), 'r--', linewidth=2, label=r'$\angle\/\Gamma_{\mathrm{rec}}$')
		#h      = h1 + h2
		#labels = [l.get_label() for l in h]
		#ax.legend(h, labels, loc=2, fontsize=13)

		#ax.set_ylim([-41, -25])
		ax.set_xticklabels('')
		#ax.set_yticks(np.arange(-39,-26,3))
		#ax.set_ylabel('$|\Gamma_{\mathrm{rec}}|$ [dB]', fontsize=16)
		#ax.text(42, -39.6, '(a)', fontsize=20)
		ax.set_ylabel(r'antenna loss [%]')#, fontsize=15)

		#ax2.set_ylim([70, 130])
		#ax2.set_xticklabels('')
		#ax2.set_yticks(np.arange(80,121,10))		
		#ax2.set_ylabel(r'$\angle\/\Gamma_{\mathrm{rec}}$ [ $^\mathrm{o}$]', fontsize=16)

		ax.set_xlim([45, 155])
		ax.tick_params(axis='x', direction='in')
		ax.set_xticks(np.arange(50, 151, 20))
		ax.set_ylim([0, 1])
		ax.set_yticks(np.arange(0.2,0.9,0.2))		

		ax.text(144, 0.07, '(c)', fontsize=14)




		Gg   = cal.ground_loss('mid_band', fe)
		flb  = np.arange(50, 111, 1)
		Gglb = cal.ground_loss('low_band', flb)
		
		


		ax     = f1.add_axes([x0, y0 + 0*dy, dx, dy])	
		h      = ax.plot(fe, (1-Gg)*100, 'b', linewidth=1.5, label='ground loss [%]')
		h      = ax.plot(flb, (1-Gglb)*100, 'r', linewidth=1.5, label='ground loss [%]')
		#ax2    = ax.twinx()
		#h2     = ax2.plot(fe, (180/np.pi)*np.unwrap(np.angle(rl)), 'r--', linewidth=2, label=r'$\angle\/\Gamma_{\mathrm{rec}}$')
		#h      = h1 + h2
		#labels = [l.get_label() for l in h]
		#ax.legend(h, labels, loc=2, fontsize=13)

		ax.set_xlabel('$\\nu$ [MHz]')#, fontsize=15)
		#ax.set_ylim([-41, -25])
		#ax.set_xticklabels('')
		#ax.set_yticks(np.arange(-39,-26,3))
		#ax.set_ylabel('$|\Gamma_{\mathrm{rec}}|$ [dB]', fontsize=16)
		#ax.text(42, -39.6, '(a)', fontsize=20)
		ax.set_ylabel(r'ground loss [%]')#, fontsize=15)
		
		#xt = np.arange(50, 151, 20)
		#ax.set_xticks(xt)		

		#ax2.set_ylim([70, 130])
		#ax2.set_xticklabels('')
		#ax2.set_yticks(np.arange(80,121,10))		
		#ax2.set_ylabel(r'$\angle\/\Gamma_{\mathrm{rec}}$ [ $^\mathrm{o}$]', fontsize=16)

		ax.set_xlim([45, 155])
		#ax.tick_params(axis='x', direction='in')
		ax.set_xticks(np.arange(50, 151, 20))
		ax.set_ylim([0.1, 0.4])
		ax.set_yticks(np.arange(0.15,0.36,0.05))		
		
		ax.text(144, 0.12, '(d)', fontsize=14)
		

		plt.savefig(path_plot_save + 'antenna_parameters.pdf', bbox_inches='tight')
		plt.close()	
		plt.close()
		plt.close()
		plt.close()









	# Antenna Beam
	if plot_number == 3:

		# Paths
		path_plot_save = edges_folder + 'results/plots/20190422/'


		# Plot

		size_x = 4
		size_y = 12 #10.5
		x0 = 0.15
		y0 = 0.09
		dx = 0.8
		dy = 0.18


		f1  = plt.figure(num=1, figsize=(size_x, size_y))		




		# Frequency
		f, il, ih = ba.frequency_edges(50, 150)
		fe = f[il:ih+1]	






		bm = cal.FEKO_blade_beam('mid_band', 0, frequency_interpolation='no', frequency=np.array([0]), AZ_antenna_axis=0)
		ff = np.arange(50,201,2)
		
		imax = 51
		fe = ff[0:imax]
		
		g_zenith = bm[:,90,0][0:imax]
		g_45_E   = bm[:,45,0][0:imax]
		g_45_H   = bm[:,45,90][0:imax]
		
	
		
		bm_inf = cal.FEKO_blade_beam('mid_band', 1, frequency_interpolation='no', frequency=np.array([0]), AZ_antenna_axis=0)
		g_inf_zenith = bm_inf[:,90,0][0:imax]
		g_inf_45_E   = bm_inf[:,45,0][0:imax]
		g_inf_45_H   = bm_inf[:,45,90][0:imax]
		


		bm_lb = oeg.FEKO_low_band_blade_beam(beam_file=2, frequency_interpolation='no', frequency=np.array([0]), AZ_antenna_axis=0)
		fcuec = np.arange(40,121,2)
		flb   = fcuec[5:36]
		
		glb_zenith = bm_lb[:,90,0][5:36]
		glb_45_E   = bm_lb[:,45,0][5:36]
		glb_45_H   = bm_lb[:,45,90][5:36]
		
		




		bm_10 = oeg.FEKO_low_band_blade_beam(beam_file=5, frequency_interpolation='no', frequency=np.array([0]), AZ_antenna_axis=0)
		fcuec = np.arange(50,121,2)
		f10   = fcuec[0:-5]
		
		g10_zenith = bm_10[:,90,0][0:-5]
		g10_45_E   = bm_10[:,45,0][0:-5]
		g10_45_H   = bm_10[:,45,90][0:-5]
		
		
		print(glb_zenith[-1])
		print(flb[-1])
		
		print(g10_zenith[-1])
		print(f10[-1])
		

		ax     = f1.add_axes([x0, y0 + 2*dy, dx, dy])
		
		h      = ax.plot(fe, g_inf_zenith,'b',linewidth=0.75, label='')
		h      = ax.plot(fe, g_zenith,'b',linewidth=1.5, label='')
		h      = ax.plot(flb, glb_zenith,'r',linewidth=1.5, label='')
		h      = ax.plot(f10, g10_zenith,'g',linewidth=1.5, label='')
		
		h      = ax.plot(fe, g_zenith,'b',linewidth=1.5, label='')
		h      = ax.plot(fe, g_45_E,'b',linewidth=1.5, label='')
		h      = ax.plot(fe, g_45_H,'b',linewidth=1.5, label='')
				
		h      = ax.plot(fe, g_inf_zenith,'b',linewidth=0.75, label='')
		h      = ax.plot(fe, g_inf_45_E,'b',linewidth=0.75, label='')
		h      = ax.plot(fe, g_inf_45_H,'b',linewidth=0.75, label='')		
		
		h      = ax.plot(flb, glb_zenith,'r',linewidth=1.5, label='')
		h      = ax.plot(flb, glb_45_E,'r',linewidth=1.5, label='')
		h      = ax.plot(flb, glb_45_H,'r',linewidth=1.5, label='')
		
		h      = ax.plot(f10, g10_zenith,'g',linewidth=1.5, label='')
		h      = ax.plot(f10, g10_45_E,'g',linewidth=1.5, label='')
		h      = ax.plot(f10, g10_45_H,'g',linewidth=1.5, label='')		
		
		
		#ax2    = ax.twinx()
		#h2     = ax2.plot(fe, off,'r--',linewidth=2, label='$C_2$')
		#h      = h1 + h2
		#labels = [l.get_label() for l in h]
		#ax.legend(h, labels, loc=0, fontsize=13)

		#ax.set_ylim([3, 5.5])
		ax.set_xticklabels('')
		#ax.set_yticks(np.arange(3.5,5.1,0.5))
		#ax.set_ylabel('$C_1$', fontsize=16)
		#ax.text(42, 3.25, '(b)', fontsize=20)
		ax.set_ylabel('gain [no units]')#, fontsize=14)

		#ax2.set_ylim([-2.4, -1.8])
		#ax2.set_xticklabels('')
		#ax2.set_yticks(np.arange(-2.3, -1.85, 0.1))
		#ax2.set_ylabel('$C_2$ [K]', fontsize=16)

		ax.set_xlim([45, 155])
		ax.tick_params(axis='x', direction='in')
		ax.set_xticks(np.arange(50, 151, 20))
		ax.set_ylim([0, 9])
		ax.set_yticks(np.arange(1,8.1,1))		
		
		ax.text(145, 0.4, '(a)', fontsize=14)
		
		ax.text(50, 5.7, 'zenith', fontsize=10)
		ax.text(50, 2.9, r'$\theta=45^{\circ}$, H-plane', fontsize=10)
		ax.text(50, 0.9, r'$\theta=45^{\circ}$, E-plane', fontsize=10)
		
		
		ax.legend(['mid-band infinite GP', 'mid-band 30mx30m GP','low-band 30mx30m GP','low-band 10mx10m GP'], fontsize=7)
	
	
	
	
	
	
		
		Nfg   = 3
	
		
		imax2 = 31
		p1 = np.polyfit(fe[0:imax2], g_zenith[0:imax2], Nfg)
		p2 = np.polyfit(fe[0:imax2], g_45_E[0:imax2], Nfg)
		p3 = np.polyfit(fe[0:imax2], g_45_H[0:imax2], Nfg)
		
		m1 = np.polyval(p1, fe[0:imax2])
		m2 = np.polyval(p2, fe[0:imax2])
		m3 = np.polyval(p3, fe[0:imax2])


		pi1 = np.polyfit(fe[0:imax2], g_inf_zenith[0:imax2], Nfg)
		pi2 = np.polyfit(fe[0:imax2], g_inf_45_E[0:imax2], Nfg)
		pi3 = np.polyfit(fe[0:imax2], g_inf_45_H[0:imax2], Nfg)
		
		mi1 = np.polyval(pi1, fe[0:imax2])
		mi2 = np.polyval(pi2, fe[0:imax2])
		mi3 = np.polyval(pi3, fe[0:imax2])




		p1 = np.polyfit(flb, glb_zenith, Nfg)
		p2 = np.polyfit(flb, glb_45_E, Nfg)
		p3 = np.polyfit(flb, glb_45_H, Nfg)
		
		mlb1 = np.polyval(p1, flb)
		mlb2 = np.polyval(p2, flb)
		mlb3 = np.polyval(p3, flb)
		
		
		p1 = np.polyfit(f10, g10_zenith, Nfg)
		p2 = np.polyfit(f10, g10_45_E, Nfg)
		p3 = np.polyfit(f10, g10_45_H, Nfg)
		
		m10_1 = np.polyval(p1, f10)
		m10_2 = np.polyval(p2, f10)
		m10_3 = np.polyval(p3, f10)		
		

		
		
		ax     = f1.add_axes([x0, y0 + 1*dy, dx, dy])
		
		h      = ax.plot(fe[0:imax2], g_zenith[0:imax2]-m1 + 0.1,'b', linewidth=1.5, label='$T_{\mathrm{unc}}$')
		h      = ax.plot(fe[0:imax2], g_45_E[0:imax2]-m2 - 0.1,'b', linewidth=1.5, label='$T_{\mathrm{unc}}$')
		h      = ax.plot(fe[0:imax2], g_45_H[0:imax2]-m3 - 0.0,'b', linewidth=1.5, label='$T_{\mathrm{unc}}$')
		
		h      = ax.plot(fe[0:imax2], g_inf_zenith[0:imax2]-mi1 + 0.1,'b', linewidth=0.75, label='$T_{\mathrm{unc}}$')
		h      = ax.plot(fe[0:imax2], g_inf_45_E[0:imax2]-mi2 - 0.1,'b', linewidth=0.75, label='$T_{\mathrm{unc}}$')
		h      = ax.plot(fe[0:imax2], g_inf_45_H[0:imax2]-mi3 - 0.0,'b', linewidth=0.75, label='$T_{\mathrm{unc}}$')
		
		h      = ax.plot(flb, glb_zenith-mlb1 + 0.1,'r', linewidth=1.5, label='$T_{\mathrm{unc}}$')
		h      = ax.plot(flb, glb_45_E-mlb2 - 0.1,'r', linewidth=1.5, label='$T_{\mathrm{unc}}$')
		h      = ax.plot(flb, glb_45_H-mlb3 - 0.0,'r', linewidth=1.5, label='$T_{\mathrm{unc}}$')		

		h      = ax.plot(f10, g10_zenith-m10_1 + 0.1,'g', linewidth=1.5, label='$T_{\mathrm{unc}}$')
		h      = ax.plot(f10, g10_45_E-m10_2 - 0.1,'g', linewidth=1.5, label='$T_{\mathrm{unc}}$')
		h      = ax.plot(f10, g10_45_H-m10_3 - 0.0,'g', linewidth=1.5, label='$T_{\mathrm{unc}}$')	

		
		
		#ax2    = ax.twinx()
		#h2     = ax2.plot(fe, TC,'r--', linewidth=2, label='$T_{\mathrm{cos}}$')
		#h3     = ax2.plot(fe, TS,'g--', linewidth=2, label='$T_{\mathrm{sin}}$')		

		#h      = h1 + h2 + h3
		#labels = [l.get_label() for l in h]
		#ax.legend(h, labels, loc=0, fontsize=13, ncol=3)

		#ax.set_ylim([178, 190])
		#ax.set_yticks(np.arange(180, 189, 2))
		#ax.set_ylabel('$T_{\mathrm{unc}}$ [K]', fontsize=16)
		#ax.set_xlabel('$\\nu$ [MHz]', fontsize=15)
		#ax.text(42, 179, '(c)', fontsize=20)
		ax.set_ylabel('gain residuals\n[0.05 per division]')#, fontsize=14)

		#ax2.set_ylim([-60, 40])
		#ax2.set_yticks(np.arange(-40, 21, 20))
		
		
		ax.set_xlim([45, 155])
		xt = np.arange(50, 151, 20)
		ax.set_xticks(xt)
		ax.tick_params(axis='x', direction='in')
		ax.set_xticklabels(xt, fontsize=10)
		ax.set_ylim([-0.175, 0.175])
		yt = np.arange(-0.15,0.16,0.05)
		ax.set_yticks(yt)
		ax.set_yticklabels(['' for i in range(len(yt))])
		
		ax.text(145, -0.156, '(b)', fontsize=14)

		ax.text(113, 0.09, 'zenith', fontsize=10)
		ax.text(113, -0.01, r'$\theta=45^{\circ}$, H-plane', fontsize=10)
		ax.text(113, -0.11, r'$\theta=45^{\circ}$, E-plane', fontsize=10)
		





		ax     = f1.add_axes([x0, y0 + 0*dy, dx, dy])
		
		ax.set_xlabel('$\\nu$ [MHz]')#, fontsize=15)
		ax.set_ylabel('place holder [X]')#, fontsize=14)

		
		ax.set_xlim([45, 155])
		xt = np.arange(50, 151, 20)
		ax.set_xticks(xt)
		#ax.set_xticklabels(['' for i in range(len(xt))], fontsize=15)
		ax.set_xticklabels(xt, fontsize=10)
		ax.set_ylim([-0.175, 0.175])
		yt = np.arange(-0.15,0.16,0.05)
		ax.set_yticks(yt)
		ax.set_yticklabels(['' for i in range(len(yt))])
		
		ax.text(145, -0.156, '(c)', fontsize=14)

		
		





		plt.savefig(path_plot_save + 'beam_gain.pdf', bbox_inches='tight')
		plt.close()	
		plt.close()
		plt.close()
		plt.close()


















	



	if plot_number == 4:
		
		# Plot

		size_x = 4.7
		size_y = 5
		
		x0   = 0.13
		y0   = 0.035
		dx   = 0.67
		dy   = 0.6
		dy1  = 0.2
		yoff = 0.05
		
		dxc = 0.03
		xoffc = 0.03
		
		panel_letter_x = 35
		panel_letter_y = 3




		beam_factor_filename = 'table_lores_mid_band_50-200MHz_90deg_alan0_haslam_gaussian_index_2.4_2.6_sigma_deg_5_reffreq_100MHz.hdf5'
		f, lst, bf           = cal.beam_factor_table_read('/home/raul/DATA2/EDGES_vol2/mid_band/calibration/beam_factors/table/' + beam_factor_filename)
		gha = lst + (24-17.76)
		gha[gha>=24] = gha[gha>=24] - 24
		IX = np.argsort(gha)
		bx1 = bf[IX]



		plt.close()
		plt.close()
		plt.close()
		f1   = plt.figure(num=1, figsize=(size_x, size_y))		




		ax     = f1.add_axes([x0, y0 + 1*(yoff+dy1), dx, dy])
		im = ax.imshow(bx1, interpolation='none', extent=[50, 200, 24, 0], aspect='auto', vmin=0.979, vmax=1.021)#, cmap='jet')
	
		ax.set_xlim([50, 120])
		ax.set_xticklabels('')
		ax.set_yticks(np.arange(0,25,3))
		ax.set_ylabel('GHA [hr]')
		
		cax    = f1.add_axes([x0 + 1*dx + xoffc, y0 + 1*(yoff+dy1), dxc, dy])
		f1.colorbar(im, cax=cax, orientation='vertical', ticks=[0.98, 0.99, 1, 1.01, 1.02])
		cax.set_title('$C$')
		
		#ax.text(panel_letter_x, panel_letter_y,  '(a)', fontsize=18)
		

		
		
		ax     = f1.add_axes([x0, y0, dx, dy1])
		ax.plot(f, bx1[0,:], 'k')
		ax.plot(f, bx1[125,:], 'k--')
		ax.plot(f, bx1[175,:], 'k:')
		
		ax.set_ylim([0.9, 1.1])
	
		#cax    = f1.add_axes([x0 + 1*dx + xoffc, y0 + 0*(yoff+dy), dxc, dy])
		#f1.colorbar(im, cax=cax, orientation='vertical')
	
		ax.set_xlim([50, 120])
		ax.set_ylim([0.95, 1.05])
		ax.set_xlabel(r'$\nu$ [MHz]')
		ax.set_yticks([0.96,1,1.04])
		ax.set_ylabel('$C$')
		#ax.text(panel_letter_x, panel_letter_y,  '(e)', fontsize=18)		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		# Saving plot
		path_plot_save = edges_folder + 'results/plots/20190422/'
		

		plt.savefig(path_plot_save + 'beam_chromaticity_correction.pdf', bbox_inches='tight')
		plt.close()	
		plt.close()
		plt.close()
		plt.close()









	if plot_number == 5:
		

		# Plot

		size_x = 4.7
		size_y = 5
		
		x0   = 0.13
		y0   = 0.035
		dx   = 0.53
		dy   = 0.55
		dy1  = 0.2
		xoff = 0.09
		
		dxc = 0.03
		xoffc = 0.03
		
		panel_letter_x = 35
		panel_letter_y = 3




		beam_factor_filename = 'table_lores_mid_band_50-200MHz_90deg_alan0_haslam_gaussian_index_2.4_2.6_sigma_deg_5_reffreq_100MHz.hdf5'
		f, lst, bf           = cal.beam_factor_table_read('/home/raul/DATA2/EDGES_vol2/mid_band/calibration/beam_factors/table/' + beam_factor_filename)
		gha = lst + (24-17.76)
		gha[gha>=24] = gha[gha>=24] - 24
		IX = np.argsort(gha)
		bx1 = bf[IX]




		beam_factor_filename = 'table_lores_mid_band_50-200MHz_90deg_alan1_haslam_gaussian_index_2.4_2.6_sigma_deg_5_reffreq_100MHz.hdf5'
		f, lst, bf           = cal.beam_factor_table_read('/home/raul/DATA2/EDGES_vol2/mid_band/calibration/beam_factors/table/' + beam_factor_filename)
		bx2 = bf[IX]





		beam_factor_filename = 'table_lores_mid_band_50-200MHz_90deg_alan0_haslam_step_index_2.5_2.6_band_deg_10_reffreq_100MHz.hdf5'
		f, lst, bf           = cal.beam_factor_table_read('/home/raul/DATA2/EDGES_vol2/mid_band/calibration/beam_factors/table/' + beam_factor_filename)
		bx3 = bf[IX]


		beam_factor_filename = 'table_lores_mid_band_50-200MHz_90deg_alan0_LW_gaussian_index_2.4_2.6_sigma_deg_5_reffreq_100MHz.hdf5'
		f, lst, bf           = cal.beam_factor_table_read('/home/raul/DATA2/EDGES_vol2/mid_band/calibration/beam_factors/table/' + beam_factor_filename)
		bx4 = bf[IX]
		
		beam_factor_filename = 'table_lores_mid_band_50-200MHz_90deg_alan0_guzman_gaussian_index_2.4_2.6_sigma_deg_5_reffreq_100MHz.hdf5'
		f, lst, bf           = cal.beam_factor_table_read('/home/raul/DATA2/EDGES_vol2/mid_band/calibration/beam_factors/table/' + beam_factor_filename)
		bx5 = bf[IX]



		plt.close()
		plt.close()
		plt.close()
		f1   = plt.figure(num=1, figsize=(size_x, size_y))		













		
		ax     = f1.add_axes([x0 + 0*(xoff+dx), y0, dx, dy])
		im = ax.imshow(bx2-bx1, interpolation='none', extent=[50, 200, 24, 0], aspect='auto', vmin=-0.0043, vmax=0.0043)#, cmap='jet')
	
		#cax    = f1.add_axes([x0 + 1*dx + xoffc, y0 + 1.5*(yoff+dy), dxc, dy])
		#f1.colorbar(im, cax=cax, orientation='vertical', ticks=[-0.004, -0.002, 0, 0.002, 0.004])
	
		ax.set_xlim([50, 120])
		#ax.set_xticklabels('')
		ax.set_xticks(np.arange(50,121,10))
		ax.set_yticks(np.arange(0,25,3))
		ax.set_xlabel(r'$\nu$ [MHz]')
		ax.set_ylabel('GHA [hr]')
		#ax.set_text(panel_letter_x, panel_letter_y,  '(b)', fontsize=18)
		ax.set_title('(a)', fontsize=18)
	
	
	
	
	
		ax     = f1.add_axes([x0 + 1*(xoff+dx), y0, dx, dy])
		im = ax.imshow(bx3-bx1, interpolation='none', extent=[50, 200, 24, 0], aspect='auto', vmin=-0.0043, vmax=0.0043)#, cmap='jet')
	
		#cax    = f1.add_axes([x0 + 1*dx + xoffc, y0 + 2*(yoff+dy), dxc, dy])
		#f1.colorbar(im, cax=cax, orientation='vertical')
	
		ax.set_xlim([50, 120])
		ax.set_yticklabels('')
		ax.set_xlabel(r'$\nu$ [MHz]')
		ax.set_xticks(np.arange(50,121,10))
		ax.set_yticks(np.arange(0,25,3))
		#ax.set_ylabel('GHA [hr]')
		#ax.text(panel_letter_x, panel_letter_y,  '(c)', fontsize=18)
		ax.set_title('(b)', fontsize=18)
	
	
	
	
	
		ax     = f1.add_axes([x0 + 2*(xoff+dx), y0, dx, dy])
		im = ax.imshow(bx4-bx1, interpolation='none', extent=[50, 200, 24, 0], aspect='auto', vmin=-0.0043, vmax=0.0043)#, cmap='jet')
	
		#cax    = f1.add_axes([x0 + 1*dx + xoffc, y0 + 1*(yoff+dy), dxc, dy])
		#f1.colorbar(im, cax=cax, orientation='vertical')
	
		ax.set_xlim([50, 120])
		ax.set_yticklabels('')
		ax.set_xlabel(r'$\nu$ [MHz]')
		ax.set_xticks(np.arange(50,121,10))
		ax.set_yticks(np.arange(0,25,3))
		#ax.set_ylabel('GHA [hr]')
		#ax.text(panel_letter_x, panel_letter_y,  '(d)', fontsize=18)
		ax.set_title('(c)', fontsize=18)
	
	
	
	
	
	
		ax     = f1.add_axes([x0 + 3*(xoff+dx), y0, dx, dy])
		im = ax.imshow(bx5-bx1, interpolation='none', extent=[50, 200, 24, 0], aspect='auto', vmin=-0.0043, vmax=0.0043)#, cmap='jet')
	
		cax    = f1.add_axes([x0 + 3.2*xoff+4*dx, y0, dxc, dy]) #  + xoffc
		f1.colorbar(im, cax=cax, orientation='vertical')
		cax.set_title(r'$\Delta C$')
	
		ax.set_xlim([50, 120])
		ax.set_yticklabels('')
		ax.set_xlabel(r'$\nu$ [MHz]')
		ax.set_xticks(np.arange(50,121,10))
		ax.set_yticks(np.arange(0,25,3))
		#ax.set_ylabel('GHA [hr]')
		#ax.text(panel_letter_x, panel_letter_y,  '(e)', fontsize=18)
		ax.set_title('(d)', fontsize=18)
	

		
		# Saving plot
		path_plot_save = edges_folder + 'results/plots/20190422/'
		

		plt.savefig(path_plot_save + 'beam_chromaticity_differences.pdf', bbox_inches='tight')
		plt.close()	
		plt.close()
		plt.close()
		plt.close()


























			
	
	if plot_number == 6:
		
		bm_all = cal.FEKO_blade_beam('mid_band', 0, frequency_interpolation='no', AZ_antenna_axis=90)
		f  = np.arange(50,201,2)
		
		el             = np.arange(0,91) 
		sin_theta      = np.sin((90-el)*(np.pi/180)) 
		sin_theta_2D_T = np.tile(sin_theta, (360, 1))
		sin_theta_2D   = sin_theta_2D_T.T		
		
		
		
		#normalization  = 1		

		sm = np.zeros(len(f))
		for i in range(len(f)):
			
			
			bm = bm_all[i,:,:]
			normalization  = np.max(np.max(bm))
			
			nb             = bm/normalization

			s_sq_deg       = np.sum(nb * sin_theta_2D) 
			sm[i]          = s_sq_deg/((180/np.pi)**2) 
			
			
			
			
		bm_all = cal.FEKO_blade_beam('mid_band', 1, frequency_interpolation='no', AZ_antenna_axis=90)
		f  = np.arange(50,201,2)
		
		el             = np.arange(0,91) 
		sin_theta      = np.sin((90-el)*(np.pi/180)) 
		sin_theta_2D_T = np.tile(sin_theta, (360, 1))
		sin_theta_2D   = sin_theta_2D_T.T		
		
		
		
		#normalization  = 1		

		smi = np.zeros(len(f))
		for i in range(len(f)):
			
			
			bm = bm_all[i,:,:]
			normalization  = np.max(np.max(bm))
			
			nb             = bm/normalization

			s_sq_deg       = np.sum(nb * sin_theta_2D) 
			smi[i]          = s_sq_deg/((180/np.pi)**2) 		
		
		





		bmlb_all = oeg.FEKO_low_band_blade_beam(beam_file=2, frequency_interpolation='no', AZ_antenna_axis=0)
		fl       = np.arange(40,121,2)
		
		sl = np.zeros(len(fl))
		for i in range(len(fl)):
			
			
			bmlb = bmlb_all[i,:,:]
			normalization  = np.max(np.max(bmlb))
			
			nb             = bmlb/normalization

			s_sq_deg       = np.sum(nb * sin_theta_2D) 
			sl[i]          = s_sq_deg/((180/np.pi)**2) 
			
		
		
		
		
		
		
		
		
			
			
		fx  = f[(f>=50) & (f<=110)] 
		smx = sm[(f>=50) & (f<=110)]
		smix = smi[(f>=50) & (f<=110)]
		slx = sl[(fl>=50) & (fl<=110)]
		
		pm = np.polyfit(fx, smx,4)
		mm = np.polyval(pm, fx)
		
		pmi = np.polyfit(fx, smix,4)
		mmi = np.polyval(pmi, fx)		
		
		pl = np.polyfit(fx, slx,4)
		ml = np.polyval(pl, fx)
		
		
		plt.close()
		plt.figure(1)
		
		plt.subplot(2,1,1)
		plt.plot(fx, smx)
		plt.plot(fx, smix, ':')
		plt.plot(fx, slx, '--')
		plt.ylabel('solid angle of\n beam above horizon [sr]')
		plt.legend(['Mid-Band 30mx30m ground plane','Mid-Band infinite ground plane','Low-Band 30mx30m ground plane'])
		
		plt.subplot(2,1,2)
		plt.plot(fx, smx - mm)
		plt.plot(fx, smix - mmi, ':')
		plt.plot(fx, slx - ml, '--')
		plt.ylabel('residuals to\n 5-term polynomial [sr]')
		plt.xlabel('frequency [MHz]')
		










		bm_all = cal.FEKO_blade_beam('mid_band', 0, frequency_interpolation='no', AZ_antenna_axis=90)
		f  = np.arange(50,201,2)
		
		el             = np.arange(0,91) 
		sin_theta      = np.sin((90-el)*(np.pi/180)) 
		sin_theta_2D_T = np.tile(sin_theta, (360, 1))
		sin_theta_2D   = sin_theta_2D_T.T		
		
		
		
		#normalization  = 1		

		sm = np.zeros(len(f))
		for i in range(len(f)):
			
			
			bm = bm_all[i,:,:]
			normalization  = 1#np.max(np.max(bm))
			
			nb             = bm/normalization

			s_sq_deg       = np.sum(nb * sin_theta_2D) 
			sm[i]          = s_sq_deg/((180/np.pi)**2) 
			
			
			
			
		bm_all = cal.FEKO_blade_beam('mid_band', 1, frequency_interpolation='no', AZ_antenna_axis=90)
		f  = np.arange(50,201,2)
		
		el             = np.arange(0,91) 
		sin_theta      = np.sin((90-el)*(np.pi/180)) 
		sin_theta_2D_T = np.tile(sin_theta, (360, 1))
		sin_theta_2D   = sin_theta_2D_T.T		
		
		
		
		#normalization  = 1		

		smi = np.zeros(len(f))
		for i in range(len(f)):
			
			
			bm = bm_all[i,:,:]
			normalization  = 1#np.max(np.max(bm))
			
			nb             = bm/normalization

			s_sq_deg       = np.sum(nb * sin_theta_2D) 
			smi[i]          = s_sq_deg/((180/np.pi)**2) 		
		
		





		bmlb_all = oeg.FEKO_low_band_blade_beam(beam_file=2, frequency_interpolation='no', AZ_antenna_axis=0)
		fl       = np.arange(40,121,2)
		
		sl = np.zeros(len(fl))
		for i in range(len(fl)):
			
			
			bmlb = bmlb_all[i,:,:]
			normalization  = 1#np.max(np.max(bmlb))
			
			nb             = bmlb/normalization

			s_sq_deg       = np.sum(nb * sin_theta_2D) 
			sl[i]          = s_sq_deg/((180/np.pi)**2) 
			
		
		
		
		
		
		
		
		
			
			
		fx  = f[(f>=50) & (f<=110)] 
		smx = sm[(f>=50) & (f<=110)]
		smix = smi[(f>=50) & (f<=110)]
		slx = sl[(fl>=50) & (fl<=110)]
		
		# normalized total radiated power
		smx  = smx/(4*np.pi)
		smix = smix/(4*np.pi)
		slx  = slx/(4*np.pi)
		
		
		
		pm = np.polyfit(fx, smx,4)
		mm = np.polyval(pm, fx)
		
		pmi = np.polyfit(fx, smix,4)
		mmi = np.polyval(pmi, fx)		
		
		pl = np.polyfit(fx, slx,4)
		ml = np.polyval(pl, fx)
		
		
		#plt.close()
		plt.figure(2)
		
		plt.subplot(2,1,1)
		plt.plot(fx, smx)
		plt.plot(fx, smix, ':')
		plt.plot(fx, slx, '--')
		plt.ylabel('normalized total radiated power\n above horizon [fraction of 4pi]')
		plt.legend(['Mid-Band 30mx30m ground plane','Mid-Band infinite ground plane','Low-Band 30mx30m ground plane'])
		
		plt.subplot(2,1,2)
		plt.plot(fx, smx - mm)
		plt.plot(fx, smix - mmi, ':')
		plt.plot(fx, slx - ml, '--')
		plt.ylabel('residuals to 5-term polynomial\n [fraction of 4pi]')
		plt.xlabel('frequency [MHz]')

























	
	
	
	return 0



	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	

def plot_daily_residuals_nominal(f, rx, wx, ydx):
	
	
	keep_index = eg.daily_nominal_filter('mid_band', 1, ydx)
	r  = rx[keep_index==1]
	w  = wx[keep_index==1]
	yd = ydx[keep_index==1]






	lr = len(r[:,0])
	for i in range(lr):
		fb, rb, wb = ba.spectral_binning_number_of_samples(f, r[i,:], w[i,:])
		if i == 0:
			rb_all = np.zeros((lr, len(fb)))
			wb_all = np.zeros((lr, len(fb)))
			
		rb_all[i,:] = rb
		wb_all[i,:] = wb
		
	

	
	
	# Settings
	# ----------------------------------

	LST_centers = list(np.arange(0.5, lr))
	LST_text    = [str(int(yd[i,1])) for i in range(lr)]#'GHA=0-5 hr', 'GHA=5-11 hr', 'GHA=11-18 hr', 'GHA=18-24 hr']
	DY          =  0.7

	FLOW_plot   =  53
	FHIGH_plot  = 122

	XTICKS      =  np.arange(60, 121, 20)
	XTEXT       =  54
	YLABEL      =  str(DY) + ' K per division'
	TITLE       = ''#'59-121 MHz, GHA=6-18 hr'
	
	figure_path = '/home/raul/Desktop/'
	figure_name = 'daily_residuals_nominal_59_121MHz_GHA_6_18hr'
	
	FIG_SX      = 6
	FIG_SY      = 15
	
	
	
	# Plotting
	x = eg.plot_residuals(fb, rb_all, wb_all, LST_text, FIG_SX=FIG_SX, FIG_SY=FIG_SY, DY=DY, FLOW=FLOW_plot, FHIGH=FHIGH_plot, XTICKS=XTICKS, XTEXT=XTEXT, YLABEL=YLABEL, TITLE=TITLE, save='yes', figure_path=figure_path, figure_name=figure_name)
	
	
	
	return 0



	
	
	
def plot_number_of_cterms_wterms():
	
	rms, cterms, wterms = cr1.calibration_RMS_read('/run/media/raul/WD_RED_6TB/EDGES_vol2/mid_band/calibration/receiver_calibration/receiver1/2018_01_25C/results/nominal/calibration_files/calibration_term_sweep_50_150MHz/calibration_term_sweep_50_150MHz.hdf5')
	
	
	plt.close()
	plt.close()
	plt.close()
	plt.close()
	plt.close()
	
	plt.figure();plt.imshow(np.flipud(rms[0,:,:]), interpolation='none', vmin=0.016, vmax=0.02, extent=[1, 15, 1, 15]);plt.colorbar() 
	plt.figure();plt.imshow(np.flipud(rms[1,:,:]), interpolation='none', vmin=0.016, vmax=0.02, extent=[1, 15, 1, 15]);plt.colorbar() 
	plt.figure();plt.imshow(np.flipud(rms[2,:,:]), interpolation='none', vmin=0.3, vmax=0.6, extent=[1, 15, 1, 15]);plt.colorbar() 
	plt.figure();plt.imshow(np.flipud(rms[3,:,:]), interpolation='none', vmin=0.3, vmax=0.6, extent=[1, 15, 1, 15]);plt.colorbar() 
	
	return 0
	
	
	
	
	
	
	
	


def plots_midband_polychord(fig):
	
	
	
	if fig==0:
		folder = edges_folder + 'mid_band/polychord/20190508/case1_nominal/foreground_model_exp/'
		getdist_samples, ww, ll, best_fit, covariance_matrix = gp.load_samples(folder + 'chain.txt', 0, label_names=[r'T_{100}\;[{\rm K}]', r'\beta', r'\gamma', r'\delta', r'\epsilon']) 
		o = gp.triangle_plot(getdist_samples, folder + 'result.pdf', legend_FS=10, label_FS=10, axes_FS=5)
		
		v, t, w, sigma, inv_sigma, det_sigma = dm.real_data(10, 60, 120)
		
		model = dm.full_model(best_fit, v, 100, model_type_signal='exp', model_type_foreground='exp', N21par=0, NFGpar=5)
	
		
	if fig==1:
		folder = edges_folder + 'mid_band/polychord/20190508/case1_nominal/foreground_model_linlog/'
		getdist_samples, ww, ll, best_fit, covariance_matrix = gp.load_samples(folder + 'chain.txt', 0, label_names=[r'a_0', r'a_1', r'a_2', r'a_3', r'a_4']) 
		o = gp.triangle_plot(getdist_samples, folder + 'result.pdf', legend_FS=10, label_FS=12, axes_FS=8)
		
		v, t, w, sigma, inv_sigma, det_sigma = dm.real_data(10, 60, 120)
		
		model = dm.full_model(best_fit, v, 100, model_type_signal='exp', model_type_foreground='linlog', N21par=0, NFGpar=5)

	
		
	if fig==2:
		# Data used:  60-67, 103-119.5
		folder = edges_folder + 'mid_band/polychord/20190508/case1_nominal/foreground_model_exp_gap/'
		getdist_samples, ww, ll, best_fit, covariance_matrix = gp.load_samples(folder + 'chain.txt', 0, label_names=[r'T_{100}\;[{\rm K}]', r'\beta', r'\gamma', r'\delta', r'\epsilon']) 
		o = gp.triangle_plot(getdist_samples, folder + 'result.pdf', legend_FS=10, label_FS=10, axes_FS=5)
		
		v, t, w, sigma, inv_sigma, det_sigma = dm.real_data(10, 60, 119.5)
		
		model = dm.full_model(best_fit, v, 100, model_type_signal='exp', model_type_foreground='exp', N21par=0, NFGpar=5)	



	if fig==3:
		# Data used:  60-65, 103-119.5
		folder = edges_folder + 'mid_band/polychord/20190508/case1_nominal/foreground_model_exp_gap2/'
		getdist_samples, ww, ll, best_fit, covariance_matrix = gp.load_samples(folder + 'chain.txt', 0, label_names=[r'T_{100}\;[{\rm K}]', r'\beta', r'\gamma', r'\delta', r'\epsilon']) 
		o = gp.triangle_plot(getdist_samples, folder + 'result.pdf', legend_FS=10, label_FS=10, axes_FS=5)
		
		v, t, w, sigma, inv_sigma, det_sigma = dm.real_data(10, 60, 119.5)
		
		model = dm.full_model(best_fit, v, 100, model_type_signal='exp', model_type_foreground='exp', N21par=0, NFGpar=5)
		
		
		

	if fig==4:
		# Data used:  60-65, 95-119.5
		folder = edges_folder + 'mid_band/polychord/20190508/case1_nominal/foreground_model_exp_gap3/'
		getdist_samples, ww, ll, best_fit, covariance_matrix = gp.load_samples(folder + 'chain.txt', 0, label_names=[r'T_{100}\;[{\rm K}]', r'\beta', r'\gamma', r'\delta', r'\epsilon']) 
		o = gp.triangle_plot(getdist_samples, folder + 'result.pdf', legend_FS=10, label_FS=10, axes_FS=5)
		
		v, t, w, sigma, inv_sigma, det_sigma = dm.real_data(10, 60, 119.5)
		
		model = dm.full_model(best_fit, v, 100, model_type_signal='exp', model_type_foreground='exp', N21par=0, NFGpar=5)
		
		
		
	if fig==5:
		# Data used:  60-65, 95-115
		folder = edges_folder + 'mid_band/polychord/20190508/case1_nominal/foreground_model_exp_gap4/'
		getdist_samples, ww, ll, best_fit, covariance_matrix = gp.load_samples(folder + 'chain.txt', 0, label_names=[r'T_{100}\;[{\rm K}]', r'\beta', r'\gamma', r'\delta', r'\epsilon']) 
		o = gp.triangle_plot(getdist_samples, folder + 'result.pdf', legend_FS=10, label_FS=10, axes_FS=5)
		
		v, t, w, sigma, inv_sigma, det_sigma = dm.real_data(10, 60, 115)
		
		model = dm.full_model(best_fit, v, 100, model_type_signal='exp', model_type_foreground='exp', N21par=0, NFGpar=5)		
	

		
	if fig==6:
		# Data used:  60-65, 100-115
		folder = edges_folder + 'mid_band/polychord/20190508/case1_nominal/foreground_model_exp_gap5/'
		getdist_samples, ww, ll, best_fit, covariance_matrix = gp.load_samples(folder + 'chain.txt', 0, label_names=[r'T_{100}\;[{\rm K}]', r'\beta', r'\gamma', r'\delta', r'\epsilon']) 
		o = gp.triangle_plot(getdist_samples, folder + 'result.pdf', legend_FS=10, label_FS=10, axes_FS=5)
		
		v, t, w, sigma, inv_sigma, det_sigma = dm.real_data(10, 60, 115)
		
		model = dm.full_model(best_fit, v, 100, model_type_signal='exp', model_type_foreground='exp', N21par=0, NFGpar=5)
		
		
		

	if fig==7:
		# Data used:  60-65, 97-115
		folder = edges_folder + 'mid_band/polychord/20190508/case1_nominal/foreground_model_exp_gap6/'
		getdist_samples, ww, ll, best_fit, covariance_matrix = gp.load_samples(folder + 'chain.txt', 0, label_names=[r'T_{100}\;[{\rm K}]', r'\beta', r'\gamma', r'\delta', r'\epsilon']) 
		o = gp.triangle_plot(getdist_samples, folder + 'result.pdf', legend_FS=10, label_FS=10, axes_FS=5)
		
		v, t, w, sigma, inv_sigma, det_sigma = dm.real_data(10, 60, 115)
		
		model = dm.full_model(best_fit, v, 100, model_type_signal='exp', model_type_foreground='exp', N21par=0, NFGpar=5)



	if fig==8:
		# Data used:  60-65, 100-115, CASE 2, cterms7, wterms8
		folder = edges_folder + 'mid_band/polychord/20190508/case1_nominal/foreground_model_exp_gap7/'
		getdist_samples, ww, ll, best_fit, covariance_matrix = gp.load_samples(folder + 'chain.txt', 0, label_names=[r'T_{100}\;[{\rm K}]', r'\beta', r'\gamma', r'\delta', r'\epsilon']) 
		o = gp.triangle_plot(getdist_samples, folder + 'result.pdf', legend_FS=10, label_FS=10, axes_FS=5)
		
		v, t, w, sigma, inv_sigma, det_sigma = dm.real_data(2, 60, 115)
		
		model = dm.full_model(best_fit, v, 100, model_type_signal='exp', model_type_foreground='exp', N21par=0, NFGpar=5)
	
	
	
	if fig==9:
		# Data used:  60-115
		folder = edges_folder + 'mid_band/polychord/20190508/case1_nominal/foreground_model_exp_signal_model_tanh/'
		getdist_samples, ww, ll, best_fit, covariance_matrix = gp.load_samples(folder + 'chain.txt', 0, label_names=[r'A\;[{\rm K}]', r'\nu_0\;[{\rm MHz}]', r'w\;[{\rm MHz}]', r'\tau_1', r'\tau_2', r'T_{100}\;[{\rm K}]', r'\beta', r'\gamma', r'\delta', r'\epsilon']) 
		o = gp.triangle_plot(getdist_samples, folder + 'result.pdf', legend_FS=10, label_FS=10, axes_FS=5)
		
		v, t, w, sigma, inv_sigma, det_sigma = dm.real_data(10, 60, 115)
		
		model = dm.full_model(best_fit, v, 100, model_type_signal='tanh', model_type_foreground='exp', N21par=5, NFGpar=5)	

	


	if fig==10:
		# Data used:  60-120
		folder = edges_folder + 'mid_band/polychord/20190508/case1_nominal/foreground_model_exp_signal_model_tanh_60_120MHz/'
		getdist_samples, ww, ll, best_fit, covariance_matrix = gp.load_samples(folder + 'chain.txt', 0, label_names=[r'A\;[{\rm K}]', r'\nu_0\;[{\rm MHz}]', r'w\;[{\rm MHz}]', r'\tau_1', r'\tau_2', r'T_{100}\;[{\rm K}]', r'\beta', r'\gamma', r'\delta', r'\epsilon']) 
		o = gp.triangle_plot(getdist_samples, folder + 'result.pdf', legend_FS=10, label_FS=13, axes_FS=7)
		
		v, t, w, sigma, inv_sigma, det_sigma = dm.real_data(10, 60, 120)
		
		model = dm.full_model(best_fit, v, 100, model_type_signal='tanh', model_type_foreground='exp', N21par=5, NFGpar=5)	


	if fig==11:
		# Data used:  60-120, CASE 2, cterms7, wterms8
		folder = edges_folder + 'mid_band/polychord/20190508/case1_nominal/foreground_model_exp_signal_model_tanh_60_120MHz_case2/'
		getdist_samples, ww, ll, best_fit, covariance_matrix = gp.load_samples(folder + 'chain.txt', 0, label_names=[r'A\;[{\rm K}]', r'\nu_0\;[{\rm MHz}]', r'w\;[{\rm MHz}]', r'\tau_1', r'\tau_2', r'T_{100}\;[{\rm K}]', r'\beta', r'\gamma', r'\delta', r'\epsilon']) 
		o = gp.triangle_plot(getdist_samples, folder + 'result.pdf', legend_FS=10, label_FS=13, axes_FS=7)
		
		v, t, w, sigma, inv_sigma, det_sigma = dm.real_data(2, 60, 120)
		
		model = dm.full_model(best_fit, v, 100, model_type_signal='tanh', model_type_foreground='exp', N21par=5, NFGpar=5)



	if fig==12:
		# Data used:  60-120
		folder = edges_folder + 'mid_band/polychord/20190508/case1_nominal/foreground_model_exp_signal_model_exp_60_120MHz/'
		getdist_samples, ww, ll, best_fit, covariance_matrix = gp.load_samples(folder + 'chain.txt', 2000, label_names=[r'A\;[{\rm K}]', r'\nu_0\;[{\rm MHz}]', r'w\;[{\rm MHz}]', r'\tau', r'\chi', r'T_{100}\;[{\rm K}]', r'\beta', r'\gamma', r'\delta', r'\epsilon']) 
		o = gp.triangle_plot(getdist_samples, folder + 'result.pdf', legend_FS=10, label_FS=13, axes_FS=7)
		
		v, t, w, sigma, inv_sigma, det_sigma = dm.real_data(10, 60, 120)
		
		model = dm.full_model(best_fit, v, 100, model_type_signal='exp', model_type_foreground='exp', N21par=5, NFGpar=5)	




	
	return v, t, w, model
	
	
	
	
	

def antsim3_calibration():

	FLOW  = 50
	FHIGH = 150


	# Spectra
	d   = np.genfromtxt(edges_folder + 'mid_band/calibration/receiver_calibration/receiver1/2018_01_25C/results/nominal/data/average_spectra_300_350.txt')
	ff  = d[:,0]
	tx  = d[:,5]

	tunc  = tx[(ff>=FLOW) & (ff<=FHIGH)]
	







	
	# Calibration quantities
	# ----------------------
	#rcv_file = edges_folder + 'mid_band/calibration/receiver_calibration/receiver1/2018_01_25C/results/nominal/calibration_files/calibration_file_receiver1_50_150MHz_cterms7_wterms8.txt'	
	
	#rcv_file = edges_folder + 'mid_band/calibration/receiver_calibration/receiver1/2018_01_25C/results/nominal/calibration_files/calibration_file_receiver1_cterms7_wterms8.txt'
	#rcv = np.genfromtxt(rcv_file)

	#fX      = rcv[:,0]
	#rcv2    = rcv[(fX>=FLOW) & (fX<=FHIGH),:]
	#f       = rcv2[:,0]
	#s11_LNA = rcv2[:,1] + 1j*rcv2[:,2]
	#C1      = rcv2[:,3]
	#C2      = rcv2[:,4]
	#TU      = rcv2[:,5]
	#TC      = rcv2[:,6]
	#TS      = rcv2[:,7]

			
	
	f, s11_LNA, C1, C2, TU, TC, TS = cal.MC_receiver('mid_band', MC_spectra_noise = np.zeros(4), MC_s11_syst = [1,1,0,0,0,0,1,1,1,1,0,0,0,0,0,0], MC_temp = np.zeros(4))
	
	
	
			
		
	# AntSim3 S11
	# -----------	
	path_s11 = edges_folder + 'mid_band/calibration/receiver_calibration/receiver1/2018_01_25C/results/nominal/s11/'
	fn  = (f-120)/60

	par        = np.genfromtxt(path_s11 + 'par_s11_simu_mag.txt')
	rsimu_mag  = ba.model_evaluate('polynomial', par, fn)
	par        = np.genfromtxt(path_s11 + 'par_s11_simu_ang.txt')
	rsimu_ang  = ba.model_evaluate('polynomial', par, fn)
	rsimu      = rsimu_mag * (np.cos(rsimu_ang) + 1j*np.sin(rsimu_ang))

	
	rsimu_MC = cal.MC_antenna_s11(f, rsimu, s11_Npar_max=14)
	
	

	



	

	# Calibrated antenna temperature with losses and beam chromaticity
	# ----------------------------------------------------------------
	tcal = cal.calibrated_antenna_temperature(tunc, rsimu_MC, s11_LNA, C1, C2, TU, TC, TS)
	
	fb, tb, wb = ba.spectral_binning_number_of_samples(f, tcal, np.ones(len(f))) 
	

		


	
	
	return f, rsimu, rsimu_MC, fb, tb







def plot_calibrated_raw_lab_data():
	

	
	sp = np.genfromtxt(edges_folder + 'mid_band/calibration/receiver_calibration/receiver1/2018_01_25C/results/nominal/calibration_files/calibration_file_receiver1_cterms7_wterms8.txt')
		
	f       = sp[:,0]
	s11_LNA = sp[:,1] + 1j*sp[:,2]
	C1      = sp[:,3]
	C2      = sp[:,4]
	TU      = sp[:,5]
	TC      = sp[:,6]
	TS      = sp[:,7]
	
	
	# S11
	path_s11 = edges_folder + 'mid_band/calibration/receiver_calibration/receiver1/2018_01_25C/results/nominal/s11/'
	fn  = (f-120)/60
	
	par     = np.genfromtxt(path_s11 + 'par_s11_LNA_mag.txt')	
	rl_mag  = ba.model_evaluate('polynomial', par, fn)
	par     = np.genfromtxt(path_s11 + 'par_s11_LNA_ang.txt')
	rl_ang  = ba.model_evaluate('polynomial', par, fn)	
	rl      = rl_mag * (np.cos(rl_ang) + 1j*np.sin(rl_ang))
	
	par     = np.genfromtxt(path_s11 + 'par_s11_amb_mag.txt')
	ra_mag  = ba.model_evaluate('fourier', par, fn)	
	par     = np.genfromtxt(path_s11 + 'par_s11_amb_ang.txt')
	ra_ang  = ba.model_evaluate('fourier', par, fn)
	ra      = ra_mag * (np.cos(ra_ang) + 1j*np.sin(ra_ang))
		
	par     = np.genfromtxt(path_s11 + 'par_s11_hot_mag.txt')
	rh_mag  = ba.model_evaluate('fourier', par, fn)
	par     = np.genfromtxt(path_s11 + 'par_s11_hot_ang.txt')
	rh_ang  = ba.model_evaluate('fourier', par, fn)
	rh      = rh_mag * (np.cos(rh_ang) + 1j*np.sin(rh_ang))
		
	par     = np.genfromtxt(path_s11 + 'par_s11_open_mag.txt')
	ro_mag  = ba.model_evaluate('fourier', par, fn)	
	par     = np.genfromtxt(path_s11 + 'par_s11_open_ang.txt')
	ro_ang  = ba.model_evaluate('fourier', par, fn)
	ro      = ro_mag * (np.cos(ro_ang) + 1j*np.sin(ro_ang))
		
	par     = np.genfromtxt(path_s11 + 'par_s11_shorted_mag.txt')
	rs_mag  = ba.model_evaluate('fourier', par, fn)	
	par     = np.genfromtxt(path_s11 + 'par_s11_shorted_ang.txt')
	rs_ang  = ba.model_evaluate('fourier', par, fn)		
	rs      = rs_mag * (np.cos(rs_ang) + 1j*np.sin(rs_ang))
	
	
		
	par         = np.genfromtxt(path_s11 + 'par_s11_sr_mag.txt')
	s11_sr_mag  = ba.model_evaluate('polynomial', par, fn)
	par         = np.genfromtxt(path_s11 + 'par_s11_sr_ang.txt')
	s11_sr_ang  = ba.model_evaluate('polynomial', par, fn)
	s11_sr      = s11_sr_mag * (np.cos(s11_sr_ang) + 1j*np.sin(s11_sr_ang))
		
	par            = np.genfromtxt(path_s11 + 'par_s12s21_sr_mag.txt')
	s12s21_sr_mag  = ba.model_evaluate('polynomial', par, fn)	
	par            = np.genfromtxt(path_s11 + 'par_s12s21_sr_ang.txt')
	s12s21_sr_ang  = ba.model_evaluate('polynomial', par, fn)
	s12s21_sr      = s12s21_sr_mag * (np.cos(s12s21_sr_ang) + 1j*np.sin(s12s21_sr_ang))
		
	par         = np.genfromtxt(path_s11 + 'par_s22_sr_mag.txt')
	s22_sr_mag  = ba.model_evaluate('polynomial', par, fn)
	par         = np.genfromtxt(path_s11 + 'par_s22_sr_ang.txt')
	s22_sr_ang  = ba.model_evaluate('polynomial', par, fn)
	s22_sr      = s22_sr_mag * (np.cos(s22_sr_ang) + 1j*np.sin(s22_sr_ang))
	
	
	
	par        = np.genfromtxt(path_s11 + 'par_s11_simu_mag.txt')
	rsimu_mag  = ba.model_evaluate('polynomial', par, fn)
	par        = np.genfromtxt(path_s11 + 'par_s11_simu_ang.txt')
	rsimu_ang  = ba.model_evaluate('polynomial', par, fn)
	rsimu      = rsimu_mag * (np.cos(rsimu_ang) + 1j*np.sin(rsimu_ang))







	d = np.genfromtxt(edges_folder + 'mid_band/calibration/receiver_calibration/receiver1/2018_01_25C/results/nominal/data/average_spectra_300_350.txt')
	Tae = d[:,1]
	The = d[:,2]
	Toe = d[:,3]
	Tse = d[:,4]
	Tqe = d[:,5]
	

	o = cal.models_calibration_spectra(1, f, MC_spectra_noise=np.zeros(4))
	mae = o[0]
	mhe = o[1]
	moe = o[2]
	mse = o[3]







	Ta = cal.calibrated_antenna_temperature(Tae, ra, s11_LNA, C1, C2, TU, TC, TS)
	Th = cal.calibrated_antenna_temperature(The, rh, s11_LNA, C1, C2, TU, TC, TS)
	To = cal.calibrated_antenna_temperature(Toe, ro, s11_LNA, C1, C2, TU, TC, TS)
	Ts = cal.calibrated_antenna_temperature(Tse, rs, s11_LNA, C1, C2, TU, TC, TS)
	Tq = cal.calibrated_antenna_temperature(Tqe, rsimu, s11_LNA, C1, C2, TU, TC, TS)




	# Temperature of hot device

	# reflection coefficient of termination
	rht = rc.gamma_de_embed(s11_sr, s12s21_sr, s22_sr, rh)

	# inverting the direction of the s-parameters,
	# since the port labels have to be inverted to match those of Pozar eqn 10.25
	s11_sr_rev = s22_sr
	s22_sr_rev = s11_sr

	# absolute value of S_21
	abs_s21 = np.sqrt(np.abs(s12s21_sr))

	# available power gain
	G = ( abs_s21**2 ) * ( 1-np.abs(rht)**2 ) / ( (np.abs(1-s11_sr_rev*rht))**2 * (1-(np.abs(rh))**2) )

	# temperature
	#Thd  = G*Th + (1-G)*Ta

	
	
	fn = (f-120)/60
	
	
	Nt=5; pp = ba.fit_polynomial_fourier('polynomial', fn, Tae, Nt)
	mae = pp[1]

	Nt=7; pp = ba.fit_polynomial_fourier('polynomial', fn, The, Nt)
	mhe = pp[1]
	
	Nt=53; pp = ba.fit_polynomial_fourier('fourier', fn, Toe, Nt)
	moe = pp[1]
	
	Nt=53; pp = ba.fit_polynomial_fourier('fourier', fn, Tse, Nt)
	mse = pp[1]
	
	
	Nt=15; pp = ba.fit_polynomial_fourier('fourier', fn, Tqe, Nt)
	mqe = pp[1]	
	
	plt.figure(1, figsize=[12, 10])
	
	plt.subplot(5, 1, 1)
	plt.ylabel('temperature\nambient [K]')
	plt.plot(f, Tae-mae)
	plt.xticks(np.arange(50,151,10))
	
	plt.subplot(5, 1, 2)
	plt.ylabel('temperature\nhot [K]')
	plt.plot(f, The-mhe)
	plt.xticks(np.arange(50,151,10))
	
	plt.subplot(5, 1, 3)
	plt.ylabel('temperature\nopen cable [K]')
	plt.plot(f, Toe-moe)
	plt.xticks(np.arange(50,151,10))
	
	plt.subplot(5, 1, 4)
	plt.ylabel('temperature\nshorted cable [K]')
	plt.plot(f, Tse-mse)
	plt.xticks(np.arange(50,151,10))
	
	
	
	plt.subplot(5, 1, 5)
	plt.ylabel('temperature\nantsim3 [K]')
	plt.xlabel('frequency [MHz]')
	
	

	
	
	
	#par = np.polyfit(f, Tqe, Nt-1); mqe = np.polyval(par, f)
	
	plt.plot(f, Tqe - mqe)
	plt.xticks(np.arange(50,151,10))
	
	plt.savefig('/home/raul/Desktop/calibration_spectra_raw.pdf', bbox_inches='tight')
	plt.close()	
	plt.close()


















	Ns = 32
	fb, rb1, wb1 = ba.spectral_binning_number_of_samples(f, Tae-mae, np.ones(len(f)), nsamples=Ns)
	fb, rb2, wb2 = ba.spectral_binning_number_of_samples(f, The-mhe, np.ones(len(f)), nsamples=Ns)
	fb, rb3, wb3 = ba.spectral_binning_number_of_samples(f, Toe-moe, np.ones(len(f)), nsamples=Ns)
	fb, rb4, wb4 = ba.spectral_binning_number_of_samples(f, Tse-mse, np.ones(len(f)), nsamples=Ns)
	fb, rb5, wb5 = ba.spectral_binning_number_of_samples(f, Tqe-mqe, np.ones(len(f)), nsamples=Ns)
	
	
	plt.figure(2, figsize=[12, 10])
	
	plt.subplot(5, 1, 1)
	plt.ylabel('temperature\nambient [K]')
	plt.plot(fb, rb1, 'r')
	plt.xticks(np.arange(50,151,10))
	
	plt.subplot(5, 1, 2)
	plt.ylabel('temperature\nhot [K]')
	plt.plot(fb, rb2, 'r')
	plt.xticks(np.arange(50,151,10))
	
	plt.subplot(5, 1, 3)
	plt.ylabel('temperature\nopen cable [K]')
	plt.plot(fb, rb3, 'r')
	plt.xticks(np.arange(50,151,10))
	
	plt.subplot(5, 1, 4)
	plt.ylabel('temperature\nshorted cable [K]')
	plt.plot(fb, rb4, 'r')
	plt.xticks(np.arange(50,151,10))
	
	
	
	plt.subplot(5, 1, 5)
	plt.ylabel('temperature\nantsim3 [K]')
	plt.xlabel('frequency [MHz]')
	#Nt=21; par = np.polyfit(f, Tqe, Nt-1); mqe = np.polyval(par, f)
	plt.plot(fb, rb5, 'r')
	plt.xticks(np.arange(50,151,10))
	
	
	plt.savefig('/home/raul/Desktop/calibration_spectra_binned.pdf', bbox_inches='tight')
	plt.close()	
	plt.close()



	
	return f, Ta, Th, To, Ts, Tq, Tae, The, Toe, Tse, Tqe, mae, mhe, moe, mse




