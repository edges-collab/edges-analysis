
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as sci

import edges as eg
import basic as ba
import rfi as rfi

import calibration as cal
import calibration_receiver1 as cr1

import reflection_coefficient as rc

import data_models as dm
import getdist_plots as gp

import healpy as hp

import astropy.units as apu
import astropy.time as apt
import astropy.coordinates as apc

import datetime as dt

from os import listdir


import os, sys
edges_folder       = os.environ['EDGES_vol3']
print('EDGES Folder: ' + edges_folder)

sys.path.insert(0, "/home/raul/edges/old")

import old_edges as oeg
import old_high_band_edges as ohb


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
	# --------------------------------
	
	if case == 0:
		flag_folder       = 'case_nominal'
		
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
		
			
	if case == 1:
		flag_folder       = 'test_rcv18_sw18_no_beam_correction' #'case_nominal_55_150MHz'
		
		receiver_cal_file = 2   # cterms=7, wterms=8 terms over 50-150 MHz
		
		antenna_s11_day   = 147
		antenna_s11_case  = 3  # taken 2+ minutes after turning on the switch
		antenna_s11_Nfit  = 14  # 14 terms over 55-150 MHz
		
		balun_correction  = 1
		ground_correction = 1
		beam_correction   = 0 # 1
		bf_case           = 0   # alan0 beam (30x30m ground plane), haslam map with gaussian lat-function for spectral index
		
		FLOW  = 55
		FHIGH = 150
		Nfg   = 5

		
		

	if case == 2:
		flag_folder       = 'calibration_2019_10_no_ground_loss_no_beam_corrections'
		
		receiver_cal_file = 100   # cterms=10, wterms=13 terms over 50-190 MHz
		
		antenna_s11_day   = 147
		antenna_s11_case  = 3   # taken 2+ minutes after turning on the switch
		antenna_s11_Nfit  = 14  # 14 terms over 55-150 MHz, 13 terms over 55-120 MHz
		
		balun_correction  = 1
		ground_correction = 0
		beam_correction   = 0
		bf_case           = 0   # alan0 beam (30x30m ground plane), haslam map with gaussian lat-function for spectral index
		
		FLOW  = 55
		FHIGH = 150
		Nfg   = 5
			



	if case == 3:
		flag_folder       = 'case_nominal_50-150MHz_no_ground_loss_no_beam_corrections'
		
		receiver_cal_file = 2   # cterms=7, wterms=8 terms over 50-150 MHz
		
		antenna_s11_day   = 147
		antenna_s11_case  = 3   # taken 2+ minutes after turning on the switch
		antenna_s11_Nfit  = 14  # 14 terms over 55-150 MHz
		
		balun_correction  = 1
		ground_correction = 0
		beam_correction   = 0
		bf_case           = 0   # alan0 beam (30x30m ground plane), haslam map with gaussian lat-function for spectral index
		
		FLOW  = 50
		FHIGH = 150
		Nfg   = 5
		
		




	if case == 4:
		flag_folder       = 'case_nominal_8_11_terms_50-150MHz_no_ground_loss_no_beam_corrections'
		
		receiver_cal_file = 26   # cterms=8, wterms=11 terms over 50-150 MHz
		
		antenna_s11_day   = 147
		antenna_s11_case  = 3   # taken 2+ minutes after turning on the switch
		antenna_s11_Nfit  = 14  # 14 terms over 55-150 MHz
		
		balun_correction  = 1
		ground_correction = 0
		beam_correction   = 0
		bf_case           = 0   # alan0 beam (30x30m ground plane), haslam map with gaussian lat-function for spectral index
		
		FLOW  = 50
		FHIGH = 150
		Nfg   = 5


	if case == 5:
		flag_folder       = 'case_nominal_14_14_terms_55-150MHz_no_ground_loss_no_beam_corrections'
		
		receiver_cal_file = 200   # cterms=8, wterms=11 terms over 50-150 MHz
		
		antenna_s11_day   = 147
		antenna_s11_case  = 3   # taken 2+ minutes after turning on the switch
		antenna_s11_Nfit  = 14  # 14 terms over 55-150 MHz
		
		balun_correction  = 1
		ground_correction = 0
		beam_correction   = 0
		bf_case           = 0   # alan0 beam (30x30m ground plane), haslam map with gaussian lat-function for spectral index
		
		FLOW  = 55
		FHIGH = 150
		Nfg   = 5




	if case == 61:
		flag_folder       = 'case_nominal_60-120MHz_7_4'
		
		receiver_cal_file = 301   # cterms=8, wterms=11 terms over 50-150 MHz
		
		antenna_s11_day   = 147
		antenna_s11_case  = 3   # taken 2+ minutes after turning on the switch
		antenna_s11_Nfit  = 13  # 13 terms over 60-120 MHz
		
		balun_correction  = 1
		ground_correction = 0
		beam_correction   = 0
		bf_case           = 0   # alan0 beam (30x30m ground plane), haslam map with gaussian lat-function for spectral index
		
		FLOW  = 60
		FHIGH = 120
		Nfg   = 5


	if case == 62:
		flag_folder       = 'case_nominal_60-120MHz_7_5'
		
		receiver_cal_file = 302   # cterms=8, wterms=11 terms over 50-150 MHz
		
		antenna_s11_day   = 147
		antenna_s11_case  = 3   # taken 2+ minutes after turning on the switch
		antenna_s11_Nfit  = 13  # 13 terms over 60-120 MHz
		
		balun_correction  = 1
		ground_correction = 0
		beam_correction   = 0
		bf_case           = 0   # alan0 beam (30x30m ground plane), haslam map with gaussian lat-function for spectral index
		
		FLOW  = 60
		FHIGH = 120
		Nfg   = 5



	
	if case == 63:
		flag_folder       = 'case_nominal_60-120MHz_7_9'
		
		receiver_cal_file = 303   # cterms=8, wterms=11 terms over 50-150 MHz
		
		antenna_s11_day   = 147
		antenna_s11_case  = 3   # taken 2+ minutes after turning on the switch
		antenna_s11_Nfit  = 13  # 13 terms over 60-120 MHz
		
		balun_correction  = 1
		ground_correction = 0
		beam_correction   = 0
		bf_case           = 0   # alan0 beam (30x30m ground plane), haslam map with gaussian lat-function for spectral index
		
		FLOW  = 60
		FHIGH = 120
		Nfg   = 5



	if case == 64:
		flag_folder       = 'case_nominal_60-120MHz_8_4'
		
		receiver_cal_file = 304   # cterms=8, wterms=11 terms over 50-150 MHz
		
		antenna_s11_day   = 147
		antenna_s11_case  = 3   # taken 2+ minutes after turning on the switch
		antenna_s11_Nfit  = 13  # 13 terms over 60-120 MHz
		
		balun_correction  = 1
		ground_correction = 0
		beam_correction   = 0
		bf_case           = 0   # alan0 beam (30x30m ground plane), haslam map with gaussian lat-function for spectral index
		
		FLOW  = 60
		FHIGH = 120
		Nfg   = 5



	if case == 65:
		flag_folder       = 'case_nominal_60-120MHz_8_5'
		
		receiver_cal_file = 305   # cterms=8, wterms=11 terms over 50-150 MHz
		
		antenna_s11_day   = 147
		antenna_s11_case  = 3   # taken 2+ minutes after turning on the switch
		antenna_s11_Nfit  = 13  # 13 terms over 60-120 MHz
		
		balun_correction  = 1
		ground_correction = 0
		beam_correction   = 0
		bf_case           = 0   # alan0 beam (30x30m ground plane), haslam map with gaussian lat-function for spectral index
		
		FLOW  = 60
		FHIGH = 120
		Nfg   = 5


	if case == 66:
		flag_folder       = 'case_nominal_60-120MHz_8_9'
		
		receiver_cal_file = 306   # cterms=8, wterms=11 terms over 50-150 MHz
		
		antenna_s11_day   = 147
		antenna_s11_case  = 3   # taken 2+ minutes after turning on the switch
		antenna_s11_Nfit  = 13  # 13 terms over 60-120 MHz
		
		balun_correction  = 1
		ground_correction = 0
		beam_correction   = 0
		bf_case           = 0   # alan0 beam (30x30m ground plane), haslam map with gaussian lat-function for spectral index
		
		FLOW  = 60
		FHIGH = 120
		Nfg   = 5


	if case == 67:
		flag_folder       = 'case_nominal_60-120MHz_5_4'
		
		receiver_cal_file = 307   # cterms=8, wterms=11 terms over 50-150 MHz
		
		antenna_s11_day   = 147
		antenna_s11_case  = 3   # taken 2+ minutes after turning on the switch
		antenna_s11_Nfit  = 13  # 13 terms over 60-120 MHz
		
		balun_correction  = 1
		ground_correction = 0
		beam_correction   = 0
		bf_case           = 0   # alan0 beam (30x30m ground plane), haslam map with gaussian lat-function for spectral index
		
		FLOW  = 60
		FHIGH = 120
		Nfg   = 5



	if case == 68:
		flag_folder       = 'case_nominal_60-120MHz_6_4'
		
		receiver_cal_file = 308   # cterms=8, wterms=11 terms over 50-150 MHz
		
		antenna_s11_day   = 147
		antenna_s11_case  = 3   # taken 2+ minutes after turning on the switch
		antenna_s11_Nfit  = 13  # 13 terms over 60-120 MHz
		
		balun_correction  = 1
		ground_correction = 0
		beam_correction   = 0
		bf_case           = 0   # alan0 beam (30x30m ground plane), haslam map with gaussian lat-function for spectral index
		
		FLOW  = 60
		FHIGH = 120
		Nfg   = 5




	if case == 69:
		flag_folder       = 'case_nominal_55-150MHz_7_7'
		
		receiver_cal_file = 21   # cterms=8, wterms=11 terms over 50-150 MHz
		
		antenna_s11_day   = 147
		antenna_s11_case  = 3   # taken 2+ minutes after turning on the switch
		antenna_s11_Nfit  = 14  # 13 terms over 60-120 MHz
		
		balun_correction  = 1
		ground_correction = 0
		beam_correction   = 0
		bf_case           = 0   # alan0 beam (30x30m ground plane), haslam map with gaussian lat-function for spectral index
		
		FLOW  = 55
		FHIGH = 150
		Nfg   = 5



	if case == 70:
		flag_folder       = 'case_nominal_55-150MHz_7_10'
		
		receiver_cal_file = 24   # cterms=8, wterms=11 terms over 50-150 MHz
		
		antenna_s11_day   = 147
		antenna_s11_case  = 3   # taken 2+ minutes after turning on the switch
		antenna_s11_Nfit  = 14  # 13 terms over 60-120 MHz
		
		balun_correction  = 1
		ground_correction = 0
		beam_correction   = 0
		bf_case           = 0   # alan0 beam (30x30m ground plane), haslam map with gaussian lat-function for spectral index
		
		FLOW  = 55
		FHIGH = 150
		Nfg   = 5



	if case == 71:
		flag_folder       = 'case_nominal_50-150MHz_7_8'
		
		receiver_cal_file = 201   # cterms=7, wterms=8 terms over 50-150 MHz
		
		antenna_s11_day   = 147
		antenna_s11_case  = 3   # taken 2+ minutes after turning on the switch
		antenna_s11_Nfit  = 14  # 14 terms over 55-150 MHz
		
		balun_correction  = 1
		ground_correction = 0
		beam_correction   = 0
		bf_case           = 0   # alan0 beam (30x30m ground plane), haslam map with gaussian lat-function for spectral index
		
		FLOW  = 50
		FHIGH = 150
		Nfg   = 5
		


	if case == 72:
		flag_folder       = 'case_nominal_50-150MHz_7_11'
		
		receiver_cal_file = 202   # cterms=7, wterms=8 terms over 50-150 MHz
		
		antenna_s11_day   = 147
		antenna_s11_case  = 3   # taken 2+ minutes after turning on the switch
		antenna_s11_Nfit  = 14  # 14 terms over 55-150 MHz
		
		balun_correction  = 1
		ground_correction = 0
		beam_correction   = 0
		bf_case           = 0   # alan0 beam (30x30m ground plane), haslam map with gaussian lat-function for spectral index
		
		FLOW  = 50
		FHIGH = 150
		Nfg   = 5
			




	if case == 73:
		flag_folder       = 'case_nominal_50-150MHz_7_8_LNA_rep1'
		
		receiver_cal_file = 203   # cterms=7, wterms=8 terms over 50-150 MHz
		
		antenna_s11_day   = 147
		antenna_s11_case  = 3   # taken 2+ minutes after turning on the switch
		antenna_s11_Nfit  = 14  # 14 terms over 55-150 MHz
		
		balun_correction  = 1
		ground_correction = 0
		beam_correction   = 0
		bf_case           = 0   # alan0 beam (30x30m ground plane), haslam map with gaussian lat-function for spectral index
		
		FLOW  = 50
		FHIGH = 150
		Nfg   = 5
			



	if case == 74:
		flag_folder       = 'case_nominal_50-150MHz_7_8_LNA_rep2'
		
		receiver_cal_file = 204   # cterms=7, wterms=8 terms over 50-150 MHz
		
		antenna_s11_day   = 147
		antenna_s11_case  = 3   # taken 2+ minutes after turning on the switch
		antenna_s11_Nfit  = 14  # 14 terms over 55-150 MHz
		
		balun_correction  = 1
		ground_correction = 0
		beam_correction   = 0
		bf_case           = 0   # alan0 beam (30x30m ground plane), haslam map with gaussian lat-function for spectral index
		
		FLOW  = 50
		FHIGH = 150
		Nfg   = 5
			

	if case == 75:
		flag_folder       = 'case_nominal_50-150MHz_7_8_LNA_rep12'
		
		receiver_cal_file = 205   # cterms=7, wterms=8 terms over 50-150 MHz
		
		antenna_s11_day   = 147
		antenna_s11_case  = 3   # taken 2+ minutes after turning on the switch
		antenna_s11_Nfit  = 14  # 14 terms over 55-150 MHz
		
		balun_correction  = 1
		ground_correction = 0
		beam_correction   = 0
		bf_case           = 0   # alan0 beam (30x30m ground plane), haslam map with gaussian lat-function for spectral index
		
		FLOW  = 50
		FHIGH = 150
		Nfg   = 5




	if case == 731:
		flag_folder       = 'case_nominal_50-150MHz_7_8_LNA_rep1_ant1'
		
		receiver_cal_file = 203   # cterms=7, wterms=8 terms over 50-150 MHz
		
		antenna_s11_day   = 147
		antenna_s11_case  = 1   # taken 2+ minutes after turning on the switch
		antenna_s11_Nfit  = 14  # 14 terms over 55-150 MHz
		
		balun_correction  = 1
		ground_correction = 0
		beam_correction   = 0
		bf_case           = 0   # alan0 beam (30x30m ground plane), haslam map with gaussian lat-function for spectral index
		
		FLOW  = 50
		FHIGH = 150
		Nfg   = 5


	if case == 735:
		flag_folder       = 'case_nominal_50-150MHz_7_8_LNA_rep1_ant5'
		
		receiver_cal_file = 203   # cterms=7, wterms=8 terms over 50-150 MHz
		
		antenna_s11_day   = 147
		antenna_s11_case  = 5   # taken 2+ minutes after turning on the switch
		antenna_s11_Nfit  = 14  # 14 terms over 55-150 MHz
		
		balun_correction  = 1
		ground_correction = 0
		beam_correction   = 0
		bf_case           = 0   # alan0 beam (30x30m ground plane), haslam map with gaussian lat-function for spectral index
		
		FLOW  = 50
		FHIGH = 150
		Nfg   = 5



	if case == 736:
		flag_folder       = 'case_nominal_50-150MHz_7_8_LNA_rep1_ant6'
		
		receiver_cal_file = 203   # cterms=7, wterms=8 terms over 50-150 MHz
		
		antenna_s11_day   = 147
		antenna_s11_case  = 6   # taken 2+ minutes after turning on the switch
		antenna_s11_Nfit  = 14  # 14 terms over 55-150 MHz
		
		balun_correction  = 1
		ground_correction = 0
		beam_correction   = 0
		bf_case           = 0   # alan0 beam (30x30m ground plane), haslam map with gaussian lat-function for spectral index
		
		FLOW  = 50
		FHIGH = 150
		Nfg   = 5










	if case == 401:
		flag_folder       = 'case_nominal_50-150MHz_LNA1_a1_h1_o1_s1_sim2'
		
		receiver_cal_file = 401   # cterms=7, wterms=8 terms over 50-150 MHz
		
		antenna_s11_day   = 147
		antenna_s11_case  = 3   # taken 2+ minutes after turning on the switch
		antenna_s11_Nfit  = 14  # 14 terms over 55-150 MHz
		
		balun_correction  = 1
		ground_correction = 0
		beam_correction   = 0
		bf_case           = 0   # alan0 beam (30x30m ground plane), haslam map with gaussian lat-function for spectral index
		
		FLOW  = 50
		FHIGH = 150
		Nfg   = 5


	if case == 402:
		flag_folder       = 'case_nominal_50-150MHz_LNA1_a1_h2_o1_s1_sim2'
		
		receiver_cal_file = 402   # cterms=7, wterms=8 terms over 50-150 MHz
		
		antenna_s11_day   = 147
		antenna_s11_case  = 3   # taken 2+ minutes after turning on the switch
		antenna_s11_Nfit  = 14  # 14 terms over 55-150 MHz
		
		balun_correction  = 1
		ground_correction = 0
		beam_correction   = 0
		bf_case           = 0   # alan0 beam (30x30m ground plane), haslam map with gaussian lat-function for spectral index
		
		FLOW  = 50
		FHIGH = 150
		Nfg   = 5


	if case == 403:
		flag_folder       = 'case_nominal_50-150MHz_LNA1_a2_h1_o1_s1_sim2'
		
		receiver_cal_file = 403   # cterms=7, wterms=8 terms over 50-150 MHz
		
		antenna_s11_day   = 147
		antenna_s11_case  = 3   # taken 2+ minutes after turning on the switch
		antenna_s11_Nfit  = 14  # 14 terms over 55-150 MHz
		
		balun_correction  = 1
		ground_correction = 0
		beam_correction   = 0
		bf_case           = 0   # alan0 beam (30x30m ground plane), haslam map with gaussian lat-function for spectral index
		
		FLOW  = 50
		FHIGH = 150
		Nfg   = 5




	if case == 404:
		flag_folder       = 'case_nominal_50-150MHz_LNA1_a2_h2_o1_s1_sim2'
		
		receiver_cal_file = 404   # cterms=7, wterms=8 terms over 50-150 MHz
		
		antenna_s11_day   = 147
		antenna_s11_case  = 3   # taken 2+ minutes after turning on the switch
		antenna_s11_Nfit  = 14  # 14 terms over 55-150 MHz
		
		balun_correction  = 1
		ground_correction = 0
		beam_correction   = 0
		bf_case           = 0   # alan0 beam (30x30m ground plane), haslam map with gaussian lat-function for spectral index
		
		FLOW  = 50
		FHIGH = 150
		Nfg   = 5
		
		
		

	if case == 405:
		flag_folder       = 'case_nominal_50-150MHz_LNA1_a2_h2_o1_s2_sim2'
		
		receiver_cal_file = 405   # cterms=7, wterms=8 terms over 50-150 MHz
		
		antenna_s11_day   = 147
		antenna_s11_case  = 3   # taken 2+ minutes after turning on the switch
		antenna_s11_Nfit  = 14  # 14 terms over 55-150 MHz
		
		balun_correction  = 1
		ground_correction = 0
		beam_correction   = 0
		bf_case           = 0   # alan0 beam (30x30m ground plane), haslam map with gaussian lat-function for spectral index
		
		FLOW  = 50
		FHIGH = 150
		Nfg   = 5



	if case == 406:
		flag_folder       = 'case_nominal_50-150MHz_LNA1_a2_h2_o2_s1_sim2'
		
		receiver_cal_file = 406   # cterms=7, wterms=8 terms over 50-150 MHz
		
		antenna_s11_day   = 147
		antenna_s11_case  = 3   # taken 2+ minutes after turning on the switch
		antenna_s11_Nfit  = 14  # 14 terms over 55-150 MHz
		
		balun_correction  = 1
		ground_correction = 0
		beam_correction   = 0
		bf_case           = 0   # alan0 beam (30x30m ground plane), haslam map with gaussian lat-function for spectral index
		
		FLOW  = 50
		FHIGH = 150
		Nfg   = 5




	if case == 407:
		flag_folder       = 'case_nominal_50-150MHz_LNA1_a2_h2_o2_s2_sim2'
		
		receiver_cal_file = 407   # cterms=7, wterms=8 terms over 50-150 MHz
		
		antenna_s11_day   = 147
		antenna_s11_case  = 3   # taken 2+ minutes after turning on the switch
		antenna_s11_Nfit  = 14  # 14 terms over 55-150 MHz
		
		balun_correction  = 1
		ground_correction = 0
		beam_correction   = 0
		bf_case           = 0   # alan0 beam (30x30m ground plane), haslam map with gaussian lat-function for spectral index
		
		FLOW  = 50
		FHIGH = 150
		Nfg   = 5




	if case == 500:
		flag_folder       = 'case_nominal_50-150MHz_LNA2_a2_h2_o2_s1_sim2_all_lc_no_bc' #'case_nominal_50-150MHz_LNA1_a2_h2_o2_s1_sim2_all_corrections'
		
		receiver_cal_file = 2   # cterms=7, wterms=8 terms over 50-150 MHz
		
		antenna_s11_day   = 147
		antenna_s11_case  = 3   # taken 2+ minutes after turning on the switch
		antenna_s11_Nfit  = 14  # 14 terms over 55-150 MHz
		
		antenna_correction = 1
		balun_correction   = 1
		ground_correction  = 1
		beam_correction    = 0
		bf_case            = 0   # alan0 beam (30x30m ground plane), haslam map with gaussian lat-function for spectral index
		
		FLOW  = 50
		FHIGH = 150
		Nfg   = 5



	if case == 501:
		flag_folder       = 'case_nominal_50-150MHz_LNA2_a2_h2_o2_s1_sim2_all_lc_yes_bc' #'case_nominal_50-150MHz_LNA1_a2_h2_o2_s1_sim2_all_corrections'
		
		receiver_cal_file = 2   # cterms=7, wterms=8 terms over 50-150 MHz
		
		antenna_s11_day   = 147
		antenna_s11_case  = 3   # taken 2+ minutes after turning on the switch
		antenna_s11_Nfit  = 14  # 14 terms over 55-150 MHz
		
		antenna_correction = 1
		balun_correction   = 1
		ground_correction  = 1
		beam_correction    = 1
		bf_case            = 0   # alan0 beam (30x30m ground plane), haslam map with gaussian lat-function for spectral index
		
		FLOW  = 50
		FHIGH = 150
		Nfg   = 5
		
	if case == 510:
		flag_folder       = 'test_A' #'case_nominal_50-150MHz_LNA1_a2_h2_o2_s1_sim2_all_corrections'
		
		receiver_cal_file = 2   # cterms=7, wterms=8 terms over 50-150 MHz
		
		antenna_s11_day   = 147
		antenna_s11_case  = 3   # taken 2+ minutes after turning on the switch
		antenna_s11_Nfit  = 14  # 14 terms over 55-150 MHz
		
		antenna_correction = 1
		balun_correction   = 1
		ground_correction  = 1
		beam_correction    = 1
		beam_correction_case = 0   # alan0 beam (30x30m ground plane), haslam map with gaussian lat-function for spectral index
		
		FLOW  = 50
		FHIGH = 150
		Nfg   = 5		
	
	if case == 511:
		flag_folder       = 'test_B' #'case_nominal_50-150MHz_LNA1_a2_h2_o2_s1_sim2_all_corrections'
		
		receiver_cal_file = 2   # cterms=7, wterms=8 terms over 50-150 MHz
		
		antenna_s11_day   = 147
		antenna_s11_case  = 3   # taken 2+ minutes after turning on the switch
		antenna_s11_Nfit  = 14  # 14 terms over 55-150 MHz
		
		antenna_correction = 1
		balun_correction   = 1
		ground_correction  = 1
		beam_correction    = 1
		beam_correction_case = 1   # alan0 beam (30x30m ground plane), haslam map with gaussian lat-function for spectral index
		
		FLOW  = 50
		FHIGH = 150
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
	for i in range(len(new_list)):  # range(15): # 
		#print(i)
		
		day = int(new_list[i][5:8])
		
		if (day >= first_day) & (day <= last_day):
			print(day)
			
			o = eg.level2_to_level3('mid_band', new_list[i], flag_folder=flag_folder, receiver_cal_file=receiver_cal_file, antenna_s11_year=2018, antenna_s11_day=antenna_s11_day, antenna_s11_case=antenna_s11_case, antenna_s11_Nfit=antenna_s11_Nfit, antenna_correction=antenna_correction, balun_correction=balun_correction, ground_correction=ground_correction, beam_correction=beam_correction, beam_correction_case=beam_correction_case, FLOW=FLOW, FHIGH=FHIGH, Nfg=Nfg)
		
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






















def VNA_comparison4():
	
	path_folder  = edges_folder + 'others/vna_comparison/keysight_P9370A/'
	

	o_p1, fx       = rc.s1p_read(path_folder + 'open.s1p')
	s_p1, fx       = rc.s1p_read(path_folder + 'short.s1p')
	m_p1, fx       = rc.s1p_read(path_folder + 'load.s1p')
	at3_p1, fx     = rc.s1p_read(path_folder + 'attn3db.s1p')
	at6_p1, fx     = rc.s1p_read(path_folder + 'attn6db.s1p')
	at10_p1, fx    = rc.s1p_read(path_folder + 'attn10db.s1p')	
	
	o_p2, fx       = rc.s1p_read(path_folder + 'port2_open.s1p')
	s_p2, fx       = rc.s1p_read(path_folder + 'port2_short.s1p')
	m_p2, fx       = rc.s1p_read(path_folder + 'port2_load.s1p')
	at3_p2, fx     = rc.s1p_read(path_folder + 'port2_attn3db.s1p')
	at6_p2, fx     = rc.s1p_read(path_folder + 'port2_attn6db.s1p')
	at10_p2, fx    = rc.s1p_read(path_folder + 'port2_attn10db.s1p')


	
	FLOW = 15e6
	f    = fx[(fx>=FLOW)]
	
	o_p1    = o_p1[(fx>=FLOW)]
	s_p1    = s_p1[(fx>=FLOW)]
	m_p1    = m_p1[(fx>=FLOW)]
	at3_p1  = at3_p1[(fx>=FLOW)]
	at6_p1  = at6_p1[(fx>=FLOW)]
	at10_p1 = at10_p1[(fx>=FLOW)]	
	
	o_p2    = o_p2[(fx>=FLOW)]
	s_p2    = s_p2[(fx>=FLOW)]
	m_p2    = m_p2[(fx>=FLOW)]
	at3_p2  = at3_p2[(fx>=FLOW)]
	at6_p2  = at6_p2[(fx>=FLOW)]
	at10_p2 = at10_p2[(fx>=FLOW)]		


	
	
	Leads = 0.004
	R50   = 48.785 - Leads
	R3    = 163.70 - Leads
	R6    = 85.04  - Leads
	R10   = 61.615 - Leads
	
	g3    = rc.impedance2gamma(R3, 50) * np.ones(len(f))
	g6    = rc.impedance2gamma(R6, 50) * np.ones(len(f))
	g10   = rc.impedance2gamma(R10, 50) * np.ones(len(f))
	
	xx  = rc.agilent_85033E(f, R50, m = 0, md_value_ps = 38)
	o_a = xx[0]
	s_a = xx[1]
	m_a = xx[2]	
	
	


	# Correction	
	at3_p1c, xx1, xx2, xx3  = rc.de_embed(o_a, s_a, m_a, o_p1, s_p1, m_p1, at3_p1)
	at6_p1c, xx1, xx2, xx3  = rc.de_embed(o_a, s_a, m_a, o_p1, s_p1, m_p1, at6_p1)
	at10_p1c, xx1, xx2, xx3 = rc.de_embed(o_a, s_a, m_a, o_p1, s_p1, m_p1, at10_p1)
	
	at3_p2c, xx1, xx2, xx3  = rc.de_embed(o_a, s_a, m_a, o_p2, s_p2, m_p2, at3_p2)
	at6_p2c, xx1, xx2, xx3  = rc.de_embed(o_a, s_a, m_a, o_p2, s_p2, m_p2, at6_p2)
	at10_p2c, xx1, xx2, xx3 = rc.de_embed(o_a, s_a, m_a, o_p2, s_p2, m_p2, at10_p2)





	# Plot
	
	plt.figure(1, figsize=(12, 10))
	
	plt.subplot(3,2,1)
	plt.plot(f/1e6, 20*np.log10(np.abs(at3_p1c)),'k')
	plt.plot(f/1e6, 20*np.log10(np.abs(at3_p2c)),'r')
	plt.plot(f/1e6, 20*np.log10(np.abs(g3)),'g--')
	
	
	
	plt.ylabel('3-dB Attn [dB]')
	plt.title('MAGNITUDE')
	plt.legend(['Port 1', 'Port 2', 'From DC resistance'])

	plt.subplot(3,2,2)
	plt.plot(f/1e6, (180/np.pi)*np.unwrap(np.angle(at3_p2c)) - (180/np.pi)*np.unwrap(np.angle(at3_p1c)),'r')
	plt.ylabel('3-dB Attn [degrees]')
	plt.title(r'$\Delta$ PHASE')




	plt.subplot(3,2,3)
	plt.plot(f/1e6, 20*np.log10(np.abs(at6_p1c)),'k')
	plt.plot(f/1e6, 20*np.log10(np.abs(at6_p2c)),'r')
	plt.plot(f/1e6, 20*np.log10(np.abs(g6)),'g--')
	
	plt.ylabel('6-dB Attn [dB]')

	plt.subplot(3,2,4)
	plt.plot(f/1e6, (180/np.pi)*np.unwrap(np.angle(at6_p2c)) - (180/np.pi)*np.unwrap(np.angle(at6_p1c)),'r')
	plt.ylabel('6-dB Attn [degrees]')




	plt.subplot(3,2,5)
	plt.plot(f/1e6, 20*np.log10(np.abs(at10_p1c)),'k')
	plt.plot(f/1e6, 20*np.log10(np.abs(at10_p2c)),'r')
	plt.plot(f/1e6, 20*np.log10(np.abs(g10)),'g--')
	
	plt.ylabel('10-dB Attn [dB]')
	plt.xlabel('frequency [MHz]')

	plt.subplot(3,2,6)
	plt.plot(f/1e6, (180/np.pi)*np.unwrap(np.angle(at10_p2c)) - (180/np.pi)*np.unwrap(np.angle(at10_p1c)),'r')
	plt.ylabel('10-dB Attn [degrees]')
	plt.xlabel('frequency [MHz]')
	
	



	plt.savefig(edges_folder + 'plots/20190612/vna_comparison.pdf', bbox_inches='tight')
	plt.close()	
	plt.close()
	plt.close()
	plt.close()



	return 0

	




























def plots_midband_paper(plot_number):
	
	
	if plot_number == 12:
		
		# Paths
		path_plot_save = edges_folder + 'plots/20200108/'
		
		FS_LABELS = 12
		FS_PANELS = 14
		
		
		
		
		f = plt.figure()
		plt.close()
		
		
		
		
		# Antenna S11
		# -----------		
		f, il, ih = ba.frequency_edges(50, 130)
		fe = f[il:ih+1]			

		ra = cal.models_antenna_s11_remove_delay('mid_band', fe, year=2018, day=147, case=5, delay_0=0.17, model_type='polynomial', Nfit=15, plot_fit_residuals='no')

		xlb1  = np.genfromtxt('/run/media/raul/SSD_4TB/EDGES_vol1/calibration/antenna_s11/low_band1/s11/corrected/2016_243/S11_blade_low_band_2016_243.txt')
		flb1  = xlb1[:,0]/1e6
		ralb1 = xlb1[:,1] + 1j*xlb1[:,2]
		
		xlb2  = np.genfromtxt('/run/media/raul/SSD_4TB/EDGES_vol1/calibration/antenna_s11/low_band2/s11/corrected/2017-06-29-low2-noshield_average/S11_blade_low_band_2017_180_NO_SHIELD.txt')		
		flb2  = xlb2[:,0]/1e6
		ralb2 = xlb2[:,1] + 1j*xlb2[:,2]		
		
		
		
		
		# Subplot 1
		# ---------
		plt.subplot(4,2,1)
		plt.plot(fe, 20*np.log10(np.abs(ra)), 'b',                              linewidth=1.3, label='')
		plt.plot(flb1[flb1<=100], 20*np.log10(np.abs(ralb1[flb1<=100])), 'r',   linewidth=1.3, label='')
		plt.plot(flb2[flb2<=100], 20*np.log10(np.abs(ralb2[flb2<=100])), 'r--', linewidth=1.3, label='')
		
		plt.ylabel('$|\Gamma_{\mathrm{ant}}|$ [dB]', fontsize=FS_LABELS)
		
		plt.xlim([48, 132])
		plt.ylim([-17, -1])
		plt.xticks(np.arange(50, 131, 10), '')		
		plt.yticks(np.arange(-16,-1,2))
		plt.text(122, -15.6, '(a)', fontsize=14)
		
		
		
		
		
		# Subplot 2
		# ---------
		plt.subplot(4,2,2)
		plt.plot(fe, (180/np.pi)*np.unwrap(np.angle(ra)), 'b',                              linewidth=1.3, label=r'')
		plt.plot(flb1[flb1<=100], (180/np.pi)*np.unwrap(np.angle(ralb1[flb1<=100])), 'r',   linewidth=1.3, label=r'')
		plt.plot(flb2[flb2<=100], (180/np.pi)*np.unwrap(np.angle(ralb2[flb2<=100])), 'r--', linewidth=1.3, label=r'')
	
		plt.ylabel(r'$\angle\/\Gamma_{\mathrm{ant}}$ [ $^\mathrm{o}$]', fontsize=FS_LABELS)
		plt.legend(['Mid-Band','Low-Band 1','Low-Band 2'], fontsize=9)
		
		plt.xlim([48, 132])
		plt.ylim([-700, 300])
		plt.xticks(np.arange(50, 131, 10), '')		
		plt.yticks(np.arange(-600,201,200))			
		plt.text(122, -620, '(b)', fontsize=FS_PANELS)






		# Receiver calibration parameters
		# -------------------------------
		rcv_file = edges_folder + 'mid_band/calibration/receiver_calibration/receiver1/2018_01_25C/results/nominal/calibration_files/calibration_file_receiver1_cterms7_wterms8.txt'

		rcv = np.genfromtxt(rcv_file)
	
		FLOW  = 50
		FHIGH = 130
		
		fX      = rcv[:,0]
		rcv2    = rcv[(fX>=FLOW) & (fX<=FHIGH),:]
		
		fe      = rcv2[:,0]
		rl      = rcv2[:,1] + 1j*rcv2[:,2]
		sca     = rcv2[:,3]
		off     = rcv2[:,4]
		TU      = rcv2[:,5]
		TC      = rcv2[:,6]
		TS      = rcv2[:,7]




		# Low-Band
		rcv1 = np.genfromtxt('/home/raul/DATA1/EDGES_vol1/calibration/receiver_calibration/low_band1/2015_08_25C/results/nominal/calibration_files/calibration_file_low_band_2015_nominal.txt')




		# Subplot 3
		# ---------
		plt.subplot(4,2,3)
		plt.plot(fe, 20*np.log10(np.abs(rl)), 'b', linewidth=1.3)
		plt.plot(rcv1[:,0], 20*np.log10(np.abs(rcv1[:,1]+1j*rcv1[:,2])))
		
		plt.xlim([48, 132])
		plt.ylim([-40, -30])
		plt.xticks(np.arange(50, 131, 10), '')
		plt.yticks(np.arange(-39,-30,2))
		plt.ylabel('$|\Gamma_{\mathrm{rec}}|$ [dB]', fontsize=FS_LABELS)
		plt.text(48.5, -39.55, '(c)', fontsize=FS_PANELS)
		#plt.text(105, -38.9, 'Mid-Band', fontweight='bold', color='b')
		#plt.text(105, -39.5, 'Low-Band 1', fontweight='bold', color='r')			
			
			
				
		# Subplot 4
		# ---------
		plt.subplot(4,2,4)
		plt.plot(fe, (180/np.pi)*np.unwrap(np.angle(rl)), 'b--', linewidth=1.3)
		plt.plot(rcv1[:,0], (180/np.pi)*np.unwrap(np.angle(rcv1[:,1]+1j*rcv1[:,2])), 'r--', linewidth=1.3)
		
		plt.xlim([48, 132])
		plt.ylim([70, 130])
		plt.xticks(np.arange(50, 131, 10), '')
		plt.yticks(np.arange(80,121,10))		
		plt.ylabel(r'$\angle\/\Gamma_{\mathrm{rec}}$ [ $^\mathrm{o}$]', fontsize=FS_LABELS)
		plt.text(48.5, 100, '(d)', fontsize=FS_PANELS)
		
		
		
		# Subplot 5
		# ---------
		plt.subplot(4,2,5)
		plt.plot(fe, sca,'b',linewidth=1.3)
		plt.plot(rcv1[:,0], rcv1[:,3],'r', linewidth=1.3)
		
		plt.xlim([48, 132])
		plt.xticks(np.arange(50, 131, 10), '')
		plt.ylim([3.3, 5.2])
		plt.yticks(np.arange(3.5,5.1,0.5))
		plt.ylabel('$C_1$', fontsize=FS_LABELS)
		plt.text(48.5, 3.38, '(e)', fontsize=FS_PANELS)		
		
		
		
		# Subplot 6
		# ---------
		plt.subplot(4,2,6)
		plt.plot(fe, off,'b--',linewidth=1.3)
		plt.plot(rcv1[:,0], rcv1[:,4],'r--', linewidth=1.3)
		
		plt.xlim([48, 132])
		plt.xticks(np.arange(50, 131, 10), '')		
		plt.ylim([-2.75, -0.75])
		plt.yticks(np.arange(-2.5, -0.85, 0.5))
		plt.ylabel('$C_2$ [K]', fontsize=FS_LABELS)
		plt.text(48.5, -1, '(f)', fontsize=FS_PANELS)	

		


		# Subplot 7
		# ---------
		plt.plot(fe, TU,'b', linewidth=1.3)
		plt.plot(rcv1[:,0], rcv1[:,5],'r', linewidth=1.3)
		
		plt.xlim([48, 132])
		plt.xticks(np.arange(50, 131, 10))		
		plt.ylim([178, 190])
		plt.yticks(np.arange(180, 189, 2))
		plt.ylabel('$T_{\mathrm{U}}$ [K]', fontsize=FS_LABELS)
		plt.xlabel('$\\nu$ [MHz]', fontsize=FS_LABELS)
		plt.text(48.5, 178.5, '(g)', fontsize=FS_PANELS)		
		
		
		
		
		# Subplot 8
		# ---------
		plt.plot(fe, TC,'b--', linewidth=1.3)
		plt.plot(rcv1[:,0], rcv1[:,6],'r--', linewidth=1.3)
		
		plt.plot(fe, TS,'b:', linewidth=1.3)
		plt.plot(rcv1[:,0], rcv1[:,7],'r:', linewidth=1.3)

		plt.xlim([48, 132])
		plt.xticks(np.arange(50, 131, 10))		
		plt.ylim([-55, 35])
		plt.yticks(np.arange(-40, 21, 20))
		plt.ylabel('$T_{\mathrm{C}}, T_{\mathrm{S}}$ [K]', fontsize=FS_LABELS)
		plt.xlabel('$\\nu$ [MHz]', fontsize=FS_LABELS)
		plt.text(48.5, 20, '(h)', fontsize=FS_PANELS)		
	






		plt.savefig(path_plot_save + 'calibration_parameters.pdf', bbox_inches='tight')
		plt.close()	
		plt.close()
		plt.close()
		plt.close()


		
		
		
		
		
		
		
		
		


		
		
		
		
	
	
	
	# Receiver calibration parameters
	if plot_number==1:
	

		# Paths
		path_plot_save = edges_folder + 'plots/20190917/'


		# Calibration parameters
		rcv_file = edges_folder + 'mid_band/calibration/receiver_calibration/receiver1/2018_01_25C/results/nominal/calibration_files/calibration_file_receiver1_cterms7_wterms8.txt'
		
		#mid_band/calibration/receiver_calibration/receiver1/2018_01_25C/results/nominal/calibration_files/calibration_file_receiver1_50_150MHz_cterms7_wterms7.txt'

		rcv = np.genfromtxt(rcv_file)
	
		FLOW  = 50
		FHIGH = 130
		
		fX      = rcv[:,0]
		rcv2    = rcv[(fX>=FLOW) & (fX<=FHIGH),:]
		
		fe      = rcv2[:,0]
		rl      = rcv2[:,1] + 1j*rcv2[:,2]
		sca     = rcv2[:,3]
		off     = rcv2[:,4]
		TU      = rcv2[:,5]
		TC      = rcv2[:,6]
		TS      = rcv2[:,7]




		# Low-Band
		rcv1 = np.genfromtxt('/home/raul/DATA1/EDGES_vol1/calibration/receiver_calibration/low_band1/2015_08_25C/results/nominal/calibration_files/calibration_file_low_band_2015_nominal.txt')


















		# Plot

		size_x = 4.5
		size_y = 7.5 #10.5
		x0 = 0.15
		y0 = 0.09
		dx = 0.7
		dy = 0.3
		
		
		FS_LABELS = 12
		FS_PANELS = 14


		f1  = plt.figure(num=1, figsize=(size_x, size_y))		


		ax     = f1.add_axes([x0, y0 + 2*dy, dx, dy])	
		h1     = ax.plot(fe, 20*np.log10(np.abs(rl)), 'b', linewidth=1.3, label='$|\Gamma_{\mathrm{rec}}|$')
		ax.plot(rcv1[:,0], 20*np.log10(np.abs(rcv1[:,1]+1j*rcv1[:,2])), 'r', linewidth=1.3)
		
		
		
		
		
		ax2    = ax.twinx()
		h2     = ax2.plot(fe, (180/np.pi)*np.unwrap(np.angle(rl)), 'b--', linewidth=1.3, label=r'$\angle\/\Gamma_{\mathrm{rec}}$')
		ax2.plot(rcv1[:,0], (180/np.pi)*np.unwrap(np.angle(rcv1[:,1]+1j*rcv1[:,2])), 'r--', linewidth=1.3)
		
		
		h      = h1 + h2
		labels = [l.get_label() for l in h]
		ax.legend(h, labels, loc=0, fontsize=10, ncol=2)

		ax.set_ylim([-40, -30])
		ax.set_xticklabels('')
		ax.set_yticks(np.arange(-39,-30,2))
		ax.set_ylabel('$|\Gamma_{\mathrm{rec}}|$ [dB]', fontsize=FS_LABELS)
		ax.text(48.5, -39.55, '(a)', fontsize=FS_PANELS)
		ax.text(105, -38.9, 'Mid-Band', fontweight='bold', color='b')
		ax.text(105, -39.5, 'Low-Band 1', fontweight='bold', color='r')

		ax2.set_ylim([70, 130])
		ax2.set_xticklabels('')
		ax2.set_yticks(np.arange(80,121,10))		
		ax2.set_ylabel(r'$\angle\/\Gamma_{\mathrm{rec}}$ [ $^\mathrm{o}$]', fontsize=FS_LABELS)

		ax.set_xlim([48, 132])
		ax.tick_params(axis='x', direction='in')
		ax.set_xticks(np.arange(50, 131, 10))
		
		
		
		



		ax     = f1.add_axes([x0, y0 + 1*dy, dx, dy])
		h1     = ax.plot(fe, sca,'b',linewidth=1.3, label='$C_1$')
		ax.plot(rcv1[:,0], rcv1[:,3],'r', linewidth=1.3)      #  <----------------------------- Low-Band
		ax2    = ax.twinx()
		h2     = ax2.plot(fe, off,'b--',linewidth=1.3, label='$C_2$')
		ax2.plot(rcv1[:,0], rcv1[:,4],'r--', linewidth=1.3)      #  <----------------------------- Low-Band
		h      = h1 + h2
		labels = [l.get_label() for l in h]
		ax.legend(h, labels, loc=0, fontsize=10, ncol=2)

		ax.set_ylim([3.3, 5.2])
		ax.set_xticklabels('')
		ax.set_yticks(np.arange(3.5,5.1,0.5))
		ax.set_ylabel('$C_1$', fontsize=FS_LABELS)
		ax.text(48.5, 3.38, '(b)', fontsize=FS_PANELS)

		#ax2.set_ylim([-2.4, -1.8])
		ax2.set_ylim([-2.75, -0.75])
		#ax2.set_ylim([-1.6, -1.4])
		ax2.set_xticklabels('')
		ax2.set_yticks(np.arange(-2.5, -0.85, 0.5))
		ax2.set_ylabel('$C_2$ [K]', fontsize=FS_LABELS)

		ax.set_xlim([48, 132])
		ax.tick_params(axis='x', direction='in')
		ax.set_xticks(np.arange(50, 131, 10))
		
		
		
		



		ax     = f1.add_axes([x0, y0 + 0*dy, dx, dy])
		h1     = ax.plot(fe, TU,'b', linewidth=1.3, label='$T_{\mathrm{U}}$')
		ax.plot(rcv1[:,0], rcv1[:,5],'r', linewidth=1.3)      #  <----------------------------- Low-Band
		
		ax2    = ax.twinx()
		h2     = ax2.plot(fe, TC,'b--', linewidth=1.3, label='$T_{\mathrm{C}}$')
		ax2.plot(rcv1[:,0], rcv1[:,6],'r--', linewidth=1.3)      #  <----------------------------- Low-Band
		
		h3     = ax2.plot(fe, TS,'b:', linewidth=1.3, label='$T_{\mathrm{S}}$')
		ax2.plot(rcv1[:,0], rcv1[:,7],'r:', linewidth=1.3)      #  <----------------------------- Low-Band

		h      = h1 + h2 + h3
		labels = [l.get_label() for l in h]
		ax.legend(h, labels, loc=0, fontsize=10, ncol=3)

		ax.set_ylim([178, 190])
		ax.set_yticks(np.arange(180, 189, 2))
		ax.set_ylabel('$T_{\mathrm{U}}$ [K]', fontsize=FS_LABELS)
		ax.set_xlabel('$\\nu$ [MHz]', fontsize=FS_LABELS)
		ax.text(48.5, 178.5, '(c)', fontsize=FS_PANELS)

		ax2.set_ylim([-55, 35])
		ax2.set_yticks(np.arange(-40, 21, 20))
		ax2.set_ylabel('$T_{\mathrm{C}}, T_{\mathrm{S}}$ [K]', fontsize=FS_LABELS)
		
		ax.set_xlim([48, 132])
		ax.set_xticks(np.arange(50, 131, 10))


		plt.savefig(path_plot_save + 'receiver_calibration.pdf', bbox_inches='tight')
		plt.close()	
		plt.close()
		plt.close()
		plt.close()










	
	# Antenna calibration parameters
	if plot_number==2:
	

		# Paths
		path_plot_save = edges_folder + 'plots/20190917/'




		# Plot
		
		FS_LABELS = 12

		size_x = 4.5
		size_y = 5.5 #10.5
		x0 = 0.15
		y0 = 0.09
		dx = 0.8
		dy = 0.4 #18


		f1  = plt.figure(num=1, figsize=(size_x, size_y))		










		# Frequency
		f, il, ih = ba.frequency_edges(50, 130)
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
		
		

		ax     = f1.add_axes([x0, y0 + 1*dy, dx, dy])	
		h      = ax.plot(fe, 20*np.log10(np.abs(ra)), 'b',                              linewidth=1.3, label='')
		h      = ax.plot(flb1[flb1<=100], 20*np.log10(np.abs(ralb1[flb1<=100])), 'r',   linewidth=1.3, label='')
		h      = ax.plot(flb2[flb2<=100], 20*np.log10(np.abs(ralb2[flb2<=100])), 'r--', linewidth=1.3, label='')
		
		
		
		
		#ax2    = ax.twinx()
		#h2     = ax2.plot(fe, (180/np.pi)*np.unwrap(np.angle(ra)), 'r--', linewidth=2, label=r'$\angle\/\Gamma_{\mathrm{ant}}$')
		#h      = h1 + h2
		#labels = [l.get_label() for l in h]
		ax.legend(['Mid-Band','Low-Band 1','Low-Band 2'], fontsize=9)

		#ax.set_ylim([-41, -25])
		ax.set_xticklabels('')
		#
		ax.set_ylabel('$|\Gamma_{\mathrm{ant}}|$ [dB]', fontsize=FS_LABELS)
		
		#
		#		
		#ax2.set_ylabel(r'$\angle\/\Gamma_{\mathrm{ant}}$ [ $^\mathrm{o}$]', fontsize=16)

		ax.set_xlim([48, 132])
		ax.set_ylim([-17, -1])
		ax.set_yticks(np.arange(-16,-1,2))
		#ax.set_xticklabels('')
		ax.tick_params(axis='x', direction='in')
		ax.set_xticks(np.arange(50, 131, 10))
		
		ax.text(122, -15.6, '(a)', fontsize=14)
		













		ax     = f1.add_axes([x0, y0 + 0*dy, dx, dy])	
		h      = ax.plot(fe, (180/np.pi)*np.unwrap(np.angle(ra)), 'b',                              linewidth=1.3, label=r'')
		h      = ax.plot(flb1[flb1<=100], (180/np.pi)*np.unwrap(np.angle(ralb1[flb1<=100])), 'r',   linewidth=1.3, label=r'')
		h      = ax.plot(flb2[flb2<=100], (180/np.pi)*np.unwrap(np.angle(ralb2[flb2<=100])), 'r--', linewidth=1.3, label=r'')
	
		
		#ax2    = ax.twinx()
		#h2     = ax2.plot(fe, (180/np.pi)*np.unwrap(np.angle(rl)), 'r--', linewidth=2, label=r'$\angle\/\Gamma_{\mathrm{rec}}$')
		#h      = h1 + h2
		#labels = [l.get_label() for l in h]
		#ax.legend(h, labels, loc=2, fontsize=13)

		#ax.set_ylim([-41, -25])
		#ax.set_xticklabels('')
		#ax.set_yticks(np.arange(-39,-26,3))
		#ax.set_ylabel('$|\Gamma_{\mathrm{rec}}|$ [dB]', fontsize=16)
		ax.set_ylabel(r'$\angle\/\Gamma_{\mathrm{ant}}$ [ $^\mathrm{o}$]', fontsize=FS_LABELS)
		#ax.text(42, -39.6, '(a)', fontsize=20)

		#ax2.set_ylim([70, 130])
		#ax2.set_xticklabels('')
		#ax2.set_yticks(np.arange(80,121,10))		
		#ax2.set_ylabel(r'$\angle\/\Gamma_{\mathrm{rec}}$ [ $^\mathrm{o}$]', fontsize=16)

		ax.set_xlim([48, 132])
		ax.tick_params(axis='x', direction='in')
		ax.set_xticks(np.arange(50, 131, 10))
		ax.set_ylim([-700, 300])
		ax.set_yticks(np.arange(-600,201,200))			
		
		ax.text(122, -620, '(b)', fontsize=14)
		ax.set_xlabel('$\\nu$ [MHz]', fontsize=13)




		plt.savefig(path_plot_save + 'antenna_reflection_coefficients.pdf', bbox_inches='tight')
		plt.close()	
		plt.close()
		plt.close()
		plt.close()










	# Balun loss
	if plot_number == 3:


		# Paths
		path_plot_save = edges_folder + 'plots/20190917/'




		# Plot
		
		FS_LABELS = 12

		size_x = 4.5
		size_y = 2.7 #10.5
		x0 = 0.15
		y0 = 0.09
		dx = 0.8
		dy = 0.8 #18


		f1  = plt.figure(num=1, figsize=(size_x, size_y))		






		# Frequency
		f, il, ih = ba.frequency_edges(50, 130)
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





		Gb, Gc = cal.balun_and_connector_loss('mid_band', fe, ra)
		Gbc    = Gb*Gc




		Gblb, Gclb = oeg.balun_and_connector_loss('low_band_2015', flb1, ralb1)
		Gbclb   = Gblb*Gclb
		
		Gblb2, Gclb2 = oeg.balun_and_connector_loss('low_band2_2017', flb2, ralb2)
		Gbclb2   = Gblb2*Gclb2





		ax     = f1.add_axes([x0, y0 + 0*dy, dx, dy])	
		h      = ax.plot(fe, (1-Gbc)*100, 'b',                              linewidth=1.3, label='')
		h      = ax.plot(flb1[flb1<=100], (1-Gbclb)[flb1<=100]*100,  'r',   linewidth=1.3, label='')
		h      = ax.plot(flb1[flb1<=100], (1-Gbclb2)[flb1<=100]*100, 'r--', linewidth=1.3, label='')
		#ax2    = ax.twinx()
		#h2     = ax2.plot(fe, (180/np.pi)*np.unwrap(np.angle(rl)), 'r--', linewidth=2, label=r'$\angle\/\Gamma_{\mathrm{rec}}$')
		#h      = h1 + h2
		#labels = [l.get_label() for l in h]
		#ax.legend(h, labels, loc=2, fontsize=13)

		#ax.set_ylim([-41, -25])
		#ax.set_xticklabels('')
		#ax.set_yticks(np.arange(-39,-26,3))
		#ax.set_ylabel('$|\Gamma_{\mathrm{rec}}|$ [dB]', fontsize=16)
		#ax.text(42, -39.6, '(a)', fontsize=20)
		ax.set_ylabel(r'antenna loss [%]')#, fontsize=15)

		#ax2.set_ylim([70, 130])
		#ax2.set_xticklabels('')
		#ax2.set_yticks(np.arange(80,121,10))		
		#ax2.set_ylabel(r'$\angle\/\Gamma_{\mathrm{rec}}$ [ $^\mathrm{o}$]', fontsize=16)

		ax.set_xlim([48, 132])
		ax.tick_params(axis='x', direction='in')
		ax.set_xticks(np.arange(50, 131, 10))
		ax.set_ylim([0, 1])
		ax.set_yticks(np.arange(0.2,0.9,0.2))		

		#ax.text(114, 0.07, '(c)', fontsize=14)
		
		ax.set_xlabel('$\\nu$ [MHz]', fontsize=13)
		ax.legend(['Mid-Band','Low-Band 1','Low-Band 2'], fontsize=9)



		plt.savefig(path_plot_save + 'balun_loss.pdf', bbox_inches='tight')
		plt.close()	
		plt.close()
		plt.close()
		plt.close()
























	# Antenna Beam
	if plot_number == 4:

		# Paths
		path_plot_save = edges_folder + 'plots/20190917/'


		# Plot

		size_x = 7.2
		size_y = 6 #10.5
		x0 = 0.15
		y0 = 0.09
		dx = 0.45
		dy = 0.4

		xoff = x0/1.3








		f1  = plt.figure(num=1, figsize=(size_x, size_y))		




		## Frequency
		#f, il, ih = ba.frequency_edges(50, 150)
		#fe = f[il:ih+1]	



		flow  = 50
		fhigh = 120


		bm = cal.FEKO_blade_beam('mid_band', 0, frequency_interpolation='no', frequency=np.array([0]), AZ_antenna_axis=0)
		ff = np.arange(50,201,2)
		
		fe       = ff[(ff>=flow) & (ff<=fhigh)]
		g_zenith = bm[:,90,0][(ff>=flow) & (ff<=fhigh)]
		g_45_E   = bm[:,45,0][(ff>=flow) & (ff<=fhigh)]
		g_45_H   = bm[:,45,90][(ff>=flow) & (ff<=fhigh)]
		
	
		
		bm_inf = cal.FEKO_blade_beam('mid_band', 1, frequency_interpolation='no', frequency=np.array([0]), AZ_antenna_axis=0)
		ff = np.arange(50,201,2)
		
		g_inf_zenith = bm_inf[:,90,0][(ff>=flow) & (ff<=fhigh)]
		g_inf_45_E   = bm_inf[:,45,0][(ff>=flow) & (ff<=fhigh)]
		g_inf_45_H   = bm_inf[:,45,90][(ff>=flow) & (ff<=fhigh)]
		


		bm_lb = oeg.FEKO_low_band_blade_beam(beam_file=2, frequency_interpolation='no', frequency=np.array([0]), AZ_antenna_axis=0)
		fcuec = np.arange(40,121,2)
		
		flb        = fcuec[(fcuec>=flow) & (fcuec<=fhigh)]
		glb_zenith = bm_lb[:,90,0][(fcuec>=flow) & (fcuec<=fhigh)]
		glb_45_E   = bm_lb[:,45,0][(fcuec>=flow) & (fcuec<=fhigh)]
		glb_45_H   = bm_lb[:,45,90][(fcuec>=flow) & (fcuec<=fhigh)]
		
		

		bm_10 = oeg.FEKO_low_band_blade_beam(beam_file=5, frequency_interpolation='no', frequency=np.array([0]), AZ_antenna_axis=0)
		fcuec      = np.arange(50,121,2)

		f10        = fcuec[(fcuec>=flow) & (fcuec<=fhigh)]	
		g10_zenith = bm_10[:,90,0][(fcuec>=flow) & (fcuec<=fhigh)]
		g10_45_E   = bm_10[:,45,0][(fcuec>=flow) & (fcuec<=fhigh)]
		g10_45_H   = bm_10[:,45,90][(fcuec>=flow) & (fcuec<=fhigh)]
		
		
		
		print(glb_zenith[-1])
		print(flb[-1])
		
		print(g10_zenith[-1])
		print(f10[-1])






		

		ax     = f1.add_axes([x0, y0 + 1*dy, dx, dy])
		
		h      = ax.plot(fe,  g_zenith,    'b',linewidth=1.0, label='')
		h      = ax.plot(fe,  g_inf_zenith,'b--',linewidth=0.5, label='')
		h      = ax.plot(flb, glb_zenith,  'r',linewidth=1.0, label='')
		h      = ax.plot(f10, g10_zenith,  'r--',linewidth=0.5, label='')
		
		h      = ax.plot(fe, g_zenith, 'b',linewidth=1.0, label='')
		h      = ax.plot(fe, g_45_E,   'b',linewidth=1.0, label='')
		h      = ax.plot(fe, g_45_H,   'b',linewidth=1.0, label='')
				
		h      = ax.plot(fe, g_inf_zenith, 'b--',linewidth=0.5, label='')
		h      = ax.plot(fe, g_inf_45_E,   'b--',linewidth=0.5, label='')
		h      = ax.plot(fe, g_inf_45_H,   'b--',linewidth=0.5, label='')		
		
		h      = ax.plot(flb, glb_zenith, 'r',linewidth=1.0, label='')
		h      = ax.plot(flb, glb_45_E,   'r',linewidth=1.0, label='')
		h      = ax.plot(flb, glb_45_H,   'r',linewidth=1.0, label='')
		
		h      = ax.plot(f10, g10_zenith, 'r--',linewidth=0.5, label='')
		h      = ax.plot(f10, g10_45_E,   'r--',linewidth=0.5, label='')
		h      = ax.plot(f10, g10_45_H,   'r--',linewidth=0.5, label='')		
		
		
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
		ax.set_ylabel('gain')#, fontsize=14)

		#ax2.set_ylim([-2.4, -1.8])
		#ax2.set_xticklabels('')
		#ax2.set_yticks(np.arange(-2.3, -1.85, 0.1))
		#ax2.set_ylabel('$C_2$ [K]', fontsize=16)

		ax.set_xlim([48, 122])
		ax.tick_params(axis='x', direction='in')
		ax.set_xticks(np.arange(50, 121, 10))
		ax.set_ylim([0, 9])
		ax.set_yticks(np.arange(1,8.1,1))		
		
		ax.text(115, 0.4, '(a)', fontsize=14)
		
		ax.text(50, 5.7, 'zenith', fontsize=10)
		ax.text(50, 2.9, r'$\theta=45^{\circ}$, H-plane', fontsize=10)
		ax.text(50, 0.9, r'$\theta=45^{\circ}$, E-plane', fontsize=10)
		
		
		ax.legend([r'Mid-Band 30x30 m$^2$ GP', 'Mid-Band infinite GP', r'Low-Band 30x30 m$^2$ GP', r'Low-Band 10x10 m$^2$ GP'], fontsize=7, ncol=2)
	
	
	
	
	
	
		
		Nfg   = 4
	
		
		p1 = np.polyfit(fe, g_zenith, Nfg)
		p2 = np.polyfit(fe, g_45_E, Nfg)
		p3 = np.polyfit(fe, g_45_H, Nfg)
		
		m1 = np.polyval(p1, fe)
		m2 = np.polyval(p2, fe)
		m3 = np.polyval(p3, fe)


		pi1 = np.polyfit(fe, g_inf_zenith, Nfg)
		pi2 = np.polyfit(fe, g_inf_45_E, Nfg)
		pi3 = np.polyfit(fe, g_inf_45_H, Nfg)
		
		mi1 = np.polyval(pi1, fe)
		mi2 = np.polyval(pi2, fe)
		mi3 = np.polyval(pi3, fe)




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
		

		
		
		ax     = f1.add_axes([x0, y0 + 0*dy, dx, dy])
		
		DY = 0.08
		h      = ax.plot(fe, g_zenith-m1 + DY,'b', linewidth=1.0, label='$T_{\mathrm{unc}}$')
		h      = ax.plot(fe, g_45_E-m2   - DY,'b', linewidth=1.0, label='$T_{\mathrm{unc}}$')
		h      = ax.plot(fe, g_45_H-m3   - 0.0,'b', linewidth=1.0, label='$T_{\mathrm{unc}}$')
		
		h      = ax.plot(fe, g_inf_zenith-mi1 + DY,'b--', linewidth=0.5, label='$T_{\mathrm{unc}}$')
		h      = ax.plot(fe, g_inf_45_E-mi2   - DY,'b--', linewidth=0.5, label='$T_{\mathrm{unc}}$')
		h      = ax.plot(fe, g_inf_45_H-mi3   - 0.0,'b--', linewidth=0.5, label='$T_{\mathrm{unc}}$')
		
		h      = ax.plot(flb, glb_zenith-mlb1 + DY,'r', linewidth=1.0, label='$T_{\mathrm{unc}}$')
		h      = ax.plot(flb, glb_45_E-mlb2   - DY,'r', linewidth=1.0, label='$T_{\mathrm{unc}}$')
		h      = ax.plot(flb, glb_45_H-mlb3   - 0.0,'r', linewidth=1.0, label='$T_{\mathrm{unc}}$')		

		h      = ax.plot(f10, g10_zenith-m10_1 + DY,'r--', linewidth=0.5, label='$T_{\mathrm{unc}}$')
		h      = ax.plot(f10, g10_45_E-m10_2   - DY,'r--', linewidth=0.5, label='$T_{\mathrm{unc}}$')
		h      = ax.plot(f10, g10_45_H-m10_3   - 0.0,'r--', linewidth=0.5, label='$T_{\mathrm{unc}}$')	

		
		
		#ax2    = ax.twinx()
		#h2     = ax2.plot(fe, TC,'r--', linewidth=2, label='$T_{\mathrm{cos}}$')
		#h3     = ax2.plot(fe, TS,'g--', linewidth=2, label='$T_{\mathrm{sin}}$')		

		#h      = h1 + h2 + h3
		#labels = [l.get_label() for l in h]
		#ax.legend(h, labels, loc=0, fontsize=13, ncol=3)

		#ax.set_ylim([178, 190])
		#ax.set_yticks(np.arange(180, 189, 2))
		#ax.set_ylabel('$T_{\mathrm{unc}}$ [K]', fontsize=16)
		ax.set_xlabel('$\\nu$ [MHz]')#, fontsize=15)
		#ax.text(42, 179, '(c)', fontsize=20)
		ax.set_ylabel('gain residuals\n['+ str(DY/2) + ' per division]')#, fontsize=14)

		#ax2.set_ylim([-60, 40])
		#ax2.set_yticks(np.arange(-40, 21, 20))
		
		
		ax.set_xlim([48, 122])
		xt = np.arange(50, 121, 10)
		ax.set_xticks(xt)
		#ax.set_xticklabels('')
		#ax.set_xticklabels(xt, fontsize=10)
		#ax.tick_params(axis='x', direction='in')
		yt = np.arange(-1.5*DY, 1.5*DY+0.0001, DY/2)
		ax.set_ylim([-0.135, 0.135])
		ax.set_yticks(yt)
		ax.set_yticklabels(['' for i in range(len(yt))])
		
		ax.text(115, -0.123, '(b)', fontsize=14)

		ax.text(50, 0.035, 'zenith', fontsize=10)
		ax.text(50, -0.03, r'$\theta=45^{\circ}$, H-plane', fontsize=10)
		ax.text(50, -0.12, r'$\theta=45^{\circ}$, E-plane', fontsize=10)
		




















		# ########################################################
		fmin = 50
		fmax = 120
		
		fmin_res = 50
		fmax_res = 120
		
		Nfg = 5
		
		
		
		el             = np.arange(0,91) 
		sin_theta      = np.sin((90-el)*(np.pi/180)) 
		sin_theta_2D_T = np.tile(sin_theta, (360, 1))
		sin_theta_2D   = sin_theta_2D_T.T			
		
		
			
		b_all = cal.FEKO_blade_beam('mid_band', 0, frequency_interpolation='no', AZ_antenna_axis=90)
		f     = np.arange(50,201,2)
		
		#b_all = cal.FEKO_blade_beam('low_band3', 1, frequency_interpolation='no', AZ_antenna_axis=90)
		#f     = np.arange(50, 121, 2)
		
		
		
		#b_all = oeg.FEKO_high_band_blade_beam_plus_shaped_finite_ground_plane(beam_file=20, frequency_interpolation='no', frequency=np.array([0]), AZ_antenna_axis=0)
		#f = np.arange(65, 201, 1)
		
		
		bint = np.zeros(len(f))
		for i in range(len(f)):
			
			b       = b_all[i,:,:]
			bint[i] = np.sum(b * sin_theta_2D)
		
		fr1 = f[(f>=fmin) & (f<=fmax)]
		bx1 = bint[(f>=fmin) & (f<=fmax)]
		b1  = bx1/np.mean(bx1)
		
		ff1 = fr1[(fr1>=fmin_res) & (fr1<=fmax_res)]
		bb1 = b1[(fr1>=fmin_res) & (fr1<=fmax_res)]
		
		x   = np.polyfit(ff1, bb1, Nfg-1)
		m   = np.polyval(x, ff1)
		r1  = bb1 - m
			
	
	
						
		b_all = cal.FEKO_blade_beam('mid_band', 1, frequency_interpolation='no', AZ_antenna_axis=90)
		f     = np.arange(50,201,2)
		
		bint  = np.zeros(len(f))
		for i in range(len(f)):
			
			b       = b_all[i,:,:]
			bint[i] = np.sum(b * sin_theta_2D)
		
		fr2 = f[(f>=fmin) & (f<=fmax)]
		bb  = bint[(f>=fmin) & (f<=fmax)]
		b2  = bb/np.mean(bb)
		
		ff2 = fr2[(fr2>=fmin_res) & (fr2<=fmax_res)]
		bb2 = b2[(fr2>=fmin_res) & (fr2<=fmax_res)]
		
		x   = np.polyfit(ff2, bb2, Nfg-1)
		m   = np.polyval(x, ff2)
		r2  = bb2 - m









		b_all = oeg.FEKO_low_band_blade_beam(beam_file=2, frequency_interpolation='no', AZ_antenna_axis=0)
		f     = np.arange(40,121,2)
		#f     = np.arange(40,101,2)
		
		bint = np.zeros(len(f))
		for i in range(len(f)):
			
			b       = b_all[i,:,:]
			bint[i] = np.sum(b * sin_theta_2D) 

		fr3 = f[(f>=fmin) & (f<=fmax)]
		bx3 = bint[(f>=fmin) & (f<=fmax)]
		b3  = bx3/np.mean(bx3)
		
		ff3 = fr3[(fr3>=fmin_res) & (fr3<=fmax_res)]
		bb3 = b3[(fr3>=fmin_res) & (fr3<=fmax_res)]
		
		x   = np.polyfit(ff3, bb3, 4)
		m   = np.polyval(x, ff3)
		r3  = bb3 - m

			
			
		
		
		b_all  = oeg.FEKO_low_band_blade_beam(beam_file=5, frequency_interpolation='no', frequency=np.array([0]), AZ_antenna_axis=0)
		f      = np.arange(50,121,2)
		
		bint   = np.zeros(len(f))
		for i in range(len(f)):
			
			b        = b_all[i,:,:]
			bint[i]  = np.sum(b * sin_theta_2D) 
	
		fr4 = f[(f>=fmin) & (f<=fmax)]
		bb  = bint[(f>=fmin) & (f<=fmax)]
		b4  = bb/np.mean(bb)
		
		ff4 = fr4[(fr4>=fmin_res) & (fr4<=fmax_res)]
		bb4 = b4[(fr4>=fmin_res) & (fr4<=fmax_res)]
		
		x   = np.polyfit(ff4, bb4, 4)
		m   = np.polyval(x, ff4)
		r4  = bb4 - m		

		
		
		
	

	
	
		
		
		#plt.close()
		ax     = f1.add_axes([x0 + 1*dx + xoff, y0 + 1*dy, dx, dy])
		
		h = ax.plot(fr1, b1, 'b',   linewidth=1.0)
		h = ax.plot(fr2, b2, 'b--', linewidth=0.5)
		h = ax.plot(fr3, b3, 'r',   linewidth=1.0)
		h = ax.plot(fr4, b4, 'r--', linewidth=0.5)
		ax.set_ylabel('normalized integrated gain')
		#plt.legend(['Mid-Band 30mx30m ground plane','Mid-Band infinite ground plane','Low-Band 30mx30m ground plane',''], fontsize=7)
		
		
		
		ax.set_xlim([48, 132])
		xt = np.arange(50, 131, 10)
		ax.set_xticks(xt)
		#ax.set_xticklabels('')
		ax.tick_params(axis='x', direction='in')
		ax.set_xticklabels(['' for i in range(len(xt))])
		
		ax.set_ylim([0.980, 1.015])
		ax.set_yticks(np.arange(0.985,1.011,0.005))
	
		#ax.text(115, 0.118, '(c)', fontsize=14)		
		
		
		#plt.subplot(2,1,2)
		#plt.plot(fx, smx - mm)
		#plt.plot(fx, smix - mmi, ':')
		#plt.plot(fx, slx - ml, '--')
		#plt.ylabel('residuals to 5-term polynomial\n [fraction of 4pi]')
		#plt.xlabel('frequency [MHz]')

		# ###################################################################################







		ax     = f1.add_axes([x0 + 1*dx + xoff, y0 + 0*dy, dx, dy])	
	
		h = ax.plot(ff1, r1,        'b',   linewidth=1.0)
		h = ax.plot(ff2, r2-0.0005, 'b--', linewidth=0.5)
		h = ax.plot(ff3, r3-0.001,  'r',   linewidth=1.0)
		h = ax.plot(ff4, r4-0.0015, 'r--', linewidth=0.5)
		
		#ax2    = ax.twinx()
		#h2     = ax2.plot(fe, (180/np.pi)*np.unwrap(np.angle(rl)), 'r--', linewidth=2, label=r'$\angle\/\Gamma_{\mathrm{rec}}$')
		#h      = h1 + h2
		#labels = [l.get_label() for l in h]
		#ax.legend(h, labels, loc=2, fontsize=13)
	

		#xt = np.arange(50, 151, 20)
		#ax.set_xticks(xt)		
	
		ax.set_ylim([-0.002, 0.0005])
		#ax2.set_xticklabels('')
		#ax2.set_yticks(np.arange(80,121,10))		
		#ax2.set_ylabel(r'$\angle\/\Gamma_{\mathrm{rec}}$ [ $^\mathrm{o}$]', fontsize=16)


		ax.set_xlabel('$\\nu$ [MHz]')#, fontsize=15)
		ax.set_ylabel('normalized integrated gain residuals\n [0.05% per division]')#, fontsize=14)
	
		ax.set_xlim([48, 132])
		#ax.tick_params(axis='x', direction='in')
		ax.set_xticks(np.arange(50, 131, 10))
			
		#ax.set_ylim([-0.175, 0.175])
		#yt = np.arange(-0.15,0.16,0.05)
		#ax.set_yticks(yt)
		ax.set_yticklabels('') #['' for i in range(len(yt))])


		#ax.text(115, 0.12, '(d)', fontsize=14)
		
		
		
		
		
		
		
		
		
		





		plt.savefig(path_plot_save + 'beam_gain.pdf', bbox_inches='tight')
		plt.close()	
		plt.close()
		plt.close()
		plt.close()







		f2 = plt.figure()
		#loss_K = 300*(1-((np.pi/180)**2)*bx1/(4*np.pi))
		sky_K = 1*((np.pi/180)**2)*bx3/(4*np.pi)
		par   = np.polyfit(fr3, sky_K, 4)
		model = np.polyval(par, fr3)
		
		plt.subplot(2,1,1)
		plt.plot(fr3, sky_K)
		plt.subplot(2,1,2)
		plt.plot(fr3, sky_K - model)
		plt.savefig(path_plot_save + 'test.pdf', bbox_inches='tight')
		plt.close()			


























	
	if plot_number == 50:
	
		ax     = f1.add_axes([x0 + 1*dx + xoff, y0 + 1*dy, dx, dy])
	
	
		Gg   = cal.ground_loss('mid_band', fe)
		flb  = np.arange(50, 121, 1)
		Gglb = cal.ground_loss('low_band', flb)
	
		h      = ax.plot(fe, (1-Gg)*100, 'b', linewidth=1.0, label='ground loss [%]')
		h      = ax.plot(flb, (1-Gglb)*100, 'r', linewidth=1.0, label='ground loss [%]')
	
		#ax.set_xlabel('$\\nu$ [MHz]')#, fontsize=15)
		#ax.set_ylim([-41, -25])
		#ax.set_xticklabels('')
		#ax.set_yticks(np.arange(-39,-26,3))
		#ax.set_ylabel('$|\Gamma_{\mathrm{rec}}|$ [dB]', fontsize=16)
		#ax.text(42, -39.6, '(a)', fontsize=20)
		ax.set_ylabel(r'ground loss [%]')#, fontsize=15)
	
	
	
		ax.set_xlim([48, 122])
		xt = np.arange(50, 121, 10)
		ax.set_xticks(xt)
		ax.set_xticklabels('')
		ax.tick_params(axis='x', direction='in')
		#ax.set_xticklabels(['' for i in range(len(xt))], fontsize=15)
		#ax.set_xticklabels(xt, fontsize=10)
		ax.set_ylim([0.1, 0.5])
		ax.set_yticks(np.arange(0.15,0.46,0.05))
	
		ax.text(115, 0.118, '(c)', fontsize=14)
	




























	



	if plot_number == 5:
		
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




		beam_factor_filename = 'table_lores_mid_band_50-200MHz_90deg_alan0_haslam_gaussian_index_2.4_2.65_sigma_deg_8.5_reffreq_90MHz.hdf5'
		f, lst, bf           = cal.beam_factor_table_read('/home/raul/DATA2/EDGES_vol2/mid_band/calibration/beam_factors/table/' + beam_factor_filename)
		gha = lst + (24-17.76)
		gha[gha>=24] = gha[gha>=24] - 24
		IX = np.argsort(gha)
		bx1 = bf[IX]

		print(gha[IX][175])

		plt.close()
		plt.close()
		plt.close()
		f1   = plt.figure(num=1, figsize=(size_x, size_y))		




		ax     = f1.add_axes([x0, y0 + 1*(yoff+dy1), dx, dy])
		im = ax.imshow(bx1, interpolation='none', extent=[50, 200, 24, 0], aspect='auto', vmin=0.979, vmax=1.021)#, cmap='jet')
	
		ax.plot(f, 6*np.ones(len(f)), 'w--', linewidth=1.5)
		ax.plot(f, 18*np.ones(len(f)), 'w--', linewidth=1.5)
	
	
		ax.set_xlim([60, 120])
		ax.set_xticklabels('')
		ax.set_yticks(np.arange(0,25,3))
		ax.set_ylabel('GHA [hr]')
		
		cax    = f1.add_axes([x0 + 1*dx + xoffc, y0 + 1*(yoff+dy1), dxc, dy])
		f1.colorbar(im, cax=cax, orientation='vertical', ticks=[0.98, 0.99, 1, 1.01, 1.02])
		cax.set_title('$C$')
		
		#ax.text(panel_letter_x, panel_letter_y,  '(a)', fontsize=18)
		

		
		
		ax     = f1.add_axes([x0, y0, dx, dy1])
		ax.plot(f, bx1[125,:], 'k')
		ax.plot(f, bx1[175,:], 'k--')
		#ax.plot(f, bx1[225,:], 'k:')
		ax.legend(['GHA=10 hr','GHA=14 hr'], fontsize=8, ncol=2)
		
		ax.set_ylim([0.9, 1.1])
	
		#cax    = f1.add_axes([x0 + 1*dx + xoffc, y0 + 0*(yoff+dy), dxc, dy])
		#f1.colorbar(im, cax=cax, orientation='vertical')
	
		ax.set_xlim([60, 120])
		ax.set_ylim([0.975, 1.025])
		ax.set_xlabel(r'$\nu$ [MHz]')
		ax.set_yticks([0.98,1,1.02])
		ax.set_ylabel('$C$')
		#ax.text(panel_letter_x, panel_letter_y,  '(e)', fontsize=18)		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		# Saving plot
		path_plot_save = edges_folder + 'plots/20190729/'
		

		plt.savefig(path_plot_save + 'beam_chromaticity_correction.pdf', bbox_inches='tight')
		plt.close()	
		plt.close()
		plt.close()
		plt.close()









	if plot_number == 6:
		

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


		beam_factor_filename = 'table_lores_mid_band_50-200MHz_90deg_alan0_haslam_gaussian_index_2.4_2.65_sigma_deg_8.5_reffreq_90MHz.hdf5'				
		f, lst, bf           = cal.beam_factor_table_read('/home/raul/DATA2/EDGES_vol2/mid_band/calibration/beam_factors/table/' + beam_factor_filename)
		gha = lst + (24-17.76)
		gha[gha>=24] = gha[gha>=24] - 24
		IX = np.argsort(gha)
		bx1 = bf[IX]




		beam_factor_filename = 'table_lores_mid_band_50-200MHz_90deg_alan1_haslam_gaussian_index_2.4_2.65_sigma_deg_8.5_reffreq_90MHz.hdf5'
		f, lst, bf           = cal.beam_factor_table_read('/home/raul/DATA2/EDGES_vol2/mid_band/calibration/beam_factors/table/' + beam_factor_filename)
		bx2 = bf[IX]

		beam_factor_filename = 'table_lores_mid_band_50-200MHz_90deg_alan0_haslam_flat_index_2.56_reffreq_90MHz.hdf5'
		f, lst, bf           = cal.beam_factor_table_read('/home/raul/DATA2/EDGES_vol2/mid_band/calibration/beam_factors/table/' + beam_factor_filename)
		bx3 = bf[IX]

		beam_factor_filename = 'table_lores_mid_band_50-200MHz_90deg_alan0_LW_gaussian_index_2.4_2.65_sigma_deg_8.5_reffreq_90MHz.hdf5'
		f, lst, bf           = cal.beam_factor_table_read('/home/raul/DATA2/EDGES_vol2/mid_band/calibration/beam_factors/table/' + beam_factor_filename)
		bx4 = bf[IX]
		
		beam_factor_filename = 'table_lores_mid_band_50-200MHz_90deg_alan0_guzman_gaussian_index_2.4_2.65_sigma_deg_8.5_reffreq_90MHz.hdf5'
		f, lst, bf           = cal.beam_factor_table_read('/home/raul/DATA2/EDGES_vol2/mid_band/calibration/beam_factors/table/' + beam_factor_filename)
		bx5 = bf[IX]



		plt.close()
		plt.close()
		plt.close()
		f1   = plt.figure(num=1, figsize=(size_x, size_y))		









		scale_max = 0.0043
		scale_min = -0.0043

		cmap = plt.cm.viridis
		rgba = cmap(0.0)
		cmap.set_under(rgba)
		
		ax     = f1.add_axes([x0 + 0*(xoff+dx), y0, dx, dy])
		im = ax.imshow(bx2-bx1, interpolation='none', extent=[50, 200, 24, 0], aspect='auto', vmin=scale_min, vmax=scale_max, cmap=cmap)#, cmap='jet')
	
		#cax    = f1.add_axes([x0 + 1*dx + xoffc, y0 + 1.5*(yoff+dy), dxc, dy])
		#f1.colorbar(im, cax=cax, orientation='vertical', ticks=[-0.004, -0.002, 0, 0.002, 0.004])
		
		plt.plot([60,120], [6,6], 'w--', linewidth=2)
		plt.plot([60,120], [18,18], 'w--', linewidth=2)
	
		ax.set_xlim([60, 120])
		#ax.set_xticklabels('')
		ax.set_xticks(np.arange(60,121,10))
		ax.set_yticks(np.arange(0,25,3))
		ax.set_xlabel(r'$\nu$ [MHz]', fontsize=14)
		ax.set_ylabel('GHA [hr]', fontsize=14)
		#ax.set_text(panel_letter_x, panel_letter_y,  '(b)', fontsize=18)
		ax.set_title('(a)', fontsize=18)
	
	
	
	
	
		ax     = f1.add_axes([x0 + 1*(xoff+dx), y0, dx, dy])
		im = ax.imshow(bx3-bx1, interpolation='none', extent=[50, 200, 24, 0], aspect='auto', vmin=scale_min, vmax=scale_max, cmap=cmap)#, cmap='jet')
	
		#cax    = f1.add_axes([x0 + 1*dx + xoffc, y0 + 2*(yoff+dy), dxc, dy])
		#f1.colorbar(im, cax=cax, orientation='vertical')

		plt.plot([60,120], [6,6], 'w--', linewidth=2)
		plt.plot([60,120], [18,18], 'w--', linewidth=2)	
	
	
		ax.set_xlim([60, 120])
		ax.set_yticklabels('')
		ax.set_xlabel(r'$\nu$ [MHz]', fontsize=14)
		ax.set_xticks(np.arange(60,121,10))
		ax.set_yticks(np.arange(0,25,3))
		#ax.set_ylabel('GHA [hr]')
		#ax.text(panel_letter_x, panel_letter_y,  '(c)', fontsize=18)
		ax.set_title('(b)', fontsize=18)
	
	
	
	
	
		ax     = f1.add_axes([x0 + 2*(xoff+dx), y0, dx, dy])
		im = ax.imshow(bx4-bx1, interpolation='none', extent=[50, 200, 24, 0], aspect='auto', vmin=scale_min, vmax=scale_max, cmap=cmap)#, cmap='jet')
	
		#cax    = f1.add_axes([x0 + 1*dx + xoffc, y0 + 1*(yoff+dy), dxc, dy])
		#f1.colorbar(im, cax=cax, orientation='vertical')
	
		plt.plot([60,120], [6,6], 'w--', linewidth=2)
		plt.plot([60,120], [18,18], 'w--', linewidth=2)	
	
	
		ax.set_xlim([60, 120])
		ax.set_yticklabels('')
		ax.set_xlabel(r'$\nu$ [MHz]', fontsize=14)
		ax.set_xticks(np.arange(60,121,10))
		ax.set_yticks(np.arange(0,25,3))
		#ax.set_ylabel('GHA [hr]')
		#ax.text(panel_letter_x, panel_letter_y,  '(d)', fontsize=18)
		ax.set_title('(c)', fontsize=18)
	
	
	
	
	
	
		ax     = f1.add_axes([x0 + 3*(xoff+dx), y0, dx, dy])
		im = ax.imshow(bx5-bx1, interpolation='none', extent=[50, 200, 24, 0], aspect='auto', vmin=scale_min, vmax=scale_max, cmap=cmap)#, cmap='jet')

		plt.plot([60,120], [6,6], 'w--', linewidth=2)
		plt.plot([60,120], [18,18], 'w--', linewidth=2)

	
		cax    = f1.add_axes([x0 + 3.2*xoff+4*dx, y0, dxc, dy]) #  + xoffc
		f1.colorbar(im, cax=cax, orientation='vertical')
		cax.set_title(r'$\Delta C$', fontsize=14)
	
		ax.set_xlim([60, 120])
		ax.set_yticklabels('')
		ax.set_xlabel(r'$\nu$ [MHz]', fontsize=14)
		ax.set_xticks(np.arange(60,121,10))
		ax.set_yticks(np.arange(0,25,3))
		#ax.set_ylabel('GHA [hr]')
		#ax.text(panel_letter_x, panel_letter_y,  '(e)', fontsize=18)
		ax.set_title('(d)', fontsize=18)
	

		
		# Saving plot
		path_plot_save = edges_folder + '/plots/20190729/'
		

		plt.savefig(path_plot_save + 'beam_chromaticity_differences.pdf', bbox_inches='tight')
		plt.close()	
		plt.close()
		plt.close()
		plt.close()


























			
	
	if plot_number == 7:
		
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









	if plot_number == 8:
		
		
		FLOW  = 58
		FHIGH = 120
		case  = 0
		vr = 90
		
		path_plot_save   = edges_folder + 'plots/20190815/'
		figure_plot_save = 'powerlog_5par_exp_4par_v1.pdf'
		
		fg = 'powerlog5'
		signal = 'exp4'
		
		
		
		
		
		#d = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case_nominal/integrated_spectrum_case_nominal_days_186_219_58-120MHz_v2.txt')
		d = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case_nominal/integrated_spectrum_case_nominal_days_186_219_58-120MHz.txt')
		
		#d = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case_nominal/integrated_spectrum_case_nominal_days_186_219_60-120MHz.txt')
		#d = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case2/integrated_spectrum_case2_days_186_219_60-120MHz.txt')
		v  = d[:,0]
		t  = d[:,1]
		w  = d[:,2]
		s  = d[:,3]
	
		t[w==0] = np.nan
		s[w==0] = np.nan		



		# Best-fit foreground model
		if fg == 'linlog5':
			#filename_foreground = edges_folder + 'mid_band/polychord/20190811/case_nominal/foreground_linlog_5par/chain.txt'
			#label_foreground    = [r'a_0', r'a_1', r'a_2', r'a_3', r'a_4']
			#getdist_samples, ww, ll, best_fit, covariance_matrix = gp.load_samples(filename_foreground, 0, label_names=label_foreground)
			#model_fg = dm.foreground_model('linlog', best_fit, v, vr, ion_abs_coeff=0, ion_emi_coeff=0)
				
			filename_foreground = edges_folder + 'mid_band/polychord/20190815/case_nominal/foreground_linlog_5par_v2/chain.txt'
			label_foreground    = [r'a_0', r'a_1', r'a_2', r'a_3', r'a_4']
			getdist_samples, ww, ll, best_fit, covariance_matrix = gp.load_samples(filename_foreground, 0, label_names=label_foreground)
			model_fg = dm.foreground_model('linlog', best_fit, v, vr, ion_abs_coeff=0, ion_emi_coeff=0)
			
			
			
			
		if fg == 'powerlog5':
			#filename_foreground = edges_folder + 'mid_band/polychord/20190811/case_nominal/foreground_powerlog_5par/chain.txt'
			#label_foreground    = [r'T_{90}\;[{\rm K}]', r'\beta', r'\gamma', r'\delta', r'\epsilon']
			#getdist_samples, ww, ll, best_fit, covariance_matrix = gp.load_samples(filename_foreground, 0, label_names=label_foreground)
			#model_fg = dm.foreground_model('powerlog', best_fit, v, vr, ion_abs_coeff=0, ion_emi_coeff=0)
			
			filename_foreground = edges_folder + 'mid_band/polychord/20190815/case_nominal/foreground_powerlog_5par/chain.txt'
			label_foreground    = [r'T_{90}\;[{\rm K}]', r'\beta', r'\gamma', r'\delta', r'\epsilon']
			getdist_samples, ww, ll, best_fit, covariance_matrix = gp.load_samples(filename_foreground, 0, label_names=label_foreground)
			model_fg = dm.foreground_model('powerlog', best_fit, v, vr, ion_abs_coeff=0, ion_emi_coeff=0)
			
			#filename_foreground = edges_folder + 'mid_band/polychord/20190815/case_nominal/foreground_powerlog_5par_v2/chain.txt'
			#label_foreground    = [r'T_{90}\;[{\rm K}]', r'\beta', r'\gamma', r'\delta', r'\epsilon']
			#getdist_samples, ww, ll, best_fit, covariance_matrix = gp.load_samples(filename_foreground, 0, label_names=label_foreground)
			#model_fg = dm.foreground_model('powerlog', best_fit, v, vr, ion_abs_coeff=0, ion_emi_coeff=0)			
		
		
		
		# Best-fit foreground model + signal model
		full = fg + '_' + signal
		if full == 'linlog5_exp4':
			#filename_foreground_plus_signal  = edges_folder + 'mid_band/polychord/20190811/case_nominal/foreground_linlog_5par_signal_exp_4par/chain.txt'
			#label_foreground_plus_signal     = [r'A\;[{\rm K}]', r'\nu_0\;[{\rm MHz}]', r'w\;[{\rm MHz}]', r'\tau', r'a_0', r'a_1', r'a_2', r'a_3', r'a_4']
			#getdist_samples, ww, ll, best_fit, covariance_matrix = gp.load_samples(filename_foreground_plus_signal, 0, label_names=label_foreground_plus_signal)
			#full_model = dm.full_model(best_fit, v, vr, model_type_signal='exp', model_type_foreground='linlog', N21par=4, NFGpar=5)
			
			filename_foreground_plus_signal  = edges_folder + 'mid_band/polychord/20190815/case_nominal/foreground_linlog_5par_signal_exp_4par_v2/chain.txt'
			label_foreground_plus_signal     = [r'A\;[{\rm K}]', r'\nu_0\;[{\rm MHz}]', r'w\;[{\rm MHz}]', r'\tau', r'a_0', r'a_1', r'a_2', r'a_3', r'a_4']
			getdist_samples, ww, ll, best_fit, covariance_matrix = gp.load_samples(filename_foreground_plus_signal, 0, label_names=label_foreground_plus_signal)
			full_model = dm.full_model(best_fit, v, vr, model_type_signal='exp', model_type_foreground='linlog', N21par=4, NFGpar=5)			

		
		if full == 'powerlog5_exp4':
			#filename_foreground_plus_signal  = edges_folder + 'mid_band/polychord/20190811/case_nominal/foreground_powerlog_5par_signal_exp_4par/chain.txt'
			#label_foreground_plus_signal     = [r'A\;[{\rm K}]', r'\nu_0\;[{\rm MHz}]', r'w\;[{\rm MHz}]', r'\tau', r'T_{90}\;[{\rm K}]', r'\beta', r'\gamma', r'\delta', r'\epsilon']
			#getdist_samples, ww, ll, best_fit, covariance_matrix = gp.load_samples(filename_foreground_plus_signal, 0, label_names=label_foreground_plus_signal)
			#full_model = dm.full_model(best_fit, v, vr, model_type_signal='exp', model_type_foreground='powerlog', N21par=4, NFGpar=5)
			
			filename_foreground_plus_signal  = edges_folder + 'mid_band/polychord/20190815/case_nominal/foreground_powerlog_5par_signal_exp_4par/chain.txt'
			label_foreground_plus_signal     = [r'A\;[{\rm K}]', r'\nu_0\;[{\rm MHz}]', r'w\;[{\rm MHz}]', r'\tau', r'T_{90}\;[{\rm K}]', r'\beta', r'\gamma', r'\delta', r'\epsilon']
			getdist_samples, ww, ll, best_fit, covariance_matrix = gp.load_samples(filename_foreground_plus_signal, 0, label_names=label_foreground_plus_signal)
			full_model = dm.full_model(best_fit, v, vr, model_type_signal='exp', model_type_foreground='powerlog', N21par=4, NFGpar=5)			
			
			#filename_foreground_plus_signal  = edges_folder + 'mid_band/polychord/20190815/case_nominal/foreground_powerlog_5par_signal_exp_4par_v2/chain.txt'
			#label_foreground_plus_signal     = [r'A\;[{\rm K}]', r'\nu_0\;[{\rm MHz}]', r'w\;[{\rm MHz}]', r'\tau', r'T_{90}\;[{\rm K}]', r'\beta', r'\gamma', r'\delta', r'\epsilon']
			#getdist_samples, ww, ll, best_fit, covariance_matrix = gp.load_samples(filename_foreground_plus_signal, 0, label_names=label_foreground_plus_signal)
			#full_model = dm.full_model(best_fit, v, vr, model_type_signal='exp', model_type_foreground='powerlog', N21par=4, NFGpar=5)		
	
		
		if full == 'powerlog5_tanh5':
			filename_foreground_plus_signal  = edges_folder + 'mid_band/polychord/20190811/case_nominal/foreground_powerlog_5par_signal_tanh_5par/chain.txt'
			label_foreground_plus_signal     = [r'A\;[{\rm K}]', r'\nu_0\;[{\rm MHz}]', r'w\;[{\rm MHz}]', r'\tau_1', r'\tau_2', r'T_{90}\;[{\rm K}]', r'\beta', r'\gamma', r'\delta', r'\epsilon']
			getdist_samples, ww, ll, best_fit, covariance_matrix = gp.load_samples(filename_foreground_plus_signal, 0, label_names=label_foreground_plus_signal)
			full_model = dm.full_model(best_fit, v, vr, model_type_signal='tanh', model_type_foreground='powerlog', N21par=5, NFGpar=5)



		
		# Best-fit signal model
		if signal == 'exp4':
			model_signal = dm.signal_model('exp', best_fit[0:4], v)	
			
		if signal == 'tanh5':
			model_signal = dm.signal_model('tanh', best_fit[0:5], v)	
	
		
		
		# Data
		#v, t, w, sigma, inv_sigma, det_sigma = dm.real_data(case, FLOW, FHIGH, gap_FLOW=0, gap_FHIGH=0)
	
		#filename = edges_folder + 'mid_band/polychord/20190810/case2/foreground_linlog_5par/chain.txt'
		
		#getdist_samples, ww, ll, best_fit, covariance_matrix = gp.load_samples(filename, 0, label_names=
				
		#filename = edges_folder + 'mid_band/polychord/20190810/case2/foreground_linlog_5par_exp_4par/chain.txt'
		
		#label_names=[r'A\;[{\rm K}]', r'\nu_0\;[{\rm MHz}]', r'w\;[{\rm MHz}]', r'\tau', r'T_{100}\;[{\rm K}]', r'\beta', r'\gamma', r'\delta', r'\epsilon']
		
		#foreground_model('exp', best_fit, v, vr, ion_abs_coeff=0, ion_emi_coeff=0)
		
		
		
		
		
		
		
		
		
		
		
		# EDGES Bowman et al. (2018)
		#model_edges2018      = dm.signal_model('exp', [-0.5, 78, 19, 7], v)
		#model_edges2018_A1 = dm.signal_model('exp', [-1, 78, 19, 7], v)
		#model_edges2018_A2 = dm.signal_model('exp', [-0.3, 78, 19, 7], v)
		
		model_edges2018, x2, limits = ba.signal_edges2018_uncertainties(v)
		model_edges2018_A1 = limits[:,0]
		model_edges2018_A2 = limits[:,1]
		
		
		
		


		
		
		x0b = 0.35
		y0b = 0.525
		dx  = 0.4
		dyb = 0.1		
		
		
		x0a = 0.35
		y0a = 0.625
		dx  = 0.4
		dya = 0.2
		

		x0c = 0.1
		y0c = 0.37
		dx  = 0.4
		dyc = 0.11
		
		
		x0d = 0.58
		y0d = 0.37
		dx  = 0.4
		dyd = 0.11
		
			
		x0e = 0.1
		y0e = 0.1
		dx  = 0.4
		dye = 0.25
		
		
		x0f = 0.58
		y0f = 0.1	
		dx  = 0.4
		dyf = 0.25
		
		
		f1 = plt.figure(1, figsize=[8,10])
		


		
		
		# Panel a
		# ---------------------------------
		ax = f1.add_axes([x0a, y0a, dx, dya])
		ax.plot(v, t, 'b', linewidth=1)
	
		ax.set_xlim([FLOW, FHIGH])
		ax.set_ylim([250, 3500])
		
		#ax.set_xticklabels('')
		ax.set_xticklabels([])
		
		ax.set_yticks(np.arange(500,3001,500))
		ax.set_yticklabels(['500','1000','1500','2000', '2500', '3000'])
		
		#ax.set_xlabel(r'$\nu$ [MHz]')
		ax.set_ylabel('T$_b$ [K]', fontsize=13)
		#ax.set_text(panel_letter_x, panel_letter_y,  '(b)', fontsize=18)
		ax.text(114,2900,'(a)', fontsize=14)#, fontweight='bold')
		#ax.grid()
	
	
		# Panel b
		# ---------------------------------
		ax = f1.add_axes([x0b, y0b, dx, dyb])
		ax.plot(v,  s, 'b', linewidth=1)
		ax.plot(v, -s, 'b', linewidth=1)
		ax.plot(v, np.zeros(len(v)), 'k--')
		ax.set_xlim([FLOW, FHIGH])
		ax.set_ylim([-0.07, 0.07])
		#ax.set_xticklabels([])
		ax.set_xticks(np.arange(60,121,10))
		ax.set_ylabel('$\sigma_b$ [K]', fontsize=13)
		ax.text(114,0.03,'(b)', fontsize=14)#, fontweight='bold')
	
	
	
	
	
	
	
	
	
	

		# Panel c
		# ---------------------------------
		ax = f1.add_axes([x0c, y0c, dx, dyc])
		ax.plot(v, t-model_fg, 'b', linewidth=1)

		ax.set_xlim([FLOW, FHIGH])
		ax.set_ylim([-0.3, 0.3])
		
		ax.set_xticks(np.arange(60,121,10))
		ax.set_xticklabels([])
		
		ax.set_yticks(np.arange(-0.2,0.21,0.2))
		#ax.set_yticklabels(['-0.6','-0.4','-0.2','0','0.2','0.4','0.6'])
		ax.set_yticklabels(['-0.2','0','0.2'])
		
		#ax.set_xlabel(r'$\nu$ [MHz]')
		ax.set_ylabel('T$_b$ [K]', fontsize=13)
		#ax.set_text(panel_letter_x, panel_letter_y,  '(b)', fontsize=18)
		ax.text(114,0.170,'(c)', fontsize=14)#, fontweight='bold')
		#ax.grid()



		# Panel d
		# ---------------------------------
		ax = f1.add_axes([x0d, y0d, dx, dyd])
		ax.plot(v, t-full_model, 'b', linewidth=1)

		ax.set_xlim([FLOW, FHIGH])
		ax.set_ylim([-0.3, 0.3])
		
		ax.set_xticks(np.arange(60,121,10))
		ax.set_xticklabels([])
		
		ax.set_yticks(np.arange(-0.2,0.21,0.2))
		#ax.set_yticklabels(['-0.6','-0.4','-0.2','0','0.2','0.4','0.6'])
		ax.set_yticklabels(['-0.2','0','0.2'])
		
		#ax.set_xlabel(r'$\nu$ [MHz]')
		#ax.set_ylabel('T$_b$ [K]')
		#ax.set_text(panel_letter_x, panel_letter_y,  '(b)', fontsize=18)
		ax.text(114,0.170,'(d)', fontsize=14)#, fontweight='bold')
		#ax.grid()


		# Panel e
		# ---------------------------------
		ax = f1.add_axes([x0e, y0e, dx, dye])
		ax.plot(v, model_signal, 'b', linewidth=1)
		ax.plot(v, model_edges2018, 'r', linewidth=0.5)
		ax.plot(v, model_edges2018_A1, 'r--', linewidth=0.5)
		ax.plot(v, model_edges2018_A2, 'r--', linewidth=0.5)
		ax.plot(v, model_signal, 'b', linewidth=1)
		
		

		ax.set_xlim([FLOW, FHIGH])
		ax.set_ylim([-1.3, 0.1])
		
		ax.set_xticks(np.arange(60,121,10))
		
		ax.set_yticks(np.arange(-1.2,0.1,0.2))
		#ax.set_yticklabels(['-0.6','-0.4','-0.2','0'])
		
		ax.set_xlabel(r'$\nu$ [MHz]', fontsize=13)
		ax.set_ylabel('T$_b$ [K]', fontsize=13)
		#ax.set_text(panel_letter_x, panel_letter_y,  '(b)', fontsize=18)
		ax.text(114,-0.25,'(e)', fontsize=14)#, fontweight='bold')
		#ax.grid()


		# Panel f
		# ---------------------------------
		ax = f1.add_axes([x0f, y0f, dx, dyf])
		ax.plot(v, model_signal + (t-full_model), 'b', linewidth=1)
		ax.plot(v, model_edges2018, 'r', linewidth=0.5)
		ax.plot(v, model_edges2018_A1, 'r--', linewidth=0.5)
		ax.plot(v, model_edges2018_A2, 'r--', linewidth=0.5)
		ax.plot(v, model_signal + (t-full_model), 'b', linewidth=1)	

		ax.set_xlim([FLOW, FHIGH])
		ax.set_ylim([-1.3, 0.1])
		
		ax.set_xticks(np.arange(60,121,10))
		
		ax.set_yticks(np.arange(-1.2,0.1,0.2))
		#ax.set_yticklabels(['-0.6','-0.4','-0.2','0'])
		
		ax.set_xlabel(r'$\nu$ [MHz]', fontsize=13)
		#ax.set_ylabel('T$_b$ [K]')
		#ax.set_text(panel_letter_x, panel_letter_y,  '(b)', fontsize=18)
		ax.text(114,-0.25,'(f)', fontsize=14)#, fontweight='bold')
		#ax.grid()
		ax.legend(['Mid-Band', 'Bowman et al. (2018)'], fontsize=9, loc=4)


		# Saving plot
		plt.savefig(path_plot_save + figure_plot_save, bbox_inches='tight')
		plt.close()	
		plt.close()
		plt.close()
		plt.close()
		
		


















	if plot_number == 9:
		
		#filename = edges_folder + 'mid_band/polychord/20190605/case2/foreground_exp_signal_tanh/chain.txt'
		#filename = edges_folder + 'mid_band/polychord/20190605/case26/foreground_exp_signal_tanh/chain.txt'
		#filename = edges_folder + 'mid_band/polychord/20190612/case2/foreground_exp_5par_signal_exp_4par/chain.txt'
		#filename = edges_folder + 'mid_band/polychord/20190612/case2/foreground_exp_5par_signal_exp_5par/chain.txt'
		#filename = edges_folder + 'mid_band/polychord/20190612/case2/foreground_exp_5par_signal_tanh_60_115MHz/chain.txt'
		filename = edges_folder + 'mid_band/polychord/20190617/case2/foreground_exp_signal_exp_4par_60_120MHz/chain.txt'
		
		#label_names=[r'A\;[{\rm K}]', r'\nu_0\;[{\rm MHz}]', r'w\;[{\rm MHz}]', r'\tau_1', r'\tau_2', r'T_{100}\;[{\rm K}]', r'\beta', r'\gamma', r'\delta', r'\epsilon']
		label_names=[r'A\;[{\rm K}]', r'\nu_0\;[{\rm MHz}]', r'w\;[{\rm MHz}]', r'\tau', r'T_{100}\;[{\rm K}]', r'\beta', r'\gamma', r'\delta', r'\epsilon']
		getdist_samples, ww, ll, best_fit, covariance_matrix = gp.load_samples(filename, 0, label_names=label_names)
		#reordered_getdist_samples = np.flip(getdist_samples)
		
		output_pdf_filename = edges_folder + 'plots/20190617/triangle_plot_exp_exp_4par_60_120MHz.pdf'
		o = gp.triangle_plot(getdist_samples, output_pdf_filename, legend_FS=10, label_FS=18, axes_FS=7)  # reordered
		#gp.rotate_xticklabels(rotation=90)
		
		
		
		
		
		
		
		
	
	
	if plot_number == 10:
		
		# Plot of histogram of GHA for integrated spectrum
		f, px, rx, wx, index, gha, ydx = eg.level4read('/home/raul/DATA2/EDGES_vol2/mid_band/spectra/level4/case26/case26.hdf5')
		
		
		keep = eg.daily_nominal_filter('mid_band', 26, ydx)
		ydy = ydx[keep>0]
		
		
		
		
		ix = np.arange(len(ydy))
		
		path_files  = edges_folder + 'mid_band/spectra/level3/case26/'
		new_list    = listdir(path_files)
		new_list.sort()
		
		index_new_list = range(len(new_list))
		
		gha_all = np.array([])
		for i in index_new_list:
			
			if len(new_list[i]) == 8:
				day = float(new_list[i][5::])			
			elif len(new_list[i]) > 8:
				day = float(new_list[i][5:8])


			
			Q = ix[ydy[:,1]==day]
			
			if len(Q)>0:
				print(new_list[i])
				f, ty, py, ry, wy, rmsy, tpy, my = eg.level3read(path_files + new_list[i])
				ii = index[i,0,0:len(my)]
				gha_i = my[ii>0,4]
				gha_i[gha_i<0] = gha_i[gha_i<0] + 24
				gha_all = np.append(gha_all, gha_i)
		
		
		
		
		
		
		

		sp = np.genfromtxt(edges_folder + 'mid_band/spectra/level5/case2/integrated_spectrum_case2.txt')
		fb = sp[:,0]
		wb = sp[:,2]
		
		fbb = fb[fb>=60]
		wbb = wb[fb>=60]
		
		
		
		
				
				
		plt.close()
		plt.close()
		fig = plt.figure(1, figsize=[3.7, 5])
		
		x0_top = 0.1 
		y0_top = 0.57
		
		x0_bottom = 0.1
		y0_bottom = 0.1
		
		dx  = 0.85
		dy  = 0.35	
		
		ax = fig.add_axes([x0_top, y0_top, dx, dy])
		ax.hist(gha_all, np.arange(6,18.1,1/6))
		plt.ylim([0, 400])
		plt.xlabel('GHA [hr]')
		plt.ylabel('number of raw spectra\nper 10-min GHA bin')
		plt.text(5.8,345,'(a)', fontsize=14)
		
		ax = fig.add_axes([x0_bottom, y0_bottom, dx, dy])
		ax.step(fbb, wbb/np.max(wbb), linewidth=1)
		plt.yticks([0, 0.25, 0.5, 0.75, 1])
		plt.ylim([0, 1.25])
		plt.xlabel(r'$\nu$ [MHz]')
		plt.ylabel('normalized weights')
		plt.text(59,1.09,'(b)', fontsize=14)
		
		plt.savefig(edges_folder + 'plots/20190730/data_statistics.pdf', bbox_inches='tight')
		plt.close()	
		plt.close()
		plt.close()
		plt.close()		
		


	if plot_number == 11:


	
	
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
		
	
	
	
	
		plt.figure(figsize=[4,6])
		#gg = [0.5, 0.5, 0.5]
		gg = 'c'
		dy = 500 # K
		lw = 1
		k  = 0.1
		for i in range(12):
			tli = tl[il[i],:]
			wli = wl[il[i],:]
			
			tmi = tm[im[i],:]
			wmi = wm[im[i],:]
			
			if i%2 == 0:
				plt.plot(fl[wmi>0], (tmi-tli)[wmi>0] - i*dy, color=gg, linewidth=lw)
			else:
				plt.plot(fl[wmi>0], (tmi-tli)[wmi>0] - i*dy, color=gg, linewidth=lw)
				
			plt.plot([50, 150], [-i*dy, -i*dy], 'k')
			if i <= 4:
				plt.text(53.6, -i*dy-k*dy, str(2*i) + ' hr')
			else:
				plt.text(52.3, -i*dy-k*dy, str(2*i) + ' hr')
			
		#yticks = dy*np.arange(-11,2)
		#plt.yticks(yticks, [str(-2*(i-11)) + ' hr'  for i in range(12)], direction='in')
		plt.yticks([])
		plt.ylabel('GHA [' + str(dy) + ' K per division]\n\n\n')
		plt.xlabel(r'$\nu$ [MHz]')
		plt.xlim([58, 102])
		plt.ylim([-12*dy, dy])
		
		
		# Saving		
		plt.savefig(edges_folder + '/plots/20190612/comparison_mid_low2.pdf', bbox_inches='tight')
		plt.close()
		plt.close()		
		





	if plot_number == 12:
		
		# Loading Haslam map
		map1, lon1, lat1, gc1 = cal.haslam_408MHz_map()
		
		# Loading LW map
		map2, lon2, lat2, gc2 = cal.LW_150MHz_map()
		
		# Loading Guzman map
		map3, lon3, lat3, gc3 = cal.guzman_45MHz_map()
			
	
		#print(v0)
	
	
	
	
		# Scaling sky map (the map contains the CMB, which has to be removed and then added back)
		# ---------------------------------------------------------------------------------------
		ipole     = 2.65
		icenter   = 2.4
		sigma_deg = 8.5
		
		i1 = ipole - (ipole-icenter) * np.exp(-(1/2)*(np.abs(lat1)/sigma_deg)**2)
		i2 = ipole - (ipole-icenter) * np.exp(-(1/2)*(np.abs(lat2)/sigma_deg)**2)
		i3 = ipole - (ipole-icenter) * np.exp(-(1/2)*(np.abs(lat3)/sigma_deg)**2)
		


		i12 = 2.56 * np.ones(len(lat1))


		band_deg = 10
		index_inband  = 2.4
		index_outband = 2.65
		i13 = np.zeros(len(lat1))
		i13[np.abs(lat1) <= band_deg] = index_inband
		i13[np.abs(lat1) > band_deg]  = index_outband
		
		
					
	
		Tcmb  = 2.725
		s1  = (map1 - Tcmb) * (90/408)**(-i1)  + Tcmb
		s2 = (map1 - Tcmb) * (90/408)**(-i12) + Tcmb
		s3 = (map1 - Tcmb) * (90/408)**(-i13) + Tcmb
		s4  = (map2 - Tcmb) * (90/150)**(-i2)  + Tcmb
		s5  = (map3 - Tcmb) * (90/45)**(-i3)   + Tcmb
		
		ss4 = hp.pixelfunc.ud_grade(s4, 512, order_in='NESTED')
		ss5 = hp.pixelfunc.ud_grade(s5, 512, order_in='NESTED')

		
		plt.close()
		plt.close()
		plt.close()
		plt.close()
		plt.close()
		plt.close()
		plt.close()
		plt.close()
		plt.close()
		plt.close()
		plt.close()
		plt.close()
		plt.close()
		plt.close()
		plt.close()
		plt.close()
		
		hp.cartview(s1, nest='yes', min=500, max=2000, cbar=False, coord='GC')
		hp.cartview(s2, nest='yes', min=500, max=2000, cbar=False, coord='GC')
		hp.cartview(ss4, nest='yes', min=500, max=2000, cbar=False, coord='GC')
		hp.cartview(ss5, nest='yes', min=500, max=2000, cbar=False, coord='GC')
		
		dLIM=500
		hp.cartview(s2 - s1, nest='yes', min=-dLIM, max=dLIM, cbar=False, coord='GC')
		hp.cartview(ss4 - s1, nest='yes', min=-dLIM, max=dLIM, cbar=False, coord='GC')
		hp.cartview(ss5 - s1, nest='yes', min=-dLIM, max=dLIM, cbar=False, coord='GC')
		
		
		
		
		
		
		
	
	
	
		

	if plot_number == 13:
		
		plt.close()
		plt.close()
		plt.close()
		plt.close()
		plt.close()
		plt.close()
		plt.close()
		plt.close()
		plt.close()
		plt.close()
		plt.close()
		
		
		
				
		# Loading Haslam map
		map408, lon, lat, gc = cal.haslam_408MHz_map()

		ipole     = 2.65
		icenter   = 2.4
		sigma_deg = 8.5
		
		i1 = ipole - (ipole-icenter) * np.exp(-(1/2)*(np.abs(lat)/sigma_deg)**2)
		
		Tcmb = 2.725
		s    = (map408 - Tcmb) * (90/408)**(-i1)  + Tcmb
		

		hp.cartview(np.log10(s), nest='true', coord=('G', 'C'), flip='geo', title='', notext='true', min=2.7, max=4.3, unit=r'log($T_{\mathrm{sky}}$)', rot=[5.76667*15,0,0], cmap='jet')   #, ,     , min=2.2, max=3.9 unit=r'log($T_{sky}$)' title='', min=2.3, max=2.6, unit=r'$\beta$'), , rot=[180,0,0]
		hp.graticule(local=True)







	
		beam   = cal.FEKO_blade_beam('mid_band', 0, frequency_interpolation='no', AZ_antenna_axis=90)
		f      = np.arange(50,201,2)
		beam90 = beam[20,:,:]
		beam90n = beam90/np.max(beam90)
		
		FWHM = np.zeros((360, 2))
		EL_raw    = np.arange(0,91,1)
		EL_new    = np.arange(0,90.01, 0.01)
		
		for j in range(len(beam90[0,:])): # Loop over AZ
			#print(j)

			func           = sci.interp1d(EL_raw, beam90n[:,j])
			beam90n_interp = func(EL_new)

			minDiff = 100
			for i in range(len(EL_new)):	
				Diff    = np.abs(beam90n_interp[i] - 0.5)
				if Diff < minDiff:
					#print(90-EL_new[i])
					minDiff   = np.copy(Diff)
					FWHM[j,0] = j
					FWHM[j,1] = 90 - EL_new[i]


		
		# Reference location
		EDGES_lat_deg  = -26.714778
		EDGES_lon_deg  = 116.605528 
		EDGES_location = apc.EarthLocation(lat=EDGES_lat_deg*apu.deg, lon=EDGES_lon_deg*apu.deg)		
		
		
		
		Time_iter_UTC_start  = np.array([2014, 1, 1,  3, 31, 0])    # 18    LST, obtained iteratively using LST = eg.utc2lst(Time_iter_UTC_start, EDGES_lon_deg)
		Time_iter_UTC_middle = np.array([2014, 1, 1,  9, 30, 0])    # 0     LST, obtained iteratively using LST = eg.utc2lst(Time_iter_UTC_start, EDGES_lon_deg)
		Time_iter_UTC_end    = np.array([2014, 1, 1,  15, 29, 0])   # 6     LST, obtained iteratively using LST = eg.utc2lst(Time_iter_UTC_start, EDGES_lon_deg)




		Time_iter_UTC_start_dt  = dt.datetime(Time_iter_UTC_start[0], Time_iter_UTC_start[1], Time_iter_UTC_start[2], Time_iter_UTC_start[3], Time_iter_UTC_start[4], Time_iter_UTC_start[5]) 
		Time_iter_UTC_middle_dt = dt.datetime(Time_iter_UTC_middle[0], Time_iter_UTC_middle[1], Time_iter_UTC_middle[2], Time_iter_UTC_middle[3], Time_iter_UTC_middle[4], Time_iter_UTC_middle[5])
		Time_iter_UTC_end_dt    = dt.datetime(Time_iter_UTC_end[0], Time_iter_UTC_end[1], Time_iter_UTC_end[2], Time_iter_UTC_end[3], Time_iter_UTC_end[4], Time_iter_UTC_end[5]) 


		# Converting Beam Contours from Local to Equatorial coordinates
		AltAz_start = apc.SkyCoord(alt = (90-FWHM[:,1])*apu.deg, az = FWHM[:,0]*apu.deg, frame = 'altaz', obstime = apt.Time(Time_iter_UTC_start_dt, format='datetime'), location = EDGES_location)
		RaDec_start = AltAz_start.icrs	
		Ra_start  = np.asarray(RaDec_start.ra)
		Dec_start = np.asarray(RaDec_start.dec)
		RaWrap_start = np.copy(Ra_start)
		RaWrap_start[Ra_start>180] = Ra_start[Ra_start>180] - 360




		AltAz_middle = apc.SkyCoord(alt = (90-FWHM[:,1])*apu.deg, az = FWHM[:,0]*apu.deg, frame = 'altaz', obstime = apt.Time(Time_iter_UTC_middle_dt, format='datetime'), location = EDGES_location)
		RaDec_middle = AltAz_middle.icrs	
		Ra_middle  = np.asarray(RaDec_middle.ra)
		Dec_middle = np.asarray(RaDec_middle.dec)
		RaWrap_middle = np.copy(Ra_middle)
		RaWrap_middle[Ra_middle>180] = Ra_middle[Ra_middle>180] - 360		




		AltAz_end = apc.SkyCoord(alt = (90-FWHM[:,1])*apu.deg, az = FWHM[:,0]*apu.deg, frame = 'altaz', obstime = apt.Time(Time_iter_UTC_end_dt, format='datetime'), location = EDGES_location)
		RaDec_end = AltAz_end.icrs	
		Ra_end  = np.asarray(RaDec_end.ra)
		Dec_end = np.asarray(RaDec_end.dec)
		RaWrap_end = np.copy(Ra_end)
		RaWrap_end[Ra_end>180] = Ra_end[Ra_end>180] - 360		





		plt.plot(np.arange(-180,181,1), -26.7*np.ones(361), 'y--', linewidth=2)
		
		plt.plot(RaWrap_start, Dec_start, 'w',     linewidth=3)
		plt.plot(RaWrap_middle, Dec_middle, 'w--', linewidth=3)
		plt.plot(RaWrap_end, Dec_end, 'w:',        linewidth=3)		


		plt.plot(-6*(360/24), -26.7, 'x', color='1', markersize=5, mew=2)
		plt.plot(0*(360/24), -26.7, 'x', color='1', markersize=5, mew=2)
		plt.plot(6*(360/24), -26.7, 'x', color='1', markersize=5, mew=2)	
	
	
	
		off_x = -4
		off_y = -12
		plt.text(-180+off_x, -90+off_y, '0')
		plt.text(-150+off_x, -90+off_y, '2')
		plt.text(-120+off_x, -90+off_y, '4')
		plt.text( -90+off_x, -90+off_y, '6')
		plt.text( -60+off_x, -90+off_y, '8')
		plt.text( -30+off_x, -90+off_y, '10')
		plt.text(  -0+off_x, -90+off_y, '12')
	
		plt.text( 30+off_x, -90+off_y, '14')
		plt.text( 60+off_x, -90+off_y, '16')
		plt.text( 90+off_x, -90+off_y, '18')
		plt.text(120+off_x, -90+off_y, '20')
		plt.text(150+off_x, -90+off_y, '22')
		plt.text(180+off_x, -90+off_y, '24')		
	
		plt.text(-60, -115, 'galactic hour angle [hr]')
	
	
	
		off_x1 = -15
		off_x2 = -10
		off_x3 = -19
		off_y  = -3
		plt.text(-180+off_x1,  90+off_y,  '90')
		plt.text(-180+off_x1,  60+off_y,  '60')
		plt.text(-180+off_x1,  30+off_y,  '30')
		plt.text(-180+off_x2,   0+off_y,   '0')
		plt.text(-180+off_x3, -30+off_y, '-30')
		plt.text(-180+off_x3, -60+off_y, '-60')
		plt.text(-180+off_x3, -90+off_y, '-90')
	
		plt.text(-210, 45, 'declination [degrees]', rotation=90)
	



	if plot_number == 14:
		f = np.arange(60, 120.5, 0.5)
		b18 = dm.signal_model('exp', [-0.5, 78, 19, 7], f)
		
		x1 = dm.signal_model('exp', [-0.5, 78, 19, 7, -0.5], f)
		x2 = dm.signal_model('exp', [-0.5, 78, 19, 7,  0.5], f)
		
		t1 = dm.signal_model('tanh', [-0.5, 78, 19, 3, 7], f)
		t2 = dm.signal_model('tanh', [-0.5, 78, 19, 7, 3], f)
		
		
		plt.close()
		plt.close()
		plt.close()
		plt.close()
		plt.close()
		plt.close()
		plt.close()
		
		
		x0_top = 0.1
		y0_top = 0.5
		x0_bottom = 0.1
		y0_bottom = 0.1
		
		dx = 0.85
		dy = 0.4
		
		
		fig = plt.figure(figsize=[3.8,4.3])
		
		ax = fig.add_axes([x0_top, y0_top, dx, dy])
		ax.plot(f, b18, 'k')
		ax.plot(f, x1, 'b--')
		ax.plot(f, x2, 'b:')
		#plt.plot(f, b18, 'k')
		plt.ylim([-0.7, 0.1])
		plt.yticks(np.arange(-0.6,0.05, 0.2))
		plt.text(59, -0.65, '(a)', fontsize=15)
		plt.legend([r'Bowman et al. (2018)',r'Exp Model, $\chi=-0.5$',r'Exp Model, $\chi=+0.5$'], fontsize=8)
		plt.ylabel('brightness\n temperature [K]')
		
		
		ax = fig.add_axes([x0_bottom, y0_bottom, dx, dy])
		ax.plot(f, b18, 'k')
		ax.plot(f, t1, 'r--')
		ax.plot(f, t2, 'r:')
		#plt.plot(f, b18, 'k')
		plt.ylim([-0.7, 0.1])
		plt.yticks(np.arange(-0.6,0.05, 0.2))
		plt.text(59, -0.65, '(b)', fontsize=15)
		plt.legend([r'Bowman et al. (2018)',r'Tanh Model, $\tau_1=3$',r'Tanh Model, $\tau_2=3$'], fontsize=8)
		plt.xlabel(r'$\nu$ [MHz]', fontsize=13)
		plt.ylabel('brightness\n temperature [K]')
		
		# Saving		
		plt.savefig(edges_folder + '/plots/20190730/absorption_models.pdf', bbox_inches='tight')
		plt.close()
		plt.close()		
	
	
	
	
	return 0 #fr3, bx3



	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	

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




def plots_midband_metadata():
	
	
	

	list_files   = listdir(edges_folder + '/mid_band/spectra/level3/case_nominal/')
	list_files.sort()
	

	plt.close()
	plt.close()
	
	
	# Processing files
	for i in range(len(list_files)):
		
		day = list_files[i][0:11]
		
		f, t, p, r, w, rms, tp, m = eg.level3read(edges_folder + '/mid_band/spectra/level3/case_nominal/' + list_files[i])

		gha = m[:,4]
		gha[gha<0] = gha[gha<0] + 24
		
		sun_el = m[:,6]
		temp   = m[:,9]
		hum    = m[:,10]
		rec_temp = m[:,11]
		
		
		index_gha_6  = -1
		index_gha_18 = -1
		for i in range(len(gha)-1):
			if (gha[i]<=6) and (gha[i+1]>6):
				index_gha_6 = i
				
			if (gha[i]<=18) and (gha[i+1]>18):
				index_gha_18 = i+1			





		plt.close()
		plt.close()


		plt.figure(figsize=[4.5,9])
		plt.subplot(5,1,1)
		plt.plot(gha, 'b', linewidth=2)
		if index_gha_6 > -1:
			plt.plot([index_gha_6, index_gha_6], [-1000, 1000], 'g--', linewidth=2)
			#plt.text(index_gha_6, 12, 'GHA=6hr', rotation=90, color='g', fontsize=12)
			plt.text(index_gha_6, 12, ' 6hr', rotation=0, color='g', fontsize=10)
		if index_gha_18 > -1:
			plt.plot([index_gha_18, index_gha_18], [-1000, 1000], 'r--', linewidth=2)
			#plt.text(index_gha_18, 14, 'GHA=18hr', rotation=90, color='r', fontsize=12)
			plt.text(index_gha_18, 12, ' 18hr', rotation=0, color='r', fontsize=10)
		plt.ylim([-1, 26])		
		plt.yticks([0,6,12,18,24])
		plt.ylabel('GHA [hr]')
		plt.grid()
		plt.title(day)
		#plt.legend(['GHA = 6 hr','GHA = 18 hr'], loc=0)
		
		
		plt.subplot(5,1,2)
		plt.plot(sun_el, 'b', linewidth=2)
		if index_gha_6 > -1:
			plt.plot([index_gha_6, index_gha_6], [-1000, 1000], 'g--', linewidth=2)
			
		if index_gha_18 > -1:
			plt.plot([index_gha_18, index_gha_18], [-1000, 1000], 'r--', linewidth=2)
		plt.ylim([-110, 110])
		plt.yticks([-90, -45, 0, 45, 90])
		plt.ylabel('sun elev [deg]')
		plt.grid()
		
		
		
		plt.subplot(5,1,3)
		plt.plot(temp, 'b', linewidth=2)
		if index_gha_6 > -1:
			plt.plot([index_gha_6, index_gha_6], [-1000, 1000], 'g--', linewidth=2)
			
		if index_gha_18 > -1:
			plt.plot([index_gha_18, index_gha_18], [-1000, 1000], 'r--', linewidth=2)
		plt.ylim([0, 40])
		plt.yticks([10,20,30])
		plt.ylabel(r'amb temp [$^{\circ}$C]')
		plt.grid()
		
		
		
		plt.subplot(5,1,4)
		plt.plot(hum, 'b', linewidth=2)
		if index_gha_6 > -1:
			plt.plot([index_gha_6, index_gha_6], [-1000, 1000], 'g--', linewidth=2)
			
		if index_gha_18 > -1:
			plt.plot([index_gha_18, index_gha_18], [-1000, 1000], 'r--', linewidth=2)
		plt.ylim([-30, 110])
		plt.yticks([-20,0,20,40,60,80,100])
		plt.ylabel('amb humid [%]')
		plt.grid()
		
		
		
		plt.subplot(5,1,5)
		plt.plot(rec_temp, 'b', linewidth=2)
		if index_gha_6 > -1:
			plt.plot([index_gha_6, index_gha_6], [-1000, 1000], 'g--', linewidth=2)
			
		if index_gha_18 > -1:
			plt.plot([index_gha_18, index_gha_18], [-1000, 1000], 'r--', linewidth=2)
			
		plt.ylim([23, 27])
		plt.yticks([24,25,26])
		plt.ylabel(r'rec temp [$^{\circ}$C]')
		plt.grid()		
		
		
		plt.xlabel('time [number of raw spectra since start of file]')




		plt.savefig(edges_folder + '/mid_band/spectra/level3/case_nominal/metadata/' + day + '.png', bbox_inches='tight')
		plt.close()	
		plt.close()
		#plt.close()
		#plt.close()
		#plt.close()
		#plt.close()
		#plt.close()
		#plt.close()
	
	
	
	
	return 0






def comparison_switch_receiver1():
	
	f       = np.arange(50,201)
	ant_s11 = np.ones(len(f))
	
	o1 = cr1.switch_correction_receiver1(ant_s11, f_in = f, case = 1)
	o2 = cr1.switch_correction_receiver1(ant_s11, f_in = f, case = 2)
	
	o10 = cr1.switch_correction_receiver1(ant_s11, f_in = f, case = 10)
	o11 = cr1.switch_correction_receiver1(ant_s11, f_in = f, case = 11)
	o12 = cr1.switch_correction_receiver1(ant_s11, f_in = f, case = 12)
	o13 = cr1.switch_correction_receiver1(ant_s11, f_in = f, case = 13)	
	
	
	fl = np.arange(50,101)
	al = np.ones(len(fl))
	ol = oeg.low_band_switch_correction(al, 25, f_in = fl)
	
	
	
	
	
	ant_1    = o1[0]
	s11_1    = o1[1]
	s12s21_1 = o1[2]
	s22_1    = o1[3]
	
	ant_2    = o2[0]
	s11_2    = o2[1]
	s12s21_2 = o2[2]
	s22_2    = o2[3]	

	ant_10    = o10[0]
	s11_10    = o10[1]
	s12s21_10 = o10[2]
	s22_10    = o10[3]
	
	ant_11    = o11[0]
	s11_11    = o11[1]
	s12s21_11 = o11[2]
	s22_11    = o11[3]	
	
	ant_12    = o12[0]
	s11_12    = o12[1]
	s12s21_12 = o12[2]
	s22_12    = o12[3]

	ant_13    = o13[0]
	s11_13    = o13[1]
	s12s21_13 = o13[2]
	s22_13    = o13[3]
	
	
	ant_l    = ol[0]
	s11_l    = ol[1]
	s12s21_l = ol[2]
	s22_l    = ol[3]	
	
	
	
	
	
	
	
	

	
	plt.subplot(3,2,1)
	plt.plot(f, 20*np.log10(np.abs(s11_1)), 'b')
	plt.plot(f, 20*np.log10(np.abs(s11_2)), 'b--')
	
	plt.plot(f, 20*np.log10(np.abs(s11_10)), 'r')
	plt.plot(f, 20*np.log10(np.abs(s11_11)), 'r')
	plt.plot(f, 20*np.log10(np.abs(s11_12)), 'r')
	plt.plot(f, 20*np.log10(np.abs(s11_13)), 'r')
	
	plt.plot(fl, 20*np.log10(np.abs(s11_l)), 'g')	
	
	
	
	plt.subplot(3,2,2)
	plt.plot(f, (180/np.pi)*np.unwrap(np.angle(s11_1)), 'b')
	plt.plot(f, (180/np.pi)*np.unwrap(np.angle(s11_2)), 'b--')
	
	plt.plot(f, (180/np.pi)*np.unwrap(np.angle(s11_10)), 'r')
	plt.plot(f, (180/np.pi)*np.unwrap(np.angle(s11_11)), 'r')
	plt.plot(f, (180/np.pi)*np.unwrap(np.angle(s11_12)), 'r')
	plt.plot(f, (180/np.pi)*np.unwrap(np.angle(s11_13)), 'r')
	
	plt.plot(fl, (180/np.pi)*np.unwrap(np.angle(s11_l)), 'g')
	
	
	
	
	
	plt.subplot(3,2,3)
	plt.plot(f, 20*np.log10(np.abs(s12s21_1)), 'b')
	plt.plot(f, 20*np.log10(np.abs(s12s21_2)), 'b--')
	
	plt.plot(f, 20*np.log10(np.abs(s12s21_10)), 'r')
	plt.plot(f, 20*np.log10(np.abs(s12s21_11)), 'r')
	plt.plot(f, 20*np.log10(np.abs(s12s21_12)), 'r')
	plt.plot(f, 20*np.log10(np.abs(s12s21_13)), 'r')
	
	plt.plot(fl, 20*np.log10(np.abs(s12s21_l)), 'g')
		

	plt.subplot(3,2,4)
	plt.plot(f, (180/np.pi)*np.unwrap(np.angle(s12s21_1)), 'b')
	plt.plot(f, (180/np.pi)*np.unwrap(np.angle(s12s21_2)), 'b--')
	
	plt.plot(f, (180/np.pi)*np.unwrap(np.angle(s12s21_10)), 'r')
	plt.plot(f, (180/np.pi)*np.unwrap(np.angle(s12s21_11)), 'r')
	plt.plot(f, (180/np.pi)*np.unwrap(np.angle(s12s21_12)), 'r')
	plt.plot(f, (180/np.pi)*np.unwrap(np.angle(s12s21_13)), 'r')
	
	plt.plot(fl, (180/np.pi)*np.unwrap(np.angle(s12s21_l)), 'g')


	
	plt.subplot(3,2,5)
	plt.plot(f, 20*np.log10(np.abs(s22_1)), 'b')
	plt.plot(f, 20*np.log10(np.abs(s22_2)), 'b--')
	
	plt.plot(f, 20*np.log10(np.abs(s22_10)), 'r')
	plt.plot(f, 20*np.log10(np.abs(s22_11)), 'r')
	plt.plot(f, 20*np.log10(np.abs(s22_12)), 'r')
	plt.plot(f, 20*np.log10(np.abs(s22_13)), 'r')
	
	plt.plot(fl, 20*np.log10(np.abs(s22_l)), 'g')
	

	plt.subplot(3,2,6)
	plt.plot(f, (180/np.pi)*np.unwrap(np.angle(s22_1)), 'b')
	plt.plot(f, (180/np.pi)*np.unwrap(np.angle(s22_2)), 'b--')
	
	plt.plot(f, (180/np.pi)*np.unwrap(np.angle(s22_10)), 'r')
	plt.plot(f, (180/np.pi)*np.unwrap(np.angle(s22_11)), 'r')
	plt.plot(f, (180/np.pi)*np.unwrap(np.angle(s22_12)), 'r')
	plt.plot(f, (180/np.pi)*np.unwrap(np.angle(s22_13)), 'r')	
	
	plt.plot(fl, (180/np.pi)*np.unwrap(np.angle(s22_l)), 'g')
	
	return 0








def plots_for_memo148(plot_number):
	
	
	# Receiver calibration parameters
	if plot_number==1:
	

		# Paths
		path_plot_save = edges_folder + 'plots/20190828/'


		# Calibration parameters
		rcv_file = edges_folder + 'mid_band/calibration/receiver_calibration/receiver1/2018_01_25C/results/nominal/calibration_files/calibration_file_receiver1_cterms7_wterms8.txt'
		
		#mid_band/calibration/receiver_calibration/receiver1/2018_01_25C/results/nominal/calibration_files/calibration_file_receiver1_50_150MHz_cterms7_wterms7.txt'

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


		rcv1 = np.genfromtxt('/home/raul/DATA2/EDGES_vol2/mid_band/calibration/receiver_calibration/receiver1/2019_04_25C/results/nominal/calibration_files/calibration_file_receiver1_50_150MHz_cterms8_wterms10.txt')




		# Low-Band
		rcv_lb = np.genfromtxt('/home/raul/DATA1/EDGES_vol1/calibration/receiver_calibration/low_band1/2015_08_25C/results/nominal/calibration_files/calibration_file_low_band_2015_nominal.txt')
		
		


















		# Plot

		size_x = 6
		size_y = 8 #10.5
		x0 = 0.15
		y0 = 0.09
		dx = 0.7
		dy = 0.3


		f1  = plt.figure(num=1, figsize=(size_x, size_y))		


		ax     = f1.add_axes([x0, y0 + 2*dy, dx, dy])	
		h1     = ax.plot(fe, 20*np.log10(np.abs(rl)), 'b', linewidth=1.5, label='$|\Gamma_{\mathrm{rec}}|$')
		ax.plot(rcv1[:,0], 20*np.log10(np.abs(rcv1[:,1]+1j*rcv1[:,2])), 'r', linewidth=1.5)
		ax.plot(rcv_lb[:,0], 20*np.log10(np.abs(rcv_lb[:,1]+1j*rcv_lb[:,2])), 'g', linewidth=1.5)
		
		
		
		
		
		ax2    = ax.twinx()
		h2     = ax2.plot(fe, (180/np.pi)*np.unwrap(np.angle(rl)), 'b--', linewidth=1.5, label=r'$\angle\/\Gamma_{\mathrm{rec}}$')
		ax2.plot(rcv1[:,0], (180/np.pi)*np.unwrap(np.angle(rcv1[:,1]+1j*rcv1[:,2])), 'r--', linewidth=1.5)
		ax2.plot(rcv_lb[:,0], (180/np.pi)*np.unwrap(np.angle(rcv_lb[:,1]+1j*rcv_lb[:,2])), 'g--', linewidth=1.5)
		
		
		h      = h1 + h2
		labels = [l.get_label() for l in h]
		ax.legend(h, labels, loc=0, fontsize=10, ncol=2)

		ax.set_ylim([-40, -28])
		ax.set_xticklabels('')
		ax.set_yticks(np.arange(-39,-28,2))
		ax.set_ylabel('$|\Gamma_{\mathrm{rec}}|$ [dB]', fontsize=14)
		ax.text(48.5, -39.5, '(a)', fontsize=16)
		ax.text(113, -37, '2015-Aug', fontweight='bold', color='g')
		ax.text(113, -38, '2018-Jan', fontweight='bold', color='b')
		ax.text(113, -39, '2019-Apr', fontweight='bold', color='r')

		ax2.set_ylim([70, 130])
		ax2.set_xticklabels('')
		ax2.set_yticks(np.arange(80,121,10))		
		ax2.set_ylabel(r'$\angle\/\Gamma_{\mathrm{rec}}$ [ $^\mathrm{o}$]', fontsize=14)

		ax.set_xlim([48, 152])
		ax.tick_params(axis='x', direction='in')
		ax.set_xticks(np.arange(50, 151, 10))
		
		
		
		



		ax     = f1.add_axes([x0, y0 + 1*dy, dx, dy])
		h1     = ax.plot(fe, sca,'b',linewidth=1.5, label='$C_1$')
		ax.plot(rcv1[:,0], rcv1[:,3],'r', linewidth=1.5)      
		ax.plot(rcv_lb[:,0], rcv_lb[:,3],'g', linewidth=1.5)      #  <----------------------------- Low-Band
		ax2    = ax.twinx()
		h2     = ax2.plot(fe, off,'b--',linewidth=1.5, label='$C_2$')
		ax2.plot(rcv1[:,0], rcv1[:,4],'r--', linewidth=1.5)      
		ax2.plot(rcv_lb[:,0], rcv_lb[:,4],'g--', linewidth=1.5)      #  <----------------------------- Low-Band
		h      = h1 + h2
		labels = [l.get_label() for l in h]
		ax.legend(h, labels, loc=0, fontsize=10, ncol=2)

		ax.set_ylim([3.3, 5.2])
		ax.set_xticklabels('')
		ax.set_yticks(np.arange(3.5,5.1,0.5))
		ax.set_ylabel('$C_1$', fontsize=14)
		ax.text(48.5, 3.38, '(b)', fontsize=16)

		#ax2.set_ylim([-2.4, -1.8])
		ax2.set_ylim([-2.75, -0])
		#ax2.set_ylim([-1.6, -1.4])
		ax2.set_xticklabels('')
		ax2.set_yticks(np.arange(-2.5, -0.05, 0.5))
		ax2.set_ylabel('$C_2$ [K]', fontsize=14)

		ax.set_xlim([48, 152])
		ax.tick_params(axis='x', direction='in')
		ax.set_xticks(np.arange(50, 151, 10))
		
		
		
		



		ax     = f1.add_axes([x0, y0 + 0*dy, dx, dy])
		h1     = ax.plot(fe, TU,'b', linewidth=1.5, label='$T_{\mathrm{unc}}$')
		ax.plot(rcv1[:,0], rcv1[:,5],'r', linewidth=1.5)      # 
		ax.plot(rcv_lb[:,0], rcv_lb[:,5],'g', linewidth=1.5)      #  <----------------------------- Low-Band
		
		ax2    = ax.twinx()
		h2     = ax2.plot(fe, TC,'b--', linewidth=1.5, label='$T_{\mathrm{cos}}$')
		ax2.plot(rcv1[:,0], rcv1[:,6],'r--', linewidth=1.5)      
		ax2.plot(rcv_lb[:,0], rcv_lb[:,6],'g--', linewidth=1.5)      #  <----------------------------- Low-Band
		
		h3     = ax2.plot(fe, TS,'b:', linewidth=1.5, label='$T_{\mathrm{sin}}$')
		ax2.plot(rcv1[:,0], rcv1[:,7],'r:', linewidth=1.5)      
		ax2.plot(rcv_lb[:,0], rcv_lb[:,7],'g:', linewidth=1.5)      #  <----------------------------- Low-Band

		h      = h1 + h2 + h3
		labels = [l.get_label() for l in h]
		ax.legend(h, labels, loc=0, fontsize=10, ncol=3)

		ax.set_ylim([178, 190])
		ax.set_yticks(np.arange(180, 189, 2))
		ax.set_ylabel('$T_{\mathrm{unc}}$ [K]', fontsize=14)
		ax.set_xlabel('$\\nu$ [MHz]', fontsize=14)
		ax.text(48.5, 178.5, '(c)', fontsize=16)

		ax2.set_ylim([-55, 35])
		ax2.set_yticks(np.arange(-40, 21, 20))
		ax2.set_ylabel('$T_{\mathrm{cos}}, T_{\mathrm{sin}}$ [K]', fontsize=14)
		
		ax.set_xlim([48, 152])
		ax.set_xticks(np.arange(50, 151, 10))


		plt.savefig(path_plot_save + 'receiver_calibration.pdf', bbox_inches='tight')
		plt.close()	
		plt.close()
		plt.close()
		plt.close()





	if plot_number == 2:
		
		# Paths
		path_plot_save = edges_folder + 'plots/20190828/'		

		

		
		flb = np.arange(50,100,1)
		alb = np.ones(len(flb))
		
		f   = np.arange(50,151,1)
		ant_s11 = np.ones(len(f))
		
		
		o = oeg.low_band_switch_correction(alb, 27.16, f_in = flb)
		v1 = o[1]
		v2 = o[2]
		v3 = o[3]		
		
		o = cr1.switch_correction_receiver1(ant_s11, f_in = f, case = 0)
		w1 = o[1]
		w2 = o[2]
		w3 = o[3]		
		
		o = cr1.switch_correction_receiver1(ant_s11, f_in = f, case = 1)
		x1 = o[1]
		x2 = o[2]
		x3 = o[3]
		
		o = cr1.switch_correction_receiver1(ant_s11, f_in = f, case = 6)
		y1 = o[1]
		y2 = o[2]
		y3 = o[3]
		


		size_x = 11
		size_y = 13.0	
	
		f1  = plt.figure(num=1, figsize=(size_x, size_y))

		plt.subplot(3,2,1)
		plt.plot(flb, 20*np.log10(np.abs(v1)), 'k--')
		plt.plot(f, 20*np.log10(np.abs(w1)), 'k:')
		plt.plot(f, 20*np.log10(np.abs(x1)), 'b')
		plt.plot(f, 20*np.log10(np.abs(y1)), 'r--')
		plt.xticks(np.arange(50, 151, 10), labels='')
		plt.ylabel(r'$|S_{11}|$ [dB]')
		plt.grid()
		plt.legend([r'sw 2015, 50.12$\Omega$', r'sw 2017, 49.85$\Omega$', r'sw 2018, 50.027$\Omega$',r'sw 2019, 50.15$\Omega$'])
		
		plt.subplot(3,2,2)
		plt.plot(flb, (180/np.pi)*np.unwrap(np.angle(v1)), 'k--')
		plt.plot(f, (180/np.pi)*np.unwrap(np.angle(w1)), 'k:')
		plt.plot(f, (180/np.pi)*np.unwrap(np.angle(x1)), 'b')
		plt.plot(f, (180/np.pi)*np.unwrap(np.angle(y1)), 'r--')
		plt.xticks(np.arange(50, 151, 10), labels='')
		plt.ylabel(r'$\angle S_{11}$ [deg]')
		plt.grid()



		plt.subplot(3,2,3)
		plt.plot(flb, 20*np.log10(np.abs(v2)), 'k--')
		plt.plot(f, 20*np.log10(np.abs(w2)), 'k:')
		plt.plot(f, 20*np.log10(np.abs(x2)), 'b')
		plt.plot(f, 20*np.log10(np.abs(y2)), 'r--')
		plt.xticks(np.arange(50, 151, 10), labels='')
		plt.ylabel(r'$|S_{12}S_{21}|$ [dB]')
		plt.grid()
		
		plt.subplot(3,2,4)
		plt.plot(flb, (180/np.pi)*np.unwrap(np.angle(v2)), 'k--')
		plt.plot(f, (180/np.pi)*np.unwrap(np.angle(w2)), 'k:')
		plt.plot(f, (180/np.pi)*np.unwrap(np.angle(x2)), 'b')
		plt.plot(f, (180/np.pi)*np.unwrap(np.angle(y2)), 'r--')
		plt.xticks(np.arange(50, 151, 10), labels='')
		plt.ylabel(r'$\angle S_{12}S_{21}$ [deg]')
		plt.grid()


	
		plt.subplot(3,2,5)
		plt.plot(flb, 20*np.log10(np.abs(v3)), 'k--')
		plt.plot(f, 20*np.log10(np.abs(w3)), 'k:')
		plt.plot(f, 20*np.log10(np.abs(x3)), 'b')
		plt.plot(f, 20*np.log10(np.abs(y3)), 'r--')
		plt.xticks(np.arange(50, 151, 10))
		plt.ylabel(r'$|S_{22}|$ [dB]')
		plt.xlabel(r'$\nu$ [MHz]', fontsize=14)
		plt.grid()
		
		plt.subplot(3,2,6)
		plt.plot(flb, (180/np.pi)*np.unwrap(np.angle(v3)), 'k--')
		plt.plot(f, (180/np.pi)*np.unwrap(np.angle(w3)), 'k:')
		plt.plot(f, (180/np.pi)*np.unwrap(np.angle(x3)), 'b')
		plt.plot(f, (180/np.pi)*np.unwrap(np.angle(y3)), 'r--')
		plt.xticks(np.arange(50, 151, 10))
		plt.ylabel(r'$\angle S_{22}$ [deg]')
		plt.xlabel(r'$\nu$ [MHz]', fontsize=14)
		plt.grid()



		plt.savefig(path_plot_save + 'sw_parameters.pdf', bbox_inches='tight')
		plt.close()	
		plt.close()
		plt.close()
		plt.close()
		





	if plot_number == 3:
		
		# Paths
		path_plot_save = edges_folder + 'plots/20190828/'		



		size_x = 11
		size_y = 13.0	

		f1  = plt.figure(num=1, figsize=(size_x, size_y))
		
		
		# Frequency
		f, il, ih = ba.frequency_edges(50, 150)
		fe = f[il:ih+1]					
		
		ra1 = cal.models_antenna_s11_remove_delay('mid_band', fe, year=2018, day=147, case=3, delay_0=0.17, model_type='polynomial', Nfit=15, plot_fit_residuals='no')
		ra2 = cal.models_antenna_s11_remove_delay('mid_band', fe, year=2018, day=147, case=36, delay_0=0.17, model_type='polynomial', Nfit=15, plot_fit_residuals='no')
		
		ra3 = cal.models_antenna_s11_remove_delay('mid_band', fe, year=2018, day=147, case=3105012, delay_0=0.17, model_type='polynomial', Nfit=15, plot_fit_residuals='no')
		ra4 = cal.models_antenna_s11_remove_delay('mid_band', fe, year=2018, day=147, case=3605012, delay_0=0.17, model_type='polynomial', Nfit=15, plot_fit_residuals='no')
		
		
		ra11 = cal.models_antenna_s11_remove_delay('mid_band', fe, year=2018, day=222, case=11, delay_0=0.17, model_type='polynomial', Nfit=15, plot_fit_residuals='no') 
		ra31 = cal.models_antenna_s11_remove_delay('mid_band', fe, year=2018, day=222, case=31, delay_0=0.17, model_type='polynomial', Nfit=15, plot_fit_residuals='no')
		
		
		
		plt.subplot(3,2,1)
		plt.plot(fe, 20*np.log10(np.abs(ra1)), 'b')
		plt.xticks(np.arange(50, 151, 10), labels='')
		plt.grid()
		plt.ylabel('magnitude [dB]')
		
		plt.subplot(3,2,2)
		plt.plot(fe, (180/np.pi)*np.unwrap(np.angle(ra1)), 'b')
		plt.xticks(np.arange(50, 151, 10), labels='')
		plt.grid()		
		plt.ylabel('phase [deg]')
	
		plt.subplot(3,2,3)
		plt.plot(fe, 20*np.log10(np.abs(ra2)) - 20*np.log10(np.abs(ra1)), 'r')
		plt.plot(fe, 20*np.log10(np.abs(ra3)) - 20*np.log10(np.abs(ra1)), 'r--')
		plt.plot(fe, 20*np.log10(np.abs(ra4)) - 20*np.log10(np.abs(ra1)), 'r:')
		plt.xticks(np.arange(50, 151, 10))
		plt.ylim([-0.04, 0.06])
		plt.grid()
		#plt.xlabel(r'$\nu$ [MHz]', fontsize=14)
		plt.ylabel(r'$\Delta$ magnitude [dB]')
		plt.legend([r'Day 147(sw 2019, 50.15$\Omega$) - Day 147(sw 2018, 50.027$\Omega$)',r'Day 147(sw 2018, 50.12$\Omega$) - Day 147(sw 2018, 50.027$\Omega$)',r'Day 147(sw 2019, 50.12$\Omega$) - Day 147(sw 2018, 50.027$\Omega$)'], fontsize=8)
	
		plt.subplot(3,2,4)
		plt.plot(fe, (180/np.pi)*np.unwrap(np.angle(ra2)) - (180/np.pi)*np.unwrap(np.angle(ra1)), 'r')
		plt.plot(fe, (180/np.pi)*np.unwrap(np.angle(ra3)) - (180/np.pi)*np.unwrap(np.angle(ra1)), 'r--')
		plt.plot(fe, (180/np.pi)*np.unwrap(np.angle(ra4)) - (180/np.pi)*np.unwrap(np.angle(ra1)), 'r:')
		plt.xticks(np.arange(50, 151, 10))
		plt.ylim([-0.3, 0.3])
		plt.grid()
		#plt.xlabel(r'$\nu$ [MHz]', fontsize=14)
		plt.ylabel(r'$\Delta$ phase [deg]')
		
		

	
		plt.subplot(3,2,5)
		plt.plot(fe, 20*np.log10(np.abs(ra11)) - 20*np.log10(np.abs(ra1)), 'g')
		plt.plot(fe, 20*np.log10(np.abs(ra31)) - 20*np.log10(np.abs(ra1)), 'g--')
		plt.xticks(np.arange(50, 151, 10))
		plt.ylim([-0.2, 0.2])
		plt.grid()
		plt.xlabel(r'$\nu$ [MHz]', fontsize=14)
		plt.ylabel(r'$\Delta$ magnitude [dB]')
		plt.legend([r'Day 222(sw 2018, 50.027$\Omega$) - Day 147(sw 2018, 50.027$\Omega$)', r'Day 222(sw 2019, 50.15$\Omega$) - Day 147(sw 2018, 50.027$\Omega$)'], fontsize=8)
	
		plt.subplot(3,2,6)
		plt.plot(fe, (180/np.pi)*np.unwrap(np.angle(ra11)) - (180/np.pi)*np.unwrap(np.angle(ra1)), 'g')
		plt.plot(fe, (180/np.pi)*np.unwrap(np.angle(ra31)) - (180/np.pi)*np.unwrap(np.angle(ra1)), 'g--')
		plt.xticks(np.arange(50, 151, 10))
		plt.ylim([-2, 2])
		plt.grid()
		plt.xlabel(r'$\nu$ [MHz]', fontsize=14)
		plt.ylabel(r'$\Delta$ phase [deg]')
		
		
		
		plt.savefig(path_plot_save + 'antenna_s11.pdf', bbox_inches='tight')
		plt.close()	
		plt.close()
		plt.close()
		plt.close()
		
		
		
	if plot_number == 4:
		
		
		# Paths
		path_plot_save = edges_folder + 'plots/20190828/'
		
		GHA1 = 6
		GHA2 = 18
		
		f, t150_low_case1, w, s150_low_case1   = eg.level3_single_file_test(edges_folder + 'mid_band/spectra/level3/tests_55_150MHz/rcv18_sw18/2018_150_00.hdf5', GHA1, GHA2, 60, 150, 'no', 'LINLOG', 5, 'no', 'name')
		f, t150_low_case2, w, s150_low_case2   = eg.level3_single_file_test(edges_folder + 'mid_band/spectra/level3/tests_55_150MHz/rcv18_sw19/2018_150_00.hdf5', GHA1, GHA2, 60, 150, 'no', 'LINLOG', 5, 'no', 'name')
		f, t150_low_case3, w, s150_low_case3   = eg.level3_single_file_test(edges_folder + 'mid_band/spectra/level3/tests_55_150MHz/rcv19_sw18/2018_150_00.hdf5', GHA1, GHA2, 60, 150, 'no', 'LINLOG', 5, 'no', 'name')
		f, t150_low_case4, w, s150_low_case4   = eg.level3_single_file_test(edges_folder + 'mid_band/spectra/level3/tests_55_150MHz/rcv19_sw19/2018_150_00.hdf5', GHA1, GHA2, 60, 150, 'no', 'LINLOG', 5, 'no', 'name')
		
		#f, t150_high_case1, w, s150_high_case1 = eg.level3_single_file_test(edges_folder + 'mid_band/spectra/level3/tests_55_150MHz/rcv18_sw18/2018_150_00.hdf5', 18, 6, 60, 150, 'no', 'LINLOG', 5, 'no', 'name')	
		#f, t150_high_case2, w, s150_high_case2 = eg.level3_single_file_test(edges_folder + 'mid_band/spectra/level3/tests_55_150MHz/rcv18_sw19/2018_150_00.hdf5', 18, 6, 60, 150, 'no', 'LINLOG', 5, 'no', 'name')
		#f, t188_high_case1, w, s188_high_case1 = eg.level3_single_file_test(edges_folder + 'mid_band/spectra/level3/tests_55_150MHz/rcv18_sw18/2018_188_00.hdf5', 18, 6, 60, 150, 'no', 'LINLOG', 5, 'no', 'name')
		#f, t188_high_case2, w, s188_high_case2 = eg.level3_single_file_test(edges_folder + 'mid_band/spectra/level3/tests_55_150MHz/rcv18_sw19/2018_188_00.hdf5', 18, 6, 60, 150, 'no', 'LINLOG', 5, 'no', 'name')
		
		f1 = plt.figure(figsize=[6,5])
		
		plt.subplot(2,1,1)
		plt.plot(f, t150_low_case2 - t150_low_case1, 'b')
		#plt.plot(f, t150_low_case3 - t150_low_case1, 'b--')
		#plt.plot(f, t150_low_case4 - t150_low_case1, 'b:')
		plt.xticks(np.arange(50, 151, 10), labels='')
		plt.xlim([58, 152])
		plt.ylim([-1,2])
		plt.title('Day 2018-150, GHA=6-18 hr')
		#plt.xlabel(r'$\nu$ [MHz]', fontsize=14)
		plt.ylabel(r'$\Delta$ temperature [K]')
		plt.legend(['Case 2 - Case 1'], fontsize=9)
		

		plt.subplot(2,1,2)
		#plt.plot(f, t150_low_case2 - t150_low_case1, 'b')
		plt.plot(f, t150_low_case3 - t150_low_case1, 'b--')
		plt.plot(f, t150_low_case4 - t150_low_case1, 'b:')
		plt.xticks(np.arange(50, 151, 10))
		plt.xlim([58, 152])
		#plt.ylim([-3,3])
		#plt.title('Day 150, GHA=6-18 hr')
		plt.xlabel(r'$\nu$ [MHz]', fontsize=14)
		plt.ylabel(r'$\Delta$ temperature [K]')
		plt.legend(['Case 3 - Case 1', 'Case 4 - Case 1'], fontsize=9)
		
		plt.savefig(path_plot_save + 'delta_temperature.pdf', bbox_inches='tight')
		plt.close()	
		plt.close()
		plt.close()
		plt.close()
		



	if plot_number == 5:
		
		
		# Paths
		path_plot_save = edges_folder + 'plots/20190828/'
		
		GHA1 = 6
		GHA2 = 18
		
		#fx, t150_low_case1, w, s150_low_case1   = eg.level3_single_file_test(edges_folder + 'mid_band/spectra/level3/tests_55_150MHz/rcv18_sw18and19_ant147/2018_150_00.hdf5', GHA1, GHA2, 55, 150, 'no', 'LINLOG', 5, 'no', 'name')
		
		fx, t150_low_case1, w, s150_low_case1   = eg.level3_single_file_test(edges_folder + 'mid_band/spectra/level3/tests_55_150MHz/rcv18_sw18/2018_150_00.hdf5',             GHA1, GHA2, 55, 150, 'no', 'LINLOG', 5, 'no', 'name')
		fx, t150_low_case2, w, s150_low_case2   = eg.level3_single_file_test(edges_folder + 'mid_band/spectra/level3/tests_55_150MHz/rcv18_sw19/2018_150_00.hdf5',             GHA1, GHA2, 55, 150, 'no', 'LINLOG', 5, 'no', 'name')
		fx, t150_low_case3, w, s150_low_case3   = eg.level3_single_file_test(edges_folder + 'mid_band/spectra/level3/tests_55_150MHz/rcv19_sw18/2018_150_00.hdf5',             GHA1, GHA2, 55, 150, 'no', 'LINLOG', 5, 'no', 'name')
		fx, t150_low_case4, w, s150_low_case4   = eg.level3_single_file_test(edges_folder + 'mid_band/spectra/level3/tests_55_150MHz/rcv19_sw19/2018_150_00.hdf5',             GHA1, GHA2, 55, 150, 'no', 'LINLOG', 5, 'no', 'name')
		
		
		#f, t150_high_case1, w, s150_high_case1 = eg.level3_single_file_test(edges_folder + 'mid_band/spectra/level3/tests_55_150MHz/rcv18_sw18/2018_150_00.hdf5', 18, 6, 60, 150, 'no', 'LINLOG', 5, 'no', 'name')	
		#f, t150_high_case2, w, s150_high_case2 = eg.level3_single_file_test(edges_folder + 'mid_band/spectra/level3/tests_55_150MHz/rcv18_sw19/2018_150_00.hdf5', 18, 6, 60, 150, 'no', 'LINLOG', 5, 'no', 'name')
		#f, t188_high_case1, w, s188_high_case1 = eg.level3_single_file_test(edges_folder + 'mid_band/spectra/level3/tests_55_150MHz/rcv18_sw18/2018_188_00.hdf5', 18, 6, 60, 150, 'no', 'LINLOG', 5, 'no', 'name')
		#f, t188_high_case2, w, s188_high_case2 = eg.level3_single_file_test(edges_folder + 'mid_band/spectra/level3/tests_55_150MHz/rcv18_sw19/2018_188_00.hdf5', 18, 6, 60, 150, 'no', 'LINLOG', 5, 'no', 'name')
			
		
		FLOW  = 60
		FHIGH = 150
		
		f              = fx[(fx>=FLOW)&(fx<=FHIGH)]
		
		t150_low_case1 = t150_low_case1[(fx>=FLOW)&(fx<=FHIGH)]
		t150_low_case2 = t150_low_case2[(fx>=FLOW)&(fx<=FHIGH)]
		t150_low_case3 = t150_low_case3[(fx>=FLOW)&(fx<=FHIGH)]
		t150_low_case4 = t150_low_case4[(fx>=FLOW)&(fx<=FHIGH)]
		
		s150_low_case1 = s150_low_case1[(fx>=FLOW)&(fx<=FHIGH)]
		s150_low_case2 = s150_low_case2[(fx>=FLOW)&(fx<=FHIGH)]
		s150_low_case3 = s150_low_case3[(fx>=FLOW)&(fx<=FHIGH)]
		s150_low_case4 = s150_low_case4[(fx>=FLOW)&(fx<=FHIGH)]



		
		
		p  = ba.fit_polynomial_fourier('LINLOG', f, t150_low_case1, 5, Weights=1/(s150_low_case1**2))
		r1 = t150_low_case1 - p[1]
		
		p  = ba.fit_polynomial_fourier('LINLOG', f, t150_low_case2, 5, Weights=1/(s150_low_case2**2))
		r2 = t150_low_case2 - p[1]
		
		p  = ba.fit_polynomial_fourier('LINLOG', f, t150_low_case3, 5, Weights=1/(s150_low_case3**2))
		r3 = t150_low_case3 - p[1]
		
		p  = ba.fit_polynomial_fourier('LINLOG', f, t150_low_case4, 5, Weights=1/(s150_low_case4**2))
		r4 = t150_low_case4 - p[1]
		
		
		
		
		
		
		
		f1 = plt.figure(figsize=[8,8])
		
		plt.subplot(4,1,1)
		plt.plot(f, r1, 'b')
		plt.xticks(np.arange(50, 151, 10), labels='')
		plt.xlim([55, 152])
		plt.ylim([-1,1])
		plt.title('Day 2018-150, GHA=6-18 hr')
		#plt.xlabel(r'$\nu$ [MHz]', fontsize=14)
		plt.ylabel(r'$\Delta$ T [K]')
		plt.legend(['Case 1'], fontsize=9)
		
		
		plt.subplot(4,1,2)
		plt.plot(f, r2, 'b')
		plt.xticks(np.arange(50, 151, 10), labels='')
		plt.xlim([55, 152])
		plt.ylim([-1,1])
		#plt.title('Day 2018-150, GHA=6-18 hr')
		#plt.xlabel(r'$\nu$ [MHz]', fontsize=14)
		plt.ylabel(r'$\Delta$ T [K]')
		plt.legend(['Case 2'], fontsize=9)
		
		
		plt.subplot(4,1,3)
		plt.plot(f, r3, 'b')
		plt.xticks(np.arange(50, 151, 10), labels='')
		plt.xlim([55, 152])
		plt.ylim([-1,1])
		#plt.title('Day 2018-150, GHA=6-18 hr')
		#plt.xlabel(r'$\nu$ [MHz]', fontsize=14)
		plt.ylabel(r'$\Delta$ T [K]')
		plt.legend(['Case 3'], fontsize=9)
		
		
		plt.subplot(4,1,4)
		plt.plot(f, r4, 'b')
		plt.xticks(np.arange(50, 151, 10))
		plt.xlim([55, 152])
		plt.ylim([-1,1])
		#plt.title('Day 2018-150, GHA=6-18 hr')
		plt.xlabel(r'$\nu$ [MHz]', fontsize=14)
		plt.ylabel(r'$\Delta$ T [K]')
		plt.legend(['Case 4'], fontsize=9)
				
		
		#plt.subplot(2,1,2)
		##plt.plot(f, t150_low_case2 - t150_low_case1, 'b')
		#plt.plot(f, t150_low_case3 - t150_low_case1, 'b--')
		#plt.plot(f, t150_low_case4 - t150_low_case1, 'b:')
		#plt.xticks(np.arange(50, 151, 10))
		#plt.xlim([58, 152])
		##plt.ylim([-3,3])
		##plt.title('Day 150, GHA=6-18 hr')
		#plt.xlabel(r'$\nu$ [MHz]', fontsize=14)
		#plt.ylabel(r'$\Delta$ temperature [K]')
		#plt.legend(['Case 3 - Case 1', 'Case 4 - Case 1'], fontsize=9)
		
		
		plt.savefig(path_plot_save + 'residuals1.pdf', bbox_inches='tight')
		plt.close()	
		plt.close()
		plt.close()
		plt.close()











	if plot_number == 6:
		
		
		# Paths
		path_plot_save = edges_folder + 'plots/20190828/'
		
		GHA1 = 6
		GHA2 = 18
		
		#fx, t150_low_case1, w, s150_low_case1   = eg.level3_single_file_test(edges_folder + 'mid_band/spectra/level3/tests_55_150MHz/rcv18_sw18and19_ant147/2018_150_00.hdf5', GHA1, GHA2, 55, 150, 'no', 'LINLOG', 5, 'no', 'name')
		
		fx, t188_low_case1, w, s188_low_case1   = eg.level3_single_file_test(edges_folder + 'mid_band/spectra/level3/tests_55_150MHz/rcv18_sw18/2018_188_00.hdf5',             GHA1, GHA2, 55, 150, 'no', 'LINLOG', 5, 'no', 'name')
		fx, t188_low_case2, w, s188_low_case2   = eg.level3_single_file_test(edges_folder + 'mid_band/spectra/level3/tests_55_150MHz/rcv18_sw19/2018_188_00.hdf5',             GHA1, GHA2, 55, 150, 'no', 'LINLOG', 5, 'no', 'name')
		fx, t188_low_case3, w, s188_low_case3   = eg.level3_single_file_test(edges_folder + 'mid_band/spectra/level3/tests_55_150MHz/rcv19_sw18/2018_188_00.hdf5',             GHA1, GHA2, 55, 150, 'no', 'LINLOG', 5, 'no', 'name')
		fx, t188_low_case4, w, s188_low_case4   = eg.level3_single_file_test(edges_folder + 'mid_band/spectra/level3/tests_55_150MHz/rcv19_sw19/2018_188_00.hdf5',             GHA1, GHA2, 55, 150, 'no', 'LINLOG', 5, 'no', 'name')
		
		
		#f, t150_high_case1, w, s150_high_case1 = eg.level3_single_file_test(edges_folder + 'mid_band/spectra/level3/tests_55_150MHz/rcv18_sw18/2018_150_00.hdf5', 18, 6, 60, 150, 'no', 'LINLOG', 5, 'no', 'name')	
		#f, t150_high_case2, w, s150_high_case2 = eg.level3_single_file_test(edges_folder + 'mid_band/spectra/level3/tests_55_150MHz/rcv18_sw19/2018_150_00.hdf5', 18, 6, 60, 150, 'no', 'LINLOG', 5, 'no', 'name')
		#f, t188_high_case1, w, s188_high_case1 = eg.level3_single_file_test(edges_folder + 'mid_band/spectra/level3/tests_55_150MHz/rcv18_sw18/2018_188_00.hdf5', 18, 6, 60, 150, 'no', 'LINLOG', 5, 'no', 'name')
		#f, t188_high_case2, w, s188_high_case2 = eg.level3_single_file_test(edges_folder + 'mid_band/spectra/level3/tests_55_150MHz/rcv18_sw19/2018_188_00.hdf5', 18, 6, 60, 150, 'no', 'LINLOG', 5, 'no', 'name')
			
		
		FLOW  = 60
		FHIGH = 150
		
		f              = fx[(fx>=FLOW)&(fx<=FHIGH)]
		
		t188_low_case1 = t188_low_case1[(fx>=FLOW)&(fx<=FHIGH)]
		t188_low_case2 = t188_low_case2[(fx>=FLOW)&(fx<=FHIGH)]
		t188_low_case3 = t188_low_case3[(fx>=FLOW)&(fx<=FHIGH)]
		t188_low_case4 = t188_low_case4[(fx>=FLOW)&(fx<=FHIGH)]
		
		s188_low_case1 = s188_low_case1[(fx>=FLOW)&(fx<=FHIGH)]
		s188_low_case2 = s188_low_case2[(fx>=FLOW)&(fx<=FHIGH)]
		s188_low_case3 = s188_low_case3[(fx>=FLOW)&(fx<=FHIGH)]
		s188_low_case4 = s188_low_case4[(fx>=FLOW)&(fx<=FHIGH)]



		
		
		p  = ba.fit_polynomial_fourier('LINLOG', f, t188_low_case1, 5, Weights=1/(s188_low_case1**2))
		r1 = t188_low_case1 - p[1]
		
		p  = ba.fit_polynomial_fourier('LINLOG', f, t188_low_case2, 5, Weights=1/(s188_low_case2**2))
		r2 = t188_low_case2 - p[1]
		
		p  = ba.fit_polynomial_fourier('LINLOG', f, t188_low_case3, 5, Weights=1/(s188_low_case3**2))
		r3 = t188_low_case3 - p[1]
		
		p  = ba.fit_polynomial_fourier('LINLOG', f, t188_low_case4, 5, Weights=1/(s188_low_case4**2))
		r4 = t188_low_case4 - p[1]
		
		
		
		
		
		
		
		f1 = plt.figure(figsize=[8,8])
		
		plt.subplot(4,1,1)
		plt.plot(f, r1, 'b')
		plt.xticks(np.arange(50, 151, 10), labels='')
		plt.xlim([55, 152])
		plt.ylim([-1,1])
		plt.title('Day 2018-188, GHA=6-18 hr')
		#plt.xlabel(r'$\nu$ [MHz]', fontsize=14)
		plt.ylabel(r'$\Delta$ T [K]')
		plt.legend(['Case 1'], fontsize=9)
		
		
		plt.subplot(4,1,2)
		plt.plot(f, r2, 'b')
		plt.xticks(np.arange(50, 151, 10), labels='')
		plt.xlim([55, 152])
		plt.ylim([-1,1])
		#plt.title('Day 2018-150, GHA=6-18 hr')
		#plt.xlabel(r'$\nu$ [MHz]', fontsize=14)
		plt.ylabel(r'$\Delta$ T [K]')
		plt.legend(['Case 2'], fontsize=9)
		
		
		plt.subplot(4,1,3)
		plt.plot(f, r3, 'b')
		plt.xticks(np.arange(50, 151, 10), labels='')
		plt.xlim([55, 152])
		plt.ylim([-1,1])
		#plt.title('Day 2018-150, GHA=6-18 hr')
		#plt.xlabel(r'$\nu$ [MHz]', fontsize=14)
		plt.ylabel(r'$\Delta$ T [K]')
		plt.legend(['Case 3'], fontsize=9)
		
		
		plt.subplot(4,1,4)
		plt.plot(f, r4, 'b')
		plt.xticks(np.arange(50, 151, 10))
		plt.xlim([55, 152])
		plt.ylim([-1,1])
		#plt.title('Day 2018-150, GHA=6-18 hr')
		plt.xlabel(r'$\nu$ [MHz]', fontsize=14)
		plt.ylabel(r'$\Delta$ T [K]')
		plt.legend(['Case 4'], fontsize=9)
				
		
		#plt.subplot(2,1,2)
		##plt.plot(f, t150_low_case2 - t150_low_case1, 'b')
		#plt.plot(f, t150_low_case3 - t150_low_case1, 'b--')
		#plt.plot(f, t150_low_case4 - t150_low_case1, 'b:')
		#plt.xticks(np.arange(50, 151, 10))
		#plt.xlim([58, 152])
		##plt.ylim([-3,3])
		##plt.title('Day 150, GHA=6-18 hr')
		#plt.xlabel(r'$\nu$ [MHz]', fontsize=14)
		#plt.ylabel(r'$\Delta$ temperature [K]')
		#plt.legend(['Case 3 - Case 1', 'Case 4 - Case 1'], fontsize=9)
		
		
		plt.savefig(path_plot_save + 'residuals2.pdf', bbox_inches='tight')
		plt.close()	
		plt.close()
		plt.close()
		plt.close()







	if plot_number == 7:
		
		
		# Paths
		path_plot_save = edges_folder + 'plots/20190828/'
		
		GHA1 = 6
		GHA2 = 18
		
		#fx, t150_low_case1, w, s150_low_case1   = eg.level3_single_file_test(edges_folder + 'mid_band/spectra/level3/tests_55_150MHz/rcv18_sw18and19_ant147/2018_150_00.hdf5', GHA1, GHA2, 55, 150, 'no', 'LINLOG', 5, 'no', 'name')
		
		fx, t150_low_case1, w, s150_low_case1   = eg.level3_single_file_test(edges_folder + 'mid_band/spectra/level3/tests_55_150MHz/rcv18_sw18/2018_150_00.hdf5',             GHA1, GHA2, 55, 150, 'no', 'LINLOG', 5, 'no', 'name')
		fx, t150_low_case2, w, s150_low_case2   = eg.level3_single_file_test(edges_folder + 'mid_band/spectra/level3/tests_55_150MHz/rcv18_sw19/2018_150_00.hdf5',             GHA1, GHA2, 55, 150, 'no', 'LINLOG', 5, 'no', 'name')
		fx, t150_low_case3, w, s150_low_case3   = eg.level3_single_file_test(edges_folder + 'mid_band/spectra/level3/tests_55_150MHz/rcv18_sw18and19_ant147/2018_150_00.hdf5',             GHA1, GHA2, 55, 150, 'no', 'LINLOG', 5, 'no', 'name')
		
		
		#f, t150_high_case1, w, s150_high_case1 = eg.level3_single_file_test(edges_folder + 'mid_band/spectra/level3/tests_55_150MHz/rcv18_sw18/2018_150_00.hdf5', 18, 6, 60, 150, 'no', 'LINLOG', 5, 'no', 'name')	
		#f, t150_high_case2, w, s150_high_case2 = eg.level3_single_file_test(edges_folder + 'mid_band/spectra/level3/tests_55_150MHz/rcv18_sw19/2018_150_00.hdf5', 18, 6, 60, 150, 'no', 'LINLOG', 5, 'no', 'name')
		#f, t188_high_case1, w, s188_high_case1 = eg.level3_single_file_test(edges_folder + 'mid_band/spectra/level3/tests_55_150MHz/rcv18_sw18/2018_188_00.hdf5', 18, 6, 60, 150, 'no', 'LINLOG', 5, 'no', 'name')
		#f, t188_high_case2, w, s188_high_case2 = eg.level3_single_file_test(edges_folder + 'mid_band/spectra/level3/tests_55_150MHz/rcv18_sw19/2018_188_00.hdf5', 18, 6, 60, 150, 'no', 'LINLOG', 5, 'no', 'name')
			
		
		FLOW  = 60
		FHIGH = 150
		
		f              = fx[(fx>=FLOW)&(fx<=FHIGH)]
		
		t150_low_case1 = t150_low_case1[(fx>=FLOW)&(fx<=FHIGH)]
		t150_low_case2 = t150_low_case2[(fx>=FLOW)&(fx<=FHIGH)]
		t150_low_case3 = t150_low_case3[(fx>=FLOW)&(fx<=FHIGH)]
		
		s150_low_case1 = s150_low_case1[(fx>=FLOW)&(fx<=FHIGH)]
		s150_low_case2 = s150_low_case2[(fx>=FLOW)&(fx<=FHIGH)]
		s150_low_case3 = s150_low_case3[(fx>=FLOW)&(fx<=FHIGH)]



		
		
		p  = ba.fit_polynomial_fourier('LINLOG', f, t150_low_case1, 5, Weights=1/(s150_low_case1**2))
		r1 = t150_low_case1 - p[1]
		
		p  = ba.fit_polynomial_fourier('LINLOG', f, t150_low_case2, 5, Weights=1/(s150_low_case2**2))
		r2 = t150_low_case2 - p[1]
		
		p  = ba.fit_polynomial_fourier('LINLOG', f, t150_low_case3, 5, Weights=1/(s150_low_case3**2))
		r3 = t150_low_case3 - p[1]
		
		
		
		
		
		
		
		
		f1 = plt.figure(figsize=[8,8])
		
		plt.subplot(3,1,1)
		plt.plot(f, r1, 'b')
		plt.xticks(np.arange(50, 151, 10), labels='')
		plt.xlim([55, 152])
		plt.ylim([-1,1])
		plt.title('Day 2018-150, GHA=6-18 hr')
		#plt.xlabel(r'$\nu$ [MHz]', fontsize=14)
		plt.ylabel(r'$\Delta$ T [K]')
		plt.legend(['Case 1'], fontsize=9)
		
		
		plt.subplot(3,1,2)
		plt.plot(f, r2, 'b')
		plt.xticks(np.arange(50, 151, 10), labels='')
		plt.xlim([55, 152])
		plt.ylim([-1,1])
		#plt.title('Day 2018-150, GHA=6-18 hr')
		#plt.xlabel(r'$\nu$ [MHz]', fontsize=14)
		plt.ylabel(r'$\Delta$ T [K]')
		plt.legend(['Case 2'], fontsize=9)
		
		
		plt.subplot(3,1,3)
		plt.plot(f, r3, 'b')
		plt.xticks(np.arange(50, 151, 10))
		plt.xlim([55, 152])
		plt.ylim([-1,1])
		#plt.title('Day 2018-150, GHA=6-18 hr')
		plt.xlabel(r'$\nu$ [MHz]', fontsize=14)
		plt.ylabel(r'$\Delta$ T [K]')
		plt.legend(['average sw 2018 & 2019'], fontsize=9)
		
		
		#plt.subplot(4,1,4)
		#plt.plot(f, r4, 'b')
		#plt.xticks(np.arange(50, 151, 10))
		#plt.xlim([55, 152])
		#plt.ylim([-1,1])
		##plt.title('Day 2018-150, GHA=6-18 hr')
		#plt.xlabel(r'$\nu$ [MHz]', fontsize=14)
		#plt.ylabel(r'$\Delta$ T [K]')
		#plt.legend(['Case 4'], fontsize=9)
				
		
		#plt.subplot(2,1,2)
		##plt.plot(f, t150_low_case2 - t150_low_case1, 'b')
		#plt.plot(f, t150_low_case3 - t150_low_case1, 'b--')
		#plt.plot(f, t150_low_case4 - t150_low_case1, 'b:')
		#plt.xticks(np.arange(50, 151, 10))
		#plt.xlim([58, 152])
		##plt.ylim([-3,3])
		##plt.title('Day 150, GHA=6-18 hr')
		#plt.xlabel(r'$\nu$ [MHz]', fontsize=14)
		#plt.ylabel(r'$\Delta$ temperature [K]')
		#plt.legend(['Case 3 - Case 1', 'Case 4 - Case 1'], fontsize=9)
		
		
		plt.savefig(path_plot_save + 'residuals3.pdf', bbox_inches='tight')
		plt.close()	
		plt.close()
		plt.close()
		plt.close()























	if plot_number == 8:
		
		
		# Paths
		path_plot_save = edges_folder + 'plots/20190828/'
		
		GHA1 = 6
		GHA2 = 18
		
		#fx, t150_low_case1, w, s150_low_case1   = eg.level3_single_file_test(edges_folder + 'mid_band/spectra/level3/tests_55_150MHz/rcv18_sw18and19_ant147/2018_150_00.hdf5', GHA1, GHA2, 55, 150, 'no', 'LINLOG', 5, 'no', 'name')
		
		fx, t188_low_case1, w, s188_low_case1   = eg.level3_single_file_test(edges_folder + 'mid_band/spectra/level3/tests_55_150MHz/rcv18_sw18/2018_188_00.hdf5',             GHA1, GHA2, 55, 150, 'no', 'LINLOG', 5, 'no', 'name')
		fx, t188_low_case2, w, s188_low_case2   = eg.level3_single_file_test(edges_folder + 'mid_band/spectra/level3/tests_55_150MHz/rcv18_sw19/2018_188_00.hdf5',             GHA1, GHA2, 55, 150, 'no', 'LINLOG', 5, 'no', 'name')
		fx, t188_low_case3, w, s188_low_case3   = eg.level3_single_file_test(edges_folder + 'mid_band/spectra/level3/tests_55_150MHz/rcv18_sw18and19_ant147/2018_188_00.hdf5',             GHA1, GHA2, 55, 150, 'no', 'LINLOG', 5, 'no', 'name')
		
		
		#f, t150_high_case1, w, s150_high_case1 = eg.level3_single_file_test(edges_folder + 'mid_band/spectra/level3/tests_55_150MHz/rcv18_sw18/2018_150_00.hdf5', 18, 6, 60, 150, 'no', 'LINLOG', 5, 'no', 'name')	
		#f, t150_high_case2, w, s150_high_case2 = eg.level3_single_file_test(edges_folder + 'mid_band/spectra/level3/tests_55_150MHz/rcv18_sw19/2018_150_00.hdf5', 18, 6, 60, 150, 'no', 'LINLOG', 5, 'no', 'name')
		#f, t188_high_case1, w, s188_high_case1 = eg.level3_single_file_test(edges_folder + 'mid_band/spectra/level3/tests_55_150MHz/rcv18_sw18/2018_188_00.hdf5', 18, 6, 60, 150, 'no', 'LINLOG', 5, 'no', 'name')
		#f, t188_high_case2, w, s188_high_case2 = eg.level3_single_file_test(edges_folder + 'mid_band/spectra/level3/tests_55_150MHz/rcv18_sw19/2018_188_00.hdf5', 18, 6, 60, 150, 'no', 'LINLOG', 5, 'no', 'name')
			
		
		FLOW  = 60
		FHIGH = 150
		
		f              = fx[(fx>=FLOW)&(fx<=FHIGH)]
		
		t188_low_case1 = t188_low_case1[(fx>=FLOW)&(fx<=FHIGH)]
		t188_low_case2 = t188_low_case2[(fx>=FLOW)&(fx<=FHIGH)]
		t188_low_case3 = t188_low_case3[(fx>=FLOW)&(fx<=FHIGH)]
		
		s188_low_case1 = s188_low_case1[(fx>=FLOW)&(fx<=FHIGH)]
		s188_low_case2 = s188_low_case2[(fx>=FLOW)&(fx<=FHIGH)]
		s188_low_case3 = s188_low_case3[(fx>=FLOW)&(fx<=FHIGH)]



		
		
		p  = ba.fit_polynomial_fourier('LINLOG', f, t188_low_case1, 5, Weights=1/(s188_low_case1**2))
		r1 = t188_low_case1 - p[1]
		
		p  = ba.fit_polynomial_fourier('LINLOG', f, t188_low_case2, 5, Weights=1/(s188_low_case2**2))
		r2 = t188_low_case2 - p[1]
		
		p  = ba.fit_polynomial_fourier('LINLOG', f, t188_low_case3, 5, Weights=1/(s188_low_case3**2))
		r3 = t188_low_case3 - p[1]
		
		
		
		
		
		
		
		
		f1 = plt.figure(figsize=[8,8])
		
		plt.subplot(3,1,1)
		plt.plot(f, r1, 'b')
		plt.xticks(np.arange(50, 151, 10), labels='')
		plt.xlim([55, 152])
		plt.ylim([-1,1])
		plt.title('Day 2018-188, GHA=6-18 hr')
		#plt.xlabel(r'$\nu$ [MHz]', fontsize=14)
		plt.ylabel(r'$\Delta$ T [K]')
		plt.legend(['Case 1'], fontsize=9)
		
		
		plt.subplot(3,1,2)
		plt.plot(f, r2, 'b')
		plt.xticks(np.arange(50, 151, 10), labels='')
		plt.xlim([55, 152])
		plt.ylim([-1,1])
		#plt.title('Day 2018-150, GHA=6-18 hr')
		#plt.xlabel(r'$\nu$ [MHz]', fontsize=14)
		plt.ylabel(r'$\Delta$ T [K]')
		plt.legend(['Case 2'], fontsize=9)
		
		
		plt.subplot(3,1,3)
		plt.plot(f, r3, 'b')
		plt.xticks(np.arange(50, 151, 10))
		plt.xlim([55, 152])
		plt.ylim([-1,1])
		#plt.title('Day 2018-150, GHA=6-18 hr')
		plt.xlabel(r'$\nu$ [MHz]', fontsize=14)
		plt.ylabel(r'$\Delta$ T [K]')
		plt.legend(['average sw 2018 & 2019'], fontsize=9)
		
		
		#plt.subplot(4,1,4)
		#plt.plot(f, r4, 'b')
		#plt.xticks(np.arange(50, 151, 10))
		#plt.xlim([55, 152])
		#plt.ylim([-1,1])
		##plt.title('Day 2018-150, GHA=6-18 hr')
		#plt.xlabel(r'$\nu$ [MHz]', fontsize=14)
		#plt.ylabel(r'$\Delta$ T [K]')
		#plt.legend(['Case 4'], fontsize=9)
				
		
		#plt.subplot(2,1,2)
		##plt.plot(f, t150_low_case2 - t150_low_case1, 'b')
		#plt.plot(f, t150_low_case3 - t150_low_case1, 'b--')
		#plt.plot(f, t150_low_case4 - t150_low_case1, 'b:')
		#plt.xticks(np.arange(50, 151, 10))
		#plt.xlim([58, 152])
		##plt.ylim([-3,3])
		##plt.title('Day 150, GHA=6-18 hr')
		#plt.xlabel(r'$\nu$ [MHz]', fontsize=14)
		#plt.ylabel(r'$\Delta$ temperature [K]')
		#plt.legend(['Case 3 - Case 1', 'Case 4 - Case 1'], fontsize=9)
		
		
		plt.savefig(path_plot_save + 'residuals4.pdf', bbox_inches='tight')
		plt.close()	
		plt.close()
		plt.close()
		plt.close()







	
	return 0





def plots_of_absorption_glitch(part_number):
	
	
	if part_number == 1:

		# ########################################################
		fmin = 50
		fmax = 200
		
		fmin_res = 50
		fmax_res = 120
		
		Nfg = 5
		
		
		
		el             = np.arange(0,91) 
		sin_theta      = np.sin((90-el)*(np.pi/180)) 
		sin_theta_2D_T = np.tile(sin_theta, (360, 1))
		sin_theta_2D   = sin_theta_2D_T.T			
		
		



		# High-Band Blade on Soil, no GP
		b_all = oeg.FEKO_high_band_blade_beam_plus_shaped_finite_ground_plane(beam_file=21, frequency_interpolation='no', frequency=np.array([0]), AZ_antenna_axis=0)
		f = np.arange(65, 201, 1)		
	
		bint = np.zeros(len(f))
		for i in range(len(f)):
			
			b       = b_all[i,:,:]
			bint[i] = np.sum(b * sin_theta_2D)
		
		ft  = f[(f>=fmin) & (f<=fmax)]
		bx  = bint[(f>=fmin) & (f<=fmax)]
		bt   = (1/(4*np.pi)) * ((np.pi/180)**2)*bx   #/np.mean(bx)
		
		fr  = ft[(ft>=fmin_res) & (ft<=fmax_res)]
		br  = bt[(ft>=fmin_res) & (ft<=fmax_res)]
		
		x   = np.polyfit(fr, br, Nfg-1)
		m   = np.polyval(x, fr)
		rr  = br - m
		
		
		ft1 = np.copy(ft)
		bt1 = np.copy(bt)
		
		fr1 = np.copy(fr)
		rr1 = np.copy(rr)


		
		# High-Band Fourpoint on Plus-Sign GP
		b_all = oeg.FEKO_high_band_fourpoint_beam(2, frequency_interpolation='no', frequency=np.array([0]), AZ_antenna_axis=0)
		f = np.arange(65, 201, 1)
		
		
		bint = np.zeros(len(f))
		for i in range(len(f)):
			
			b       = b_all[i,:,:]
			bint[i] = np.sum(b * sin_theta_2D)
		
		ft  = f[(f>=fmin) & (f<=fmax)]
		bx  = bint[(f>=fmin) & (f<=fmax)]
		bt   = (1/(4*np.pi)) * ((np.pi/180)**2)*bx   #/np.mean(bx)
		
		fr  = ft[(ft>=fmin_res) & (ft<=fmax_res)]
		br  = bt[(ft>=fmin_res) & (ft<=fmax_res)]
		
		x   = np.polyfit(fr, br, Nfg-1)
		m   = np.polyval(x, fr)
		rr  = br - m
		
		
		ft2 = np.copy(ft)
		bt2 = np.copy(bt)
		
		fr2 = np.copy(fr)
		rr2 = np.copy(rr)
		
		
		
		# High-Band Blade on Plus-Sign GP
		b_all = oeg.FEKO_high_band_blade_beam_plus_shaped_finite_ground_plane(beam_file=20, frequency_interpolation='no', frequency=np.array([0]), AZ_antenna_axis=0)
		f = np.arange(65, 201, 1)		
	
		bint = np.zeros(len(f))
		for i in range(len(f)):
			
			b       = b_all[i,:,:]
			bint[i] = np.sum(b * sin_theta_2D)
		
		ft  = f[(f>=fmin) & (f<=fmax)]
		bx  = bint[(f>=fmin) & (f<=fmax)]
		bt   = (1/(4*np.pi)) * ((np.pi/180)**2)*bx   #/np.mean(bx)
		
		fr  = ft[(ft>=fmin_res) & (ft<=fmax_res)]
		br  = bt[(ft>=fmin_res) & (ft<=fmax_res)]
		
		x   = np.polyfit(fr, br, Nfg-1)
		m   = np.polyval(x, fr)
		rr  = br - m
		
		
		ft3 = np.copy(ft)
		bt3 = np.copy(bt)
		
		fr3 = np.copy(fr)
		rr3 = np.copy(rr)
			
	
	
	
	
		# Low-Band 3, Blade on Plus-Sign GP
		b_all = cal.FEKO_blade_beam('low_band3', 1, frequency_interpolation='no', AZ_antenna_axis=90)
		f     = np.arange(50, 121, 2)	
	
	
		bint = np.zeros(len(f))
		for i in range(len(f)):
			
			b       = b_all[i,:,:]
			bint[i] = np.sum(b * sin_theta_2D)
		
		ft  = f[(f>=fmin) & (f<=fmax)]
		bx  = bint[(f>=fmin) & (f<=fmax)]
		bt   = (1/(4*np.pi)) * ((np.pi/180)**2)*bx   #/np.mean(bx)
		
		fr  = ft[(ft>=fmin_res) & (ft<=fmax_res)]
		br  = bt[(ft>=fmin_res) & (ft<=fmax_res)]
		
		x   = np.polyfit(fr, br, Nfg-1)
		m   = np.polyval(x, fr)
		rr  = br - m
		
		
		ft4 = np.copy(ft)
		bt4 = np.copy(bt)
		
		fr4 = np.copy(fr)
		rr4 = np.copy(rr)
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		# Figure 2
		# #########################################################################################
		
		
		
		
		# Mid-Band, Blade, infinite GP
		b_all = cal.FEKO_blade_beam('mid_band', 1, frequency_interpolation='no', AZ_antenna_axis=90)
		f     = np.arange(50,201,2)		
		
		bint = np.zeros(len(f))
		for i in range(len(f)):
			
			b       = b_all[i,:,:]
			bint[i] = np.sum(b * sin_theta_2D)
		
		ft  = f[(f>=fmin) & (f<=fmax)]
		bx  = bint[(f>=fmin) & (f<=fmax)]
		bt   = (1/(4*np.pi)) * ((np.pi/180)**2)*bx   #/np.mean(bx)
		
		fr  = ft[(ft>=fmin_res) & (ft<=fmax_res)]
		br  = bt[(ft>=fmin_res) & (ft<=fmax_res)]
		
		x   = np.polyfit(fr, br, Nfg-1)
		m   = np.polyval(x, fr)
		rr  = br - m
		
		
		ft5 = np.copy(ft)
		bt5 = np.copy(bt)
		
		fr5 = np.copy(fr)
		rr5 = np.copy(rr)			
		
		
		
		
		# Low-Band, Blade on 10m x 10m GP
		b_all  = oeg.FEKO_low_band_blade_beam(beam_file=5, frequency_interpolation='no', frequency=np.array([0]), AZ_antenna_axis=0)
		f      = np.arange(50,121,2)
		
		
		bint = np.zeros(len(f))
		for i in range(len(f)):
			
			b       = b_all[i,:,:]
			bint[i] = np.sum(b * sin_theta_2D)
		
		ft  = f[(f>=fmin) & (f<=fmax)]
		bx  = bint[(f>=fmin) & (f<=fmax)]
		bt   = (1/(4*np.pi)) * ((np.pi/180)**2)*bx   #/np.mean(bx)
		
		fr  = ft[(ft>=fmin_res) & (ft<=fmax_res)]
		br  = bt[(ft>=fmin_res) & (ft<=fmax_res)]
		
		x   = np.polyfit(fr, br, Nfg-1)
		m   = np.polyval(x, fr)
		rr  = br - m
		
		
		ft6 = np.copy(ft)
		bt6 = np.copy(bt)
		
		fr6 = np.copy(fr)
		rr6 = np.copy(rr)		
		
		
		
		
		
		# Low-Band, Blade on 30m x 30m GP
		b_all = oeg.FEKO_low_band_blade_beam(beam_file=2, frequency_interpolation='no', AZ_antenna_axis=0)
		f     = np.arange(40,121,2)		
		
		bint = np.zeros(len(f))
		for i in range(len(f)):
			
			b       = b_all[i,:,:]
			bint[i] = np.sum(b * sin_theta_2D)
		
		ft  = f[(f>=fmin) & (f<=fmax)]
		bx  = bint[(f>=fmin) & (f<=fmax)]
		bt   = (1/(4*np.pi)) * ((np.pi/180)**2)*bx   #/np.mean(bx)
		
		fr  = ft[(ft>=fmin_res) & (ft<=fmax_res)]
		br  = bt[(ft>=fmin_res) & (ft<=fmax_res)]
		
		x   = np.polyfit(fr, br, Nfg-1)
		m   = np.polyval(x, fr)
		rr  = br - m
		
		
		ft7 = np.copy(ft)
		bt7 = np.copy(bt)
		
		fr7 = np.copy(fr)
		rr7 = np.copy(rr)				
		
		
		
		
		
		# Low-Band, Blade on 30m x 30m GP, NIVEDITA
		b_all = oeg.FEKO_low_band_blade_beam(beam_file=0, frequency_interpolation='no', AZ_antenna_axis=0)
		f     = np.arange(40,101,2)		
		
		bint = np.zeros(len(f))
		for i in range(len(f)):
			
			b       = b_all[i,:,:]
			bint[i] = np.sum(b * sin_theta_2D)
		
		ft  = f[(f>=fmin) & (f<=fmax)]
		bx  = bint[(f>=fmin) & (f<=fmax)]
		bt   = (1/(4*np.pi)) * ((np.pi/180)**2)*bx   #/np.mean(bx)
		
		fr  = ft[(ft>=fmin_res) & (ft<=fmax_res)]
		br  = bt[(ft>=fmin_res) & (ft<=fmax_res)]
		
		x   = np.polyfit(fr, br, Nfg-1)
		m   = np.polyval(x, fr)
		rr  = br - m
		
		
		ft8 = np.copy(ft)
		bt8 = np.copy(bt)
		
		fr8 = np.copy(fr)
		rr8 = np.copy(rr)			
		
		
		
		
		
		
		
		
		
		# Mid-Band, Blade, on 30m x 30m GP
		b_all = cal.FEKO_blade_beam('mid_band', 0, frequency_interpolation='no', AZ_antenna_axis=90)
		f     = np.arange(50,201,2)		
		
		bint = np.zeros(len(f))
		for i in range(len(f)):
			
			b       = b_all[i,:,:]
			bint[i] = np.sum(b * sin_theta_2D)
		
		ft  = f[(f>=fmin) & (f<=fmax)]
		bx  = bint[(f>=fmin) & (f<=fmax)]
		bt   = (1/(4*np.pi)) * ((np.pi/180)**2)*bx   #/np.mean(bx)
		
		fr  = ft[(ft>=fmin_res) & (ft<=fmax_res)]
		br  = bt[(ft>=fmin_res) & (ft<=fmax_res)]
		
		x   = np.polyfit(fr, br, Nfg-1)
		m   = np.polyval(x, fr)
		rr  = br - m
		
		
		ft9 = np.copy(ft)
		bt9 = np.copy(bt)
		
		fr9 = np.copy(fr)
		rr9 = np.copy(rr)		
		
		
		
		
		
		# Mid-Band, Blade, on 30m x 30m GP, NIVEDITA
		b_all = cal.FEKO_blade_beam('mid_band', 100, frequency_interpolation='no', AZ_antenna_axis=90)
		f     = np.arange(60,201,2)
		
		bint = np.zeros(len(f))
		for i in range(len(f)):
			
			b       = b_all[i,:,:]
			bint[i] = np.sum(b * sin_theta_2D)
		
		ft  = f[(f>=fmin) & (f<=fmax)]
		bx  = bint[(f>=fmin) & (f<=fmax)]
		bt   = (1/(4*np.pi)) * ((np.pi/180)**2)*bx   #/np.mean(bx)
		
		fr  = ft[(ft>=fmin_res) & (ft<=fmax_res)]
		br  = bt[(ft>=fmin_res) & (ft<=fmax_res)]
		
		x   = np.polyfit(fr, br, Nfg-1)
		m   = np.polyval(x, fr)
		rr  = br - m
		
		
		ft10 = np.copy(ft)
		bt10 = np.copy(bt)
		
		fr10 = np.copy(fr)
		rr10 = np.copy(rr)		
		
		
		
		
		
		
		
		
		
		
	
	
	
	
		plt.figure(1)
	
		plt.subplot(4,2,1)
		plt.plot(ft1, bt1)
		plt.xlim([45, 205])
		plt.title(r'Integrated gain above horizon / $4\pi$')
		plt.ylabel('High-Band Blade\n no GP')
		
		plt.subplot(4,2,2)
		plt.plot(fr1, rr1)
		plt.xlim([45, 125])
		plt.ylim(-0.00029, 0.00029)
		plt.title('Residuals')
		
		
		plt.subplot(4,2,3)
		plt.plot(ft2, bt2)
		plt.xlim([45, 205])
		plt.ylabel('High-Band Fourpoint\n Plus-sign GP')
		
		plt.subplot(4,2,4)
		plt.plot(fr2, rr2)		
		plt.xlim([45, 125])
		plt.ylim(-0.00029, 0.00029)
		
		plt.subplot(4,2,5)
		plt.plot(ft3, bt3)
		plt.xlim([45, 205])
		plt.ylabel('High-Band Blade\n Plus-sign GP')
		
		plt.subplot(4,2,6)
		plt.plot(fr3, rr3)		
		plt.xlim([45, 125])
		plt.ylim(-0.00029, 0.00029)
			
		plt.subplot(4,2,7)
		plt.plot(ft4, bt4)
		plt.xlim([45, 205])
		plt.ylabel('Low-Band 3 Blade\n Plus-sign GP')
		plt.xlabel('frequency [MHz]')
		
		plt.subplot(4,2,8)
		plt.plot(fr4, rr4)			
		plt.xlim([45, 125])
		plt.ylim(-0.00029, 0.00029)
		plt.xlabel('frequency [MHz]')
		
		
		
	
	
	
	
	
		plt.figure(2)
	
		plt.subplot(6,2,1)
		plt.plot(ft5, bt5)
		plt.xlim([45, 205])		
		plt.title(r'Integrated gain above horizon / $4\pi$')
		plt.ylabel('Mid-Band Blade\n Infinite GP')
		
		plt.subplot(6,2,2)
		plt.plot(fr5, rr5)
		plt.xlim([45, 125])
		plt.ylim(-0.00029, 0.00029)
		plt.title('Residuals')
		
		plt.subplot(6,2,3)
		plt.plot(ft6, bt6)
		plt.xlim([45, 205])
		plt.ylabel('Low-Band Blade\n 10m x 10m GP')
		
		plt.subplot(6,2,4)
		plt.plot(fr6, rr6)
		plt.ylim(-0.00029, 0.00029)
		plt.xlim([45, 125])			
		
		
		plt.subplot(6,2,5)
		plt.plot(ft7, bt7)
		plt.xlim([45, 205])
		plt.ylabel('Low-Band Blade\n 30m x 30m GP')
		
		plt.subplot(6,2,6)
		plt.plot(fr7, rr7)
		plt.ylim(-0.00029, 0.00029)
		plt.xlim([45, 125])
		
		
		plt.subplot(6,2,7)
		plt.plot(ft8, bt8)
		plt.xlim([45, 205])
		plt.ylabel('Low-Band Blade\n 30m x 30m GP\n NIVEDITA')
		
		plt.subplot(6,2,8)
		plt.plot(fr8, rr8)
		plt.ylim(-0.00029, 0.00029)
		plt.xlim([45, 125])		
		
		
		plt.subplot(6,2,9)
		plt.plot(ft9, bt9)
		plt.xlim([45, 205])
		plt.ylabel('Mid-Band Blade\n 30m x 30m GP')
		
		plt.subplot(6,2,10)
		plt.plot(fr9, rr9)
		plt.ylim(-0.00029, 0.00029)
		plt.xlim([45, 125])		
		
		
		plt.subplot(6,2,11)
		plt.plot(ft10, bt10)
		plt.xlim([45, 205])
		plt.xlabel('frequency [MHz]')
		plt.ylabel('Mid-Band Blade\n 30m x 30m GP\n NIVEDITA')
		
		plt.subplot(6,2,12)
		plt.plot(fr10, rr10)
		plt.ylim(-0.00029, 0.00029)
		plt.xlim([45, 125])
		plt.xlabel('frequency [MHz]')
		
		
		
		
		
		
		
		

	
	
	
	
	elif part_number == 2:
	
		lst = np.genfromtxt('/run/media/raul/WD_RED_6TB/EDGES_vol2/low_band3/calibration/beam_factors/raw/gain_glitch_test_LST.txt')
		f   = np.genfromtxt('/run/media/raul/WD_RED_6TB/EDGES_vol2/low_band3/calibration/beam_factors/raw/gain_glitch_test_freq.txt')
		t   = np.genfromtxt('/run/media/raul/WD_RED_6TB/EDGES_vol2/low_band3/calibration/beam_factors/raw/gain_glitch_test_tant.txt')
		
		gah = np.genfromtxt('/run/media/raul/WD_RED_6TB/EDGES_vol2/low_band3/calibration/beam_factors/raw/gain_glitch_test_int_gain_above_horizon.txt')
		pah = np.genfromtxt('/run/media/raul/WD_RED_6TB/EDGES_vol2/low_band3/calibration/beam_factors/raw/gain_glitch_test_pixels_above_horizon.txt')
		
		bf  = np.genfromtxt('/run/media/raul/WD_RED_6TB/EDGES_vol2/low_band3/calibration/beam_factors/raw/gain_glitch_test_data.txt')
	
	
		
		t10 = t[11]
		t53 = t[53]
		
		bf10 = bf[11]
		bf53 = bf[53]
		
		fr  = f[(f>=60) & (f<=120)]
		t10 = t10[(f>=60) & (f<=120)]
		t53 = t53[(f>=60) & (f<=120)]
		
		gg  = gah[(f>=60) & (f<=120)]
		
		bf10 = bf10[(f>=60) & (f<=120)]
		bf53 = bf53[(f>=60) & (f<=120)]
		
		
		t10_corr = t10*gg/3145728 + 300*(1-(gg/3145728))
		t53_corr = t53*gg/3145728 + 300*(1-(gg/3145728))
		
		bf10_corr = bf10 * (gg/gg[13])
		bf53_corr = bf53 * (gg/gg[13])
		
		
		
		p10_1 = ba.fit_polynomial_fourier('LINLOG', fr, t10_corr, 6)   
		p10_2 = ba.fit_polynomial_fourier('LINLOG', fr, t10, 6)   
		
		p53_1 = ba.fit_polynomial_fourier('LINLOG', fr, t53_corr, 6)   
		p53_2 = ba.fit_polynomial_fourier('LINLOG', fr, t53, 6) 	
		
	
	
		p_bf_10_1 = ba.fit_polynomial_fourier('LINLOG', fr, bf10_corr, 6)   
		p_bf_10_2 = ba.fit_polynomial_fourier('LINLOG', fr, bf10, 6)  
	
		p_bf_53_1 = ba.fit_polynomial_fourier('LINLOG', fr, bf53_corr, 6)   
		p_bf_53_2 = ba.fit_polynomial_fourier('LINLOG', fr, bf53, 6) 
		
		
		
		
		
		k10_corr = (t10_corr - 300*(1-(gg/3145728)))/bf10_corr
		k10      = t10_corr/bf10
		
		k53_corr = (t53_corr - 300*(1-(gg/3145728)))/bf53_corr
		k53      = t53_corr/bf53	
		
		
		p_k10_corr = ba.fit_polynomial_fourier('LINLOG', fr, k10_corr, 6)  
		p_k10      = ba.fit_polynomial_fourier('LINLOG', fr, k10, 6)  
		
		p_k53_corr = ba.fit_polynomial_fourier('LINLOG', fr, k53_corr, 6)  
		p_k53      = ba.fit_polynomial_fourier('LINLOG', fr, k53, 6) 	
		
		
		
		plt.close()
		
		plt.figure(1)
		
		plt.subplot(2,2,1)
		plt.plot(fr, t10_corr)
		plt.plot(fr, t10, '--')
		plt.ylabel('temperature [K]')
		plt.title('Low Foreground (GHA = 10 hr)')
		
		plt.subplot(2,2,2)
		plt.plot(fr, t53_corr)
		plt.plot(fr, t53, '--')
		#plt.ylabel('temperature [K]')
		plt.title('High Foreground (GHA = 0 hr)')
		
		plt.subplot(2,2,3)
		plt.plot(fr, t10_corr - p10_1[1])
		plt.plot(fr, t10 - p10_2[1], '--')
		plt.ylim([-0.5, 0.5])
		plt.xlabel('frequency [MHz]')
		plt.ylabel(r'$\Delta$ temperature [K]')
		plt.legend(['correct','incorrect'])
		
		
		plt.subplot(2,2,4)
		plt.plot(fr, t53_corr - p53_1[1])
		plt.plot(fr, t53 - p53_2[1], '--')	
		plt.ylim([-0.5, 0.5])
		plt.xlabel('frequency [MHz]')
		#plt.ylabel(r'\Delta temperature [K]')
		
		
		
		plt.figure(2)
		
		plt.subplot(2,2,1)
		plt.plot(fr, bf10_corr)
		plt.plot(fr, bf10, '--')
		plt.ylabel('beam correction factor')
		plt.title('Low Foreground (GHA = 10 hr)')
		
		plt.subplot(2,2,2)
		plt.plot(fr, bf53_corr)
		plt.plot(fr, bf53, '--')
		#plt.ylabel('temperature [K]')
		plt.title('High Foreground (GHA = 0 hr)')
		
		plt.subplot(2,2,3)
		plt.plot(fr, bf10_corr - p_bf_10_1[1])
		plt.plot(fr, bf10 - p_bf_10_2[1], '--')
		#plt.plot(fr, t10 - p10_2[1])
		plt.ylim([-0.0004, 0.0004])
		plt.xlabel('frequency [MHz]')
		plt.ylabel(r'$\Delta$ temperature [K]')
		plt.legend(['correct','incorrect'])
		
		
		plt.subplot(2,2,4)
		plt.plot(fr, bf53_corr - p_bf_53_1[1])
		plt.plot(fr, bf53 - p_bf_53_2[1], '--')	
		#plt.plot(fr, t53_corr - p53_1[1])
		#plt.plot(fr, t53 - p53_2[1])	
		plt.ylim([-0.0004, 0.0004])
		plt.xlabel('frequency [MHz]')	
		
	
	
	
	
	
		plt.figure(3)
		
		plt.subplot(2,2,1)
		plt.plot(fr, k10_corr)
		#plt.plot(fr, k10)
		plt.ylabel('temperature [K]')
		plt.title('Low Foreground (GHA = 10 hr)')
		
		plt.subplot(2,2,2)
		plt.plot(fr, t53_corr)
		##plt.ylabel('temperature [K]')
		plt.title('High Foreground (GHA = 0 hr)')
		
		plt.subplot(2,2,3)
		plt.plot(fr, k10_corr - p_k10_corr[1])
		plt.plot(fr, k10 - p_k10[1], '--')
		plt.ylim([-0.5, 0.5])
		plt.xlabel('frequency [MHz]')
		plt.ylabel(r'$\Delta$ temperature [K]')
		plt.legend(['correct','incorrect'])
		
		
		plt.subplot(2,2,4)
		plt.plot(fr, k53_corr - p_k53_corr[1])
		plt.plot(fr, k53 - p_k53[1], '--')	
		plt.ylim([-0.5, 0.5])
		plt.xlabel('frequency [MHz]')
		
		







	
	return 0 #f, t, lst, bf, gah








def high_band_2015_reanalysis():


	# import old_high_band_edges as ohb
	
	
	LST1 = 1
	LST2 = 11
	
	FLOW  = 80
	FHIGH = 140
	
	Nfg = 5	
	
	
	#f, r, p, w, rms, m = ohb.level3read('/run/media/raul/WD_BLACK_6TB/EDGES_vol1/spectra/level3/high_band_2015/2018_analysis/case101/2015_288_00.hdf5')
	
	filename_list = ['2015_251_00.hdf5', '2015_252_00.hdf5', '2015_253_00.hdf5', '2015_254_00.hdf5', '2015_256_00.hdf5', '2015_257_00.hdf5', '2015_258_00.hdf5', '2015_259_00.hdf5']
	
	
	for i in range(len(filename_list)):
		print(filename_list[i])
		
		f, r, p, w, rms, m = ohb.level3read('/run/media/raul/WD_BLACK_6TB/EDGES_vol1/spectra/level3/high_band_2015/2018_analysis/case101/' + filename_list[i])

		index          = np.arange(len(r[:,0]))
		index_selected = index[(m[:,3]>=LST1) & (m[:,3]<=LST2)]
		
		avr, avw       = ba.spectral_averaging(r[index_selected,:], w[index_selected,:])
		avp            = np.mean(p[index,:], axis=0)
		
		if i == 0:
			r_all = np.zeros((len(filename_list), len(f)))
			w_all = np.zeros((len(filename_list), len(f)))
			p_all = np.zeros((len(filename_list), len(avp)))
			
			
		r_all[i,:] = avr
		w_all[i,:] = avw
		p_all[i,:] = avp
	
	
	
	
	rr, ww = ba.spectral_averaging(r_all, w_all)
	pp     = np.mean(p_all, axis=0)
	
	
	fb, rb, wb, sb = ba.spectral_binning_number_of_samples(f, rr, ww, nsamples=128)
	
	
	av_mf     = np.polyval(pp, fb/200)
	avmb      = av_mf * ((fb/200)**(-2.5)) 
	
	
	tb = rb + avmb
	
	fk = fb[(fb>= FLOW) & (fb<= FHIGH)]
	tk = tb[(fb>= FLOW) & (fb<= FHIGH)]
	wk = wb[(fb>= FLOW) & (fb<= FHIGH)]
	sk = sb[(fb>= FLOW) & (fb<= FHIGH)]
	
	
	p1     = ba.fit_polynomial_fourier('LINLOG', fk, tk, Nfg, Weights=(1/sk)**2)
	
	signal = dm.signal_model('tanh', [ -0.7, 79, 22, 7, 8], fk)
	p2     = ba.fit_polynomial_fourier('LINLOG', fk, tk-signal, Nfg, Weights=(1/sk)**2)
	
	
	plt.close()
	plt.close()
	plt.plot(fk, tk - p1[1])
	plt.plot(fk, tk-signal - p2[1]-0.6)
	plt.ylim([-0.8, 0.4])
	plt.yticks([-0.6, -0.4, -0.2, 0, 0.2], labels=['','',''])

	
	plt.xlabel('frequency [MHz]')
	plt.ylabel(r'$\Delta$ temperature [0.2 K per division]')
	plt.legend(['Foreground','Foreground + Signal'])


	#plt.plot(fb, rb)


	return fk, tk #0 #fb, rb










def comparison_FEKO_HFSS():
	
	
	theta, phi, b60 = ba.HFSS_beam_read('/run/media/raul/WD_RED_6TB/EDGES_vol2/mid_band/calibration/beam/raul/hfss/20191002/pec_60MHz.csv', 'linear', theta_min=0, theta_max=180, theta_resolution=1, phi_min=0, phi_max=359, phi_resolution=1)
	H60 = b60[theta<=90,:]
	
	theta, phi, b90 = ba.HFSS_beam_read('/run/media/raul/WD_RED_6TB/EDGES_vol2/mid_band/calibration/beam/raul/hfss/20191002/pec_90MHz.csv', 'linear', theta_min=0, theta_max=180, theta_resolution=1, phi_min=0, phi_max=359, phi_resolution=1)
	H90 = b90[theta<=90,:]
	
	theta, phi, b120 = ba.HFSS_beam_read('/run/media/raul/WD_RED_6TB/EDGES_vol2/mid_band/calibration/beam/raul/hfss/20191002/pec_120MHz.csv', 'linear', theta_min=0, theta_max=180, theta_resolution=1, phi_min=0, phi_max=359, phi_resolution=1)
	H120 = b120[theta<=90,:]
	
	
	bm   = cal.FEKO_blade_beam('mid_band', 1, frequency_interpolation='no', frequency=np.array([0]), AZ_antenna_axis=0)
	F60  = np.flipud(bm[5])
	F90  = np.flipud(bm[20])
	F120 = np.flipud(bm[35])
	
	
	plt.figure(1)
	plt.subplot(2,1,1)
	plt.imshow(F60, interpolation=None, aspect='auto');   cbar = plt.colorbar(); cbar.set_label('directive gain [linear]', rotation=90)
	plt.title('Feko, 60 MHz')
	plt.ylabel('theta [deg]')
	
	plt.subplot(2,1,2)
	plt.imshow(H60 - F60, interpolation=None, aspect='auto');   cbar = plt.colorbar(); cbar.set_label(r'$\Delta$ directive gain [linear]', rotation=90)
	plt.title('(HFSS - Feko), 60 MHz')
	plt.ylabel('theta [deg]')
	plt.xlabel('phi [deg]')
	
	
	plt.figure(2)
	plt.subplot(2,1,1)
	plt.imshow(F90, interpolation=None, aspect='auto');   cbar = plt.colorbar(); cbar.set_label('directive gain [linear]', rotation=90)
	plt.title('Feko, 90 MHz')
	plt.ylabel('theta [deg]')
	
	plt.subplot(2,1,2)
	plt.imshow(H90 - F90, interpolation=None, aspect='auto');   cbar = plt.colorbar(); cbar.set_label(r'$\Delta$ directive gain [linear]', rotation=90)
	plt.title('(HFSS - Feko), 90 MHz')
	plt.ylabel('theta [deg]')
	plt.xlabel('phi [deg]')	
	
	
	plt.figure(3)
	plt.subplot(2,1,1)
	plt.imshow(F120, interpolation=None, aspect='auto');   cbar = plt.colorbar(); cbar.set_label('directive gain [linear]', rotation=90)
	plt.title('Feko, 120 MHz')
	plt.ylabel('theta [deg]')
	
	plt.subplot(2,1,2)
	plt.imshow(H120 - F120, interpolation=None, aspect='auto'); cbar = plt.colorbar(); cbar.set_label(r'$\Delta$ directive gain [linear]', rotation=90)
	plt.title('(HFSS - Feko), 120 MHz')
	plt.ylabel('theta [deg]')
	plt.xlabel('phi [deg]')	
	
	
	
	return H60, H90, H120, F60, F90, F120



















def comparison_FEKO_WIPLD():
	
	plt.close()
	plt.close()
	plt.close()
	plt.close()
	plt.close()
	plt.close()
	plt.close()
	plt.close()
	plt.close()
	plt.close()
	plt.close()
	
	path_plot_save = edges_folder + 'plots/20191015/'
	
	
	
	
	# WIPL-D
	filename = '/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191012/blade_dipole_infinite_PEC.ra1'
	f, thetaX, phiX, beam  = ba.WIPLD_beam_read(filename)
	
	theta = thetaX[thetaX<=90]
	m = beam[:,thetaX<=90,:]
	W60  = m[f==60,:,:][0]
	W90  = m[f==90,:,:][0]
	W120 = m[f==120,:,:][0]
	
	#x = beam[f==60,:,:][0]
	#W60 = x[theta<=90,:]
	
	#x = beam[f==90,:,:][0]
	#W90 = x[theta<=90,:]	
	
	#x = beam[f==120,:,:][0]
	#W120 = x[theta<=90,:]	
	
	
	sin_theta      = np.sin(theta*(np.pi/180))
	sin_theta_2D_T = np.tile(sin_theta, (360, 1))
	sin_theta_2D   = sin_theta_2D_T.T
	
	bint = np.zeros(len(f))
	for i in range(len(f)):
		
		b       = m[i,:,:]
		bint[i] = np.sum(b * sin_theta_2D)
	
	btW  = (1/(4*np.pi)) * ((np.pi/180)**2)*bint   #/np.mean(bx)
	
	Nfg = 7
	fHIGH = 150
	fX    = f[f<=fHIGH]
	btWX  = btW[f<=fHIGH]
	
	x   = np.polyfit(fX, btWX, Nfg-1)
	model   = np.polyval(x, fX)
	rtWX = btWX - model
	
	
	deltaW = np.zeros((len(m[:,0,0])-1, len(m[0,:,0]), len(m[0,0,:])))
	for i in range(len(f)-1):
		deltaW[i,:,:] = m[i+1,:,:] - m[i,:,:]
	
	
	
	
	
	
	# HFSS
	thetaX, phi, b60 = ba.HFSS_beam_read(edges_folder + 'others/beam_simulations/hfss/20191002/mid_band_infinite_pec/60MHz.csv', 'linear', theta_min=0, theta_max=180, theta_resolution=1, phi_min=0, phi_max=359, phi_resolution=1)
	theta = thetaX[thetaX<=90]
	H60   = b60[thetaX<=90,:]
	
	thetaX, phi, b90 = ba.HFSS_beam_read(edges_folder + 'others/beam_simulations/hfss/20191002/mid_band_infinite_pec/90MHz.csv', 'linear', theta_min=0, theta_max=180, theta_resolution=1, phi_min=0, phi_max=359, phi_resolution=1)
	H90 = b90[thetaX<=90,:]
	
	thetaX, phi, b120 = ba.HFSS_beam_read(edges_folder + 'others/beam_simulations/hfss/20191002/mid_band_infinite_pec/120MHz.csv', 'linear', theta_min=0, theta_max=180, theta_resolution=1, phi_min=0, phi_max=359, phi_resolution=1)
	H120 = b120[thetaX<=90,:]
	
	

	
	thetaX, phi, b65 = ba.HFSS_beam_read(edges_folder + 'others/beam_simulations/hfss/20191002/mid_band_infinite_pec/65MHz.csv', 'linear', theta_min=0, theta_max=90, theta_resolution=1, phi_min=0, phi_max=359, phi_resolution=1)
	H65 = b65[thetaX<=90,:]
	
	thetaX, phi, b66 = ba.HFSS_beam_read(edges_folder + 'others/beam_simulations/hfss/20191002/mid_band_infinite_pec/66MHz.csv', 'linear', theta_min=0, theta_max=90, theta_resolution=1, phi_min=0, phi_max=359, phi_resolution=1)
	H66 = b66[thetaX<=90,:]
	
	thetaX, phi, b67 = ba.HFSS_beam_read(edges_folder + 'others/beam_simulations/hfss/20191002/mid_band_infinite_pec/67MHz.csv', 'linear', theta_min=0, theta_max=90, theta_resolution=1, phi_min=0, phi_max=359, phi_resolution=1)
	H67 = b67[thetaX<=90,:]
	
	thetaX, phi, b68 = ba.HFSS_beam_read(edges_folder + 'others/beam_simulations/hfss/20191002/mid_band_infinite_pec/68MHz.csv', 'linear', theta_min=0, theta_max=90, theta_resolution=1, phi_min=0, phi_max=359, phi_resolution=1)
	H68 = b68[thetaX<=90,:]
	
	thetaX, phi, b69 = ba.HFSS_beam_read(edges_folder + 'others/beam_simulations/hfss/20191002/mid_band_infinite_pec/69MHz.csv', 'linear', theta_min=0, theta_max=90, theta_resolution=1, phi_min=0, phi_max=359, phi_resolution=1)
	H69 = b69[thetaX<=90,:]
	
	thetaX, phi, b70 = ba.HFSS_beam_read(edges_folder + 'others/beam_simulations/hfss/20191002/mid_band_infinite_pec/70MHz.csv', 'linear', theta_min=0, theta_max=90, theta_resolution=1, phi_min=0, phi_max=359, phi_resolution=1)
	H70 = b70[thetaX<=90,:]	
	
	thetaX, phi, b71 = ba.HFSS_beam_read(edges_folder + 'others/beam_simulations/hfss/20191002/mid_band_infinite_pec/71MHz.csv', 'linear', theta_min=0, theta_max=90, theta_resolution=1, phi_min=0, phi_max=359, phi_resolution=1)
	H71 = b71[thetaX<=90,:]	

	thetaX, phi, b72 = ba.HFSS_beam_read(edges_folder + 'others/beam_simulations/hfss/20191002/mid_band_infinite_pec/72MHz.csv', 'linear', theta_min=0, theta_max=90, theta_resolution=1, phi_min=0, phi_max=359, phi_resolution=1)
	H72 = b72[thetaX<=90,:]	

	thetaX, phi, b73 = ba.HFSS_beam_read(edges_folder + 'others/beam_simulations/hfss/20191002/mid_band_infinite_pec/73MHz.csv', 'linear', theta_min=0, theta_max=90, theta_resolution=1, phi_min=0, phi_max=359, phi_resolution=1)
	H73 = b73[thetaX<=90,:]	

	thetaX, phi, b74 = ba.HFSS_beam_read(edges_folder + 'others/beam_simulations/hfss/20191002/mid_band_infinite_pec/74MHz.csv', 'linear', theta_min=0, theta_max=90, theta_resolution=1, phi_min=0, phi_max=359, phi_resolution=1)
	H74 = b74[thetaX<=90,:]	

	thetaX, phi, b75 = ba.HFSS_beam_read(edges_folder + 'others/beam_simulations/hfss/20191002/mid_band_infinite_pec/75MHz.csv', 'linear', theta_min=0, theta_max=90, theta_resolution=1, phi_min=0, phi_max=359, phi_resolution=1)
	H75 = b75[thetaX<=90,:]	




	sin_theta      = np.sin(theta*(np.pi/180))
	sin_theta_2D_T = np.tile(sin_theta, (360, 1))
	sin_theta_2D   = sin_theta_2D_T.T
	
	bint60 = np.sum(H60 * sin_theta_2D)
	btH60  = (1/(4*np.pi)) * ((np.pi/180)**2)*bint60 
	
	bint90 = np.sum(H90 * sin_theta_2D)
	btH90  = (1/(4*np.pi)) * ((np.pi/180)**2)*bint90 
	
	bint120 = np.sum(H120 * sin_theta_2D)
	btH120  = (1/(4*np.pi)) * ((np.pi/180)**2)*bint120 
	
	
	bint65 = np.sum(H65 * sin_theta_2D)
	btH65  = (1/(4*np.pi)) * ((np.pi/180)**2)*bint65 	
	
	bint66 = np.sum(H66 * sin_theta_2D)
	btH66  = (1/(4*np.pi)) * ((np.pi/180)**2)*bint66 

	bint67 = np.sum(H67 * sin_theta_2D)
	btH67  = (1/(4*np.pi)) * ((np.pi/180)**2)*bint67 
	
	bint68 = np.sum(H68 * sin_theta_2D)
	btH68  = (1/(4*np.pi)) * ((np.pi/180)**2)*bint68 
	
	bint69 = np.sum(H69 * sin_theta_2D)
	btH69  = (1/(4*np.pi)) * ((np.pi/180)**2)*bint69 
	
	bint70 = np.sum(H70 * sin_theta_2D)
	btH70  = (1/(4*np.pi)) * ((np.pi/180)**2)*bint70 	
	
	bint71 = np.sum(H71 * sin_theta_2D)
	btH71  = (1/(4*np.pi)) * ((np.pi/180)**2)*bint71 	

	bint72 = np.sum(H72 * sin_theta_2D)
	btH72  = (1/(4*np.pi)) * ((np.pi/180)**2)*bint72 	

	bint73 = np.sum(H73 * sin_theta_2D)
	btH73  = (1/(4*np.pi)) * ((np.pi/180)**2)*bint73 	

	bint74 = np.sum(H74 * sin_theta_2D)
	btH74  = (1/(4*np.pi)) * ((np.pi/180)**2)*bint74 	

	bint75 = np.sum(H75 * sin_theta_2D)
	btH75  = (1/(4*np.pi)) * ((np.pi/180)**2)*bint75 	
	
	
	fHX  = np.array([60, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 90, 120])
	btHX = np.array([btH60, btH65, btH66, btH67, btH68, btH69, btH70, btH71, btH72, btH73, btH74, btH75, btH90, btH120])
	
	Nfg = 4
	x     = np.polyfit(fHX, btHX, Nfg-1)
	model = np.polyval(x, fHX)
	rtHX  = btHX - model	
	
	
	
	


	# FEKO
	bm   = cal.FEKO_blade_beam('mid_band', 1, frequency_interpolation='no', frequency=np.array([0]), AZ_antenna_axis=0)
	F60  = np.flipud(bm[5])
	F90  = np.flipud(bm[20])
	F120 = np.flipud(bm[35])
	

	f              = np.arange(50,201,2)
	el             = np.arange(0,91) 
	sin_theta      = np.sin((90-el)*(np.pi/180)) 
	sin_theta_2D_T = np.tile(sin_theta, (360, 1))
	sin_theta_2D   = sin_theta_2D_T.T			
	
	bint = np.zeros(len(f))
	for i in range(len(f)):
		
		b       = bm[i,:,:]
		bint[i] = np.sum(b * sin_theta_2D)
	
	btF  = (1/(4*np.pi)) * ((np.pi/180)**2)*bint   #/np.mean(bx)
	
	Nfg = 7
	fX    = f[f<=fHIGH]
	btFX  = btF[f<=fHIGH]
	
	x     = np.polyfit(fX, btFX, Nfg-1)
	model = np.polyval(x, fX)
	rtFX  = btFX - model
	
	
	deltaF = np.zeros((len(bm[:,0,0])-1, len(bm[0,:,0]), len(bm[0,0,:])))
	for i in range(len(f)-1):
		XX  = bm[i+1,:,:] - bm[i,:,:]
		XXX = np.flipud(XX) 
		deltaF[i,:,:] = XXX

	
	
	
	
	#plt.figure(1, figsize=(9,12))
	#MIN = -0.05
	#MAX =  0.05
	
	#plt.subplot(4,1,1)
	#plt.imshow(F60, interpolation=None, aspect='auto');   cbar = plt.colorbar(); cbar.set_label('directive gain [linear]', rotation=90)
	#plt.title('FEKO, 60 MHz')
	#plt.ylabel('theta [deg]')
	#plt.yticks([0, 30, 60, 90])
	#plt.xticks([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360], labels=[])
	
	#plt.subplot(4,1,2)
	#plt.imshow(W60 - F60, interpolation=None, aspect='auto', vmin=MIN, vmax=MAX);   cbar = plt.colorbar(); cbar.set_label(r'$\Delta$ directive gain [linear]', rotation=90)
	#plt.title('(WIPL-D - FEKO), 60 MHz')
	#plt.ylabel('theta [deg]')
	#plt.yticks([0, 30, 60, 90])
	#plt.xticks([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360], labels=[])	
	
	#plt.subplot(4,1,3)
	#plt.imshow(H60 - F60, interpolation=None, aspect='auto', vmin=MIN, vmax=MAX);   cbar = plt.colorbar(); cbar.set_label(r'$\Delta$ directive gain [linear]', rotation=90)
	#plt.title('(HFSS - FEKO), 60 MHz')
	#plt.ylabel('theta [deg]')
	#plt.yticks([0, 30, 60, 90])
	#plt.xticks([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360], labels=[])
	
	#plt.subplot(4,1,4)
	#plt.imshow(W60 - H60, interpolation=None, aspect='auto', vmin=MIN, vmax=MAX);   cbar = plt.colorbar(); cbar.set_label(r'$\Delta$ directive gain [linear]', rotation=90)
	#plt.title('(WIPL-D - HFSS), 60 MHz')
	#plt.ylabel('theta [deg]')
	#plt.xlabel('phi [deg]')
	#plt.yticks([0, 30, 60, 90])
	#plt.xticks([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360])
	
	#plt.savefig(path_plot_save + 'fig1.pdf', bbox_inches='tight')
	#plt.close()	
	#plt.close()
	
	
	
	
	#plt.figure(2, figsize=(9,12))
	#MIN = -0.1
	#MAX =  0.1
	
	#plt.subplot(4,1,1)
	#plt.imshow(F90, interpolation=None, aspect='auto');   cbar = plt.colorbar(); cbar.set_label('directive gain [linear]', rotation=90)
	#plt.title('FEKO, 90 MHz')
	#plt.ylabel('theta [deg]')
	#plt.yticks([0, 30, 60, 90])
	#plt.xticks([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360], labels=[])
	
	#plt.subplot(4,1,2)
	#plt.imshow(W90 - F90, interpolation=None, aspect='auto', vmin=MIN, vmax=MAX);   cbar = plt.colorbar(); cbar.set_label(r'$\Delta$ directive gain [linear]', rotation=90)
	#plt.title('(WIPL-D - FEKO), 90 MHz')
	#plt.ylabel('theta [deg]')
	#plt.yticks([0, 30, 60, 90])
	#plt.xticks([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360], labels=[])
	
	#plt.subplot(4,1,3)
	#plt.imshow(H90 - F90, interpolation=None, aspect='auto', vmin=MIN, vmax=MAX);   cbar = plt.colorbar(); cbar.set_label(r'$\Delta$ directive gain [linear]', rotation=90)
	#plt.title('(HFSS - FEKO), 90 MHz')
	#plt.ylabel('theta [deg]')
	#plt.yticks([0, 30, 60, 90])
	#plt.xticks([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360], labels=[])	
	
	#plt.subplot(4,1,4)
	#plt.imshow(W90 - H90, interpolation=None, aspect='auto', vmin=MIN, vmax=MAX);   cbar = plt.colorbar(); cbar.set_label(r'$\Delta$ directive gain [linear]', rotation=90)
	#plt.title('(WIPL-D - HFSS), 90 MHz')
	#plt.ylabel('theta [deg]')
	#plt.xlabel('phi [deg]')	
	#plt.yticks([0, 30, 60, 90])
	#plt.xticks([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360])

	#plt.savefig(path_plot_save + 'fig2.pdf', bbox_inches='tight')
	#plt.close()	
	#plt.close()	
	
	
	
	
	
	#plt.figure(3, figsize=(9,12))
	#MIN = -0.2
	#MAX =  0.2
	
	#plt.subplot(4,1,1)
	#plt.imshow(F120, interpolation=None, aspect='auto');   cbar = plt.colorbar(); cbar.set_label('directive gain [linear]', rotation=90)
	#plt.title('FEKO, 120 MHz')
	#plt.ylabel('theta [deg]')
	#plt.yticks([0, 30, 60, 90])
	#plt.xticks([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360], labels=[])	
	
	#plt.subplot(4,1,2)
	#plt.imshow(W120 - F120, interpolation=None, aspect='auto', vmin=MIN, vmax=MAX); cbar = plt.colorbar(); cbar.set_label(r'$\Delta$ directive gain [linear]', rotation=90)
	#plt.title('(WIPL-D - FEKO), 120 MHz')
	#plt.ylabel('theta [deg]')
	#plt.yticks([0, 30, 60, 90])
	#plt.xticks([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360], labels=[])	
	
	#plt.subplot(4,1,3)
	#plt.imshow(H120 - F120, interpolation=None, aspect='auto', vmin=MIN, vmax=MAX); cbar = plt.colorbar(); cbar.set_label(r'$\Delta$ directive gain [linear]', rotation=90)
	#plt.title('(HFSS - FEKO), 120 MHz')
	#plt.ylabel('theta [deg]')
	#plt.yticks([0, 30, 60, 90])
	#plt.xticks([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360], labels=[])	
	
	#plt.subplot(4,1,4)
	#plt.imshow(W120 - H120, interpolation=None, aspect='auto', vmin=MIN, vmax=MAX); cbar = plt.colorbar(); cbar.set_label(r'$\Delta$ directive gain [linear]', rotation=90)
	#plt.title('(WIPL-D - HFSS), 120 MHz')
	#plt.ylabel('theta [deg]')
	#plt.xlabel('phi [deg]')	
	#plt.yticks([0, 30, 60, 90])
	#plt.xticks([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360])	
	
	#plt.savefig(path_plot_save + 'fig3.pdf', bbox_inches='tight')
	#plt.close()	
	#plt.close()
	

	


	
	
	#plt.figure(4, figsize=(7,7))
	#plt.subplot(2,1,1)
	#plt.imshow(deltaF[:,:,0].T, interpolation='none', aspect='auto', extent=[50, 200, 90, 0], vmin=-0.05, vmax=0.05);
	#cbar = plt.colorbar()
	#plt.yticks([0, 30, 60, 90])
	#plt.xticks([50, 75, 100, 125, 150, 175, 200], labels=[])
	#cbar.set_label(r'$\Delta$ directive gain per MHz', rotation=90)
	#plt.ylabel('theta [deg]')
	#plt.title('FEKO, phi=0 [deg]')
	
	#plt.subplot(2,1,2)
	#plt.imshow(deltaF[:,:,90].T, interpolation='none', aspect='auto', extent=[50, 200, 90, 0], vmin=-0.05, vmax=0.05);
	#cbar = plt.colorbar()
	#plt.yticks([0, 30, 60, 90])
	#plt.xticks([50, 75, 100, 125, 150, 175, 200])
	#cbar.set_label(r'$\Delta$ directive gain per MHz', rotation=90)
	#plt.ylabel('theta [deg]')
	#plt.xlabel('frequency [MHz]')
	#plt.title('FEKO, phi=90 [deg]')

	#plt.savefig(path_plot_save + 'fig4.pdf', bbox_inches='tight')
	#plt.close()	
	#plt.close()
	
	

	#plt.figure(5, figsize=(7,7))
	#plt.subplot(2,1,1)
	#plt.imshow(deltaW[:,:,0].T, interpolation='none', aspect='auto', extent=[50, 200, 90, 0], vmin=-0.05, vmax=0.05);
	#cbar = plt.colorbar()
	#plt.yticks([0, 30, 60, 90])
	#plt.xticks([50, 75, 100, 125, 150, 175, 200], labels=[])
	#cbar.set_label(r'$\Delta$ directive gain per MHz', rotation=90)
	#plt.ylabel('theta [deg]')
	#plt.title('WIPL-D, phi=0 [deg]')
	
	#plt.subplot(2,1,2)
	#plt.imshow(deltaW[:,:,90].T, interpolation='none', aspect='auto', extent=[50, 200, 90, 0], vmin=-0.05, vmax=0.05);
	#cbar = plt.colorbar()
	#plt.yticks([0, 30, 60, 90])
	#plt.xticks([50, 75, 100, 125, 150, 175, 200])
	#cbar.set_label(r'$\Delta$ directive gain per MHz', rotation=90)
	#plt.ylabel('theta [deg]')
	#plt.xlabel('frequency [MHz]')
	#plt.title('WIPL-D, phi=90 [deg]')

	#plt.savefig(path_plot_save + 'fig5.pdf', bbox_inches='tight')
	#plt.close()	
	#plt.close()


	
	plt.figure(6, figsize=(8,10))
	plt.subplot(2,1,1)
	plt.plot(f, btF, 'b')
	plt.plot(f, btW, 'r--')
	plt.plot(fHX, btHX, 'g.-')
	plt.xticks([50, 75, 100, 125, 150, 175, 200], labels=[])
	plt.ylabel('integrated gain above horizon [fraction of 4 pi]')
	plt.legend(['FEKO','WIPL-D','HFSS-IE'], loc=3)
	
	plt.subplot(2,1,2)
	plt.plot([70, 70], [-1.5e-5, 1.5e-5], 'c:')
	plt.plot([95, 95], [-1.5e-5, 1.5e-5], 'c:')	
	plt.text(67, -0.5e-5, '70 MHz', rotation=90)
	plt.text(92, -0.5e-5, '95 MHz', rotation=90)
	plt.plot(fX, rtWX+0e-5, 'r--')
	plt.plot(fX, rtFX+1e-5, 'b')
	plt.plot(fHX, rtHX-1e-5, 'g.-')
	plt.ylim([-1.5e-5, 1.5e-5])
	plt.xticks([50, 75, 100, 125, 150, 175, 200])
	plt.yticks(np.arange(-1e-5, 1.1e-5, 5e-6), labels=[])
	plt.xlabel('frequency [MHz]')
	plt.ylabel('fit residuals [fraction of 4 pi]\n(5e-6 per division)')
	
	plt.savefig(path_plot_save + 'fig6.pdf', bbox_inches='tight')
	plt.close()	
	plt.close()	

	
	
	
	
	
	return W60, W90, W120, F60, F90, F120, f, btW, btF, fX, rtWX, rtFX, btH60, btH90, btH120, deltaF, deltaW









def plots_beam_gain_derivative():
	
	
	# Low-Band, Blade on 10m x 10m GP
	b_all  = oeg.FEKO_low_band_blade_beam(beam_file=5, frequency_interpolation='no', frequency=np.array([0]), AZ_antenna_axis=0)
	f      = np.arange(50,121,2)
	
	delta = np.zeros((len(b_all[:,0,0])-1, len(b_all[0,:,0]), len(b_all[0,0,:])))
	for i in range(len(f)-1):
		XX  = b_all[i+1,:,:] - b_all[i,:,:]
		XXX = np.flipud(XX)
		delta[i,:,:] = XXX



	## Low-Band, Blade on 10m x 10m GP
	#b_all  = oeg.FEKO_low_band_blade_beam(beam_file=4, frequency_interpolation='no', frequency=np.array([0]), AZ_antenna_axis=0)
	#f      = np.arange(40,101,2)
	
	#delta = np.zeros((len(b_all[:,0,0])-1, len(b_all[0,:,0]), len(b_all[0,0,:])))
	#for i in range(len(f)-1):
		#XX  = b_all[i+1,:,:] - b_all[i,:,:]
		#XXX = np.flipud(XX)
		#delta[i,:,:] = XXX



	
	## Low-Band, Blade on 30m x 30m GP
	#b_all = oeg.FEKO_low_band_blade_beam(beam_file=2, frequency_interpolation='no', AZ_antenna_axis=0)
	#f     = np.arange(40,121,2)
	
	#delta = np.zeros((len(b_all[:,0,0])-1, len(b_all[0,:,0]), len(b_all[0,0,:])))
	#for i in range(len(f)-1):
		#XX  = b_all[i+1,:,:] - b_all[i,:,:]
		#XXX = np.flipud(XX)
		#delta[i,:,:] = XXX	

	
	## Low-Band, Blade on 30m x 30m GP, NIVEDITA
	#b_all = oeg.FEKO_low_band_blade_beam(beam_file=0, frequency_interpolation='no', AZ_antenna_axis=0)
	#f     = np.arange(40,101,2)
	
	#delta = np.zeros((len(b_all[:,0,0])-1, len(b_all[0,:,0]), len(b_all[0,0,:])))
	#for i in range(len(f)-1):
		#XX  = b_all[i+1,:,:] - b_all[i,:,:]
		#XXX = np.flipud(XX)
		#delta[i,:,:] = XXX

	
	

	## Mid-Band, Blade, on 30m x 30m GP
	#b_all = cal.FEKO_blade_beam('mid_band', 0, frequency_interpolation='no', AZ_antenna_axis=90)
	#f     = np.arange(50,201,2)
	
	#delta = np.zeros((len(b_all[:,0,0])-1, len(b_all[0,:,0]), len(b_all[0,0,:])))
	#for i in range(len(f)-1):
		#XX  = b_all[i+1,:,:] - b_all[i,:,:]
		#XXX = np.flipud(XX)
		#delta[i,:,:] = XXX	

	
	

	## Mid-Band, Blade, on 30m x 30m GP, NIVEDITA
	#b_all = cal.FEKO_blade_beam('mid_band', 100, frequency_interpolation='no', AZ_antenna_axis=90)
	#f     = np.arange(60,201,2)

	#delta = np.zeros((len(b_all[:,0,0])-1, len(b_all[0,:,0]), len(b_all[0,0,:])))
	#for i in range(len(f)-1):
		#XX  = b_all[i+1,:,:] - b_all[i,:,:]
		#XXX = np.flipud(XX)
		#delta[i,:,:] = XXX



	## FEKO
	#bm   = cal.FEKO_blade_beam('mid_band', 1, frequency_interpolation='no', frequency=np.array([0]), AZ_antenna_axis=0)
	#f    = np.arange(50,201,2)
	
	#delta = np.zeros((len(bm[:,0,0])-1, len(bm[0,:,0]), len(bm[0,0,:])))
	#for i in range(len(f)-1):
		#XX  = bm[i+1,:,:] - bm[i,:,:]
		#XXX = np.flipud(XX) 
		#delta[i,:,:] = XXX
	



	
	return delta, b_all, f












def integrated_antenna_gain_WIPLD_try1():
	
	plt.close()
	plt.close()
	plt.close()
	plt.close()
	plt.close()
	plt.close()
	plt.close()
	plt.close()
	plt.close()
	plt.close()
	plt.close()
	
	path_plot_save = edges_folder + 'plots/20191022/'
	
	
	
	
	# WIPL-D
	#filename = '/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191022/blade_dipole_infinite_soil.ra1'
	#filename = '/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191022/blade_dipole_infinite_soil_metal_GP_1m_side_0.01m_height.ra1'
	
	
	#filename = '/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191022/blade_dipole_infinite_soil_metal_GP_10m_side_0.05m_height_90-100MHz.ra1'
	#filename = '/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191022/blade_dipole_infinite_soil_metal_GP_10m_side_0.01m_height_90-100MHz.ra1'
	#filename = '/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191022/blade_dipole_infinite_soil_metal_GP_10m_side_0.001m_height_90-100MHz.ra1'
	#filename = '/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191022/blade_dipole_infinite_soil_metal_GP_10m_side_0.0001m_height_90-100MHz.ra1'
	#filename = '/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191022/blade_dipole_infinite_soil_metal_GP_10m_side_0.00001m_height_90-100MHz.ra1'
	
	
	#filename = '/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191022/blade_dipole_infinite_soil_metal_GP_10m_side_0.01m_height_65-75MHz.ra1'
	#filename = '/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191022/blade_dipole_infinite_soil_metal_GP_10m_side_0.01m_height_65-75MHz.ra1'
	
	#filename = '/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191022/blade_dipole_infinite_soil_metal_GP_4m_side_0.01m_height.ra1'
	#filename = '/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191022/blade_dipole_infinite_soil_metal_GP_4m_side_0.05m_height.ra1'
	
	#filename = '/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191022/blade_dipole_free_space.ra1'
	
	
	
	
	
	
	
	
	filename = '/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191022/blade_dipole_infinite_soil_metal_GP_0.00001m_60-78MHz.ra1'
	
	f1, thetaX, phiX, beam1 = ba.WIPLD_beam_read(filename)
	
	
	filename = '/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191022/blade_dipole_infinite_soil_metal_GP_0.00001m_80-98MHz.ra1'
	
	f2, thetaX, phiX, beam2 = ba.WIPLD_beam_read(filename)
	
	
	filename = '/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191022/blade_dipole_infinite_soil_metal_GP_0.00001m_100-108MHz.ra1'
	
	f3, thetaX, phiX, beam3 = ba.WIPLD_beam_read(filename)
	
	
	
	f    = np.concatenate((f1, f2, f3))
	beam = np.concatenate((beam1, beam2, beam3))




	
	
	#filename = '/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191022/blade_dipole_infinite_soil_metal_GP_0.00001m_height_65-83MHz.ra1'

#	filename = '/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191022/blade_dipole_infinite_soil_metal_GP_0.00001m_height_85-103MHz.ra1'



	
	#f, thetaX, phiX, beam = ba.WIPLD_beam_read(filename)
	
	theta = thetaX[thetaX<90]
	m = beam[:,thetaX<90,:]

	
	
	
	
	sin_theta      = np.sin(theta*(np.pi/180))
	sin_theta_2D_T = np.tile(sin_theta, (360, 1))
	sin_theta_2D   = sin_theta_2D_T.T
	
	bint = np.zeros(len(f))
	for i in range(len(f)):
		
		b       = m[i,:,:]
		bint[i] = np.sum(b * sin_theta_2D)
	
	btW  = (1/(4*np.pi)) * ((np.pi/180)**2)*bint   #/np.mean(bx)
	
	Nfg   = 5
	fLOW  = 50
	fHIGH = 120
	fX    = f[(f>=fLOW) & (f<=fHIGH)]
	btWX  = btW[(f>=fLOW) & (f<=fHIGH)]
	
	x   = np.polyfit(fX, btWX, Nfg-1)
	model   = np.polyval(x, fX)
	rtWX = btWX - model
	
	
	deltaW = np.zeros((len(m[:,0,0])-1, len(m[0,:,0]), len(m[0,0,:])))
	for i in range(len(f)-1):
		deltaW[i,:,:] = m[i+1,:,:] - m[i,:,:]
	
	
	return f, btW, fX, rtWX



















def integrated_antenna_gain_WIPLD(case, Nfg):
	
			
	# WIPL-D
	filename0 = '/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191023/blade_dipole_free_space.ra1'
	
	filename1 = '/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191023/blade_dipole_infinite_soil_3.5_0.02.ra1'
	filename2 = '/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191023/blade_dipole_infinite_soil_2.5_0.02.ra1'
	filename3 = '/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191023/blade_dipole_infinite_soil_4.5_0.02.ra1'
	filename4 = '/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191023/blade_dipole_infinite_soil_3.5_0.002.ra1'
	filename5 = '/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191023/blade_dipole_infinite_soil_3.5_0.2.ra1'
	
	
	filename11 = '/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191023/blade_dipole_infinite_soil_metal_GP_1m_side_0.0000001m_height.ra1'
	filename12 = '/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191023/blade_dipole_infinite_soil_metal_GP_1m_side_0.01m_height.ra1'
	filename13 = '/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191023/blade_dipole_infinite_soil_metal_GP_1m_side_0.0000001m_height_2.5_0.02.ra1'
	filename14 = '/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191023/blade_dipole_infinite_soil_metal_GP_1m_side_0.0000001m_height_4.5_0.02.ra1'
	filename15 = '/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191023/blade_dipole_infinite_soil_metal_GP_1m_side_0.0000001m_height_3.5_0.002.ra1'
	filename16 = '/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191023/blade_dipole_infinite_soil_metal_GP_1m_side_0.0000001m_height_3.5_0.2.ra1'
	
	
	
	filename51 = '/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191023/blade_dipole_infinite_soil_metal_GP_5m_side_0.0000001m_height.ra1'
	filename52 = '/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191023/blade_dipole_infinite_soil_metal_GP_5m_side_0.01m_height.ra1'
	filename53 = '/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191023/blade_dipole_infinite_soil_metal_GP_5m_side_0.0000001m_height_2.5_0.02.ra1'
	filename54 = '/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191023/blade_dipole_infinite_soil_metal_GP_5m_side_0.0000001m_height_4.5_0.02.ra1'
	filename55 = '/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191023/blade_dipole_infinite_soil_metal_GP_5m_side_0.0000001m_height_3.5_0.002.ra1'
	filename56 = '/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191023/blade_dipole_infinite_soil_metal_GP_5m_side_0.0000001m_height_3.5_0.2.ra1'

	
	
	filename101 = '/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191023/blade_dipole_infinite_soil_metal_GP_10m_side_0.0000001m_height.ra1'
	filename102 = '/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191023/blade_dipole_infinite_soil_metal_GP_10m_side_0.01m_height.ra1'
	filename103 = '/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191023/blade_dipole_infinite_soil_metal_GP_10m_side_0.0000001m_height_2.5_0.02.ra1'
	filename104 = '/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191023/blade_dipole_infinite_soil_metal_GP_10m_side_0.0000001m_height_4.5_0.02.ra1'
	filename105 = '/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191023/blade_dipole_infinite_soil_metal_GP_10m_side_0.0000001m_height_3.5_0.002.ra1'
	filename106 = '/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191023/blade_dipole_infinite_soil_metal_GP_10m_side_0.0000001m_height_3.5_0.2.ra1'
	
	
	filename200 = '/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191030/blade_dipole_infinite_soil_metal_GP_30mx30m_0.0000001m_height_3.5_0.02.ra1'
	filename201 = '/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191030/blade_dipole_infinite_soil_metal_GP_single_precision.ra1'
	filename202 = '/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191030/blade_dipole_infinite_soil_metal_GP_double_precision.ra1'
	filename203 = '/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191030/blade_dipole_infinite_soil_metal_GP_0.01_single_precision.ra1'
	filename204 = '/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191030/blade_dipole_infinite_soil_metal_GP_0.01_double_precision.ra1'
	filename205 = '/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191030/blade_dipole_infinite_soil_metal_GP_190-200MHz_single_precision.ra1'
	filename206 = '/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191030/blade_dipole_infinite_soil_metal_GP_190-200MHz_double_precision.ra1'


	filename300 = '/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191030/blade_dipole_infinite_PEC.ra1'
	filename301 = '/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191030/blade_dipole_infinite_soil_metal_GP_10mx10m.ra1'
	filename302 = '/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191030/blade_dipole_infinite_soil_metal_GP_15mx15m.ra1'
	filename303 = '/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191030/blade_dipole_infinite_soil_metal_GP_20mx20m.ra1'
	filename304 = '/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191030/blade_dipole_infinite_soil_metal_GP_30mx30m.ra1'
	filename305 = '/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191030/blade_dipole_infinite_soil_real_metal_GP_15mx15m.ra1'
	filename306 = '/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191030/blade_dipole_infinite_soil_real_metal_GP_30mx30m.ra1'
	
	filename400 = '/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191101/blade_dipole_infinite_PEC_50-120MHz.ra1'
	filename401 = '/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191101/blade_dipole_infinite_soil_metal_GP_10mx10m_50-120MHz.ra1'
	filename402 = '/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191101/blade_dipole_infinite_soil_metal_GP_15mx15m_50-120MHz.ra1'
	filename403 = '/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191101/blade_dipole_infinite_soil_metal_GP_20mx20m_50-120MHz.ra1'
	filename404 = '/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191101/blade_dipole_infinite_soil_metal_GP_30mx30m_50-120MHz.ra1'
	filename405 = '/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191101/blade_dipole_infinite_soil_real_metal_GP_15mx15m_50-120MHz.ra1'
	filename406 = '/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191101/blade_dipole_infinite_soil_real_metal_GP_30mx30m_50-120MHz.ra1'	
	
	filename500 = '/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191101/EDGES_low_band_30mx30m.ra1'
	filename501 = '/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191101/EDGES_low_band_10mx10m.ra1'
	
	filename601 = '/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191112/blade_dipole_infinite_soil_metal_GP_auto_mesh_MLFMM.ra1'
	#filename602 = '/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191112/blade_dipole_infinite_soil_metal_GP_auto_mesh_MLFMM_200MHz.ra1'
	filename602 = '/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191112/blade_dipole_infinite_soil_metal_GP_0.1m.ra1'
	


	filename701 = '/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191114/blade_dipole_infinite_soil_metal_GP_1cm.ra1'
	filename702 = '/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191114/blade_dipole_infinite_soil_metal_GP_1cm_double.ra1'
	filename703 = '/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191114/blade_dipole_infinite_soil_metal_GP_integral_accuracy_enhanced3_matrix_precision_double.ra1'


	filename801 = '/run/media/raul/WD_RED_6TB/EDGES_vol2/others/beam_simulations/wipl-d/20191122/mid_band_perf_30x30.ra1'
	
	
	
	
	
	if case == 0:
		filename = filename0

	if case == 1:
		filename = filename1

	if case == 2:
		filename = filename2
		
	if case == 3:
		filename = filename3
		
	if case == 4:
		filename = filename4
		
	if case == 5:
		filename = filename5
		



	if case == 11:
		filename = filename11
		
	if case == 12:
		filename = filename12
		
	if case == 13:
		filename = filename13
		
	if case == 14:
		filename = filename14
		
	if case == 15:
		filename = filename15

	if case == 16:
		filename = filename16
		
		
	

	if case == 51:
		filename = filename51
		
	if case == 52:
		filename = filename52
		
	if case == 53:
		filename = filename53
		
	if case == 54:
		filename = filename54
		
	if case == 55:
		filename = filename55

	if case == 56:
		filename = filename56




	if case == 101:
		filename = filename101
		
	if case == 102:
		filename = filename102
		
	if case == 103:
		filename = filename103
		
	if case == 104:
		filename = filename104
		
	if case == 105:
		filename = filename105

	if case == 106:
		filename = filename106
		



	if case == 200:
		filename = filename200

	if case == 201:
		filename = filename201

	if case == 202:
		filename = filename202

	if case == 203:
		filename = filename203

	if case == 204:
		filename = filename204

	if case == 205:
		filename = filename205

	if case == 206:
		filename = filename206



	if case == 300:
		filename = filename300

	if case == 301:
		filename = filename301
		
	if case == 302:
		filename = filename302
		
	if case == 303:
		filename = filename303

	if case == 304:
		filename = filename304
		
	if case == 305:
		filename = filename305

	if case == 306:
		filename = filename306




	if case == 400:
		filename = filename400

	if case == 401:
		filename = filename401
		
	if case == 402:
		filename = filename402
		
	if case == 403:
		filename = filename403

	if case == 404:
		filename = filename404
		
	if case == 405:
		filename = filename405

	if case == 406:
		filename = filename406




	if case == 500:
		filename = filename500

	if case == 501:
		filename = filename501




	if case == 601:
		filename = filename601

	if case == 602:
		filename = filename602



	if case == 701:
		filename = filename701

	if case == 702:
		filename = filename702

	if case == 703:
		filename = filename703
		
		
		
	if case == 801:
		filename = filename801





		
	f, thetaX, phiX, beam = ba.WIPLD_beam_read(filename)
	
	if case == 0:
		theta = np.copy(thetaX)
		m     = np.copy(beam)
	else:
		theta = thetaX[thetaX<90]
		m     = beam[:,thetaX<90,:]

	
	
	
	
	sin_theta      = np.sin(theta*(np.pi/180))
	sin_theta_2D_T = np.tile(sin_theta, (360, 1))
	sin_theta_2D   = sin_theta_2D_T.T
	
	bint = np.zeros(len(f))
	for i in range(len(f)):
		
		bt      = m[i,:,:]
		bint[i] = np.sum(bt * sin_theta_2D)
	
	b = (1/(4*np.pi)) * ((np.pi/180)**2)*bint   #/np.mean(bx)
	
	#Nfg   = 5
	fLOW  = 50
	fHIGH = 200
	fX    = f[(f>=fLOW) & (f<=fHIGH)]
	bX    = b[(f>=fLOW) & (f<=fHIGH)]
	
	x     = np.polyfit(fX, bX, Nfg-1)
	model = np.polyval(x, fX)
	rX    = bX - model
	
	
	#deltaW = np.zeros((len(m[:,0,0])-1, len(m[0,:,0]), len(m[0,0,:])))
	#for i in range(len(f)-1):
		#deltaW[i,:,:] = m[i+1,:,:] - m[i,:,:]
	
	
	return f, b, fX, rX, thetaX, phiX, beam




















def plots_for_memo_153(figname):
	
	path_plot_save = edges_folder + 'plots/20191025/'
	
	
	sx = 8
	sy = 10
	
	if figname == 1:
		
		Nfg = 5
		f, b0, f0, r0 = integrated_antenna_gain_WIPLD(0, Nfg)

		plt.close()
		plt.close()
		plt.close()
		plt.figure(figsize=[sx, sy])
		
		plt.subplot(2,1,1)
		plt.plot(f, b0, 'k')
		plt.ylim([0.999, 1.001])
		plt.ylabel('integrated gain over the full sphere\n [fraction of 4pi]')
	
		plt.subplot(2,1,2)
		plt.plot(f0, r0, 'k')
		plt.ylim([-0.000005, 0.000005])
		plt.ylabel('residuals')
		plt.xlabel('frequency [MHz]')
		
		plt.savefig(path_plot_save + 'fig1.pdf', bbox_inches='tight')
		plt.close()	
		plt.close()		


	if figname == 2:
		
		Nfg = 5
		f, b1, f1, r1 = integrated_antenna_gain_WIPLD(1, Nfg)
		f, b2, f2, r2 = integrated_antenna_gain_WIPLD(2, Nfg)
		f, b3, f3, r3 = integrated_antenna_gain_WIPLD(3, Nfg)
		f, b4, f4, r4 = integrated_antenna_gain_WIPLD(4, Nfg)
		f, b5, f5, r5 = integrated_antenna_gain_WIPLD(5, Nfg)	

		plt.close()
		plt.close()
		plt.close()
		plt.figure(figsize=[sx, sy])
		
		plt.subplot(2,1,1)
		plt.plot(f, b1, 'k')
		plt.plot(f, b2, 'b--')
		plt.plot(f, b3, 'b:')
		plt.plot(f, b4, 'r--')
		plt.plot(f, b5, 'r:')
		plt.ylabel('integrated gian above horizon\n [fraction of 4pi]')
		plt.legend([r'$\epsilon_r$=3.5, $\sigma$=0.02', r'$\epsilon_r$=2.5, $\sigma$=0.02', r'$\epsilon_r$=4.5, $\sigma$=0.02', r'$\epsilon_r$=3.5, $\sigma$=0.002', r'$\epsilon_r$=3.5, $\sigma$=0.2'], ncol=2, loc=0)
		
	
		plt.subplot(2,1,2)
		plt.plot(f1, r1, 'k')
		plt.plot(f2, r2, 'b--')
		plt.plot(f3, r3, 'b:')
		plt.plot(f4, r4, 'r--')
		plt.plot(f5, r5, 'r:')
		plt.ylim([-0.00002, 0.00002])
		plt.ylabel('residuals')
		plt.xlabel('frequency [MHz]')
		
		plt.savefig(path_plot_save + 'fig2.pdf', bbox_inches='tight')
		plt.close()	
		plt.close()
	

	if figname == 3:
		
		Nfg = 7
		f, b1, f1, r1 = integrated_antenna_gain_WIPLD(11, Nfg)
		f, b2, f2, r2 = integrated_antenna_gain_WIPLD(12, Nfg)
		f, b3, f3, r3 = integrated_antenna_gain_WIPLD(13, Nfg)
		f, b4, f4, r4 = integrated_antenna_gain_WIPLD(14, Nfg)
		f, b5, f5, r5 = integrated_antenna_gain_WIPLD(15, Nfg)
		f, b6, f6, r6 = integrated_antenna_gain_WIPLD(16, Nfg)

		plt.close()
		plt.close()
		plt.close()
		plt.figure(figsize=[sx, sy])
		
		plt.subplot(2,1,1)
		plt.plot(f, b1, 'k')
		#plt.plot(f, b2, 'k:')
		plt.plot(f, b3, 'b--')
		plt.plot(f, b4, 'b:')
		plt.plot(f, b5, 'r--')
		plt.plot(f, b6, 'r:')
		plt.ylabel('integrated gian above horizon\n [fraction of 4pi]')
		plt.legend([r'$\epsilon_r$=3.5, $\sigma$=0.02', r'$\epsilon_r$=2.5, $\sigma$=0.02', r'$\epsilon_r$=4.5, $\sigma$=0.02', r'$\epsilon_r$=3.5, $\sigma$=0.002', r'$\epsilon_r$=3.5, $\sigma$=0.2'], ncol=2, loc=0)	
		
	
		plt.subplot(2,1,2)
		plt.plot(f1, r1, 'k')
		#plt.plot(f2, r2, 'k:')
		plt.plot(f3, r3, 'b--')
		plt.plot(f4, r4, 'b:')
		plt.plot(f5, r5, 'r--')
		plt.plot(f6, r6, 'r:')
		plt.ylim([-0.0006, 0.0006])
		plt.ylabel('residuals')
		plt.xlabel('frequency [MHz]')
		
		plt.savefig(path_plot_save + 'fig3.pdf', bbox_inches='tight')
		plt.close()	
		plt.close()
	
	
	
	
	if figname == 4:
		
		Nfg = 7
		f, b1, f1, r1 = integrated_antenna_gain_WIPLD(51, Nfg)
		f, b2, f2, r2 = integrated_antenna_gain_WIPLD(52, Nfg)
		f, b3, f3, r3 = integrated_antenna_gain_WIPLD(53, Nfg)
		f, b4, f4, r4 = integrated_antenna_gain_WIPLD(54, Nfg)
		f, b5, f5, r5 = integrated_antenna_gain_WIPLD(55, Nfg)
		f, b6, f6, r6 = integrated_antenna_gain_WIPLD(56, Nfg)

		plt.close()
		plt.close()
		plt.close()
		plt.figure(figsize=[sx, sy])
		
		plt.subplot(2,1,1)
		plt.plot(f, b1, 'k')
		#plt.plot(f, b2, 'k:')
		plt.plot(f, b3, 'b--')
		plt.plot(f, b4, 'b:')
		plt.plot(f, b5, 'r--')
		plt.plot(f, b6, 'r:')
		plt.ylabel('integrated gian above horizon\n [fraction of 4pi]')
		plt.legend([r'$\epsilon_r$=3.5, $\sigma$=0.02', r'$\epsilon_r$=2.5, $\sigma$=0.02', r'$\epsilon_r$=4.5, $\sigma$=0.02', r'$\epsilon_r$=3.5, $\sigma$=0.002', r'$\epsilon_r$=3.5, $\sigma$=0.2'], ncol=2, loc=0)
		
	
		plt.subplot(2,1,2)
		plt.plot(f1, r1, 'k')
		#plt.plot(f2, r2, 'k:')
		plt.plot(f3, r3, 'b--')
		plt.plot(f4, r4, 'b:')
		plt.plot(f5, r5, 'r--')
		plt.plot(f6, r6, 'r:')
		plt.ylim([-0.0004, 0.0004])
		plt.ylabel('residuals')
		plt.xlabel('frequency [MHz]')
			
		plt.savefig(path_plot_save + 'fig4.pdf', bbox_inches='tight')
		plt.close()	
		plt.close()
	
	
	
	

	if figname == 5:
		
		Nfg = 7
		f, b1, f1, r1 = integrated_antenna_gain_WIPLD(101, Nfg)
		f, b2, f2, r2 = integrated_antenna_gain_WIPLD(102, Nfg)
		f, b3, f3, r3 = integrated_antenna_gain_WIPLD(103, Nfg)
		f, b4, f4, r4 = integrated_antenna_gain_WIPLD(104, Nfg)
		f, b5, f5, r5 = integrated_antenna_gain_WIPLD(105, Nfg)
		f, b6, f6, r6 = integrated_antenna_gain_WIPLD(106, Nfg)

		plt.close()
		plt.close()
		plt.close()
		plt.figure(figsize=[sx, sy])
		
		plt.subplot(2,1,1)
		plt.plot(f, b1, 'k')
		#plt.plot(f, b2, 'k:')
		plt.plot(f, b3, 'b--')
		plt.plot(f, b4, 'b:')
		plt.plot(f, b5, 'r--')
		plt.plot(f, b6, 'r:')
		plt.ylabel('integrated gian above horizon\n [fraction of 4pi]')
		plt.legend([r'$\epsilon_r$=3.5, $\sigma$=0.02', r'$\epsilon_r$=2.5, $\sigma$=0.02', r'$\epsilon_r$=4.5, $\sigma$=0.02', r'$\epsilon_r$=3.5, $\sigma$=0.002', r'$\epsilon_r$=3.5, $\sigma$=0.2'], ncol=2, loc=0)
		
	
		plt.subplot(2,1,2)
		plt.plot(f1, r1, 'k')
		#plt.plot(f2, r2, 'k:')
		plt.plot(f3, r3, 'b--')
		plt.plot(f4, r4, 'b:')
		plt.plot(f5, r5, 'r--')
		plt.plot(f6, r6, 'r:')
		plt.ylim([-0.0002, 0.0002])
		plt.ylabel('residuals')
		plt.xlabel('frequency [MHz]')
			
		plt.savefig(path_plot_save + 'fig5.pdf', bbox_inches='tight')
		plt.close()	
		plt.close()


	
	
	
	return 0









def plots_for_memo_155():
	
	path_plot_save = edges_folder + 'plots/20191105/'
	
	
	fA, b300, fx, rx, theta, phi, beam300= integrated_antenna_gain_WIPLD(300, 2)
	fA, b301, fx, rx, theta, phi, beam301= integrated_antenna_gain_WIPLD(301, 2)
	fA, b302, fx, rx, theta, phi, beam302= integrated_antenna_gain_WIPLD(302, 2)
	fA, b303, fx, rx, theta, phi, beam303= integrated_antenna_gain_WIPLD(303, 2)
	fA, b304, fx, rx, theta, phi, beam304= integrated_antenna_gain_WIPLD(304, 2)
	fA, b305, fx, rx, theta, phi, beam305= integrated_antenna_gain_WIPLD(305, 2)
	fA, b306, fx, rx, theta, phi, beam306= integrated_antenna_gain_WIPLD(306, 2)
	
	
	fB, b400, fx, rx, theta, phi, beam400= integrated_antenna_gain_WIPLD(400, 2)
	fB, b401, fx, rx, theta, phi, beam401= integrated_antenna_gain_WIPLD(401, 2)
	fB, b402, fx, rx, theta, phi, beam402= integrated_antenna_gain_WIPLD(402, 2)
	fB, b403, fx, rx, theta, phi, beam403= integrated_antenna_gain_WIPLD(403, 2)
	fB, b404, fx, rx, theta, phi, beam404= integrated_antenna_gain_WIPLD(404, 2)
	fB, b405, fx, rx, theta, phi, beam405= integrated_antenna_gain_WIPLD(405, 2)
	fB, b406, fx, rx, theta, phi, beam406= integrated_antenna_gain_WIPLD(406, 2)
	
	fC, b500, fx, rx, theta, phi, beam500= integrated_antenna_gain_WIPLD(500, 2)
	fC, b501, fx, rx, theta, phi, beam501= integrated_antenna_gain_WIPLD(501, 2)
	





	# --------------------------------------------
	# FEKO
	# --------------------------------------------

	fmin = 1
	fmax = 300
	
	el             = np.arange(0,91) 
	sin_theta      = np.sin((90-el)*(np.pi/180)) 
	sin_theta_2D_T = np.tile(sin_theta, (360, 1))
	sin_theta_2D   = sin_theta_2D_T.T			
	


	# Low-Band, Blade on 10m x 10m GP
	b_all  = oeg.FEKO_low_band_blade_beam(beam_file=5, frequency_interpolation='no', frequency=np.array([0]), AZ_antenna_axis=0)
	f      = np.arange(50,121,2)
	#f      = np.arange(40,101,2.5)
	
	
	bint = np.zeros(len(f))
	for i in range(len(f)):
		
		b       = b_all[i,:,:]
		bint[i] = np.sum(b * sin_theta_2D)
	
	ft  = f[(f>=fmin) & (f<=fmax)]
	bx  = bint[(f>=fmin) & (f<=fmax)]
	bt   = (1/(4*np.pi)) * ((np.pi/180)**2)*bx   #/np.mean(bx)
	
	flow10 = np.copy(ft)
	blow10 = np.copy(bt)
	beam_low10 = np.copy(b_all)
	
	
	
	# Low-Band, Blade on 30m x 30m GP
	b_all = oeg.FEKO_low_band_blade_beam(beam_file=2, frequency_interpolation='no', AZ_antenna_axis=0)
	f     = np.arange(40,121,2)		
	
	bint = np.zeros(len(f))
	for i in range(len(f)):
		
		b       = b_all[i,:,:]
		bint[i] = np.sum(b * sin_theta_2D)
	
	ft  = f[(f>=fmin) & (f<=fmax)]
	bx  = bint[(f>=fmin) & (f<=fmax)]
	bt   = (1/(4*np.pi)) * ((np.pi/180)**2)*bx   #/np.mean(bx)

	flow30 = np.copy(ft)
	blow30 = np.copy(bt)		
	beam_low30 = np.copy(b_all)



	# Mid-Band, Blade, on 30m x 30m GP
	b_all = cal.FEKO_blade_beam('mid_band', 0, frequency_interpolation='no', AZ_antenna_axis=90)
	f     = np.arange(50,201,2)		
	
	bint = np.zeros(len(f))
	for i in range(len(f)):
		
		b       = b_all[i,:,:]
		bint[i] = np.sum(b * sin_theta_2D)
	
	ft  = f[(f>=fmin) & (f<=fmax)]
	bx  = bint[(f>=fmin) & (f<=fmax)]
	bt   = (1/(4*np.pi)) * ((np.pi/180)**2)*bx   #/np.mean(bx)
	
	fmid30 = np.copy(ft)
	bmid30 = np.copy(bt)
	beam_mid30 = np.copy(b_all)
	
	








	
	
	
	# Figure 1
	# --------------------------------------------
		
	plt.figure(1, figsize=[10,12])
	plt.plot(fA, b301, 'r')
	plt.plot(fA, b305, 'y')
	plt.plot(fA, b302, 'm')
	plt.plot(fA, b303, 'c')
	plt.plot(fA, b306, 'b')
	plt.plot(fA, b304, 'g')
	plt.plot(fA, b300, 'k')
	
	plt.plot(fB, b401, 'r--')
	plt.plot(fB, b405, 'y--')
	plt.plot(fB, b402, 'm--')
	plt.plot(fB, b403, 'c--')
	plt.plot(fB, b406, 'b--')
	plt.plot(fB, b404, 'g--')
	plt.plot(fB, b400, 'k--')
	
	
	
	
	
	
	
	
	
	
	
	
	plt.ylim([0.988, 1.002])
	
	plt.xlabel('frequency [MHz]')
	plt.ylabel('integrated gain above horizon\n [fraction of 4pi]')


	plt.legend(['10mx10m', 'perf 15mx15m', '15mx15m', '20mx20m', 'perf 30mx30m', '30mx30m', 'Inf PEC'], loc=0)
	
	plt.savefig(path_plot_save + 'fig1.pdf', bbox_inches='tight')
	plt.close()	
	plt.close()
	
	


	# Figure 2
	# --------------------------------------------
	
	plt.figure(2, figsize=[10,8])
	plt.plot(fA, b301, 'r')
	plt.plot(fC, b501, 'r--')


	plt.plot(fA, b306, 'b')
	plt.plot(fC, b500, 'b--')

	plt.xlabel('frequency [MHz]')
	plt.ylabel('integrated gain above horizon\n [fraction of 4pi]')
	
	plt.legend(['10mx10m, mid-band', '10mx10m, low-band', 'perf 30mx30m, mid-band', 'perf 30mx30m, low-band'], loc=0)
	
	plt.savefig(path_plot_save + 'fig2.pdf', bbox_inches='tight')
	plt.close()	
	plt.close()
	
	
	
	
	
	# Figure 3
	# --------------------------------------------
	
	plt.figure(3, figsize=[10,8])
	plt.plot(fA, 100*(1 - b301), 'r')
	plt.plot(fC, 100*(1 - b501), 'r--')

	plt.plot(fA, 100*(1 - b306), 'b')
	plt.plot(fC, 100*(1 - b500), 'b--')

	plt.xlabel('frequency [MHz]')
	plt.ylabel('loss [%]')
	
	plt.legend(['10mx10m, mid-band', '10mx10m, low-band', 'perf 30mx30m, mid-band', 'perf 30mx30m, low-band'], loc=0)	
	
	plt.savefig(path_plot_save + 'fig3.pdf', bbox_inches='tight')
	plt.close()	
	plt.close()
		
	
	
	
	# Figure 4
	# --------------------------------------------
	
	plt.figure(4, figsize=[10,13])
	plt.plot(fA, b301, 'r')
	plt.plot(fC, b501, 'r--')
	plt.plot(flow10, blow10, 'm--')


	plt.plot(fA, b306, 'b')
	plt.plot(fmid30, bmid30, 'c')
	plt.plot(fC, b500, 'b--')
	plt.plot(flow30, blow30, 'c--')

	plt.xlabel('frequency [MHz]')
	plt.ylabel('integrated gain above horizon\n [fraction of 4pi]')
	
	plt.legend(['10mx10m, mid-band', '10mx10m, low-band', '10mx10m, low-band FEKO', 'perf 30mx30m, mid-band', 'perf 30mx30m, mid-band FEKO', 'perf 30mx30m, low-band', 'perf 30mx30m, low-band FEKO'], loc=0)
	
	plt.savefig(path_plot_save + 'fig4.pdf', bbox_inches='tight')
	plt.close()	
	plt.close()	
	
	

	
	
	# Figure 5
	# --------------------------------------------	
	
	plt.figure(5, figsize=[10,8])
	plt.plot(fA, beam301[:,0,0], 'r')
	plt.plot(fC, beam501[:,0,0], 'r--')
	plt.plot(fA, beam306[:,0,0], 'b')
	plt.plot(fC, beam500[:,0,0], 'b--')
	
	plt.plot(fA, beam301[:,45,0], 'r')
	plt.plot(fC, beam501[:,45,0], 'r--')
	plt.plot(fA, beam306[:,45,0], 'b')
	plt.plot(fC, beam500[:,45,0], 'b--')	
	
	plt.plot(fA, beam301[:,45,90], 'r')
	plt.plot(fC, beam501[:,45,90], 'r--')
	plt.plot(fA, beam306[:,45,90], 'b')
	plt.plot(fC, beam500[:,45,90], 'b--')
	
	plt.xlim([50, 140])
	plt.ylim([0, 8])
	
	plt.xlabel('frequency [MHz]')
	plt.ylabel('gain')
	
	plt.legend(['10mx10m, mid-band', '10mx10m, low-band', 'perf 30mx30m, mid-band', 'perf 30mx30m, low-band'], loc=0)
	
	plt.text(60,7.3, 'Zenith', fontsize=16)
	plt.text(60,4.3, r'$\theta=45^{\circ}$, $\phi=90^{\circ}$', fontsize=16)
	plt.text(60,2.1, r'$\theta=45^{\circ}$, $\phi=0^{\circ}$', fontsize=16)
	
	plt.savefig(path_plot_save + 'fig5.pdf', bbox_inches='tight')
	plt.close()	
	plt.close()	
	
	
	
	
	
	
	
	# Figure 6
	# --------------------------------------------	
	
	plt.figure(6, figsize=[10,8])
	plt.plot(flow10, beam_low10[:,-1,0], 'r--')
	plt.plot(fmid30, beam_mid30[:,-1,0], 'b')
	plt.plot(flow30, beam_low30[:,-1,0], 'b--')
	
	plt.plot(flow10, beam_low10[:,45,0], 'r--')
	plt.plot(fmid30, beam_mid30[:,45,0], 'b')
	plt.plot(flow30, beam_low30[:,45,0], 'b--')
	
	plt.plot(flow10, beam_low10[:,45,90], 'r--')
	plt.plot(fmid30, beam_mid30[:,45,90], 'b')
	plt.plot(flow30, beam_low30[:,45,90], 'b--')	

	
	plt.xlim([50, 140])
	plt.ylim([0, 8])
	
	plt.xlabel('frequency [MHz]')
	plt.ylabel('gain')
	
	plt.legend(['10mx10m, low-band', 'perf 30mx30m, mid-band', 'perf 30mx30m, low-band'], loc=0)
	
	plt.text(60,7.3, 'Zenith', fontsize=16)
	plt.text(60,4.3, r'$\theta=45^{\circ}$, $\phi=90^{\circ}$', fontsize=16)
	plt.text(60,2.1, r'$\theta=45^{\circ}$, $\phi=0^{\circ}$', fontsize=16)	
	
	plt.savefig(path_plot_save + 'fig6.pdf', bbox_inches='tight')
	plt.close()	
	plt.close()	
	
		
	
	
	
	
	
	
	
	
	
	
	
	

	# Figure 7
	# --------------------------------------------	
	# Mid-Band, 10mx10m
	
	delta = np.zeros((len(beam301[:,0,0])-1, len(beam301[0,:,0]), len(beam301[0,0,:])))
	for i in range(len(fA)-1):
		XX  = beam301[i+1,:,:] - beam301[i,:,:]
		XXX = np.flipud(XX)
		delta[i,:,:] = XXX	
	
	plt.figure(7, figsize=(7,7))
	
	plt.subplot(2,1,1)
	plt.imshow(np.flipud(delta[:,:,0].T), interpolation='none', aspect='auto', extent=[50, 200, 90, 0], vmin=-0.05, vmax=0.05);
	cbar = plt.colorbar()
	plt.yticks([0, 30, 60, 90])
	plt.xticks([50, 75, 100, 125, 150, 175, 200], labels=[])
	cbar.set_label(r'$\Delta$ directive gain per MHz', rotation=90)
	plt.ylabel('theta [deg]')
	plt.title('phi=0 [deg]')	
	
	plt.subplot(2,1,2)
	plt.imshow(np.flipud(delta[:,:,90].T), interpolation='none', aspect='auto', extent=[50, 200, 90, 0], vmin=-0.05, vmax=0.05);
	cbar = plt.colorbar()
	plt.yticks([0, 30, 60, 90])
	plt.xticks([50, 75, 100, 125, 150, 175, 200])
	cbar.set_label(r'$\Delta$ directive gain per MHz', rotation=90)
	plt.ylabel('theta [deg]')
	plt.xlabel('frequency [MHz]')
	plt.title('phi=90 [deg]')

	plt.savefig(path_plot_save + 'fig7.pdf', bbox_inches='tight')
	plt.close()	
	plt.close()
	
	
	
	
	# Figure 8
	# --------------------------------------------	
	# Mid-Band, 30mx30m
	
	delta = np.zeros((len(beam306[:,0,0])-1, len(beam306[0,:,0]), len(beam306[0,0,:])))
	for i in range(len(fA)-1):
		XX  = beam306[i+1,:,:] - beam306[i,:,:]
		XXX = np.flipud(XX)
		delta[i,:,:] = XXX	
	
	plt.figure(8, figsize=(7,7))
	
	plt.subplot(2,1,1)
	plt.imshow(np.flipud(delta[:,:,0].T), interpolation='none', aspect='auto', extent=[50, 200, 90, 0], vmin=-0.05, vmax=0.05);
	cbar = plt.colorbar()
	plt.yticks([0, 30, 60, 90])
	plt.xticks([50, 75, 100, 125, 150, 175, 200], labels=[])
	cbar.set_label(r'$\Delta$ directive gain per MHz', rotation=90)
	plt.ylabel('theta [deg]')
	plt.title('phi=0 [deg]')	
	
	plt.subplot(2,1,2)
	plt.imshow(np.flipud(delta[:,:,90].T), interpolation='none', aspect='auto', extent=[50, 200, 90, 0], vmin=-0.05, vmax=0.05);
	cbar = plt.colorbar()
	plt.yticks([0, 30, 60, 90])
	plt.xticks([50, 75, 100, 125, 150, 175, 200])
	cbar.set_label(r'$\Delta$ directive gain per MHz', rotation=90)
	plt.ylabel('theta [deg]')
	plt.xlabel('frequency [MHz]')
	plt.title('phi=90 [deg]')

	plt.savefig(path_plot_save + 'fig8.pdf', bbox_inches='tight')
	plt.close()	
	plt.close()	
	
	
	
	
	
	

	# Figure 9
	# --------------------------------------------	
	# Low-Band, 10mx10m
	
	delta = np.zeros((len(beam501[:,0,0])-1, len(beam501[0,:,0]), len(beam501[0,0,:])))
	for i in range(len(fC)-1):
		XX  = beam501[i+1,:,:] - beam501[i,:,:]
		XXX = np.flipud(XX)
		delta[i,:,:] = XXX	
	
	plt.figure(9, figsize=(7,7))
	
	plt.subplot(2,1,1)
	plt.imshow(np.flipud(delta[:,:,0].T), interpolation='none', aspect='auto', extent=[50, 200, 90, 0], vmin=-0.05, vmax=0.05);
	cbar = plt.colorbar()
	plt.yticks([0, 30, 60, 90])
	plt.xticks([50, 75, 100, 125, 150, 175, 200], labels=[])
	cbar.set_label(r'$\Delta$ directive gain per MHz', rotation=90)
	plt.ylabel('theta [deg]')
	plt.title('phi=0 [deg]')	
	
	plt.subplot(2,1,2)
	plt.imshow(np.flipud(delta[:,:,90].T), interpolation='none', aspect='auto', extent=[50, 200, 90, 0], vmin=-0.05, vmax=0.05);
	cbar = plt.colorbar()
	plt.yticks([0, 30, 60, 90])
	plt.xticks([50, 75, 100, 125, 150, 175, 200])
	cbar.set_label(r'$\Delta$ directive gain per MHz', rotation=90)
	plt.ylabel('theta [deg]')
	plt.xlabel('frequency [MHz]')
	plt.title('phi=90 [deg]')

	plt.savefig(path_plot_save + 'fig9.pdf', bbox_inches='tight')
	plt.close()	
	plt.close()
	
	
	
	# Figure 10
	# --------------------------------------------	
	# Low-Band, 30mx30m
	
	delta = np.zeros((len(beam500[:,0,0])-1, len(beam500[0,:,0]), len(beam500[0,0,:])))
	for i in range(len(fC)-1):
		XX  = beam500[i+1,:,:] - beam500[i,:,:]
		XXX = np.flipud(XX)
		delta[i,:,:] = XXX	
	
	plt.figure(10, figsize=(7,7))
	
	plt.subplot(2,1,1)
	plt.imshow(np.flipud(delta[:,:,0].T), interpolation='none', aspect='auto', extent=[50, 200, 90, 0], vmin=-0.05, vmax=0.05);
	cbar = plt.colorbar()
	plt.yticks([0, 30, 60, 90])
	plt.xticks([50, 75, 100, 125, 150, 175, 200], labels=[])
	cbar.set_label(r'$\Delta$ directive gain per MHz', rotation=90)
	plt.ylabel('theta [deg]')
	plt.title('phi=0 [deg]')	
	
	plt.subplot(2,1,2)
	plt.imshow(np.flipud(delta[:,:,90].T), interpolation='none', aspect='auto', extent=[50, 200, 90, 0], vmin=-0.05, vmax=0.05);
	cbar = plt.colorbar()
	plt.yticks([0, 30, 60, 90])
	plt.xticks([50, 75, 100, 125, 150, 175, 200])
	cbar.set_label(r'$\Delta$ directive gain per MHz', rotation=90)
	plt.ylabel('theta [deg]')
	plt.xlabel('frequency [MHz]')
	plt.title('phi=90 [deg]')

	plt.savefig(path_plot_save + 'fig10.pdf', bbox_inches='tight')
	plt.close()	
	plt.close()	
	
	
	
	
	
	# Figure 11
	# --------------------------------------------	
	# Mid-Band, 30mx30m
	
	delta = np.zeros((len(beam_mid30[:,0,0])-1, len(beam_mid30[0,:,0]), len(beam_mid30[0,0,:])))
	for i in range(len(fmid30)-1):
		XX  = beam_mid30[i+1,:,:] - beam_mid30[i,:,:]
		XXX = np.flipud(XX)
		delta[i,:,:] = XXX	
	
	plt.figure(11, figsize=(7,7))
	
	plt.subplot(2,1,1)
	plt.imshow(delta[:,:,0].T, interpolation='none', aspect='auto', extent=[50, 200, 90, 0], vmin=-0.05, vmax=0.05);
	cbar = plt.colorbar()
	plt.yticks([0, 30, 60, 90])
	plt.xticks([50, 75, 100, 125, 150, 175, 200], labels=[])
	cbar.set_label(r'$\Delta$ directive gain per MHz', rotation=90)
	plt.ylabel('theta [deg]')
	plt.title('phi=0 [deg]')	
	
	plt.subplot(2,1,2)
	plt.imshow(delta[:,:,90].T, interpolation='none', aspect='auto', extent=[50, 200, 90, 0], vmin=-0.05, vmax=0.05);
	cbar = plt.colorbar()
	plt.yticks([0, 30, 60, 90])
	plt.xticks([50, 75, 100, 125, 150, 175, 200])
	cbar.set_label(r'$\Delta$ directive gain per MHz', rotation=90)
	plt.ylabel('theta [deg]')
	plt.xlabel('frequency [MHz]')
	plt.title('phi=90 [deg]')

	plt.savefig(path_plot_save + 'fig11.pdf', bbox_inches='tight')
	plt.close()	
	plt.close()	



	
	

	# Figure 12
	# --------------------------------------------	
	# Low-Band, 30mx30m
	
	delta = np.zeros((len(beam_low30[:,0,0])-1, len(beam_low30[0,:,0]), len(beam_low30[0,0,:])))
	for i in range(len(flow30)-1):
		XX  = beam_low30[i+1,:,:] - beam_low30[i,:,:]
		XXX = np.flipud(XX)
		delta[i,:,:] = XXX	
	
	plt.figure(12, figsize=(7,7))
	
	plt.subplot(2,1,1)
	plt.imshow(delta[:,:,0].T, interpolation='none', aspect='auto', extent=[40, 120, 90, 0], vmin=-0.05, vmax=0.05);
	cbar = plt.colorbar()
	plt.yticks([0, 30, 60, 90])
	plt.xticks([40, 60, 80, 100, 120], labels=[])
	cbar.set_label(r'$\Delta$ directive gain per MHz', rotation=90)
	plt.ylabel('theta [deg]')
	plt.title('phi=0 [deg]')	
	
	plt.subplot(2,1,2)
	plt.imshow(delta[:,:,90].T, interpolation='none', aspect='auto', extent=[40, 120, 90, 0], vmin=-0.05, vmax=0.05);
	cbar = plt.colorbar()
	plt.yticks([0, 30, 60, 90])
	plt.xticks([40, 60, 80, 100, 120])
	cbar.set_label(r'$\Delta$ directive gain per MHz', rotation=90)
	plt.ylabel('theta [deg]')
	plt.xlabel('frequency [MHz]')
	plt.title('phi=90 [deg]')

	plt.savefig(path_plot_save + 'fig12.pdf', bbox_inches='tight')
	plt.close()	
	plt.close()	





	# Figure 13
	# --------------------------------------------	
	# Low-Band, 10mx10m
	
	delta = np.zeros((len(beam_low10[:,0,0])-1, len(beam_low10[0,:,0]), len(beam_low10[0,0,:])))
	for i in range(len(flow10)-1):
		XX  = beam_low10[i+1,:,:] - beam_low10[i,:,:]
		XXX = np.flipud(XX)
		delta[i,:,:] = XXX	
	
	plt.figure(13, figsize=(7,7))
	
	plt.subplot(2,1,1)
	plt.imshow(delta[:,:,0].T, interpolation='none', aspect='auto', extent=[50, 120, 90, 0], vmin=-0.05, vmax=0.05);
	cbar = plt.colorbar()
	plt.yticks([0, 30, 60, 90])
	plt.xticks([50, 60, 70, 80, 90, 100, 110, 120], labels=[])
	cbar.set_label(r'$\Delta$ directive gain per MHz', rotation=90)
	plt.ylabel('theta [deg]')
	plt.title('phi=0 [deg]')	
	
	plt.subplot(2,1,2)
	plt.imshow(delta[:,:,90].T, interpolation='none', aspect='auto', extent=[50, 120, 90, 0], vmin=-0.05, vmax=0.05);
	cbar = plt.colorbar()
	plt.yticks([0, 30, 60, 90])
	plt.xticks([50, 60, 70, 80, 90, 100, 110, 120])
	cbar.set_label(r'$\Delta$ directive gain per MHz', rotation=90)
	plt.ylabel('theta [deg]')
	plt.xlabel('frequency [MHz]')
	plt.title('phi=90 [deg]')

	plt.savefig(path_plot_save + 'fig13.pdf', bbox_inches='tight')
	plt.close()	
	plt.close()	








	delta = np.zeros((len(beam[:,0,0])-1, len(beam[0,:,0]), len(beam[0,0,:])))
	for i in range(len(f)-1):
		XX  = beam[i+1,:,:] - beam[i,:,:]
		XXX = np.flipud(XX)
		delta[i,:,:] = XXX

	
	
	
	
	return fA, beam301, delta, flow10, beam_low10







def plot_calibration_s11():
	
	path_plot_save = edges_folder + 'plots/20191112/'
	
	
	# Main folder
	main_folder     = edges_folder + 'mid_band/calibration/receiver_calibration/receiver1/2019_10_25C/'

	# Paths for source data
	path_s11        = main_folder + 'data/s11/corrected/'
	
	d1 = np.genfromtxt(path_s11 + 's11_calibration_mid_band_LNA25degC_2019-11-09-19-13-48_second_set_of_measurements.txt')
	d2 = np.genfromtxt(path_s11 + 's11_calibration_mid_band_LNA25degC_2019-11-12-21-08-17_first_set_of_measurements.txt')


	plt.figure(1, figsize=(12,14))
	
	plt.subplot(3,2,1)
	plt.plot(d2[:,0], 20*np.log10(np.abs(d2[:,3]+1j*d2[:,4])), 'b')
	plt.plot(d1[:,0], 20*np.log10(np.abs(d1[:,3]+1j*d1[:,4])), 'b--')
	plt.plot(d2[:,0], 20*np.log10(np.abs(d2[:,5]+1j*d2[:,6])), 'r')
	plt.plot(d1[:,0], 20*np.log10(np.abs(d1[:,5]+1j*d1[:,6])), 'r--')
	plt.ylabel('magnitude [dB]')
	plt.legend(['ambient rep1', 'ambient rep2','hot rep1', 'hot rep2'])
	
	plt.subplot(3,2,2)
	plt.plot(d1[:,0], (180/np.pi)*np.unwrap(np.angle(d1[:,3]+1j*d1[:,4])), 'b--')
	plt.plot(d2[:,0], (180/np.pi)*np.unwrap(np.angle(d2[:,3]+1j*d2[:,4])), 'b')
	plt.plot(d1[:,0], (180/np.pi)*np.unwrap(np.angle(d1[:,5]+1j*d1[:,6])), 'r--')
	plt.plot(d2[:,0], (180/np.pi)*np.unwrap(np.angle(d2[:,5]+1j*d2[:,6])), 'r')
	plt.ylabel('phase [deg]')




	plt.subplot(3,2,3)
	plt.plot(d2[:,0], 20*np.log10(np.abs(d2[:,7]+1j*d2[:,8])), 'b')
	plt.plot(d1[:,0], 20*np.log10(np.abs(d1[:,7]+1j*d1[:,8])), 'b--')
	plt.plot(d2[:,0], 20*np.log10(np.abs(d2[:,9]+1j*d2[:,10])), 'r')
	plt.plot(d1[:,0], 20*np.log10(np.abs(d1[:,9]+1j*d1[:,10])), 'r--')
	plt.legend(['open rep1','open rep2','shorted rep1','shorted rep2'])
	plt.ylabel('magnitude [dB]')
	
	
	plt.subplot(3,2,4)
	plt.plot(d1[:,0], (180/np.pi)*np.unwrap(np.angle(d1[:,7]+1j*d1[:,8])), 'b--')
	plt.plot(d2[:,0], (180/np.pi)*np.unwrap(np.angle(d2[:,7]+1j*d2[:,8])), 'b')
	plt.plot(d1[:,0], (180/np.pi)*np.unwrap(np.angle(d1[:,9]+1j*d1[:,10])), 'r--')
	plt.plot(d2[:,0], (180/np.pi)*np.unwrap(np.angle(d2[:,9]+1j*d2[:,10])), 'r')
	plt.ylabel('phase [deg]')
	

	
	plt.subplot(3,2,5)
	plt.plot(d2[:,0], 20*np.log10(np.abs(d2[:,17]+1j*d2[:,18])), 'b')
	plt.plot(d1[:,0], 20*np.log10(np.abs(d1[:,17]+1j*d1[:,18])), 'b--')
	plt.plot(d2[:,0], 20*np.log10(np.abs(d2[:,19]+1j*d2[:,20])), 'r')
	plt.plot(d1[:,0], 20*np.log10(np.abs(d1[:,19]+1j*d1[:,20])), 'r--')
	plt.legend(['sim2 rep1','sim2 rep2', 'sim3 rep1','sim3 rep2'])
	plt.ylabel('magnitude [dB]')
	plt.xlabel('frequency [MHz]')
	
	
	plt.subplot(3,2,6)
	plt.plot(d2[:,0], (180/np.pi)*np.unwrap(np.angle(d2[:,17]+1j*d2[:,18])), 'b')
	plt.plot(d1[:,0], (180/np.pi)*np.unwrap(np.angle(d1[:,17]+1j*d1[:,18])), 'b--')
	plt.plot(d1[:,0], (180/np.pi)*np.unwrap(np.angle(d1[:,19]+1j*d1[:,20])), 'r--')
	plt.plot(d2[:,0], (180/np.pi)*np.unwrap(np.angle(d2[:,19]+1j*d2[:,20])), 'r')
	plt.ylabel('phase [deg]')
	plt.xlabel('frequency [MHz]')
	
	plt.savefig(path_plot_save + 'fig1.pdf', bbox_inches='tight')
	plt.close()	
	plt.close()	







	plt.figure(2, figsize=(12,14))
	
	plt.subplot(3,2,1)
	plt.plot(d1[:,0], 20*np.log10(np.abs(d1[:,3]+1j*d1[:,4])) - 20*np.log10(np.abs(d2[:,3]+1j*d2[:,4])), 'b')
	plt.plot(d1[:,0], 20*np.log10(np.abs(d1[:,5]+1j*d1[:,6])) - 20*np.log10(np.abs(d2[:,5]+1j*d2[:,6])), 'r')
	plt.ylabel(r'$\Delta$ magnitude [dB]')
	plt.legend(['ambient','hot'])
	
	plt.subplot(3,2,2)
	plt.plot(d1[:,0], (180/np.pi)*np.unwrap(np.angle(d1[:,3]+1j*d1[:,4])) - (180/np.pi)*np.unwrap(np.angle(d2[:,3]+1j*d2[:,4])), 'b')
	plt.plot(d1[:,0], (180/np.pi)*np.unwrap(np.angle(d1[:,5]+1j*d1[:,6])) - (180/np.pi)*np.unwrap(np.angle(d2[:,5]+1j*d2[:,6])), 'r')
	plt.ylabel(r'$\Delta$ phase [deg]')




	plt.subplot(3,2,3)
	plt.plot(d1[:,0], 20*np.log10(np.abs(d1[:,7]+1j*d1[:,8])) - 20*np.log10(np.abs(d2[:,7]+1j*d2[:,8])), 'b')
	plt.plot(d1[:,0], 20*np.log10(np.abs(d1[:,9]+1j*d1[:,10])) - 20*np.log10(np.abs(d2[:,9]+1j*d2[:,10])), 'r')
	plt.ylabel(r'$\Delta$ magnitude [dB]')
	plt.legend(['open','shorted'])
	plt.ylim([-0.006, 0.06])
	
	plt.subplot(3,2,4)
	plt.plot(d1[:,0], (180/np.pi)*np.unwrap(np.angle(d1[:,7]+1j*d1[:,8])) - (180/np.pi)*np.unwrap(np.angle(d2[:,7]+1j*d2[:,8])), 'b')
	plt.plot(d1[:,0], (180/np.pi)*np.unwrap(np.angle(d1[:,9]+1j*d1[:,10])) - (180/np.pi)*np.unwrap(np.angle(d2[:,9]+1j*d2[:,10])), 'r')
	plt.ylabel(r'$\Delta$ phase [deg]')
	

	
	plt.subplot(3,2,5)
	plt.plot(d1[:,0], 20*np.log10(np.abs(d1[:,17]+1j*d1[:,18])) - 20*np.log10(np.abs(d2[:,17]+1j*d2[:,18])), 'b')
	plt.plot(d1[:,0], 20*np.log10(np.abs(d1[:,19]+1j*d1[:,20])) - 20*np.log10(np.abs(d2[:,19]+1j*d2[:,20])), 'r')
	plt.ylabel(r'$\Delta$ magnitude [dB]')
	plt.xlabel('frequency [MHz]')
	plt.legend(['sim2','sim3'])
	plt.ylim([-0.02, 0.05])
	
	plt.subplot(3,2,6)
	plt.plot(d1[:,0], (180/np.pi)*np.unwrap(np.angle(d1[:,17]+1j*d1[:,18])) - (180/np.pi)*np.unwrap(np.angle(d2[:,17]+1j*d2[:,18])), 'b')
	plt.plot(d1[:,0], (180/np.pi)*np.unwrap(np.angle(d1[:,19]+1j*d1[:,20])) - (180/np.pi)*np.unwrap(np.angle(d2[:,19]+1j*d2[:,20])), 'r')
	plt.ylabel(r'$\Delta$ phase [deg]')
	plt.xlabel('frequency [MHz]')

	plt.savefig(path_plot_save + 'fig2.pdf', bbox_inches='tight')
	plt.close()	
	plt.close()	


	
	return 0






def plot_mid_band_GHA_14_16():
	
	
	fb, tb, rb, wb, sb = eg.level4_integration(3,[14, 15], 147, 200, 60, 135, 5)
	
	
	plt.close()
	plt.close()
	plt.close()
	plt.close()
	
	plt.figure(1, figsize=[9,4])
	
	plt.plot(fb[wb>0], rb[wb>0], 'b')
	
	sig   = dm.signal_model('tanh', [-0.7,79,19.5,7.5,4.5], fb)
	model = ba.fit_polynomial_fourier('LINLOG', fb, tb-sig,5, Weights=wb)
	mb    = ba.model_evaluate('LINLOG', model[0], fb)
	
	plt.plot(fb[wb>0], (tb-sig-mb)[wb>0]-0.8, 'b')
	
	plt.xticks(np.arange(60, 135, 10), fontsize=12)
	plt.ylim([-1.1, 0.4])
	plt.yticks([-1, -0.8, -0.6, -0.2, 0, 0.2], ['-0.2','0','0.2','-0.2','0','0.2'], fontsize=12)
	plt.xlabel(r'$\nu$ [MHz]', fontsize=14)
	plt.ylabel(r'$T_b$ [K]', fontsize=14)
	
	plt.savefig('/home/raul/Desktop/GHA_14-16hr.pdf', bbox_inches='tight')
	plt.close()	
	plt.close()		
	
	
	return 0







def plot_signal_residuals(FLOW, FHIGH, A21, model_type, Ntotal):
	
	
	#f, t, r, w, s = eg.level4_integration(501, [18,19,20,21,22,23,0,1,2,3,4,5], 148, 167, FLOW, FHIGH, Nfg)
	
	#pc       = ba.fit_polynomial_fourier(model_type, f/200, t, Nfg)
	#model_fg = ba.model_evaluate(model_type, pc[0], f/200)	
	
	
	
	
	
	f  = np.arange(FLOW, FHIGH+1, 1)
	A  = 1500
	f0 = 75
	model_fg = A*(f/f0)** ( (-2.5) + 0.1*np.log(f/f0) )
	
	sig   = dm.signal_model('exp', [A21,79,19,7], f)
	
	total = model_fg + sig
	
	
	pc = ba.fit_polynomial_fourier(model_type, f/200, total, Ntotal)
	m  = ba.model_evaluate(model_type, pc[0], f/200)
	r  = total - m
	
	
	return f, r, model_fg















