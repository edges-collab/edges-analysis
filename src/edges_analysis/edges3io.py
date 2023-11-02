"""
IO script for EDGES-3 data
Last modified: Jul 28, 2023
Author: Akshatha Vydula
Email: vydula@asu.edu

Script based on Alan's reads1p1.c
"""


"""
INFO on EDGES-3 file structure:


    We need four S11 and four spectra (Ambient, Hot, Open and Short), and temperature logger.

    Currently we have the following:

    Data root directory: ~/data5/edges/data/EDGES3_data/MRO

        1. S11 files: root_directory/YYYY_DOY_HH_{amb|ant|hot|open|short|lna|lna_S|lna_O|lna_L|L|O|S}.s1p}
        2. Spectra: root_directory/mro/{amb|ant|hot|open|short}/{2022|2023}/YYYY_DOY_HH_MM_{amb|ant|hot|open|short}.acq
        3. Temperature logger: root_directory/temperature_logger/temperature.log


"""

import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from edges_cal.s11 import StandardsReadings, VNAReading
from glob import glob
from edges_analysis import GSData
import cmath
import matplotlib.pyplot as plt
import numpy as np
from edges_cal.s11 import StandardsReadings, VNAReading
from glob import glob
root_dir = "/data5/edges/data/EDGES3_data/MRO"

# constants from reads1p1.c


loadps = 38
openps = shortps = 33
ps = 33


def get_s1p_files(load, year, day):
    """
    Take the load (amb, ant, hot, open, short or lna) and return a list of .s1p files

    """

    file_list = {}

    # first get the input and then LOS

    file_temp = glob(
        root_dir + "/" + year + "_" + day + "*" + "[00-99]_" + load + ".s1p"
    )
    assert len(file_temp) == 1
    file_list["input"] = file_temp[0]

    for name, label in {"open": "O", "short": "S", "match": "L"}.items():
        if load == "lna":
            file_temp = glob(
                root_dir
                + "/"
                + year
                + "_"
                + day
                + "*"
                + "[00-99]_lna_"
                + label
                + ".s1p"
            )
            assert len(file_temp) == 1
            file_list[name] = file_temp[0]

        else:
            file_temp = glob(
                root_dir + "/" + year + "_" + day + "*" + "[00-99]_" + label + ".s1p"
            )
            assert len(file_temp) == 1
            file_list[name] = file_temp[0]

    # throw an error if len(file_lest !4)
    return file_list


def agilent(freq, res, delayps):
    """
    Following Alan's code here

    Keeping the commented lines as is. We might need it later

    """

    # Agilent approx. follow

    loss = 2.30 * 1e9
    #    print("loss %f Gohm/s\n",loss/1e9);
    Zcab = 50 + (1 - 1j) * (loss / (2 * 2 * np.pi * freq * 1e6)) * np.sqrt(
        freq * 1e6 / 1e9
    )
    #    delay = 33e-12;
    delay = delayps * 1e-12
    rload = res
    #    print("Zcab %f %f\n",creal(Zcab),cimag(Zcab));
    gl = loss * (delay / (2 * 50)) * np.sqrt(freq * 1e6 / 1e9) + 1j * (
        2 * np.pi * freq * 1e6 * delay
        + (loss * delay / (2 * 50)) * np.sqrt(freq * 1e6 / 1e9)
    )
    T = (rload - Zcab) / (rload + Zcab)
    T = T * np.exp(-2 * gl)
    Z = Zcab * (1 + T) / (1 - T)
    Tload = (Z - 50) / (Z + 50)
    #  print("Tload %f %f\n",creal(Tload),cimag(Tload));
    return Tload


def cabsparams(Topen, Tshort, Tload, Tobsopen, Tobsshort, Tobsload):
    b0 = Tobsopen
    b1 = Tobsshort
    b2 = Tobsload

    a00 = 1
    a01 = Topen
    a02 = Topen * Tobsopen
    a10 = 1
    a11 = Tshort
    a12 = Tshort * Tobsshort
    a20 = 1
    a21 = Tload
    a22 = Tload * Tobsload

    d = (
        a00 * a11 * a22
        + a10 * a21 * a02
        + a20 * a01 * a12
        - a20 * a11 * a02
        - a10 * a01 * a22
        - a21 * a12 * a00
    )

    aa00 = (a11 * a22 - a21 * a12) / d
    aa01 = -(a01 * a22 - a21 * a02) / d
    aa02 = (a01 * a12 - a11 * a02) / d
    aa10 = -(a10 * a22 - a20 * a12) / d
    aa11 = (a00 * a22 - a20 * a02) / d
    aa12 = -(a00 * a12 - a10 * a02) / d
    aa20 = (a10 * a21 - a20 * a11) / d
    aa21 = -(a00 * a21 - a20 * a01) / d
    aa22 = (a00 * a11 - a10 * a01) / d

    s11 = aa00 * b0 + aa01 * b1 + aa02 * b2
    s1221 = aa10 * b0 + aa11 * b1 + aa12 * b2
    s22 = aa20 * b0 + aa21 * b1 + aa22 * b2
    s1221 += (s11) * (s22)

    return (s11, s1221, s22)


# delay correction -- from corrcsv.c


def cabl(freq, delay, Tin, lossf, dielf):
    """
    Apply path legth correction for LNA


    The 8-position switch memo is 303 and the correction for the path to the
    LNA for the calibration of the LNA s11 is described in memos 367 and 392.

    corrcsv.c corrects lna s11 file for the different vna path to lna args:
    s11.csv -cablen -cabdiel -cabloss outputs c_s11.csv

    The actual numbers are slightly temperature dependent

    corrcsv s11.csv -cablen 4.26 -cabdiel -1.24 -cabloss -91.5

    and need to be determined using a calibration test like that described in
    memos 369 and 361. Basically the path length corrections can be "tuned" by
    minimizing the ripple on the calibrated spectrum of the open or shorted
    cable.

    cablen --> length in inches
    cabloss --> loss correction percentage
    cabdiel --> dielectric correction in percentage

    """

    b = 0.1175 * 2.54e-2 * 0.5
    a = 0.0362 * 2.54e-2 * 0.5
    diel = 2.05 * dielf  # UT-141C-SP
    d2 = np.sqrt(
        1.0 / (np.pi * 4.0 * np.pi * 1e-7 * 5.96e07 * 0.8 * lossf)
    )  #   // for tinned copper
    d = np.sqrt(
        1.0 / (np.pi * 4.0 * np.pi * 1e-7 * 5.96e07 * lossf)
    )  #  // skin depth at 1 Hz for copper

    L = (4.0 * np.pi * 1e-7 / (2.0 * np.pi)) * np.log(b / a)
    C = 2.0 * np.pi * 8.854e-12 * diel / np.log(b / a)

    La = 4.0 * np.pi * 1e-7 * d / (4.0 * np.pi * a)
    Lb = 4.0 * np.pi * 1e-7 * d2 / (4.0 * np.pi * b)
    disp = (La + Lb) / L
    R = 2.0 * np.pi * L * disp * np.sqrt(freq)
    L = L * (1.0 + disp / np.sqrt(freq))
    G = 0

    if diel > 1.2:
        G = 2.0 * np.pi * C * freq * 2e-4  # // 2e-4 is the loss tangent for teflon

    Zcab = np.sqrt((1j * 2 * np.pi * freq * L + R) / (1j * 2 * np.pi * freq * C + G))
    g = np.sqrt((1j * 2 * np.pi * freq * L + R) * (1j * 2 * np.pi * freq * C + G))

    T = (50.0 - Zcab) / (50.0 + Zcab)
    Vin = np.exp(+g * delay * 3e08) + T * np.exp(-g * delay * 3e08)
    Iin = (np.exp(+g * delay * 3e08) - T * np.exp(-g * delay * 3e08)) / Zcab
    Vout = 1 + T  # // Iout = (1 - T)/Zcab
    s11 = s22 = ((Vin / Iin) - 50) / ((Vin / Iin) + 50)
    VVin = Vin + 50.0 * Iin
    s12 = s21 = 2 * Vout / VVin

    Z = 50.0 * (1 + Tin) / (1 - Tin)
    T = (Z - Zcab) / (Z + Zcab)
    T = T * np.exp(-g * 2 * delay * 3e08)
    Z = Zcab * (1 + T) / (1 - T)
    T = (Z - 50.0) / (Z + 50.0)

    return T, s11, s12


def gets11(load, year, day, cablen=0, cabloss=0, cabdiel=0):
    """
    Take the load (amb, ant, hot, open, short or lna), file day and year, optional cable length correction
    parameters for lna and return s11

    """

    res = {
        "open": 1e9,
        "short": 0,
        "load": 49.962,
    }  # resistance of the calibration modes

    s1p_files = get_s1p_files(load, year, day)

    vna_load = VNAReading.from_s1p(s1p_files["match"])  # load
    vna_open = VNAReading.from_s1p(s1p_files["open"])  # open
    vna_short = VNAReading.from_s1p(s1p_files["short"])  # short
    vna_input = VNAReading.from_s1p(s1p_files["input"])  # input

    """
    alan converts the values from the .s1p files in dB to amp and phase (complex number)
    This is taken care of in VNAReading.from_s1p. Look for the function read() in class S1P in edges-io/io.py
    """

    t_gamma_load = agilent(vna_load.freq.freq.value / 1e6, res["load"], ps)
    t_gamma_open = agilent(vna_open.freq.freq.value / 1e6, res["open"], ps)
    t_gamma_short = agilent(vna_short.freq.freq.value / 1e6, res["short"], ps)

    s11, s1221, s22 = cabsparams(
        t_gamma_open,
        t_gamma_short,
        t_gamma_load,
        vna_open.s11,
        vna_short.s11,
        vna_load.s11,
    )

    # print(vna_open.s11)

    ss11ant = (vna_input.s11 - s11) / (s1221 + s22 * (vna_input.s11 - s11))

    if load == "lna":
        """
        cable length correction only for lna

        cablen --> length in inches
        cabloss --> loss correction percentage
        cabdiel --> dielectric correction in percentage

        """

        # correct_delay # --> from corrcsv.c
        # cabl(fre,fabs(cablen)*2.54e-2/3e08,0,&s11,&s12,1+cabloss*0.01,1+cabdiel*0.01)

        T, s11, s12 = cabl(
            vna_input.freq.freq.value,
            np.abs(cablen) * 2.54e-2 / 3e08,
            0,
            1 + cabloss * 0.01,
            1 + cabdiel * 0.01,
        )

        s22 = s11
        Ta = ss11ant

        if cablen > 0.0:
            Ta = s11 + (s12 * s12 * Ta) / (1 - s22 * Ta)

        if cablen < 0.0:
            Ta = (Ta - s11) / (s12 * s12 - s11 * s22 + s22 * Ta)

        ss11ant = Ta

    return ss11ant, vna_load.freq.freq.value, vna_input.s11



#temperature logger related functions

def extract_dates(anc_obj):


    '''
    Take ancilliary data from acq file and return the start and end time as datetime objects.

    '''

    start_time = anc_obj.data['times'][0][0].decode('utf-8') #first instance of time
    end_time = anc_obj.data['times'][-1][0].decode('utf-8')  #second instance of time

    date_format = "%Y:%j:%H:%M:%S"



    # Parse the string into a datetime object
    start_time = datetime.strptime(start_time, date_format)

    end_time = datetime.strptime(end_time, date_format)


    return(start_time, end_time)




def extract_temp_values_from_logger(temperature_file=root_dir+"/temperature_logger/temperature.log"):



    '''
        Easiest way (i think) is to read one line at a time, check conditions in each line,
        extract a datetime object and temperature readouts.
        I am throwing away the readouts that don't have complete information.
        If such read out is encountered, count is 'reset' meaning that readout won't be appended to the dataframe

    '''

    #lets first create a panda dataframe

    df = pd.DataFrame({'Time':[], 'Front End temperature':[], 'Amb load temperature':[], 'Hot load temperature':[],
                      'Inner box temperature':[], 'Thermal Control':[], 'Battery Voltage':[],
                       'PR59 Current':[]})

    with open(temperature_file, 'r') as file:



        for line in file:

            this_line = line
            #print(this_line)

            if '_' in this_line:
                #print(this_line)

                values = this_line.split('_')  # this is the line that looks like 2022_318_03
                if(len(values)==3):
                    count =0 # reset everytime the first line is encountered

                    try:
                        year = int(values[0])
                        doy = int(values[1])
                        count +=1
                    except:
                        #skip this data record
                        count = count #do nothing


            if 'UTC' in this_line:

                values = this_line.split(' ') #this_line is alredy split with single space

                # this is the line that looks like "Mon Nov 14 03:20:26 UTC 2022"
                                        #we only want time from this
                if(len(values)==6):

                    time = values[3]
                    values = time.split(':')  # we take the time that looks like "03:20:26"

                    try:
                        hh = int(values[0])
                        mm = int(values[1])
                        ss = int(values[2])

                        #create datetime object using year, day of the year, hh,mm,ss information.

                        date_object = datetime(year, 1, 1, hh, mm, ss) + timedelta(days=doy - 1)
                        count +=1

                    except:
                        #skip this data record
                        count = count #do nothing

            if '0 ' in this_line:      #this is the line that looks like "0 +3.000000e+01" --> we don't need this now
                count += 1 # do nothing

            if '100 ' in this_line:
                values = this_line.split(" ")        #this line is for sensor 100
                if(len(values)==2):
                    #check if there are two values in that line -- one for the sensor and one for the value

                    try:
                        front_end_box_temp = float(values[1].strip())
                        count +=1
                    except:
                        #skip this data record
                        count = count #do nothing

            if '101 ' in this_line:
                values = this_line.split(" ")        #this line is for sensor 101
                if(len(values)==2):  #check if there are two values in that line -- one for the sensor and one for the value

                    try:
                        amb_load_temp = float(values[1].strip())
                        count +=1

                    except ValueError:
                        #skip this data record
                        count = count #do nothing

            if '102 ' in this_line:
                values = this_line.split(" ")        #this line is for sensor 102
                if(len(values)==2):

                    try:
                        hot_load_temp = float(values[1].strip())
                        count +=1

                    except ValueError:
                        #skip this data record
                        count = count #do nothing

            if '103 ' in this_line:
                values = this_line.split(" ")        #this line is for sensor 103
                if(len(values)==2):

                    try:
                        innerbox_temp = float(values[1].strip())
                        count +=1

                    except ValueError:
                        #skip this data record
                        count = count #do nothing

            if '106 ' in this_line:
                values = this_line.split(" ")        #this line is for sensor 106
                if(len(values)==2):

                    try:
                        therm_control = float(values[1].strip())
                        count +=1

                    except ValueError:
                        #skip this data record
                        count = count #do nothing

            if '150 ' in this_line:
                values = this_line.split(" ")        #this is battery voltage 150
                if(len(values)==2):

                    try:
                        battery_voltage = float(values[1].strip())
                        count +=1

                    except ValueError:
                        #skip this data record
                        count = count #do nothing



            if '152 ' in this_line:
                values = this_line.split(" ")        #this is PR59 current 152
                if(len(values)==2):

                    try:
                        pr59_current = float(values[1].strip())
                        count +=1

                    except ValueError:
                        #skip this data record
                        count = count #do nothing

            if(count ==10):

                temp_df = pd.DataFrame({'Time':[date_object], 'Front End temperature':[front_end_box_temp],
                                            'Amb load temperature':[amb_load_temp],
                                            'Hot load temperature':[hot_load_temp],
                                            'Inner box temperature':[innerbox_temp],
                                            'Thermal Control':[therm_control],
                                            'Battery Voltage':[battery_voltage],
                                            'PR59 Current':[pr59_current]})



                # Concatenate the this DataFrame with the original DataFrame
                df = pd.concat([df, temp_df], ignore_index=True)
                count=0 #reset



    df.to_csv('meta_data_files/temperature_data.csv')


    return('meta_data_files/temperature_data.csv')

def extract_temperature(anc_obj, load='box', extract_log= False, temperature_file='meta_data_files/temperature_data.csv'):

    '''
    Take start and end time from the ancillary data and return the average temperature in that time range    
    --  For hot load, temperature from hot load temperature sensor is used (register 102) that
    sits directly on the hot load at the end of 8 position switch
    --  For amb, long cable open and short, the register 101 is used which is the temperature of the ambient load
    '''

    start_time, end_time = extract_dates(anc_obj)

    if(extract_log == True):

        '''
        extract the data from log only if required.
        Default is False, meaning the pre-extracted file will be used
        This is don to avoid repeted file parsing

        '''
        temperature_file = extract_temp_values_from_logger(root_dir+'/temperature_logger/temperature.log')

        #default file is temperature_data.csv


    df = pd.read_csv(temperature_file)

    df['Time'] = pd.to_datetime(df['Time'])

    front_end_temp = df[(df['Time'] >= start_time) & (df['Time'] <= end_time)]['Front End temperature']
    amb_load_temp = df[(df['Time'] >= start_time) & (df['Time'] <= end_time)]['Amb load temperature']
    hot_load_temp = df[(df['Time'] >= start_time) & (df['Time'] <= end_time)]['Hot load temperature']
    inner_box_temp = df[(df['Time'] >= start_time) & (df['Time'] <= end_time)]['Inner box temperature']
    therm_control = df[(df['Time'] >= start_time) & (df['Time'] <= end_time)]['Thermal Control']
    battery_voltage = df[(df['Time'] >= start_time) & (df['Time'] <= end_time)]['Battery Voltage']
    battery_current = df[(df['Time'] >= start_time) & (df['Time'] <= end_time)]['PR59 Current']

    if load =='hot':
        
        temperature = hot_load_temp.values
        #print('hot', temperature)
    
    elif load == 'open' or 'short' or 'box' or 'amb':
        
        temperature =amb_load_temp.values  #this is default if no load is specified


    if load =='amb':

        temperature = amb_load_temp.values
        #print('amb', temperature)

    elif load =='hot':

        temperature = hot_load_temp.values
        #print('hot', temperature)

    elif load == 'open' or 'short' or 'box':

        temperature =front_end_temp.values  #this is default if no load is specified
        #print('default', temperature)

    temperature_deg_C = temperature * u.deg_C
    temperature_K = temperature_deg_C.to(u.K,  equivalencies=u.temperature())


    return(temperature_K.mean())

