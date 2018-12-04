

import sys
sys.path.insert(0, '/home/raul/GetDist-0.2.8.4.2')

import numpy as np
import matplotlib.pyplot as plt
#import getdist as getdist
from getdist import MCSamples, plots




def triangle_plot(input_textfile, output_pdf, label_names=[], legend_FS=10, label_FS=10, axes_FS=10):

    # filename = '/home/raul/Desktop/test.txt'


    d = np.genfromtxt(input_textfile)

    ss = d[:,2::]
    ww = d[:,0]
    ll = d[:,1]/2

    # names  = ['x1', 'x2', 'x3', 'x4', 'x5']
    # labels = [r'a_0', r'a_1', r'a_2', r'a_3', r'a_4']

    Npar = len(ss[0,:])
    names = ['x'+str(i+1) for i in range(Npar)]
    
    if len(label_names)==0:
        labels = [r'a_'+str(i) for i in range(Npar)]
    else:
        labels = label_names
    
    samples = MCSamples(samples=ss, weights=ww, loglikes=ll, names=names, labels=labels, label=r'only')

 

    g = plots.getSubplotPlotter(subplot_size=1.5)
    
    g.settings.legend_fontsize = legend_FS
    g.settings.lab_fontsize    = label_FS
    g.settings.axes_fontsize   = axes_FS
    
    g.triangle_plot([samples], filled=True) #, param_limits={'x1':[1498, 1502], 'x2':[-2.504, -2.496]}, filled=True, legend_loc='upper right')
    g.export(output_pdf)
    plt.close()
    plt.close()


    return samples






