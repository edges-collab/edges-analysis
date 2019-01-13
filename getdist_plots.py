

import sys
sys.path.insert(0, '/home/raul/GetDist-0.2.8.4.2')

import numpy as np
import matplotlib.pyplot as plt
#import getdist as getdist
from getdist import MCSamples, plots







def load_samples(input_textfile, index_good, label_names=[]):
	
	
	# Loading data
	dd = np.genfromtxt(input_textfile)
	d = dd[index_good::,:]

	ww = d[:,0]    # Weights
	ll = d[:,1]/2  # Minus Log Likelihood	
	ss = d[:,2::]  # Parameter samples
	
	

	# Parameter names and labels
	Npar = len(ss[0,:])
	names = ['x'+str(i+1) for i in range(Npar)]

	if len(label_names)==0:
		labels = [r'a_'+str(i) for i in range(Npar)]
	else:
		labels = label_names

	
	
	# Convert samples into GETDIST format 
	getdist_samples = MCSamples(samples=ss, weights=ww, loglikes=ll, names=names, labels=labels, label=r'only')


	# Best fit and covariance matrix
	IX = np.argmin(d[:,1])    # the maximum likelihood point
	best_fit = d[IX, 2::]
	
	covariance_matrix = getdist_samples.cov()
	
	



	return getdist_samples, ww, ll, best_fit, covariance_matrix












def triangle_plot(getdist_samples, output_pdf_filename, legend_FS=10, label_FS=10, axes_FS=10):

	g = plots.getSubplotPlotter(subplot_size=1.5)

	g.settings.legend_fontsize = legend_FS
	g.settings.lab_fontsize    = label_FS
	g.settings.axes_fontsize   = axes_FS

	g.triangle_plot([getdist_samples], filled=True) #, param_limits={'x1':[1498, 1502], 'x2':[-2.504, -2.496]}, filled=True, legend_loc='upper right')
	g.export(output_pdf_filename)
	plt.close()
	plt.close()


	return 0





