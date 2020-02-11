

import numpy as np
import scipy as sp
import PyPolyChord
import sys
sys.path.insert(0, '/home/raul/GetDist-0.2.8.4.2')

from PyPolyChord.settings import PolyChordSettings
from PyPolyChord.priors import UniformPrior


#import getdist as getdist
from getdist import MCSamples, plots

#import getdist.plots as gdp
import matplotlib.pyplot as plt


#plt.rcParams['text.usetex']=True






# Loading data as global quantities
#v, d, inv_sigma, det_sigma = data()

v  = np.arange(60,151,1)
v0 = 100

N21 = 4 
Nfg = 5


Nparameters = N21+Nfg
Nderived    = 0


















def simulated_data(theta):

	#v          = np.arange(50, 101, 1)
	#T75        = 1500
	#beta       = -2.5
	#std_dev_vec  = 2*np.ones(len(v))   # frequency dependent or independent
	#std_dev = 1
	#noise   = np.random.normal(0, std_dev, size=len(v))
	
	
	std_dev_vec   = 0.03*(v/v0)**(-2.5)
	#std_dev_vec   = 0.05*np.ones(len(v))
	
	
	sigma         = np.diag(std_dev_vec**2)     # uncertainty covariance matrix
	inv_sigma     = sp.linalg.inv(sigma)
	det_sigma     = np.linalg.det(sigma)
	
	noise         = np.random.multivariate_normal(np.zeros(len(v)), sigma)
	
	d_no_noise    = model(theta)
	d             = d_no_noise + noise
	
	N             = len(v)


	return v, d, sigma, inv_sigma, det_sigma









def model_eor_flattened_gaussian(model_type=1, T21=1, vr=75, dv=20, tau0=4, tilt=0):

	# Memo 220 and 226
	if model_type == 1:
		b  = -np.log(-np.log( (1 + np.exp(-tau0))/2 )/tau0)
		K1 = T21 * (1 - np.exp( -tau0 * np.exp( (-b*(v-vr)**2) / ((dv**2)/4))))
		K2 = 1 + (tilt * (v - vr) / dv)
		K3 = 1 - np.exp(-tau0)
		T  = K1 * K2 / K3

	# Memo 226
	if model_type == 2:
		K1 = np.tanh( (1/(v + dv/2) - 1/vr) / (dv/(tau0*(vr**2))) )
		K2 = np.tanh( (1/(v - dv/2) - 1/vr) / (dv/(tau0*(vr**2))) )
		T = -(T21/2) * (K1 - K2) 

	return T   # The amplitude is equal to T21, not to -T21









def model(theta):
	
	
	model_21 = 'flattened_gaussian'
	model_fg = 'LINLOG'  # 'NONE'
	
		
	
	if (model_21 == 'flattened_gaussian'):
		if N21 == 4:
			model_21 = model_eor_flattened_gaussian(model_type=1, T21=theta[0], vr=theta[1], dv=theta[2], tau0=theta[3], tilt=0)
			
		if N21 == 5:
			model_21 = model_eor_flattened_gaussian(model_type=1, T21=theta[0], vr=theta[1], dv=theta[2], tau0=theta[3], tilt=theta[4])		
	
	
	
	
	#if (model_21 == 'mirocha2016'):	
		
	
		
	if (model_fg == 'LINLOG'):
		
		model_fg = 0
		for i in range(Nfg):
			j = N21 + i
			model_fg = model_fg      +     theta[j] * ((v/v0)**(-2.5)) * ((np.log(v/v0))**i)
			
			
	if (model_fg == 'NONE'):
		model_fg = 0
			
		
		
			
	model = model_21 + model_fg
	
			
	return model

















#v, d, sigma, inv_sigma, det_sigma = simulated_data([1000, 1, 1, -1, 4])  # power law
v, d, sigma, inv_sigma, det_sigma = simulated_data([-0.5, 78, 20, 7, 1000, 1, 1, -1, 4]) # flattened gaussian






def loglikelihood(theta):
	
	N = len(v)
	
	# Evaluating model
	m = model(theta)
	
	
	# Log-likelihood
	DELTA = d-m
	lnL2 = -(1/2)*np.dot(np.dot(DELTA, inv_sigma), DELTA)      -(N/2)*np.log(2*np.pi)      -(1/2)*np.log(det_sigma)
	#lnL2 =  #-(1/2)*np.log(det_sigma)
	
	
	# This solves numerical errors
	if np.isnan(lnL2) == True:
		print('True')
		lnL2 = -np.infty
		
	
	#print(lnL)
	print(lnL2)
	#print('hola')
	
	return lnL2, 0





#def prior(hypercube):
	#""" Uniform prior from [-1,1]^D. """
	#return UniformPrior(-1e4, 1e4)(hypercube)








def prior(cube):
	
	"""
	
	A function defining the tranform between the parameterisation in the unit hypercube to the true parameters.
	
	Args: cube (array, list): a list containing the parameters as drawn from a unit hypercube.
	
	Returns:
	list: the transformed parameters.
	
	"""
	
	
	# Unpack the parameters (in their unit hypercube form)
	T21_prime = cube[0]
	vr_prime  = cube[1]
	dv_prime  = cube[2]
	tau_prime = cube[3]
	
	a0_prime  = cube[4]
	a1_prime  = cube[5]
	a2_prime  = cube[6]
	a3_prime  = cube[7]
	a4_prime  = cube[8]
	
	
	
	T21_min = -10  # lower bound on uniform prior 
	T21_max =  10  # upper bound on uniform prior 
	
	vr_min =  60   # lower bound on uniform prior 
	vr_max = 150   # upper bound on uniform prior 
	
	dv_min =   2   # lower bound on uniform prior 
	dv_max = 100   # upper bound on uniform prior 
	
	tau_min =  0  # lower bound on uniform prior 
	tau_max = 30  # upper bound on uniform prior 
	
	
	a0_min =   900 # lower bound on uniform prior 
	a0_max =  1100 # upper bound on uniform prior 	
	
	a1_min = -1e4  # lower bound on uniform prior 
	a1_max =  1e4  # upper bound on uniform prior 
	
	a2_min = -1e4  # lower bound on uniform prior 
	a2_max =  1e4  # upper bound on uniform prior 
	
	a3_min = -1e4  # lower bound on uniform prior 
	a3_max =  1e4  # upper bound on uniform prior 
	
	a4_min = -1e4  # lower bound on uniform prior 
	a4_max =  1e4  # upper bound on uniform prior 	
	
	
	
	T21 = T21_prime * (T21_max - T21_min) + T21_min
	vr  = vr_prime  * (vr_max - vr_min)   + vr_min
	dv  = dv_prime  * (dv_max - dv_min)   + dv_min
	tau = tau_prime * (tau_max - tau_min) + tau_min
	
	a0  = a0_prime  * (a0_max - a0_min)   + a0_min
	a1  = a1_prime  * (a1_max - a1_min)   + a1_min
	a2  = a2_prime  * (a2_max - a2_min)   + a2_min
	a3  = a3_prime  * (a3_max - a3_min)   + a3_min
	a4  = a4_prime  * (a4_max - a4_min)   + a4_min
	
	
	# mmu = 0.     # mean of Gaussian prior on m
	# msigma = 10. # standard deviation of Gaussian prior on m
	# m = mmu + msigma*ndtri(mprime) # convert back to m
	
	
	
	theta = [T21, vr, dv, tau, a0, a1, a2, a3, a4]
	
	return theta














def dumper(live, dead, logweights, logZ, logZerr):
	print(dead[-1])









def run(root_name):
	
	# in python, or ipython >>  
	

	settings               = PolyChordSettings(Nparameters, Nderived)
	settings.base_dir      = '/home/raul/Desktop/'
	settings.file_root     = root_name
	settings.do_clustering = True
	settings.read_resume   = False
	output                 = PyPolyChord.run_polychord(loglikelihood, Nparameters, Nderived, settings, prior, dumper)
	
	return 0





















def triangle_plot():
	
	#d = np.genfromtxt('/home/raul/Desktop/hahaha.txt')
	#d = np.genfromtxt('/home/raul/Desktop/ha.txt')
	d = np.genfromtxt('/home/raul/Desktop/hu.txt')
	
	ss = d[:,2::]
	ww = d[:,0]
	ll = d[:,1]/2
	
	#names  = ['x1', 'x2', 'x3', 'x4', 'x5']
	#labels = [r'a_0', r'a_1', r'a_2', r'a_3', r'a_4']
	
	names  = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9']
	labels = [r'T_{21}', r'\nu_r', r'\Delta\nu', r'\tau', r'a_0', r'a_1', r'a_2', r'a_3', r'a_4']	
	
	samples = MCSamples(samples=ss, weights=ww, loglikes=ll, names=names, labels=labels, label=r'only')
	
	#samples2 = MCSamples(samples=ss2, weights=ww2, loglikes=ll2, names=names, labels=labels, label=r'$\sigma=2\;\;\rm[K]$')
	#samples4 = MCSamples(samples=ss4, weights=ww4, loglikes=ll4, names=names, labels=labels, label=r'$\sigma = \left(\frac{\nu}{75\;\;\rm{MHz}}\right)^{-2.5}\;\;\rm{[K]}$')
	
	# Printing 68% limits for the two parameters, and each set
	#lim3_par0 = samples3.twoTailLimits(0,68)
	#lim3_par1 = samples3.twoTailLimits(1,68)
	#lim2_par0 = samples2.twoTailLimits(0,68)
	#lim2_par1 = samples2.twoTailLimits(1,68)
	#lim4_par0 = samples4.twoTailLimits(0,68)
	#lim4_par1 = samples4.twoTailLimits(1,68)
	
	#print(np.append(samples2.mean(0), samples2.mean(1)))
	#print(np.append(samples3.mean(0), samples3.mean(1)))
	#print(np.append(samples4.mean(0), samples4.mean(1)))
	
	#print(np.append(lim2_par0, lim2_par1))
	#print(np.append(lim3_par0, lim3_par1))
	#print(np.append(lim4_par0, lim4_par1))
	

	

	
	g = plots.getSubplotPlotter(subplot_size=3.3)
	g.settings.legend_fontsize = 15
	g.settings.lab_fontsize = 15
	g.settings.axes_fontsize = 15
	#g.settings.tight_layout=False
	g.triangle_plot([samples], filled=True) #, param_limits={'x1':[1498, 1502], 'x2':[-2.504, -2.496]}, filled=True, legend_loc='upper right')
	g.export('/home/raul/Desktop/output_file_X.pdf')
	plt.close()
	plt.close()
	

	return samples






def model_plot():
	
	t  = model([1500, -2.5])
	s3 = model([1, -2.5])
	
	s1 = 2*np.ones(len(v))
	s2 = 1*np.ones(len(v))
	
	
	plt.close()
	plt.close()
	
	plt.figure(figsize=[6,8])
	plt.subplot(2,1,1)
	plt.plot(v, t, 'k')
	
	plt.ylabel('temperature [K]')
	plt.ylim([0, 5000])
	
	plt.subplot(2,1,2)
	plt.plot(v, s1, color=[0.7, 0.7, 0.7])
	plt.plot(v, s2, 'r')
	plt.plot(v, s3, 'b')
	
	plt.xlabel('frequency [MHz]')
	plt.ylabel('temperature [K]')
	plt.legend([r'$\sigma=2\;\;\rm{[K]}$',r'$\sigma=1\;\;\rm{[K]}$',r'$\sigma = \left(\frac{\nu}{75\;\;\rm{MHz}}\right)^{-2.5}\;\;\rm{[K]}$'])
	
	plt.ylim([0, 3])
	
	plt.savefig('models.pdf', bbox_inches='tight')
	plt.close()
	plt.close()	
	
	return 0









def chains_plot():
	
	d3 = np.genfromtxt('/home/ramo7131/Desktop/chains/test3.txt')
	d2 = np.genfromtxt('/home/ramo7131/Desktop/chains/test2.txt')
	d4 = np.genfromtxt('/home/ramo7131/Desktop/chains/test4.txt')
	
	
	
	ss3 = d3[:,2::]
	ss2 = d2[:,2::]
	ss4 = d4[:,2::]
	
	FS = 15
	
	plt.close()
	plt.close()
	plt.figure()
	plt.subplot(2,1,1)
	plt.plot(ss2[:,0], color=[0.7, 0.7, 0.7])
	plt.plot(ss3[:,0], 'r')
	plt.plot(ss4[:,0], 'b')
	plt.ylabel(r'$T_{75}\;\; \rm [K]$', fontsize=FS)
	plt.legend([r'$\sigma=2\;\;\rm{[K]}$',r'$\sigma=1\;\;\rm{[K]}$',r'$\sigma = \left(\frac{\nu}{75\;\;\rm{MHz}}\right)^{-2.5}\;\;\rm{[K]}$'])
	
	plt.subplot(2,1,2)
	plt.plot(ss2[:,1], color=[0.7, 0.7, 0.7])
	plt.plot(ss3[:,1], 'r')
	plt.plot(ss4[:,1], 'b')
	plt.ylabel(r'$\beta$', fontsize=FS)
	
	plt.xlabel('sample number')
	plt.savefig('chains.pdf', bbox_inches='tight')
	plt.close()
	plt.close()
	
	
	return 0










