
import numpy as np
import scipy as sp
import PyPolyChord
import sys
sys.path.insert(0, '/home/ramo7131/GetDist-0.2.8.4.2')

from PyPolyChord.settings import PolyChordSettings
from PyPolyChord.priors import UniformPrior


#import getdist as getdist
from getdist import MCSamples, plots

#import getdist.plots as gdp
import matplotlib.pyplot as plt


#plt.rcParams['text.usetex']=True


v          = np.arange(50, 101, 1)
T75        = 1500
beta       = -2.5

#std_dev_vec   = 2*np.ones(len(v))   # frequency dependent or independent
std_dev_vec   = (v/75)**(beta)
sigma     = np.diag(std_dev_vec**2)     # uncertainty covariance matrix
inv_sigma = sp.linalg.inv(sigma)

det_sigma = np.linalg.det(sigma)
noise = np.random.multivariate_normal(np.zeros(len(v)), sigma)


#std_dev = 1
#noise   = np.random.normal(0, std_dev, size=len(v))



d_no_noise = T75*((v/75)**(beta))
d          = d_no_noise + noise


N           = len(v)
Nparameters = 2
Nderived    = 0





def model(theta):
	T75_i  = theta[0]
	beta_i = theta[1]    # -2.5
	m = T75_i*((v/75)**(beta_i))
	
	return m





def loglikelihood(theta):
		
	m = model(theta)
	DELTA = d-m
	#lnL = -(1/2)*np.sum( (DELTA/std_dev)**2 )    -(1/2)*N*np.log(2*np.pi*(std_dev**2))
	lnL2 = -(1/2)*np.dot(np.dot(DELTA, inv_sigma), DELTA)      -(N/2)*np.log(2*np.pi)      -(1/2)*np.log(det_sigma)
	
	# This solves numerical errors
	if np.isnan(lnL2) == True:
		print('True')
		lnL2 = -np.infty
		
	
	#print(lnL)
	print(lnL2)
	#print('hola')
	
	return lnL2, 0





def prior(hypercube):
	""" Uniform prior from [-1,1]^D. """
	return UniformPrior(-1e4, 1e4)(hypercube)






def dumper(live, dead, logweights, logZ, logZerr):
	print(dead[-1])






def run(root_name):
	settings = PolyChordSettings(Nparameters, Nderived)
	settings.file_root = root_name
	settings.do_clustering = True
	settings.read_resume = False
	output = PyPolyChord.run_polychord(loglikelihood, Nparameters, Nderived, settings, prior, dumper)
	
	return 0


def triangle_plot():
	
	d3 = np.genfromtxt('/home/ramo7131/Desktop/chains/test3.txt')
	d2 = np.genfromtxt('/home/ramo7131/Desktop/chains/test2.txt')
	d4 = np.genfromtxt('/home/ramo7131/Desktop/chains/test4.txt')
	
	
	
	ss3 = d3[:,2::]
	ss2 = d2[:,2::]
	ss4 = d4[:,2::]
	
	ww3 = d3[:,0]
	ww2 = d2[:,0]
	ww4 = d4[:,0]
	
	ll3 = d3[:,1]/2
	ll2 = d2[:,1]/2
	ll4 = d4[:,1]/2	

	names = ['x1', 'x2']
	labels = [r'$T_{75}\;\;\rm{[K]}$', r'$\beta$']
	
	samples3 = MCSamples(samples=ss3, weights=ww3, loglikes=ll3, names=names, labels=labels, label=r'$\sigma=1\;\;\rm{[K]}$')
	samples2 = MCSamples(samples=ss2, weights=ww2, loglikes=ll2, names=names, labels=labels, label=r'$\sigma=2\;\;\rm[K]$')
	samples4 = MCSamples(samples=ss4, weights=ww4, loglikes=ll4, names=names, labels=labels, label=r'$\sigma = \left(\frac{\nu}{75\;\;\rm{MHz}}\right)^{-2.5}\;\;\rm{[K]}$')
	
	
	# Printing 68% limits for the two parameters, and each set
	lim3_par0 = samples3.twoTailLimits(0,68)
	lim3_par1 = samples3.twoTailLimits(1,68)
	lim2_par0 = samples2.twoTailLimits(0,68)
	lim2_par1 = samples2.twoTailLimits(1,68)
	lim4_par0 = samples4.twoTailLimits(0,68)
	lim4_par1 = samples4.twoTailLimits(1,68)
	
	print(np.append(samples2.mean(0), samples2.mean(1)))
	print(np.append(samples3.mean(0), samples3.mean(1)))
	print(np.append(samples4.mean(0), samples4.mean(1)))
	
	print(np.append(lim2_par0, lim2_par1))
	print(np.append(lim3_par0, lim3_par1))
	print(np.append(lim4_par0, lim4_par1))
	

	

	
	# g = plots.getSubplotPlotter(subplot_size=3.3)
	# g.settings.legend_fontsize = 15
	# g.settings.lab_fontsize = 15
	# g.settings.axes_fontsize = 15
	# #g.settings.tight_layout=False
	# g.triangle_plot([samples2, samples3, samples4], param_limits={'x1':[1498, 1502], 'x2':[-2.504, -2.496]}, filled=True, legend_loc='upper right')
	# g.export('output_file.pdf')
	# plt.close()
	# plt.close()
	

	return samples3






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




