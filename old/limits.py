

import numpy as np
import matplotlib.pyplot as plt




def plot_limits():
	
	SPT = 2.64 # Liu 2013
	Planck_best_fit = 8.8
	Planck_1sigma   = 0.9	
	EDGES_2010 = 0.06
	
	z = np.arange(11, 6, -0.01)
	EDGES_2017 = 0.2*np.random.normal(0,np.ones(len(z)))+0.5	
	
	
	
	
	plt.close()
	
	
	# Constraints
	ax = plt.gca()
	ax.fill([6, 6, 8, 6], [0, 4, 4, 0], fill=True, color='green')   # Gunn-Peterson
	ax.tick_params(axis='x', direction='out')
	ax.tick_params(axis='y', direction='out')  
	ax.fill([11, 6, 6, 11], [SPT, SPT, 4, 4], fill=True, color='red', alpha=0.7)
	ax.fill([11, Planck_best_fit + Planck_1sigma, Planck_best_fit + Planck_1sigma, 11], [0, 0, 4, 4], fill=False, hatch='\\', zorder=6)
	ax.fill([Planck_best_fit - Planck_1sigma, 6, 6, Planck_best_fit - Planck_1sigma], [0, 0, 4, 4], fill=False, hatch='\\', zorder=6)
	ax.plot([Planck_best_fit, Planck_best_fit],[0, 4], 'k--', linewidth=2, zorder=6)
	ax.fill_between(z, EDGES_2017, where=EDGES_2017>=np.zeros(len(z)), interpolate=True, color='blue', zorder=5)
	ax.fill([11, 6, 6, 11], [0, 0, EDGES_2010, EDGES_2010], fill=True, color=[0.2, 0.2, 0.2], zorder=5)
	
	ax.plot(8.14, (9.21-7.26), 'dk', markerfacecolor='none', mew=2)
	ax.plot(7.89, (8.89-7.07), 'xk', markerfacecolor='none', mew=2)
	ax.plot(8.13, (9.21-7.32), 'xk', markerfacecolor='none', mew=2)
	ax.plot(7.76, (8.70-6.94), 'xk', markerfacecolor='none', mew=2)
	ax.plot(8.01, (9.02-7.20), 'xk', markerfacecolor='none', mew=2)	
	ax.plot(7.57, (8.52-6.82), 'xk', markerfacecolor='none', mew=2)	
	
	
	
	


	# Text and arrows
	plt.text(6.8, 2.1,   'Gunn-'+'\n'+'Peterson'+'\n'+'(2001)', backgroundcolor='white', zorder=6)
	plt.text(9.48, 2.97,   r'SPT $2\sigma$'+'\n'+'(2012)', backgroundcolor='white', zorder=6)
	plt.text(7.81, 1.25,   r'Planck $1\sigma$'+'\n'+'(2016)', backgroundcolor='white', zorder=6)
	plt.text(10.43, 1.25,  r'Planck $1\sigma$'+'\n'+'(2016)', backgroundcolor='white', zorder=6)
	plt.text(9.43, 1.53,  r'$z_r=8.8$', fontsize=15)
	plt.text(9.29, 1.25,   'Planck' +'\n'+ '(2016)', zorder=6)
	plt.text(8.6, 2.02,    'Greig &' + '\n' + 'Mesinger' +'\n'+ '(2016)', zorder=6)
	plt.text(10.65, 0.8,  'EDGES $2\sigma$'+'\n'+'(2010)', backgroundcolor='white', zorder=6)
	plt.arrow(10.35, 0.7, 0.0, -0.5, fc="k", ec="k", head_width=0.1, head_length=0.1, linewidth=2, zorder=6)
	plt.text(8.68, 0.3,   'EDGES $2\sigma$'+'\n'+'(2017)', backgroundcolor='white', zorder=6)
	
	
	
		
	plt.xlim([11, 6])
	plt.ylim([0, 3.5])
	plt.xlabel('$z_r$', fontsize=20)
	plt.ylabel('$\Delta z$', fontsize=20)
	
	plt.savefig('/home/ramo7131/Desktop/test.pdf', bbox_inches='tight')
	
	return 0

