


# Standard 

%matplotlib
import matplotlib.pyplot as plt
import ares


sim = ares.simulations.Global21cm(fX=1)
sim.run()

history = sim.history
dTb     = history['dTb']
v       = history['nu']
plt.plot(v, dTb)







# Faster Cooling of IGM

%matplotlib
import matplotlib.pyplot as plt
import ares

cold = \
{
 'approx_thermal_history': 'exp',
 'load_ics': 'parametric',
 'inits_Tk_p0': 194.002300947,
 'inits_Tk_p1': 1.20941098917,
 'inits_Tk_p2': -6.0088645858,
}


sim = ares.simulations.Global21cm(fX=1, **cold)
sim.run()

history = sim.history
dTb     = history['dTb']
v       = history['nu']
plt.plot(v, dTb)












# More elaborate
import numpy as np
import matplotlib.pyplot as plt
import ares

ax = None
for model in ['cold', 'radio']:
    pars = ares.util.ParameterBundle('mirocha2018:base') + ares.util.ParameterBundle('mirocha2018:{}'.format(model))
         
    sim = ares.simulations.Global21cm(**pars) 
    sim.run()
    ax, zax = sim.GlobalSignature(ax=ax, lw=3, label='lfcal'+model, ymin=-750)
    
    
    
    
    
    
    
# Plot the EDGES data
b18 = ares.util.read_lit('bowman2018')
ax, zax = b18.plot_recovered(ax=ax, alpha=0.5, color='k')

# The 'radio' model has a single free parameter: 'pop_rad_yield{2}'
# This is the 1.4 GHz (rest-frame) luminosity per unit SFR.
# Nearby galaxies have L_R ~1e29 erg/s/Hz/SFR.

# The 'cold' model has three parameters: inits_Tk_p0, inits_Tk_p1, inits_Tk_p2
# These are the z_0, alpha, and beta of my Eq. 4.
# You can change to different parameterizations using the 'approx_thermal_history'
# parameter. Currently it is set to 'exp', though a few other models are available,
# such as 'exp+gauss', which adds in an optional gaussian feature (see below). 
# Have a look at the 'log_cooling_rate' function in ares.physics.Cosmology to 
# see the equations and parameters.

# You can also add in a radio background to the fcoll models. 
# See the following page on the docs for more about the astrophysics side
# of that model (https://ares.readthedocs.io/en/latest/example_gs_standard.html).

# First, define the spectral range where this emission is generated.
# Convert to eV since that's what ARES uses internally
E21 = nu_0_mhz * 1e6 * h_p / erg_per_ev
nu_min = 1e7  # in Hz
nu_max = 1e12
Emin = nu_min * (h_p / erg_per_ev)   # 1 GHz -> eV
Emax = nu_max * (h_p / erg_per_ev)   # 3 GHz -> eV

radio = \
{
 'pop_sfr_model{3}': 'link:sfrd:0',
 
 # Assume a power-law with spectral index -0.7
 'pop_sed{3}': 'pl',
 'pop_alpha{3}': -0.7,
 'pop_Emin{3}': Emin,
 'pop_Emax{3}': Emax,
 'pop_EminNorm{3}': None,
 'pop_EmaxNorm{3}': None,
 'pop_Enorm{3}': E21, # 1.4 GHz: frequency at which pop_rad_yield is defined.
 'pop_rad_yield_units{3}': 'erg/s/sfr/hz', 
 
 # Free parameter: output per unit SFR.
 'pop_rad_yield{3}': 1e32,
 
 # Optional free parameter: shut off redshift for radio emission
 'pop_zdead{3}': 15.,
 
 # X-ray emission: highly degenerate with radio emission,
 # needs to be increased as radio emission is increased
 'pop_rad_yield{1}': 1e40,
 
 # Solution method, make sure this new population *only* emits in the radio
 'pop_solve_rte{3}': True,
 'pop_radio_src{3}': True,
 'pop_lw_src{3}': False,
 'pop_lya_src{3}': False,
 'pop_heat_src_igm{3}': False,
 'pop_ion_src_igm{3}': False,
 'pop_ion_src_cgm{3}': False,
}

sim = ares.simulations.Global21cm(**radio)
sim.run()

ax, zax = sim.GlobalSignature(ls='--', lw=3, ax=ax, ymin=-750,
    label='fcoll+radio')

# Lastly, some alternative parameterizations for the cooling.

# This is the standard 'exp' model
cold_1 = ares.util.ParameterBundle('mirocha2018:cold')

# This is a model where the temperature need not be monotonically declining.
# To do this, we add an optional Gaussian component to the cooling rate, which
# enables a preferred epoch of 'extra' cooling.
# See 'log_cooling_rate' function in ares.physics.Cosmology to see the
# equation and parameters for this approach.
cold_2 = \
{
 'approx_thermal_history': 'exp+gauss', 
 'inits_Tk_p0': 600.,
 'inits_Tk_p1': 8.,
 'inits_Tk_p2': 0.4, 
 'inits_Tk_p3': 300., 
 'inits_Tk_p4': 200,
}

# Run it
sim = ares.simulations.Global21cm(fX=2., fstar=0.02, **cold_2)
sim.run()
sim.GlobalSignature(ls=':', label='exp+gauss', ymin=-750, label='fcoll+cold')

ax.legend(fontsize=12)









