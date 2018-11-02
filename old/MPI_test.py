


import sys
import numpy as np
import scipy as sp
import scipy.interpolate as spi
import matplotlib.pyplot as plt

import emcee
import matlab.engine
from emcee.utils import MPIPool



matlab_engine = matlab.engine.start_matlab()

# Run this program from the terminal as follows:
# 
# $ mpirun -np 2 python MPI_test.py
#
# It requires an MPI implementation. Best option is mpich. Install as: $ sudo apt-get install mpich




# Loading High-band data
data  = np.genfromtxt('/home/ramo7131/DATA/EDGES/results/high_band/products/average_spectrum/high_band_2015_LST_0.26_6.26_dates_2015_250_299_nominal.txt')
vb    = data[:,0]
db    = data[:,1]
wb    = data[:,2]


# Cutting data within desired range
flow  = 90
fhigh = 190
v     = vb[(vb>=flow) & (vb<=fhigh)]
d     = db[(vb>=flow) & (vb<=fhigh)]
w     = wb[(vb>=flow) & (vb<=fhigh)]


# Normalized frequency
vn       = flow + (fhigh-flow)/2


# Uncertainty StdDev
sigma_noise = 0.035*np.ones(len(v))


# Foreground model
model_fg        = 'EDGES_polynomial'
Nfg             = 5


# 21cm fit parameters 
P1 = np.log10(0.001) + (np.log10(0.5)-np.log10(0.001))/2
P2 = np.log10(16.5) + (np.log10(76.5)-np.log10(16.5))/2
P3 = np.log10(0.00001) + (np.log10(10)-np.log10(0.00001))/2
P4 = 0.055 + (0.1-0.055)/2
P5 = 1 + (1.5-1)/2
P6 = 0.1 + (3-0.1)/2
P7 = 10 + (50-10)/2

parameters_21cm = [-1, -1, P3, P4, P5, P6, P7]   # Parameters with -1 are fit parameters. Other parameters take the Px fixed value.



# MCMC chain parameters
MCMC_Nchain            = 500
MCMC_rejected_fraction = 0.3





























def fit_polynomial_fourier(model_type, xdata, ydata, nterms, Weights=1, external_model_in_K=0):

        """

	2017-10-02
	This a cheap, short version of the same function in "edges.py"

	"""


        # Initializing "design" matrix
        AT  = np.zeros((nterms, len(xdata)))

        # Initializing auxiliary output array	
        aux = (0, 0)

        # Evaluating design matrix with foreground basis
        if (model_type == 'EDGES_polynomial') or (model_type == 'EDGES_polynomial_plus_external'):
                for i in range(nterms):
                        AT[i,:] = xdata**(-2.5+i)

        # Adding external model to the design matrix
        if model_type == 'EDGES_polynomial_plus_external':
                AT = np.append(AT, external_model_in_K.reshape(1,-1), axis=0)





        # Applying General Least Squares Formalism, and Solving using QR decomposition
        # ----------------------------------------------------------------------------
        # ----------------------------------------------------------------------------
        # See: http://www2.imm.dtu.dk/pubdb/views/edoc_download.php/2804/pdf/imm2804.pdf

        # If no weights are given
        if np.isscalar(Weights):
                W = np.eye(len(xdata))

        # If a vector is given
        elif np.ndim(Weights) == 1:
                W = np.diag(Weights)

        # If a matrix is given
        elif np.ndim(Weights) == 2:
                W = Weights


        # Sqrt of weight matrix
        sqrtW = np.sqrt(W)


        # Transposing matrices so 'frequency' dimension is along columns
        A     = AT.T
        ydata = np.reshape(ydata, (-1,1))


        # A and ydata "tilde"
        WA     = np.dot(sqrtW, A)
        Wydata = np.dot(sqrtW, ydata)


        # Solving system using 'short' QR decomposition (see R. Butt, Num. Anal. Using MATLAB)
        Q1, R1      = sp.linalg.qr(WA, mode='economic') # returns

        param       = sp.linalg.solve(R1, np.dot(Q1.T, Wydata))

        model       = np.dot(A, param)
        error       = ydata - model
        DF          = len(xdata)-len(param)-1
        wMSE        = (1./DF) * np.dot(error.T, np.dot(W, error))  # This is correct because we also have the Weight matrix in the (AT * W * A)^-1.
        wRMS        = np.sqrt( np.dot(error.T, np.dot(W, error)) / np.sum(np.diag(W)))
        #inv_pre_cov = np.linalg.lstsq(np.dot(R1.T, R1), np.eye(nterms))   # using "lstsq" to compute the inverse: inv_pre_cov = (R1.T * R1) ^ -1
        #cov         = MSE * inv_pre_cov[0]
        inv_pre_cov = np.linalg.inv(np.dot(R1.T, R1))
        cov         = wMSE * inv_pre_cov


        # Back to input format
        ydata = ydata.flatten()
        model = model.flatten()
        param = param.flatten()


        return param, model, wRMS, cov, wMSE, aux     # wMSE = reduced chi square









def model_evaluate(model_type, par, xdata):

        """

	2017-10-02
	This a cheap, short version of the same function in "edges.py"

	"""


        if (model_type == 'EDGES_polynomial'):
                summ = 0
                for i in range(len(par)):
                        summ = summ + par[i] * xdata**(-2.5+i)


        else:
                summ = 0


        model = np.copy(summ)
        return model









def fialkov_model_21cm(f_new, values_21cm, interpolation_kind='linear'):

        """


	2017-10-02
	Function that produces a 21cm model in MATLAB and sends it to Python

	Webpage: https://www.mathworks.com/help/matlab/matlab-engine-for-python.html


	"""


        # Copying parameter list
        all_lin_values_21cm    = values_21cm[:]


        # Transforming first three parameters from Log10 to Linear
        all_lin_values_21cm[0] = 10**(values_21cm[0])
        all_lin_values_21cm[1] = 10**(values_21cm[1])
        all_lin_values_21cm[2] = 10**(values_21cm[2])


        # Calling MATLAB function
        # matlab_engine = matlab.engine.start_matlab()
        #print all_lin_values_21cm
        #future_model_md, future_Z, future_flags_md = matlab_engine.Global21cm.run(matlab.double(all_lin_values_21cm), nargout=3, async=True)
        #model_md = future_model_md.result()
        #Z        = future_Z.result()
        #flags_md = future_flags_md.result()
        
        
        model_md, Z, flags_md = matlab_engine.Global21cm.run(matlab.double(all_lin_values_21cm), nargout=3)
        


        # Converting MATLAB outputs to numpy arrays
        model_raw = np.array(model_md._data.tolist())
        flags     = np.array(flags_md._data.tolist())


        # Redshift and frequency
        z_raw = np.arange(5,50.1,0.1)
        c     = 299792458	# wikipedia, m/s
        f21   = 1420.40575177e6  # wikipedia,    
        l21   = c / f21          # frequency to wavelength, as emitted
        l     = l21 * (1 + z_raw)
        f_raw = c / (l * 1e6)


        # Interpolation
        func_model = spi.interp1d(f_raw, model_raw, kind=interpolation_kind)
        model_new  = func_model(f_new)	


        return model_new/1000, Z, flags






def fialkov_priors_foreground(theta):

        # a0, a1, a2, a3 = theta

        # Counting the parameters with allowable values
        flag = 0
        for i in range(len(theta)):
                if (-1e5 <= theta[i] <= 1e5):
                        flag = flag + 1

        # Assign likelihood
        if flag == len(theta):
                out = 0
                print 'good_fg'
        else:
                out = -1e10

        return out





def fialkov_priors_21cm(theta, flags):

        P1, P2, P3, P4, P5, P6, P7 = theta

        if (np.log10(0.001) <= P1 <= np.log10(0.5)) \
           and (np.log10(16.5) <= P2 <= np.log10(76.5)) \
           and (np.log10(0.00001) <= P3 <= np.log10(10)) \
           and (0.055 <= P4 <= 0.088) \
           and (1 <= P5 <= 1.5) \
           and (0.1 <= P6 <= 3) \
           and (10 <= P7 <= 50) \
           and (flags[0] == 0) \
		   and (flags[2] == 0):
                out = 0

                print 'good_21'
        else:
                out = -1e10
                print 'bad_21'

        print theta
        #print flags

        return out







def fialkov_log_likelihood(theta):

        # Evaluating model foregrounds
        if Nfg == 0:
                Tfg = 0
                log_priors_fg = 0

        elif Nfg > 0:
                Tfg           = model_evaluate(model_fg, theta[0:Nfg], v/vn)
                log_priors_fg = fialkov_priors_foreground(theta[0:Nfg])

        # Evaluating model 21cm
        j = -1
        values_21cm = parameters_21cm[:]
        for i in range(len(parameters_21cm)):
                if parameters_21cm[i] == -1:
                        j = j + 1
                        values_21cm[i] = theta[Nfg + j]

        T21, Z, flags = fialkov_model_21cm(v, values_21cm, interpolation_kind='linear')
        log_priors_21 = fialkov_priors_21cm(values_21cm, flags)

        # Full model
        Tfull = Tfg + T21

        # Log-likelihood
        log_likelihood =  -(1/2)  *  np.sum(  ((d-Tfull)/sigma_noise)**2  )

        # Log-priors + Log-likelihood
        return log_priors_fg + log_priors_21 + log_likelihood




































#def lnprob(x):
#   k = eng.mean(matlab.double([1, 2, 3]))
#	p21_center = [P1_center, P2_center, P3_center, P4_center, P5_center, P6_center, P7_center]
#	p21_center[0] = 10**(p21_center[0])
#	p21_center[1] = 10**(p21_center[1])
#	p21_center[2] = 10**(p21_center[2])
#	model_md, Z, flags_md = fialkov_model_21cm(v, p21_center, interpolation_kind='linear')
#	print 'xxx: ' + str(np.random.normal()) + ' \n'
#	return -0.5 * np.sum(x ** 2) + np.sum(model_md)




























# Pool for MCMC
pool = MPIPool()
if not pool.is_master():
	pool.wait()
	sys.exit(0)




# Counting number of fit 21cm parameters
N21 = 0
for i in range(len(parameters_21cm)):
        if parameters_21cm[i] == -1:
                N21 = N21 + 1

# Number of parameters (dimensions) and walkers
Ndim     = Nfg + N21
Nwalkers = 2*Ndim + 2




# Random starting point
# p0      = [np.random.rand(Ndim) for i in xrange(Nwalkers)]
p0 = (np.random.uniform(size=Ndim*Nwalkers).reshape((Nwalkers, Ndim)) - 0.5) / 10

# Appropriate offset to the random starting point of some parameters
# First are foreground model coefficients
x        = fit_polynomial_fourier(model_fg, v/vn, d, Nfg, Weights=w)
poly_par = x[0]	
for i in range(Nfg):
        p0[:,i] = p0[:,i] + poly_par[i]

# Then, are the 21cm parameters
P21_offset = [P1, P2, P3, P4, P5, P6, P7]
j=-1
for i in range(len(parameters_21cm)):
        if parameters_21cm[i] == -1:
                j = j+1
                p0[:,Nfg+j] = p0[:,Nfg+j] + P21_offset[i]




# MCMC processing
sampler = emcee.EnsembleSampler(Nwalkers, Ndim, fialkov_log_likelihood, pool=pool)
sampler.run_mcmc(p0, MCMC_Nchain)

# Dimension of accepted sampes is (MCMC_rejected_fraction*MCMC_Nchain*Nwalkers), (Ndim)
samples_cut = sampler.chain[:, (np.int(MCMC_rejected_fraction*MCMC_Nchain)):, :].reshape((-1, Ndim))

print samples_cut.shape
pool.close()





plt.figure()
plt.subplot(2,5,1)
plt.hist(samples_cut[:,0])

plt.subplot(2,5,2)
plt.hist(samples_cut[:,1])

plt.subplot(2,5,3)
plt.hist(samples_cut[:,2])

plt.subplot(2,5,4)
plt.hist(samples_cut[:,3])

plt.subplot(2,5,5)
plt.hist(samples_cut[:,4])

plt.subplot(2,5,6)
plt.hist(samples_cut[:,5])

plt.subplot(2,5,7)
plt.hist(samples_cut[:,6])


plt.figure()
plt.plot(samples_cut[:,5], samples_cut[:,6], '.')
plt.show()








