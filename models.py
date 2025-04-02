import numpy as np
from scipy.stats import norm
from utils import *

## Circadian rhythm model functions
# Obtain the phase of a periodic function
def phi(t, T, t0=0):
    x = (t - t0)/T
    x_frac = x - np.floor(x)
    return 2*np.pi * x_frac

# Define circadian PRC by summing Gaussian PDFs
def PRC(phase):
    return norm.pdf(phase,loc=np.pi/2,scale=np.pi/4)-norm.pdf(phase,loc=2*np.pi,scale=np.pi/4)-norm.pdf(phase,loc=0,scale=np.pi/4)

# Define Light nonlinearity as a function of internal clock phase
def f_L(phi, L):
    return PRC(phi) * L

# Define dynamics of T: circadian rhythm period
def dT(T, mu, sigma): 
    # Original: dtheta_dt = freq + f_L(theta % (2*np.pi), L_input(t,light_params)) - .001 * theta
    # Old: -(T - mu) * (T - (mu-1))**2 * (T - (mu+1))**2
    # Old: np.exp(-(T-mu)**2) * (T - mu)*(T - (mu-1))*(T - (mu+1))
    # Simple: -(T-mu)/TAU_T
    # Sigmoidal simple: -np.tanh(T-mu)/TAU_T
    return -np.tanh(T-mu) * np.exp(-(T-mu)**2/(2*sigma**2))

# Define system dynamics
def circadian(t, y, params):
    T    = y[0]
    L    = input_daily(t, params['light'])
    dTdt =  dT(T, mu=params['mu'], sigma=params['sigma']) - params['alpha'] * f_L(phi(t,T=T), L)
    return [dTdt]