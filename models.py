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

def f_T(T, t):
    # return (1 - np.cos(2*np.pi/T * t)) / 2
    return -np.cos(2*np.pi/T * t)

# Define dynamics of T: circadian rhythm period
def dT(T, mu, tau): 
    # Original: dtheta_dt = freq + f_L(theta % (2*np.pi), L_input(t,light_params)) - .001 * theta
    # Old: -(T - mu) * (T - (mu-1))**2 * (T - (mu+1))**2
    # Old: np.exp(-(T-mu)**2) * (T - mu)*(T - (mu-1))*(T - (mu+1))
    # Simple: -(T-mu)/TAU_T
    # Sigmoidal simple: -np.tanh(T-mu)/TAU_T
    # return -np.tanh(T-mu) * np.exp(-(T-mu)**2/(2*sigma**2))
    return  -np.tanh(T-mu)/tau

# Define system dynamics
def circadian(t, y, params):
    T    = y[0]
    L    = params['light']['L'](t, params['light'])
    dTdt =  dT(T, mu=params['mu'], tau=params['tau']) - params['alpha'] * f_L(phi(t,T=T), L)
    return [dTdt]


## Sleep model functions
def f_S(S):
    return S**3

def dS(S, tau, S_max=2):
    return (S_max-S)/tau

def sleep(t, y, params):
    S    = y[0]
    dSdt = dS(S, params['tau_S']) - params['rest']['R'](t, params['rest'])/params['tau_R']
    return [dSdt]

## Combined model
def energy(t, y, params):
    T    = y[0]
    S    = y[1]
    # Circadian dynamics
    R = params['S']['rest']['R']
    L    = params['C']['light']['L'](t, params['C']['light']) * (R==0)
    dTdt =  dT(T, mu=params['C']['mu'], tau=params['C']['tau']) - params['C']['alpha'] * f_L(phi(t,T=T), L)
    # Sleep dynamics
    dSdt = ( dS(S, params['S']['tau_S']) - R(t, params['S']['rest'])/params['S']['tau_R'] ) + f_T(T,t)/params['C']['tau']
    return [dTdt, dSdt]