from scipy import signal
import numpy as np
from astropy import units as u, constants as const

def rotational_broadening(dv=0.1, N=1e3, vrot=5, epsilon=0.5):
        
    vrot   = vrot.to('km/s').value
    dv   = dv.to('km/s').value
    v    = np.arange(N)*dv 
    v   -= v[-1]/2
    y    = 1 - (v/vrot)**2 
    inds = y > 0
    
    K    = np.zeros_like(v)
    K[inds] = (2*(1-epsilon)*np.sqrt(y[inds])+np.pi*epsilon/2.*y[inds])/(np.pi*vrot*(1-epsilon/3.0)) 
    K /= K.sum()
    
    return K

def gaussian_broadening(dv=100*u.m/u.s, N=1e3, R=300_000):

    std    = (const.c/R/dv/(2*np.sqrt(np.log(4)))).decompose().value
    G      = signal.windows.gaussian(N, std=std)
    G     /= G.sum()
    
    return G 


def velocity_shift(spec, dv, v_shift):

    N       = len(spec)
    vv      = v_shift.to(dv.unit).value
    F       = np.fft.rfft(spec, n=N)
    omega   = -(2*np.pi*1.j) * np.fft.rfftfreq(N, d=dv.value)
    shifted = np.fft.irfft(F*np.exp(vv*omega), n=N)

    return shifted