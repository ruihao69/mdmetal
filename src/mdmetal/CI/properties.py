# %%
import numpy as np
from numba import njit

def compute_diabatic_populations_from_adiabatic_psi(psi_adiabatic, evecs):
    psi_diabatic = np.dot(evecs, psi_adiabatic)
    return np.abs(psi_diabatic)**2

@njit
def reduce_state_populations(psi_CI, norbital, states):
    populations = np.zeros(norbital, dtype=np.float64)
    for istate, state in enumerate(states):
        pop_istate = np.abs(psi_CI[istate])**2
        for ii in state:
            populations[ii] += pop_istate  
    return populations

def compute_orbital_populations(psi_CI_adiabatic, evecs, norbital, states):
    psi_CI_diabatic = np.dot(evecs, psi_CI_adiabatic)
    populations = reduce_state_populations(psi_CI_diabatic, norbital, states)
    return populations

def compute_KE(P, mass):
    return 0.5 * P**2 / mass

def compute_PE_ehrenfest(evals, c):
    return np.dot(evals * c.conj(), c).real