# %%
import numpy as np
from numpy.typing import NDArray
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

@njit
def compute_surface_hopping_pop(
    active_state: int,
    c: np.ndarray,
    U: np.ndarray,
) -> np.ndarray:
    ns = c.shape[0]
    rho = np.outer(c, c.conj()) 
    populations = np.zeros(ns, dtype=np.float64)
    for istate in range(ns):
        populations[istate] += np.abs(U[istate, active_state])**2
        for jj in range(ns):
            for kk in range(jj+1, ns):
                populations[istate] += 2.0 * np.real(U[istate, jj] * np.conj(U[istate, kk]) * rho[jj, kk])
    return populations  

# @njit
# def compute_surface_hopping_pop2(
#     active_state: int,
#     evecs: NDArray[np.float64], 
#     states: NDArray[np.int64]
# ):
#     ns, ne = states.shape
#     istate = states[active_state]
#     c_diab = np.zeros(ns, dtype=np.complex128)
#     for jj, jstate in enumerate(states):
#         tmp_val: complex = 1.0
#         for kk in range(ne):
#             i, j = istate[kk], jstate[kk]
#             tmp_val *= evecs[i, j]
#         c_diab[jj] += tmp_val
#     return reduce_state_populations(c_diab, evecs.shape[0], states)
               

def compute_surface_hopping_pop2(
    active_state: int,
    U_orb: np.ndarray,
    states: NDArray[np.int64]
) -> np.ndarray:
    # the active state is:  
    # a_1^{\dagger} ... a_n^{\dagger} | >, where n is the number of electrons
    # thus, the simplist equation for the corresponding diabatic orbitals is:
    # pop(iorbital) = \sum_{ielec} \sum_j^{norbs} |U[j, state[ielec]]|^2
    # However, the off diagonal contributions from the coherence terms 
    # will be ignored for now.
    ns, ne = states.shape
    no = U_orb.shape[1]
    state = states[active_state]
    populations = np.zeros(no, dtype=np.float64)
    for io in range(no):
        for ie in range(ne):
            active_orb = state[ie]
            populations[io] += np.abs(U_orb[active_orb, io])**2
    return populations