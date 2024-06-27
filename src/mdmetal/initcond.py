# %%
import numpy as np

from mdmetal.parse_inp import Method
from mdmetal.hamiltonian import NewnsAndersonHarmonic

import math
from itertools import combinations

# initial conditions for the classical dof

def boltzmann_sampling(n_samples: int, kT: float, mass: float, Omega_B: float) -> np.ndarray:
    sigma_R = np.sqrt(kT / (mass * Omega_B**2))
    sigma_P = np.sqrt(kT * mass * Omega_B**2)
    R_list = np.random.normal(0, sigma_R, n_samples)
    P_list = np.random.normal(0, sigma_P, n_samples)
    return R_list, P_list

def wigner_sampling(n_samples: int, kT: float, mass: float, Omega_B: float) -> np.ndarray:
    beta = 1 / kT
    
    sigma_dimless = np.sqrt(0.5 / np.tanh(0.5 * beta * Omega_B))
    dimless_to_P = np.sqrt(Omega_B * mass)
    dimless_to_R = np.sqrt(1.0 / (mass * Omega_B))
    
    sigma_P = sigma_dimless * dimless_to_P
    sigma_R = sigma_dimless * dimless_to_R
    R_list = np.random.normal(0, sigma_R, n_samples)
    P_list = np.random.normal(0, sigma_P, n_samples)
    return R_list, P_list

def init_diab_amplitudes(
    no: int,
    ne: int,
    diabatic_state: np.ndarray,
    method: Method,
) -> np.ndarray:
    # initialize the amplitudes based on the initial diabatic state and the method
    if method.is_CI:
        nstates = math.comb(no, ne)
        psi0 = np.zeros(nstates, dtype=np.complex128)  
        for istate, state in enumerate(combinations(range(no), ne)):
            if np.all(state == diabatic_state):
                psi0[istate] = 1.0
                break
    elif method.is_IESH:
        psi0 = np.zeros((no, ne), dtype=np.complex128)
        for ie, io in enumerate(diabatic_state):
            psi0[io, ie] = 1.0
    else:
        raise ValueError(f"method {method} not implemented. It should be either CI or IESH")    
    return psi0

def init_amplitudes(
    no: int,
    ne: int,
    diabatic_state: np.ndarray,
    method: Method,
    R0: np.ndarray,
    hami: NewnsAndersonHarmonic, 
) -> np.ndarray:
    ntrajs: int = R0.shape[0]
    # from initial diab state to initial diabatic state
    psi0_diab = init_diab_amplitudes(no, ne, diabatic_state, method)
    if method.is_diabatic:
        return np.array([psi0_diab for _ in range(ntrajs)]) 
    else:
        if method != Method.CI_FSSH_A:
            psi0_adiab = []
            for r0 in R0:
                H, _ = hami.evaluate(r0, is_CI=method.is_CI)
                _, evecs = np.linalg.eigh(H)
                # transform the diabatic state to adiabatic state
                psi0_adiab.append(np.dot(evecs.T.conj(), psi0_diab))
            return np.array(psi0_adiab)
        else:
            from mdmetal.CI.fssh2 import permutation_order_and_list, get_U_state
            perm, order = permutation_order_and_list(hami.states, ne)
            psi0_adiab = []
            for r0 in R0:
                H, _ = hami.evaluate(R0[0], is_CI=False)
                _, evecs = np.linalg.eigh(H)
                # construct the U_state matrix
                U_state = get_U_state(evecs, hami.states, perm, order)
                psi0_adiab.append(np.dot(U_state.T.conj(), psi0_diab))
            return np.array(psi0_adiab)