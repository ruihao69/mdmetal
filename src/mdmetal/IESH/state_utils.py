# %%
import numpy as np
from numba import njit, jit
from numpy.typing import NDArray

from mdmetal.hamiltonian import numba_setdiff1d

def get_iesh_targets(
    state: NDArray[np.int64],
    states: NDArray[np.int64],
    no: int,
) -> np.ndarray:
    target_states = [] 
    one_electron_terms = [] # bdagger_i b_j coupling
    init_state_vec = np.zeros(no, dtype=np.int64)
    targ_state_vec = np.zeros(no, dtype=np.int64)
    init_state_vec[state] = 1
    
    for target in states:
        targ_state_vec[target] = 1
        diff = targ_state_vec - init_state_vec
        # IESH considers only 1-electron excitations
        if np.sum(np.abs(diff)) == 2:
            target_states.append(target)
            i = np.where(diff == -1)[0][0] 
            j = np.where(diff == 1)[0][0]
            one_electron_terms.append(np.array([i, j]))
        targ_state_vec[target] = 0
        
    return np.array(target_states), np.array(one_electron_terms)
    # return target_states, one_electron_terms

def map_state_to_index(
    state: NDArray[np.int64],
    states: NDArray[np.int64],
) -> int:
    return np.where(np.all(states == state, axis=1))[0][0]

@njit
def one_electron_overlap(
    c: NDArray[np.complex128],  
    k: NDArray[np.int64],
) -> np.complex128:
    ne = k.shape[0]
    S = np.zeros((ne, ne), dtype=np.complex128)
    for ii in range(ne):
        for jj in range(ne):
            S[ii, jj] = c[k[ii], jj]
    return np.linalg.det(S) 

@njit
def evaluate_Akj(
    c: NDArray[np.complex128], # independent-electron wavefunctions (n_orbital, n_electron)
    state_k: NDArray[np.int64], # bra state
    state_j: NDArray[np.int64], # ket state
) -> NDArray[np.complex128]:
    j_psi = one_electron_overlap(c, state_j)
    psi_k = np.conj(one_electron_overlap(c, state_k))
    return j_psi * psi_k

@njit
def get_hopping_prob(
    c: NDArray[np.complex128],
    active_state: NDArray[np.int64], # k-state
    target_state: NDArray[np.int64], # j-state
    elem_1d: NDArray[np.int64], # 1-electron terms
    v_dot_d: NDArray[np.float64],
    dt: float,
) -> np.float64:
    Akk = evaluate_Akj(c, active_state, active_state).real
    Akj = evaluate_Akj(c, active_state, target_state)
    p, q = elem_1d 
    Bjk = -2.0 * np.real(np.conj(Akj) * v_dot_d[p, q])
    gjk = max(0, dt * Bjk / Akk)
    return gjk
            
        
    
# %%