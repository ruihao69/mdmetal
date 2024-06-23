import numpy as np
from numpy.typing import NDArray
from numba import njit
from typing import Tuple

@njit
def evaluate_nonadiabatic_couplings_1d(
    dHdR: NDArray[np.float64],
    evals: NDArray[np.float64],
    evecs: NDArray[np.float64],
    RTOL: float = 1.5e-3,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    dE = evals.max() - evals.min()

    n_elec = dHdR.shape[0]
    F = np.zeros(n_elec, dtype=np.float64)
    d = np.zeros((n_elec, n_elec), dtype=np.float64)
    F_hellmann_feynman = np.zeros((n_elec, n_elec), dtype=np.float64)

    F_hellmann_feynman[:] = -np.dot(evecs.T, np.dot(dHdR, evecs))

    for ii in range(dHdR.shape[0]):
        F[ii] = np.real(F_hellmann_feynman[ii, ii])
        for jj in range(ii+1, dHdR.shape[0]):
            # if (np.abs(evals[ii] - evals[jj]) < TOL) and (np.abs(F_hellmann_feynman[ii, jj]) < TOL):
            if (np.abs(evals[ii] - evals[jj]) < RTOL * dE): # adhoc filteration of trivial crossings
                continue
            E_diff = np.abs(evals[ii] - evals[jj])
            F_abs = np.abs(F_hellmann_feynman[ii, jj])
            # if (E_diff < epsilon):
            #     print(f"Warning: small energy difference {E_diff:.2e} between states {ii} and {jj} with Fij = {F_abs:.2e}")
            if F_abs / E_diff > 10:
                # print(f"Warning: large nonadiabatic coupling {F_abs:.2e} between states {ii} and {jj} with energy difference {E_diff:.2e}")
                continue
            d[ii, jj] = F_hellmann_feynman[ii, jj] / (evals[ii] - evals[jj])
            d[jj, ii] = -d[ii, jj]
    return d, F, F_hellmann_feynman

def evaluate_F_hellmann_feynman(dHdR: NDArray, evecs: NDArray):
    return -np.dot(evecs.conj().T, np.dot(dHdR, evecs))

def approximate_nac_matrix(evecs_last: NDArray, evecs_curr: NDArray, dt: float):
    """Use the adiabatic wavefunctions to approximate the nonadiabatic coupling matrix.

    Args:
        evecs_last (NDArray): the current eigenvectors (adiabatic wavefunctions)
        evecs_curr (NDArray): the previous eigenvectors (adiabatic wavefunctions)
        dt (float): time step (for classical update)

    Returns:
        NDArray: The approximate nonadiabatic coupling matrix.
    
    Reference:
        Hammes-Schiffer, S.; Tully, J. C. Proton Transfer in Solution: Molecular Dynamics with Quantum Transitions. J. Chem. Phys. 1994, 101, 4657-4667.
        Akimov. J. Phys. Chem. Lett. 2018, 9, 6096-6102
    """
    last_curr_term = np.dot(evecs_last.conj().T, evecs_curr)
    curr_last_term = np.dot(evecs_curr.conj().T, evecs_last)
    return 1.0 / (2.0 * dt) * (last_curr_term - curr_last_term)


@njit
def get_phase_correction_real(prev_evecs, curr_evecs):
    TOL = 1e-3
    phase_correction = np.zeros(curr_evecs.shape[0], dtype=np.float64)
    for ii in range(curr_evecs.shape[1]):
        tmp1 = np.ascontiguousarray(prev_evecs[:, ii])
        tmp2 = np.ascontiguousarray(curr_evecs[:, ii])
        dotval: float = np.dot(tmp1, tmp2)
        if np.abs(dotval) < TOL:
           raise ValueError(f"Small dot product value {dotval} between the consequtive {ii}-th states")
        phase_correction[ii] = np.sign(dotval) 
    return phase_correction

@njit
def get_phase_correction_complex(prev_evecs, curr_evecs):
    TOL = 1e-10
    phase_correction = np.zeros(curr_evecs.shape[0], dtype=np.complex128)
    tmp1 = np.zeros(curr_evecs.shape[0], dtype=np.complex128)
    tmp2 = np.zeros(curr_evecs.shape[0], dtype=np.complex128)
    for ii in range(curr_evecs.shape[1]):
        # tmp1 = np.ascontiguousarray(prev_evecs[:, ii], dtype=np.complex128)
        # tmp2 = np.ascontiguousarray(curr_evecs[:, ii], dtype=np.complex128)
        tmp1[:] = prev_evecs[:, ii] 
        tmp2[:] = curr_evecs[:, ii]
        tmpval: np.complex128 = np.dot(tmp1.conjugate(), tmp2)
        phase_correction[ii] = 1.0 if np.abs(tmpval) < TOL else tmpval / np.abs(tmpval)
    return phase_correction

def get_phase_correction(prev_evecs, curr_evecs):   
    if np.iscomplexobj(prev_evecs) or np.iscomplexobj(curr_evecs):
        raise NotImplementedError("Complex phase correction is not implemented yet")
    else:
        return get_phase_correction_real(prev_evecs, curr_evecs)
    
def get_corrected_psi(psi, phase_correction):
    # This phase correction *is* the Eq.4 of J. Phys. Chem. Lett. 2018, 9, 6096−6102
    return psi * phase_correction

@njit    
def phase_correct_nac(nac, phase_correction):
    for ii in range(nac.shape[0]):
        for jj in range(1+ii, nac.shape[1]):
            phase_corr: float = phase_correction[jj] * phase_correction[ii]
            nac[ii, jj] *= phase_corr
            nac[jj, ii] *= phase_corr
    return nac

@njit
def phase_correct_fhf(F_hellmann_feynman, phase_correction):
    for ii in range(F_hellmann_feynman.shape[0]):
        for jj in range(1+ii, F_hellmann_feynman.shape[1]):
            phase_corr: float = phase_correction[jj] * phase_correction[ii]
            F_hellmann_feynman[ii, jj] *= phase_corr
            F_hellmann_feynman[jj, ii] *= phase_corr
    return F_hellmann_feynman

@njit
def state_reordering(evecs_last, evecs_curr) -> Tuple[NDArray[np.int64]]:
    S = np.dot(evecs_last.T.conj(), evecs_curr)
    dim = S.shape[0]
    perm = np.arange(dim) 
    reorder_tuples = [] 
    reorder_cols = []
    for icol in range(dim):
        if icol in reorder_cols:
            continue
        maxval = -9999.0
        maxidx = -1
        for idx in range(icol, dim):
            tmp = max(np.abs(S[idx, icol]), maxval)
            if tmp > maxval:
                maxval = tmp
                maxidx = idx
        if perm[icol] != maxidx:
            # print(f"Reordering states {icol} and {maxidx}")
            reorder_tuples.append((icol, maxidx))
            reorder_cols.append(icol)
            reorder_cols.append(maxidx)
            
    # print(f"{reorder_tuples=}") 
    for (i, j) in reorder_tuples:
        perm[i], perm[j] = perm[j], perm[i]
    return perm 
        
        
        
    