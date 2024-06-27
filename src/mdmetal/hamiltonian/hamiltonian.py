# %%
import numpy as np
from numpy.typing import NDArray
from numba import njit
from scipy.special import roots_legendre
import scipy.linalg as LA

from mdmetal.hamiltonian.potential import U0, U1, grad_U0, grad_U1

import math
from dataclasses import dataclass
from itertools import combinations
import warnings
from typing import Tuple

@njit
def one_electron_newns_anderson(
    U0: float,
    U1: float,
    V: float,
    ek: NDArray[np.float64],
    vk: NDArray[np.float64],
) -> NDArray[np.float64]:
    no = ek.size + 1
    H = np.zeros((no, no), dtype=np.float64)
    for ii in range(no):
        if ii == 0:
            H[ii, ii] += (U1 - U0) # only have the state inidependent
                                   # potential in the Hamiltonian
        else:
            H[ii, ii] += ek[ii-1] 
            H[ii, 0] = H[0, ii] = V * vk[ii-1]
    return H

@njit
def one_electron_newns_anderson_grad(
    dU0: float,
    dU1: float,
    no: int,
) -> NDArray[np.float64]:
    grad_H = np.zeros((no, no), dtype=np.float64)
    for ii in range(no):
        if ii == 0:
            grad_H[ii, ii] = dU1 - dU0 # only have the state inidependent
                                       # potential in the Hamiltonian
    return grad_H

@njit
def numba_setdiff1d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # np.setdiff1d(a, b) is not supported by Numba
    # simple implementation
    diff = np.zeros_like(a)
    diff_count = 0
    for i in a:
        for j in b:
            if i == j:
                break
        else:
            diff_count += 1
            diff[diff_count - 1] = i
    diff_out = np.zeros(diff_count, dtype=a.dtype)
    for i in range(diff_count):
        diff_out[i] = diff[i]
    return diff_out

@njit
def get_quad_hamiltonian_CI(
    h: NDArray[np.float64],
    states: NDArray[np.int64],
) -> NDArray[np.float64]:
    # h = \sum_{ij} h_{ij} |i><j|
    ns, ne = states.shape

    H_CI = np.zeros((ns, ns), dtype=np.float64)

    # diagonal elements
    for ii in range(ns):
        for jj in range(ne):
            orbit_ij = states[ii, jj]
            H_CI[ii, ii] += h[orbit_ij, orbit_ij]

    # off-diagonal elements
    for ii, istate in enumerate(states):
        for jj, jstate in enumerate(states):
            if ii == jj:
                continue
            idiff = numba_setdiff1d(istate, jstate)
            if idiff.size != 1: # only one element should be different
                continue
            jdiff = numba_setdiff1d(jstate, istate)
            i, j = idiff[0], jdiff[0]
            H_CI[ii, jj] = h[i, j]
    return H_CI



def align_phase(prev_evecs, curr_evecs):
    """Algorithm to align the phase of eigen vectors, originally proposed
    by Graham (Gaohan) Miao and Zeyu Zhou. (the Hamiltonian is changed adiabatically)

    Args:
        prev_evecs (ArrayLike): the eigen vectors of the previous iteration
        curr_evecs (ArrayLike): the eigen vectors of the current iteration

    Returns:
        ArrayLike: the phased-corrected eigen vectors

    Reference:
        [1]. `FSSHND` by Graham (Gaohan) Miao. git@github.com:Eplistical/FSSHND.git.
             See the 'Hamiltonian::cal_info()' function in 'src/hamiltonian.cpp'
    """
    # if np.iscomplexobj(prev_evecs) or np.iscomplexobj(curr_evecs):
    #     tmp = np.dot(prev_evecs.conjugate().T, curr_evecs)
    # else:
    #     tmp = np.dot(prev_evecs.T, curr_evecs)
    if prev_evecs is None:
        return curr_evecs
    else:
        tmp = np.dot(prev_evecs.conjugate().T, curr_evecs)
        diag_tmp = np.diag(tmp)
        phase_factors = np.ones_like(diag_tmp)
        # mask = np.logical_not(np.isclose(diag_tmp.real, 0))
        # phase_factors[mask] = np.sign(diag_tmp[mask].real)
        mask = np.logical_not(np.isclose(diag_tmp, 0))
        phase_factors[mask] = diag_tmp[mask] / np.abs(diag_tmp[mask])
        aligned_evecs = curr_evecs / phase_factors
        return aligned_evecs

@dataclass
class NewnsAndersonHarmonic:
    """
    Newns-Anderson Hamiltonian with harmonic classical degrees of freedom.
    """
    nk: int                   # gaussian quadrature points for the electornic bath states
    ek: NDArray[np.float64]   # electronic bath energies
    vk: NDArray[np.float64]   # system-electronic bath couplings
    no: int                   # number of orbitals
    ne: int                   # number of electrons
    states: NDArray[np.int64] # electronic states
    mass: float = 2000.0      # harmonic oscillator mass
    omega_B: float = 2e-4     # harmonic oscillator frequency
    # gamma: float = 1.0        # langevin bath friction
    gamma: float = omega_B * 2.0
    # Gamma: float = 1e-5       # metal-molecule coupling (wide-band limit)
    Gamma: float = 1e-4       # metal-molecule coupling (wide-band limit)
    # Gamma: float = 2e-4
    Er: float = 0.00125       # reorganization energy
    dG: float = -0.0038       # electronic energy difference
    kT: float = 0.00095       # thermal energy
    # W: float = 16e-3          # electron bath bandwidth
    W: float = 2e-2         # electron bath bandwidth
    g: float = (2*Er / mass / omega_B**2)**0.5 # electron-vibration coupling
    # g: float = 20.6097
    V: float = np.sqrt(0.5 * Gamma / np.pi) # system-electronic bath coupling

    @classmethod
    def initialize(cls, nk: int, ne: int, kT: float, W: float, Gamma: float, mass: float) -> "NewnsAndersonHarmonic":
        gird_points, weights = cls.get_gauss_quad(nk)
        ek = W / 4 * (1 + gird_points)
        ek = np.concatenate((ek, -ek))
        vk = np.sqrt(W * weights) / 2
        vk = np.concatenate((vk, vk))

        argsort = np.argsort(ek)
        ek = ek[argsort]
        vk = vk[argsort]

        no = nk * 2 + 1 # number of orbitals = bath orbitals + system orbital (1-level)
        state_list = []
        for state in combinations(range(no), ne):
            state_list.append(state)
        if len(state_list) > 10000:
            warnings.warn("Number of states is large, consider reducing the number of electrons.")

        states = np.array(state_list, dtype=np.int64)

        # return cls(nk, ek, vk, no, ne, states)
        return cls(nk, ek, vk, no, ne, states, mass=mass, kT=kT, W=W, Gamma=Gamma)

    @staticmethod
    def get_gauss_quad(nk: int):
        gird_points, weights = roots_legendre(nk)
        return gird_points, weights

    def get_H_one_electron(
        self,
        R: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        u0 = U0(R, self.mass, self.omega_B)
        u1 = U1(R, self.g, self.mass, self.omega_B, self.dG)
        return one_electron_newns_anderson(u0, u1, self.V, self.ek, self.vk)

    def get_grad_H_one_electron(
        self,
        R: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        dU0 = grad_U0(R, self.mass, self.omega_B)
        dU1 = grad_U1(R, self.g, self.mass, self.omega_B)
        return one_electron_newns_anderson_grad(dU0, dU1, self.no)

    def get_H_CI(
        self,
        h: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        if self.states.shape[0] > 300:
            raise ValueError("Number of states is large, CI simulation is impractical.")
        return get_quad_hamiltonian_CI(h, self.states)

    def get_grad_H_CI(
        self,
        grad_h: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        if self.states.shape[0] > 300:
            raise ValueError("Number of states is large, CI simulation is impractical.")
        return get_quad_hamiltonian_CI(grad_h, self.states)
    
    def get_U0(
        self,
        R: float,
    ) -> float:
        return U0(R, self.mass, self.omega_B)   
    
    def get_grad_U0(
        self,
        R: float,
    ) -> float:
        return grad_U0(R, self.mass, self.omega_B)

    def evaluate(
        self,
        R: float,
        is_CI: bool,
    ):
        H = self.get_H_one_electron(R)
        grad_H = self.get_grad_H_one_electron(R)
        if is_CI:
            H_CI = self.get_H_CI(H)
            grad_H_CI = self.get_grad_H_CI(grad_H)
            return H_CI, grad_H_CI
        else:
            return H, grad_H



def main():
    import matplotlib.pyplot as plt
    from mdmetal.hamiltonian.utils import evaluate_nonadiabatic_couplings_1d

    nk = 2
    ne = 2
    model = NewnsAndersonHarmonic.initialize(nk, ne, 0.00095, 16e-3, 1e-5, 2000.0)
    print(model.ek)
    print(model.V * model.vk)
    
    print(model.g)

    R_TEST = 1.0
    V_TEST = 1.0
    H_test = model.get_H_one_electron(R_TEST)
    gradH_test = model.get_grad_H_one_electron(R_TEST)
    evals, evecs = LA.eigh(H_test)
    E_cl = 0.5 * model.mass * model.omega_B**2*R_TEST**2
    grad_cl = model.mass * model.omega_B**2*R_TEST
    print(f"{H_test.flatten()=}")
    print(f"{evals=}")
    print(f"{evals-E_cl=}")
    print(f"{gradH_test.flatten()=}")
    grad_H_diff = gradH_test - np.diagflat([grad_cl for _ in range(gradH_test.shape[0])])
    print(f"{grad_H_diff=}")
    # nac, F, F_hf= evaluate_nonadiabatic_couplings_1d(grad_H_diff, evals, evecs)
    nac, F, F_hf= evaluate_nonadiabatic_couplings_1d(gradH_test, evals, evecs)
    print(f"{nac.flatten()=}")
    vdotd = nac * V_TEST
    print(f"{vdotd.flatten()=}")
    state = np.array([1, 2])
    acc = sum(F[iorb] / model.mass for iorb in state )
    print(f"acc={acc}")
    F_cl = -grad_cl / model.mass
    print(f"{F_cl=}")

    L = 30
    R = np.linspace(-L, L, 5000)

    H_list = np.zeros((R.size, model.no, model.no))
    E_list = np.zeros((R.size, model.no))
    E0_list = np.zeros((R.size, model.no)) # state-independent potential
    evecs_list = np.zeros((R.size, model.no, model.no))
    grad_H_list = np.zeros((R.size, model.no, model.no))
    nac_list = np.zeros((R.size, model.no, model.no))
    for ii, r in enumerate(R):
        E0_list[ii] = model.get_U0(r)
        H_list[ii] = model.get_H_one_electron(r)
        # E_list[ii], evecs_list[ii] = np.linalg.eigh(H_list[ii])
        E_list[ii], evecs_list[ii] = LA.eigh(H_list[ii])
        if ii > 0:
            evecs_list[ii] = align_phase(evecs_list[ii-1], evecs_list[ii])
        # grad_H = model.get_grad_H_one_electron(r)
        grad_H_list[ii] = model.get_grad_H_one_electron(r)
        nac_list[ii] = evaluate_nonadiabatic_couplings_1d(grad_H_list[ii], E_list[ii], evecs_list[ii])[0]
    
    E_list += E0_list

    fig = plt.figure(figsize=(16, 6), dpi=300)
    ax = fig.add_subplot(121)
    for ii in range(model.no):
    # for ii in [2, 3]:
        ax.plot(R, E_list[:, ii], label=f"Orbital {ii}")
    ax.set_xlabel("R")
    ax.set_ylabel("Energy")
    ax.legend()
    ax.set_xlim(-10, 20)
    ax.set_ylim(-0.02, 0.05)
    ax = fig.add_subplot(122)
    for ii in range(model.no):
        for jj in range(ii+1, model.no):
            ax.plot(R, nac_list[:, ii, jj], label=f"Orbital {ii} -> {jj}")
    # ax.set_xlim(-10, 20)
    ax.legend() 
    plt.show()

    H_CI_list = np.zeros((R.size, model.states.shape[0], model.states.shape[0]))
    E_CI_list = np.zeros((R.size, model.states.shape[0]))
    E0_list = np.zeros((R.size, model.states.shape[0]))
    evecs_CI_list = np.zeros((R.size, model.states.shape[0], model.states.shape[0]))
    grad_H_CI_list = np.zeros((R.size, model.states.shape[0], model.states.shape[0]))
    nac_CI_list = np.zeros((R.size, model.states.shape[0], model.states.shape[0]))
    for ii, r in enumerate(R):
        E0_list[ii] = model.get_U0(r)
        H_CI_list[ii] = model.get_H_CI(H_list[ii])
        # E_CI_list[ii], evecs_CI_list[ii] = np.linalg.eigh(H_CI_list[ii])
        E_CI_list[ii], evecs_CI_list[ii] = LA.eigh(H_CI_list[ii])
        if ii > 0:
            evecs_CI_list[ii] = align_phase(evecs_CI_list[ii-1], evecs_CI_list[ii])
        # grad_H_CI = model.get_grad_H_CI(grad_H_list[ii])
        # grad_H_CI_list[ii] = grad_H_CI
        # nac_CI, _, _ = evaluate_nonadiabatic_couplings(grad_H_CI[..., None], E_CI_list[ii], evecs_CI_list[ii])
        # nac_CI_list[ii] = nac_CI[:, :, 0]
        grad_H_CI_list[ii] = model.get_grad_H_CI(grad_H_list[ii])
        nac_CI_list[ii] = evaluate_nonadiabatic_couplings_1d(grad_H_CI_list[ii], E_CI_list[ii], evecs_CI_list[ii])[0]
    
    E_CI_list += E0_list


    fig = plt.figure(figsize=(16, 6), dpi=300)
    ax = fig.add_subplot(121)
    for ii in range(model.states.shape[0]):
    # for ii in [2, 3]:
        ax.plot(R, E_CI_list[:, ii], label=f"State {model.states[ii]}")
    ax.set_xlabel("R")
    ax.set_ylabel("Energy")
    ax.set_xlim(-10, 20)
    ax.set_ylim(-0.02, 0.05)
    ax.legend()
    ax = fig.add_subplot(122)
    for ii in range(model.states.shape[0]):
        for jj in range(ii+1, model.states.shape[0]):
    # ii = 0
    # for jj in range(ii+1, model.states.shape[0]):
            ax.plot(R, nac_CI_list[:, ii, jj], label=f"State {model.states[ii]} -> {model.states[jj]}")
    # ax.set_xlim(-5, 15)

    plt.show()


# %%
if __name__ == "__main__":
    main()
# %%
