# %%
import numpy as np
from numpy.typing import NDArray
from numba import njit
from scipy.special import roots_legendre
import scipy.linalg as LA

from mdmetal.hamiltonian.potential import U0, U1, grad_U0, grad_U1
from mdmetal.hamiltonian.hamiltonian import one_electron_newns_anderson, one_electron_newns_anderson_grad, numba_setdiff1d, align_phase
from mdmetal.hamiltonian.hamiltonian import NewnsAndersonHarmonic

import math
from dataclasses import dataclass
from itertools import combinations
import warnings
from typing import Tuple

@njit
def get_quad_hamiltonian_CI_diag(
    E: NDArray[np.float64],
    states: NDArray[np.int64],
) -> NDArray[np.float64]:
    ns, ne = states.shape

    E_CI = np.zeros(ns, dtype=np.float64)
    # diagonal elements
    for ii in range(ns):
        for jj in range(ne):
            orbit_ij = states[ii, jj]
            E_CI[ii] += E[orbit_ij]
    return E_CI

@njit 
def get_quad_NAC_CI(    
    tilde_grad_H: NDArray[np.float64],
    E_CI: NDArray[np.float64],
    states: NDArray[np.int64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    ns, ne = states.shape
    
    D = np.zeros((ns, ns), dtype=np.float64)
    F = np.zeros(ns, dtype=np.float64) 
    F_hellman_feynman = np.zeros((ns, ns), dtype=np.float64)
    
    # there are no on-diagonal elements in D
    
    # off-diagonal elements
    for ii, istate in enumerate(states):
        for jj, jstate in enumerate(states):
            if ii == jj:
                for kk, orbit_k in enumerate(istate):
                    F[ii] -= tilde_grad_H[orbit_k, orbit_k]
                    F_hellman_feynman[ii, ii] -= tilde_grad_H[orbit_k, orbit_k]
                continue 
            idiff = numba_setdiff1d(istate, jstate)
            # only one element should be different
            # since we are dealing with qudratic Hamiltonian
            if idiff.size != 1: 
                continue
            jdiff = numba_setdiff1d(jstate, istate)
            i, j = idiff[0], jdiff[0]
            F_hellman_feynman[ii, jj] -= tilde_grad_H[i, j]
            D[ii, jj] += tilde_grad_H[i, j] / (E_CI[jj] - E_CI[ii])
    return D, F, F_hellman_feynman
            

def get_CI_matricies_in_adiab_orb(
    evals: NDArray[np.float64],
    evecs: NDArray[np.float64],
    grad_H_diab: NDArray[np.float64],
    states: NDArray[np.int64]
) -> Tuple[NDArray[np.float64]]:
    # diagonalize the single-particle Hamiltonian  
    # (this should be done before calling this function)
    # the diagonalization basically transforms the Hamiltonian
    # h = \sum_i \sun_j h_{ij} b_i^\dagger b_j (diabatic basis)
    # to the adiabatic basis
    # \tilde{h} = \sum_k \epsilon_k a_k^\dagger a_k (adiabatic basis)

    
    # use single-particle eigenstates to construct the CI Hamiltonian
    # H_{CI} = \sum_{I} \sum{J} |I> <I| \tilde{h} |J> <J|
    # since the single-particle Hamiltonian is diagonal in the adiabatic basis
    # the CI Hamiltonian is also diagonal
    E_CI = get_quad_hamiltonian_CI_diag(evals, states)
    
    # construct the non-adiabatic coupling matrix
    # \grad H = \sum_{i} \sum{j} \grad h_ij b_i^\dagger b_j
    # \tilde{\grad H} = U^{\dagger} \grad H U = 
    # \sum_{k} \sum_{l} \tilde{\grad h}_{kl} a_k^\dagger a_l, not necessarily diagonal
    # Then, we can construct the non-adiabatic coupling matrix 
    # in the CI basis as follows (the Hellman-Feynman theorem):
    # D_{IJ} = <I| \tilde{\grad H} |J> / (E_I - E_J)
    # meanwhile, the diagonal elements of D are zero.
    # In the implementation, we will also calculate diagonal forces
    # F_I = <I| \tilde{\grad H} |I> and the general Hellman-Feynman forces
    # F_{IJ} = <I| \tilde{\grad H} |J>
    tilde_grad_H = np.dot(np.conj(evecs.T), np.dot(grad_H_diab, evecs))
    D, F, F_hellman_feynman = get_quad_NAC_CI(tilde_grad_H, E_CI, states)
    
    return E_CI, D, F, F_hellman_feynman
   

@dataclass
class NewnsAndersonHarmonic2: # ad-hoc class to test the adiabatic implementation
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
    
    def evaluate_one_electron(
        self,
        R: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64]]:
        H = self.get_H_one_electron(R)
        grad_H = self.get_grad_H_one_electron(R)
        return H, grad_H
    
    def evaluate_many_body(
        self,
        evals: NDArray[np.float64],
        evecs: NDArray[np.float64],
        grad_H: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        return get_CI_matricies_in_adiab_orb(evals, evecs, grad_H, self.states)
        
    
    @classmethod
    def from_NewnsAndersonHarmonic(
        cls,
        newns_anderson_harmonic: NewnsAndersonHarmonic,    
    ) -> "NewnsAndersonHarmonic2":
        return cls(
            nk=newns_anderson_harmonic.nk,
            ek=newns_anderson_harmonic.ek,
            vk=newns_anderson_harmonic.vk,
            no=newns_anderson_harmonic.no,
            ne=newns_anderson_harmonic.ne,
            states=newns_anderson_harmonic.states,
            mass=newns_anderson_harmonic.mass,
            omega_B=newns_anderson_harmonic.omega_B,
            gamma=newns_anderson_harmonic.gamma,
            Gamma=newns_anderson_harmonic.Gamma,
            Er=newns_anderson_harmonic.Er,
            dG=newns_anderson_harmonic.dG,
            kT=newns_anderson_harmonic.kT,
            W=newns_anderson_harmonic.W,
            g=newns_anderson_harmonic.g,
            V=newns_anderson_harmonic.V
        )
        

def _test_main():
    import matplotlib.pyplot as plt
    from mdmetal.hamiltonian.utils import evaluate_nonadiabatic_couplings_1d

    nk = 2
    ne = 2
    model1 = NewnsAndersonHarmonic.initialize(nk, ne, 0.00095, 16e-3, 1e-5, 2000.0)
    model = NewnsAndersonHarmonic2.from_NewnsAndersonHarmonic(model1)
    
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
    
    E_CI_list = np.zeros((R.size, model.states.shape[0]))
    E0_list = np.zeros((R.size, model.states.shape[0]))
    nac_CI_list = np.zeros((R.size, model.states.shape[0], model.states.shape[0]))
    for ii, r in enumerate(R):
        E0_list[ii] = model.get_U0(r)   
        # E_CI_list[ii], nac_CI_list[ii], _, _ = model.evaluate(r)
        H_list[ii], grad_H_list[ii] = model.evaluate_one_electron(r)
        E_list[ii], evecs_list[ii] = LA.eigh(H_list[ii])
        if ii > 0:
            evecs_list[ii] = align_phase(evecs_list[ii-1], evecs_list[ii])
        E_CI_list[ii], nac_CI_list[ii], _, _ = model.evaluate_many_body(E_list[ii], evecs_list[ii], grad_H_list[ii])
        
        
    
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
    _test_main()
    
# %%
