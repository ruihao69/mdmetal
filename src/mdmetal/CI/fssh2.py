# %%
import numpy as np
from numpy.typing import NDArray
from numba import njit
import scipy.linalg as LA

from mdmetal.hamiltonian import NewnsAndersonHarmonic
from mdmetal.hamiltonian.hamiltonian_adiab_orb import NewnsAndersonHarmonic2
from mdmetal.utils import schrodinger_adiabatic
from mdmetal.CI.properties import compute_KE,  reduce_state_populations, compute_surface_hopping_pop2
from mdmetal.CI.ehrenfest_adiabatic import get_phase_correction
from mdmetal.CI.fssh import P_dot, hopping_check, quantum_rk4, momentum_rescale 

from typing import Tuple 
from copy import deepcopy

def P_dot(
    F: NDArray[np.float64],
    active_state: int
) -> np.float64:
    return F[active_state]

def evalute_hamiltonian_adiabatic(
    R: float,
    P: float,
    h_obj: NewnsAndersonHarmonic2,
    last_evecs: NDArray[np.float64],
    last_phase_corr: NDArray[np.float64],
) -> NDArray[np.float64]:
    # print(f"{R=}, {P=}") 
    H, grad_H = h_obj.evaluate_one_electron(R)
    F0 = - h_obj.get_grad_U0(R)
    
    # correct the phase of the eigenvectors 
    evals, evecs = LA.eigh(H)
    phase_corr = get_phase_correction(last_evecs, evecs) if last_evecs is not None else np.ones(evecs.shape[1])
    evecs = evecs / phase_corr
    
    # evaluate the non-adiabatic couplings
    E, D, F, F_hellmann_feynman = h_obj.evaluate_many_body(evals, evecs, grad_H)
    
    # v_dot_d
    v_dot_d = D * P / h_obj.mass
    
    return E, evecs, D, F, F0, v_dot_d, phase_corr

def verlet_adiabatic(
    t: float,
    R: float,
    P: float,
    c: NDArray[np.complex128],
    active_state: int,
    dt: float,
    h_obj: NewnsAndersonHarmonic,
    last_evals: NDArray[np.float64],
    last_v_dot_d: NDArray[np.float64],
    last_F: NDArray[np.float64],
    last_F0: NDArray[np.float64], # state-independent force
    last_evecs: NDArray[np.float64],
    last_phase_corr: NDArray[np.float64],
    last_d: NDArray[np.float64]
) -> Tuple[float, float, float, NDArray[np.complex128], NDArray[np.float64], NDArray[np.float64]]:
    # unpack some parameters 
    mass = h_obj.mass
    
    # evaluate the langevin forces
    gamma = h_obj.gamma
    D = h_obj.kT * gamma * mass
    sigma = np.sqrt(2 * D / dt)
    dP_langevin = dt * (-gamma * P + sigma * np.random.normal(0, 1))
    dP_langevin = 0
    
    # first half step for c and P
    P += 0.5 * dt * P_dot(last_F, active_state) + 0.5 * dP_langevin + 0.5 * dt * last_F0
   
    # update c 
    c, hopping_flag, target_state = quantum_rk4(c, active_state, last_evals, last_v_dot_d, 0.5*dt, h_obj.states)
    
    if hopping_flag:
        KE = 0.5 * P**2 / mass
        is_valid_hop, dE = hopping_check(active_state, target_state, last_evals, KE)
        if is_valid_hop:
            direction = last_d[active_state, target_state]
            P = momentum_rescale(P, mass, direction, dE)
            active_state = target_state
        else:
            P = -P
            
    # update R
    R += dt * P / mass
    
    # re-evaluate the hamiltonian
    evals, evecs, d, F, F0, v_dot_d, phase_corr = evalute_hamiltonian_adiabatic(R, P, h_obj, last_evecs, last_phase_corr)
    
    # update c
    c, hopping_flag, target_state = quantum_rk4(c, active_state, evals, v_dot_d, 0.5 * dt, h_obj.states)
    
    if hopping_flag:
        KE = 0.5 * P**2 / mass
        is_valid_hop, dE = hopping_check(active_state, target_state, evals, KE)
        if is_valid_hop:
            direction = d[active_state, target_state]
            P = momentum_rescale(P, mass, direction, dE)
            active_state = target_state
        else:
            P = -P
    
    # second half step for P
    P += 0.5 * dt * P_dot(F, active_state) + 0.5 * dP_langevin  + 0.5 * dt * F0
    
    # re-evaluate the v_dot_d
    v_dot_d = P * d / mass
    
    t += dt
    
    return t, R, P, c, active_state, evals, evecs, d, F, F0, v_dot_d, phase_corr 
    

def dynamics_one(
    R0: float,
    P0: float,
    c0: NDArray[np.complex128],
    h_obj: NewnsAndersonHarmonic,
    tf: float,
    dt: float,
    n_quantum_steps: int = 1,
    out_freq: int = 100,  
):
    # unpack some parameters
    mass = h_obj.mass
    nstates = h_obj.states.shape[0]
    
    nsteps = int(tf / dt)
    n_out = int(nsteps / out_freq)
    
    hami = deepcopy(h_obj)
    
    # pre-allocate the output arrays
    t_out = np.zeros(n_out)
    R_out = np.zeros(n_out)
    P_out = np.zeros(n_out)
    pop_out = np.zeros((n_out, hami.no), dtype=np.float64)
    KE_out = np.zeros(n_out)
    PE_out = np.zeros(n_out)
    
    
    # prepare the initial conditions
    t = 0
    R = R0
    P = P0
    c = c0
    
    prob = np.abs(c)**2
    active_state = np.random.choice(nstates, p=prob)
    
    # evaluate the hamiltonian once to start the dynamics
    evals, evecs, d, F, F0, v_dot_d, phase_corr = evalute_hamiltonian_adiabatic(R, P, h_obj, None, None)
    
    # main loop
    for istep in range(nsteps):
        if istep % out_freq == 0:
            iout = int(istep / out_freq)
            t_out[iout] = t
            R_out[iout] = R
            P_out[iout] = P
            pop_out[iout, :] = compute_surface_hopping_pop2(active_state, evecs, hami.states)
            KE_out[iout] = compute_KE(P, mass)
            PE_out[iout] = h_obj.get_U0(R) + evals[active_state]
        
        # update the dynamics
        t, R, P, c, active_state, evals, evecs, d, F, F0, v_dot_d, phase_corr = verlet_adiabatic(
            t, R, P, c, active_state, dt, h_obj, evals, v_dot_d, F, F0, evecs, phase_corr, d
        )
        # print(f"{t=}, {R=}, {P=}, {active_state=}")
        # print(f"{t=}, {np.sum(np.abs(c)**2)=}, {active_state=}")
        
    return t_out, R_out, P_out, KE_out, PE_out, pop_out
    
def _test_main():
    hamiltonian =  NewnsAndersonHarmonic.initialize(2, 2, 0.00095, 16e-3, 1e-5, 2000.0)
    hamiltonian = NewnsAndersonHarmonic2.from_NewnsAndersonHarmonic(hamiltonian)    
    
    np.random.seed(0)
    
    sigma_R = np.sqrt(hamiltonian.kT / hamiltonian.mass) / hamiltonian.omega_B
    sigma_P = np.sqrt(hamiltonian.kT * hamiltonian.mass)
    
    R0, P0 = np.random.normal(0, sigma_R), np.random.normal(0, sigma_P)
    
    nstates = hamiltonian.states.shape[0]
    psi0 = np.zeros(nstates, dtype=np.complex128)
    state = np.array([1, 2], dtype=np.int64) # initialized at state |01100>
    # diabatic state
    istate = np.where(np.all(hamiltonian.states == state, axis=1))[0][0]
    psi0[istate] = 1.0
    
    H, gradH = hamiltonian.evaluate_one_electron(R0)
    evals, evecs = LA.eigh(H)
    
    # use independent particle approximation for the initial amplitude
    # overlap matrix
    psi_dummy_list = []
    for ie in range(hamiltonian.ne):
        psi_dummy = np.zeros(hamiltonian.no, dtype=np.complex128)
        psi_dummy[state[ie]] = 1.0
        psi_dummy = np.dot(evecs, psi_dummy)
        psi_dummy_list.append(psi_dummy)
    psi_dummy_list = np.array(psi_dummy_list).T
        
    for istate, state in enumerate(hamiltonian.states):
        S = np.zeros((hamiltonian.ne, hamiltonian.ne), dtype=np.complex128)
        for ie in range(hamiltonian.ne):
            for je in range(hamiltonian.ne):
                S[ie, je] += psi_dummy_list[state[ie], je]
        psi0[istate] = np.linalg.det(S)

    # psi0 = np.dot(evecs.T.conj(), psi0)
    
    t, R, P, KE, PE, pop = dynamics_one(R0, P0, psi0, hamiltonian, tf=100000, dt=1, out_freq=100)
    
    import matplotlib.pyplot as plt 
    fig = plt.figure(figsize=(10, 5), dpi=300)
    ax = fig.add_subplot(121)
    ax.plot(t*hamiltonian.omega_B, R)
    ax.set_xlabel('Time (1/w)')
    ax.set_ylabel('R (a.u.)')
    ax = fig.add_subplot(122)
    ax.plot(t*hamiltonian.omega_B, P)
    ax.set_xlabel('Time (1/w)')
    ax.set_ylabel('P (a.u.)')
    plt.show()
    
    fig = plt.figure(figsize=(6, 5), dpi=300)
    ax = fig.add_subplot(111)
    for i in range(hamiltonian.no):
        ax.plot(t*hamiltonian.omega_B, pop[:, i], label=f"orbital {i}")
    ax.set_xlabel('Time (1/w)')
    ax.set_ylabel('Population')
    ax.legend()
    plt.show()
    
    fig = plt.figure(figsize=(6, 5), dpi=300)
    ax = fig.add_subplot(111)
    ax.plot(t*hamiltonian.omega_B, np.sum(pop, axis=1), label="state 0")
    ax.set_ylim([1.9, 2.1])
    
    
    fig = plt.figure(figsize=(15, 5), dpi=300)
    ax = fig.add_subplot(131)
    ax.plot(t*hamiltonian.omega_B, KE)
    ax.set_xlabel('Time (1/w)')
    ax.set_ylabel('KE (a.u.)')
    
    ax = fig.add_subplot(132)
    ax.plot(t*hamiltonian.omega_B, PE)
    ax.set_xlabel('Time (1/w)')
    
    ax = fig.add_subplot(133)
    ax.plot(t*hamiltonian.omega_B, KE + PE)
    ax.set_xlabel('Time (1/w)')
    ax.set_ylabel('TE (a.u.)')
    plt.show()
    
# %%
if __name__ == "__main__":
    _test_main()
# %%
    