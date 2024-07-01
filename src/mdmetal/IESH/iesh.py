# %%
import numpy as np
from numba import njit
from numpy.typing import NDArray
import scipy.linalg as LA

from mdmetal.hamiltonian import NewnsAndersonHarmonic, phase_correct_nac, get_phase_correction, evaluate_F_hellmann_feynman, approximate_nac_matrix, evaluate_nonadiabatic_couplings_1d, state_reordering, phase_correct_fhf
from mdmetal.IESH.state_utils import get_iesh_targets, map_state_to_index, get_hopping_prob
from mdmetal.IESH.properties import populations_iesh_landry

from typing import Callable, Tuple, Optional
from copy import deepcopy

def schrodinger_adiabatic_iesh(
    E: NDArray[np.float64],
    v_dot_d: NDArray[np.float64],
    psi: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    iH = -1.j * np.diagflat(E)  - v_dot_d
    return np.dot(iH, psi)
    # return -1.j * E[:, None] * psi - np.dot(v_dot_d, psi)
    # out = np.zeros_like(psi)
    # for ie in range(psi.shape[1]):
    #     out[:, ie] = -1.j * E * psi[:, ie] - np.dot(v_dot_d, psi[:, ie])
    # return out

def evaluate_hamiltonian_adiabatic(
    R: float,
    P: float,
    c: NDArray[np.complex128],
    h_obj: NewnsAndersonHarmonic,
    last_evecs: NDArray[np.float64],
    last_phase_corr: NDArray[np.float64],
) -> NDArray[np.float64]:
    H, grad_H = h_obj.evaluate(R, is_CI=False)
    # E0 = h_obj.get_U0(R)
    F0 = -h_obj.get_grad_U0(R)
    # E0 = F0 = 0.0
    
    evals, evecs = LA.eigh(H)
    
    d, F, _ = evaluate_nonadiabatic_couplings_1d(grad_H, evals, evecs)
    
    phase_corr = get_phase_correction(last_evecs, evecs) if last_evecs is not None else np.ones(evecs.shape[1])
    evecs = evecs / phase_corr
    # print(f"{np.iscomplexobj(phase_corr)=}")  
    evecs = evecs / phase_corr
    d = phase_correct_nac(d, phase_corr)
    c = c * (phase_corr / last_phase_corr)[:, None] if last_evecs is not None else c * phase_corr[:, None]
    
    # phase_corr = np.ones(evecs.shape[1])
    
    v_dot_d = P * d / h_obj.mass
    
    return c, evals, evecs, d, F, v_dot_d, phase_corr, F0

def P_dot(
    F: NDArray[np.float64],
    F0: NDArray[np.float64],
    active_state: NDArray[np.int64],
) -> NDArray[np.float64]:
    return sum(F[i] for i in active_state) + F0

def IESH_hop(
    c: NDArray[np.complex128],
    active_state: NDArray[np.int64],
    v_dot_d: NDArray[np.float64],
    dt: float,
    states: NDArray[np.int64],
    no: int,
) -> Tuple[bool, NDArray[np.int64], NDArray[np.int64]]:
    target_states, target_1e_terms = get_iesh_targets(active_state, states, no)
    try: 
        hopping_rates = np.array([get_hopping_prob(c, active_state, targ, term, v_dot_d, dt) for (targ, term) in zip(target_states, target_1e_terms)])
    except np.linalg.LinAlgError:
        print(f"{c=}")
        
        
    final_state_ind = hopping(hopping_rates)
    if final_state_ind == -1:
        return False, target_states, None 
    else:
        return True, target_states[final_state_ind], target_1e_terms[final_state_ind]

@njit   
def hopping(
    hopping_rates: NDArray[np.float64],
) -> NDArray[np.int64]:
    rand = np.random.rand() # random number between 0 and 1
    final_state = -1
    cum_prob: float = 0.0
    for i, rate in enumerate(hopping_rates):
        cum_prob += rate
        if rand < cum_prob:
            final_state = i
            break
    return final_state

def hopping_check(
    active_state: NDArray[np.int64],
    target_state: NDArray[np.int64],
    evals: NDArray[np.float64], 
    KE_before: float,
) -> Tuple[bool, float]:
    E_before = np.sum(evals[active_state])
    E_after = np.sum(evals[target_state])
    KE_after = (E_before + KE_before) - E_after # conservation of energy
    return (KE_after > 0), E_after - E_before

def momentum_rescale(
    P: float,
    mass: float,
    dij: float,    
    dE: float,
) -> NDArray[np.float64]:
    a = 0.5 * dij**2 / mass
    b = P * dij / mass
    c = dE
    
    roots = np.roots([a, b, c])
    
    argmin = np.argmin(np.abs(roots))
    kappa = roots[argmin]
    
    return P + kappa * dij


def quantum_rk4(
    c: NDArray[np.complex128],
    active_state: NDArray[np.int64],
    evals: NDArray[np.float64],
    v_dot_d: NDArray[np.float64],
    dt: float,
    states: NDArray[np.int64],
    n_quantum_steps: int = 1,
) -> Tuple[NDArray[np.complex128], bool, NDArray[np.int64], NDArray[np.int64]]:
    dt_inner = dt / n_quantum_steps
    no = v_dot_d.shape[0]
    hopping_flag = False
    term_1e = None
    target_state = np.copy(active_state)
    for _ in range(n_quantum_steps):
        k1 = schrodinger_adiabatic_iesh(evals, v_dot_d, c)
        k2 = schrodinger_adiabatic_iesh(evals, v_dot_d, c + 0.5 * dt_inner * k1)
        k3 = schrodinger_adiabatic_iesh(evals, v_dot_d, c + 0.5 * dt_inner * k2)
        k4 = schrodinger_adiabatic_iesh(evals, v_dot_d, c + 1 * dt_inner * k3)
        c +=  dt_inner / 6 * (k1 + 2*k2 + 2*k3 + k4)
        if not hopping_flag:
            hopping_flag, target_state, term_1e = IESH_hop(c, active_state, v_dot_d, dt_inner, states, no)
    return c, hopping_flag, target_state, term_1e

def verlet_iesh(
    t: float,
    R: float,
    P: float,
    c: NDArray[np.complex128],
    active_state: NDArray[np.int64],
    dt: float,
    h_obj: NewnsAndersonHarmonic,
    last_evals: NDArray[np.float64],
    last_evecs: NDArray[np.float64],
    last_d: NDArray[np.float64],
    last_v_dot_d: NDArray[np.float64],
    last_F: NDArray[np.float64],
    last_phase_corr: NDArray[np.float64],
    last_F0: float,
):
    # unpack some parameters 
    mass = h_obj.mass
    
    # evaluate the langevin forces
    gamma = h_obj.gamma
    D = h_obj.kT * gamma * mass
    sigma = np.sqrt(2 * D / dt)
    dP_langevin = dt * (-gamma * P + sigma * np.random.normal(0, 1))
    # dP_langevin = 0
    
    # first half step for c and P
    P += 0.5 * dt * P_dot(last_F, last_F0, active_state) + 0.5 * dP_langevin
    
    c, hopping_flag, target_state, term_1e = quantum_rk4(c, active_state, last_evals, last_v_dot_d, 0.5*dt, h_obj.states)
    
    
    
    # first hopping check
    if hopping_flag:
        KE = 0.5 * P**2 / mass
        is_valid_hop, dE = hopping_check(active_state, target_state, last_evals, KE)
        if is_valid_hop:
            direction = last_d[term_1e[0], term_1e[1]]
            P = momentum_rescale(P, mass, direction, dE)
            active_state = target_state
        else:
            P = -P # as suggested by Tully, always reverse the momentum
    
    # update R 
    R += dt * P / mass
    
    # re-evaluate the hamiltonian
    c, evals, evecs, d, F, v_dot_d, phase_corr, F0 = evaluate_hamiltonian_adiabatic(R, P, c, h_obj, last_evecs, last_phase_corr)
    
    
    # second half step for c and P
    c, hopping_flag, target_state, term_1e = quantum_rk4(c, active_state, evals, v_dot_d, 0.5*dt, h_obj.states)
    
    # print(f"{np.sum(np.abs(c)**2, axis=0)=}")
    
    # second hopping check
    if hopping_flag:
        KE = 0.5 * P**2 / mass
        is_valid_hop, dE = hopping_check(active_state, target_state, evals, KE)
        if is_valid_hop:
            direction = d[term_1e[0], term_1e[1]]
            P = momentum_rescale(P, mass, direction, dE)
            active_state = target_state
        else:
            P = -P # as suggested by Tully, always reverse the momentum
    
    P += 0.5 * dt * P_dot(F, F0, active_state) + 0.5 * dP_langevin
    
    # re-evaluate the v_dot_d
    v_dot_d = P * d / mass
    
    t += dt
    
    # raise ValueError("Stop here")
    
    return t, R, P, c, active_state, evals, evecs, d, F, v_dot_d, phase_corr, F0

def dynamics_one(
    R0: float,
    P0: float,
    c0: NDArray[np.complex128],
    hamiltonian: NewnsAndersonHarmonic,
    tf: float,
    dt: float,
    n_quantum_steps: int = 1,   
    out_freq: int = 100,  
) -> Tuple[float, float, float, NDArray[np.complex128], NDArray[np.int64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    # unpack some parameters
    mass = hamiltonian.mass
    ne = hamiltonian.ne
    no = hamiltonian.no
    
    # copy the initial conditions
    hami = deepcopy(hamiltonian)
    
    nsteps = int(tf / dt)
    n_out = int(nsteps / out_freq)
    
    # prepare the storage
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
    
    # random sampling of the initial state
    # state = []
    active_state = np.zeros(ne, dtype=np.int64)
    active_state[:] = -1
    for ie in range(ne):
        cinit_i = c0[:, ie]
        prob_i = np.abs(cinit_i)**2
        # use the numpy random choice to sample the initial state
        orbit_i = -1
        while np.any(orbit_i == active_state): # avoid duplicate orbitals for different electrons
            orbit_i = np.random.choice(no, p=prob_i) 
        active_state[ie] = orbit_i
        
    
    c, evals, evecs, d, F, v_dot_d, phase_corr, F0 = evaluate_hamiltonian_adiabatic(R, P, c, hami, None, None)
    
    for istep in range(nsteps):
        if istep % out_freq == 0:
            iout = istep // out_freq
            t_out[iout] = t
            R_out[iout] = R
            P_out[iout] = P
            KE_out[iout] = 0.5 * P**2 / mass 
            PE_out[iout] = np.sum(evals[active_state]) + hami.get_U0(R)
            pop_out[iout] = populations_iesh_landry(c, evecs, active_state)
            
        t, R, P, c, active_state, evals, evecs, d, F, v_dot_d, phase_corr, F0 = verlet_iesh(t, R, P, c, active_state, dt, hami, evals, evecs, d, v_dot_d, F, phase_corr, F0)
        # print(f"{np.sum(np.abs(c)**2, axis=0)=}")
        # print(f"{R=}, {P=}, {np.sum(np.abs(c)**2, axis=0)=}")
        
    return t_out, R_out, P_out, pop_out, KE_out, PE_out 

def _test_main():
    # parameters
    hamiltonian =  NewnsAndersonHarmonic.initialize(2, 2, 0.00095, 16e-3, 1e-5, 2000.0)
    no = hamiltonian.no
    ne = hamiltonian.ne
    
    np.random.seed(2)
    
    sigma_R = np.sqrt(hamiltonian.kT / hamiltonian.mass) / hamiltonian.omega_B
    sigma_P = np.sqrt(hamiltonian.kT * hamiltonian.mass)
    
    R0, P0 = np.random.normal(0, sigma_R), np.random.normal(0, sigma_P)
    
     
    psi0 = np.zeros((no, ne), dtype=np.complex128)
    state = np.array([1, 2], dtype=np.int64) # initialized at state |01100>
    for ie in range(ne):
        psi0[state[ie], ie] = 1.0
    
    H, gradH = hamiltonian.evaluate(R0, is_CI=False)
    evals, evecs = LA.eigh(H)
    
    psi0 = np.dot(evecs.T.conj(), psi0)
    
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
    
             
        
    
        
    
        
            
    
    