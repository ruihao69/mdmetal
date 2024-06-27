# %%
import numpy as np
from numpy.typing import NDArray
from numba import njit
import scipy.linalg as LA

from mdmetal.hamiltonian import NewnsAndersonHarmonic
from mdmetal.utils import schrodinger_adiabatic
from mdmetal.CI.properties import compute_KE,  reduce_state_populations, compute_surface_hopping_pop
from mdmetal.CI.ehrenfest_adiabatic import evalute_hamiltonian_adiabatic

from typing import Tuple 
from copy import deepcopy

def P_dot(
    F: NDArray[np.float64],
    active_state: int
) -> np.float64:
    return F[active_state, active_state]

@njit
def compute_hopping_prob(
    c: NDArray[np.complex128],
    active_state: int,
    v_dot_d: NDArray[np.float64],
    dt: float
) -> np.float64:
    nstates = c.shape[0]
    hopping_prob = np.zeros(nstates, dtype=np.float64)
    kk = active_state
    akk = np.abs(c[kk])**2
    for ll in range(nstates):
        akl_conj = np.conj(c[kk]) * c[ll] 
        prob = - 2.0 * np.real(akl_conj * v_dot_d[kk, ll]) / akk * dt
        hopping_prob[ll] = prob 
    return hopping_prob 
        
@njit
def hop(
    active_state: int,
    hopping_prob: NDArray[np.float64], 
) -> int:
    cum_prob = 0.0  
    for target_state, prob in enumerate(hopping_prob):
        if target_state == active_state:
            continue
        
        cum_prob += prob
        if cum_prob > np.random.rand():
            return target_state
        
    return active_state

def CI_hop(
    c: NDArray[np.complex128],
    active_state: int,
    v_dot_d: NDArray[np.float64],
    dt: float
) -> Tuple[bool, np.int64]:
    hopping_prob = compute_hopping_prob(c, active_state, v_dot_d, dt)
    target_state = hop(active_state, hopping_prob)
    
    flag_hop = target_state != active_state
    
    return flag_hop, target_state

def hopping_check(
    active_state: int,
    target_state: int,
    evals: NDArray[np.float64],
    KE_before: np.float64,
) -> Tuple[bool, np.float64]:
    E_before = evals[active_state]
    E_after = evals[target_state]
    KE_after = KE_before + E_before - E_after
    return (KE_after > 0), E_after - E_before

def momentum_rescale(
    P: float,
    mass: float,
    dij: float,
    dE: float,
) -> float:
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
) -> Tuple[NDArray[np.complex128], bool, int]:
    dt_inner = dt / n_quantum_steps
    no = v_dot_d.shape[0]
    hopping_flag = False
    target_state = np.copy(active_state)
    for _ in range(n_quantum_steps):
        k1 = schrodinger_adiabatic(evals, v_dot_d, c)
        k2 = schrodinger_adiabatic(evals, v_dot_d, c + 0.5 * dt_inner * k1)
        k3 = schrodinger_adiabatic(evals, v_dot_d, c + 0.5 * dt_inner * k2)
        k4 = schrodinger_adiabatic(evals, v_dot_d, c + 1 * dt_inner * k3)
        c +=  dt_inner / 6 * (k1 + 2*k2 + 2*k3 + k4)
        if not hopping_flag:
            hopping_flag, target_state = CI_hop(c, active_state, v_dot_d, dt_inner)
    return c, hopping_flag, target_state

def verlet_ci_fssh(
    t: float,
    R: float,
    P: float,
    c: NDArray[np.complex128],
    active_state: int,
    dt: float,
    h_obj: NewnsAndersonHarmonic,   
    last_order: NDArray[np.int64],
    last_evals: NDArray[np.float64],
    last_evecs: NDArray[np.float64],
    last_d: NDArray[np.float64],
    last_v_dot_d: NDArray[np.float64],
    last_F: NDArray[np.float64],
    last_F0: NDArray[np.float64],
    last_phase_corr: NDArray[np.float64],
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
    P += 0.5 * dt * P_dot(last_F, active_state) + 0.5 * dP_langevin + 0.5 * dt * last_F0
    
    c, hopping_flag, target_state = quantum_rk4(c, active_state, last_evals, last_v_dot_d, dt, last_order)
    
    if hopping_flag:
        KE = 0.5 * P**2 / mass
        is_valid_hop, dE = hopping_check(active_state, target_state, last_evals, KE)
        if is_valid_hop:
            direction = last_d[active_state, target_state]
            P = momentum_rescale(P, mass, direction, dE)
            active_state = target_state
        else:
            P = -P # reverse the momentum
            
    # update R
    R += dt * P / mass
    
    # re-evaluate the hamiltonian
    c, evals, evecs, d, F, F0, v_dot_d, phase_corr, order = evalute_hamiltonian_adiabatic(R, P, c, h_obj, last_evecs, last_phase_corr)
    
    # update c
    c, hopping_flag, target_state = quantum_rk4(c, active_state, evals, v_dot_d, 0.5 * dt, order)
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
    P += 0.5 * dt * P_dot(F, active_state) + 0.5 * dP_langevin + 0.5 * dt * F0
    
    # re-evaluate the v_dot_d
    v_dot_d = P * d / mass
    
    t += dt
    
    return t, R, P, c, active_state, evals, evecs, d, F, F0, v_dot_d, phase_corr, order

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
    
    # preallocate the arrays
    t_out = np.zeros(n_out)
    R_out = np.zeros(n_out)
    P_out = np.zeros(n_out)
    pop_out = np.zeros((n_out, h_obj.no), dtype=np.float64)
    KE_out = np.zeros(n_out)    
    PE_out = np.zeros(n_out)
    
    # prepare the initial conditions 
    t = 0
    R = R0
    P = P0
    c = c0
    
    prob = np.abs(c)**2 
    active_state = np.random.choice(nstates, p=prob)
    
    # evaluate the hamiltonian
    c, evals, evecs, d, F, F0, v_dot_d, phase_corr, order = evalute_hamiltonian_adiabatic(R, P, c, h_obj, None, None)
    
    for istep in range(nsteps):
        if istep % out_freq == 0:
            iout = int(istep / out_freq)
            t_out[iout] = t
            R_out[iout] = R
            P_out[iout] = P
            diab_pop = compute_surface_hopping_pop(active_state, c, evecs)
            pop_out[iout] = reduce_state_populations(diab_pop, h_obj.no, h_obj.states)
            KE_out[iout] = compute_KE(P, mass)
            PE_out[iout] = evals[active_state] + hami.get_U0(R)
        
        t, R, P, c, active_state, evals, evecs, d, F, F0, v_dot_d, phase_corr, order = verlet_ci_fssh(t, R, P, c, active_state, dt, hami, order, evals, evecs, d, v_dot_d, F, F0, phase_corr)
    
    return t_out, R_out, P_out, KE_out, PE_out, pop_out


def _test_main():
    hamiltonian =  NewnsAndersonHarmonic.initialize(2, 2, 0.00095, 16e-3, 1e-5, 2000.0)
    
    np.random.seed(0)
    
    sigma_R = np.sqrt(hamiltonian.kT / hamiltonian.mass) / hamiltonian.omega_B
    sigma_P = np.sqrt(hamiltonian.kT * hamiltonian.mass)
    
    R0, P0 = np.random.normal(0, sigma_R), np.random.normal(0, sigma_P)
    
    nstates = hamiltonian.states.shape[0]
    psi0 = np.zeros(nstates, dtype=np.complex128)
    state = np.array([1, 2], dtype=np.int64) # initialized at state |01100>
    istate = np.where(np.all(hamiltonian.states == state, axis=1))[0][0]
    psi0[istate] = 1.0
    
    H, gradH = hamiltonian.evaluate(R0, is_CI=True)
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
    
    


    
        
    
    
    
    
