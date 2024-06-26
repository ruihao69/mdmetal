# %%
import numpy as np
from numpy.typing import NDArray    
import scipy.linalg as LA

from mdmetal.hamiltonian import NewnsAndersonHarmonic, phase_correct_nac, get_phase_correction, evaluate_F_hellmann_feynman, approximate_nac_matrix, evaluate_nonadiabatic_couplings_1d, state_reordering, phase_correct_fhf
from mdmetal.utils import schrodinger_adiabatic
from mdmetal.CI.properties import compute_KE, compute_orbital_populations 

from typing import Callable, Tuple, Optional
from copy import deepcopy

def _state_reordering(evecs_last, evecs_curr) -> Tuple[NDArray[np.int64]]:
    perm = np.arange(evecs_last.shape[1])
    for icol in range(evecs_last.shape[1]):
        maxidx = np.argmax(np.abs(np.dot(evecs_last[:, icol], evecs_curr)))
        perm[icol] = maxidx
        # if maxidx != icol:
        #     print(f"{np.dot(evecs_last[:, icol], evecs_curr[:, maxidx])=}, {np.dot(evecs_last[:, icol], evecs_curr[:, icol])=}")
    return perm

def _phase_correction(evecs_last, evecs_curr, order):
    # print(f"{order=}")
    phase_corr = np.ones(evecs_last.shape[1])
    for icol in range(evecs_last.shape[1]):
        val = np.dot(evecs_last[:, icol], evecs_curr[:, order[icol]].conj())   
        aval = np.abs(val)
        if aval > 0.001:
            phase_corr[order[icol]] = val / aval
        else:
            print(f"Warning: phase correction is zero for column {icol}")
    return phase_corr
        

def evalute_hamiltonian_adiabatic(
    R: float,
    P: float,
    c: NDArray[np.complex128],
    h_obj: NewnsAndersonHarmonic,
    last_evecs: NDArray[np.float64],
    last_phase_corr: NDArray[np.float64],
) -> NDArray[np.float64]:
    # print(f"{R=}, {P=}") 
    H, grad_H = h_obj.evaluate(R, is_CI=True)
    
    evals, evecs = LA.eigh(H)
    
    d, _, F_hellmann_feynman = evaluate_nonadiabatic_couplings_1d(grad_H, evals, evecs)
    
    # state reordering  
    # order = state_reordering(last_evecs, evecs) if last_evecs is not None else np.arange(len(evals))
    order = _state_reordering(last_evecs, evecs) if last_evecs is not None else np.arange(len(evals))
    # print(f"{order=}")

    # if not np.allclose(order, np.arange(len(evals))):
    #     if not np.allclose(evals[order], evals, atol=1e-4*(evals.max() - evals.min()), rtol=1e-4):
    #         c = c[order]    
    #         evecs = evecs[:, order]
    #         evals = evals[order]
    #         d = d[:, order][order, :]
    #         F_hellmann_feynman = F_hellmann_feynman[:, order][order, :]
        
    # phase correction
    # phase_corr = get_phase_correction(last_evecs, evecs) if last_evecs is not None else np.ones(len(evals))
    phase_corr = _phase_correction(last_evecs, evecs, order) if last_evecs is not None else np.ones(len(evals))
    evecs = evecs / phase_corr
    d = phase_correct_nac(d, phase_corr)
    F_hellmann_feynman = phase_correct_fhf(F_hellmann_feynman, phase_corr)
    c = c * phase_corr / last_phase_corr if last_phase_corr is not None else c * phase_corr
    
    # phase_corr = np.arange(len(evals))
    
    v_dot_d = P * d / h_obj.mass 
    
    return c, evals, evecs, d, F_hellmann_feynman, v_dot_d, phase_corr, order

def P_dot(
    F_hellmann_feynman: NDArray[np.float64],
    c: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    return np.dot(c.conj(), np.dot(F_hellmann_feynman, c)).real

def quantum_rk4(
    c: NDArray[np.complex128],
    evals: NDArray[np.float64],
    v_dot_d: NDArray[np.float64],
    dt: float,
    n_quantum_steps: int = 1,   
) -> NDArray[np.complex128]:
    dt_inner = dt / n_quantum_steps
    for _ in range(n_quantum_steps):
        k1 = schrodinger_adiabatic(evals, v_dot_d, c)
        k2 = schrodinger_adiabatic(evals, v_dot_d, c + 0.5 * dt_inner * k1)
        k3 = schrodinger_adiabatic(evals, v_dot_d, c + 0.5 * dt_inner * k2)
        k4 = schrodinger_adiabatic(evals, v_dot_d, c + dt_inner * k3)
        c += dt_inner / 6 * (k1 + 2 * k2 + 2 * k3 + k4) 
    return c

def verlet_adiabatic(
    t: float,
    R: float,
    P: float,
    c: NDArray[np.complex128],
    dt: float,
    h_obj: NewnsAndersonHarmonic,
    last_order: NDArray[np.int64],
    last_evals: NDArray[np.float64],
    last_v_dot_d: NDArray[np.float64],
    last_F_hellmann_feynman: NDArray[np.float64],
    last_evecs: NDArray[np.float64],
    last_phase_corr: NDArray[np.float64],
) -> Tuple[float, float, float, NDArray[np.complex128], NDArray[np.float64], NDArray[np.float64]]:
    # unpack some parameters 
    mass = h_obj.mass
    
    # evaluate the langevin forces
    gamma = h_obj.gamma
    D = h_obj.kT * gamma * mass
    sigma = np.sqrt(2 * D / dt)
    dP_langevin = dt * (-gamma * P + sigma * np.random.normal(0, 1))
    # dP_langevin = 0
    
    # first half step for P
    P += 0.5 * dt * P_dot(last_F_hellmann_feynman, c) + 0.5 * dP_langevin
    c = quantum_rk4(c, last_evals, last_v_dot_d, 0.5 * dt)
    
    # update R 
    R += dt * P / mass
    
    # re-evaluate the hamiltonian
    c, evals, evecs, d, F_hellmann_feynman, v_dot_d, phase_corr, order = evalute_hamiltonian_adiabatic(R, P, c, h_obj, last_evecs, last_phase_corr)
    
    # update c
    c = quantum_rk4(c, evals, v_dot_d, 0.5 * dt)
    
    # second half step for P
    P += 0.5 * dt * P_dot(F_hellmann_feynman, c) + 0.5 * dP_langevin    
    
    # re-evaluate the v_dot_d
    v_dot_d = P * d / mass
    
    t += dt
    return t, R, P, c, evals, evecs, v_dot_d, F_hellmann_feynman, phase_corr, order

def dynamics_one(
    R0: float,
    P0: float,
    c0: NDArray[np.complex128],
    h_obj: NewnsAndersonHarmonic,
    tf: float,  
    dt: float,
    n_quantum_steps: int = 1,
    out_freq: int = 100,
) -> Tuple[NDArray[np.float64]]:
    # copy the initial conditions
    hami = deepcopy(h_obj)
    
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
    
    c, evals, evecs, d, F_hellmann_feynman, v_dot_d, phase_corr, order = evalute_hamiltonian_adiabatic(R, P, c, hami, None, None)
    # print(f"{P_dot(F_hellmann_feynman, c)=}")
    
    for istep in range(nsteps):
        if istep % out_freq == 0:
            iout = istep // out_freq
            t_out[iout] = t
            R_out[iout] = R
            P_out[iout] = P
            KE_out[iout] = compute_KE(P, hami.mass)
            PE_out[iout] = np.sum(evals * np.abs(c)**2) 
            pop_out[iout, :] = compute_orbital_populations(c, evecs, hami.no, hami.states)
            
        t, R, P, c, evals, evecs, v_dot_d, F_hellmann_feynman, phase_corr, order = verlet_adiabatic(t, R, P, c, dt, hami, order, evals, v_dot_d, F_hellmann_feynman, evecs, phase_corr) 
        
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
    
    


    
        
