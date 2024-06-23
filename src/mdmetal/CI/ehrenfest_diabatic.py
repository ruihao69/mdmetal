# %%
import numpy as np
from numpy.typing import NDArray    
import scipy.linalg as LA

from mdmetal.hamiltonian import NewnsAndersonHarmonic
from mdmetal.utils import schrodinger_diabatic
from mdmetal.CI.properties import compute_KE, compute_PE_ehrenfest, reduce_state_populations

from typing import Callable, Tuple, Optional
from copy import deepcopy

def evaluate_hamiltonian_diabatic(
    R: float,
    P: float,
    h_obj: NewnsAndersonHarmonic,
) -> NDArray[np.float64]:
    return h_obj.evaluate(R, is_CI=True)
    

def P_dot(
    grad_H_CI: NDArray[np.float64],
    c: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    return -np.dot(c.conj(), np.dot(grad_H_CI, c)).real 

def quantum_rk4(
    c: NDArray[np.complex128],   
    H: NDArray[np.float64],
    dt: float,
    n_quantum_steps: int = 1,
) -> NDArray[np.complex128]:
    dt_inner = dt / n_quantum_steps
    for _ in range(n_quantum_steps):
        k1 = schrodinger_diabatic(H, c)
        k2 = schrodinger_diabatic(H, c + 0.5 * dt_inner * k1)
        k3 = schrodinger_diabatic(H, c + 0.5 * dt_inner * k2)
        k4 = schrodinger_diabatic(H, c + dt_inner * k3)
        c += dt_inner / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return c

def verlet_diabatic(
    t: float,
    R: float,
    P: float,
    c: NDArray[np.complex128],
    dt: float,
    h_obj: NewnsAndersonHarmonic,
    last_H_CI: NDArray[np.float64],
    last_grad_H_CI: NDArray[np.float64],
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
    P += 0.5 * dt * P_dot(last_grad_H_CI, c) + 0.5 * dP_langevin
    c = quantum_rk4(c, last_H_CI, dt * 0.5)
    
    # update R
    R += dt * P / mass
    
    # re-evaluate the hamiltonian 
    H_CI, grad_H_CI = evaluate_hamiltonian_diabatic(R, P, h_obj)
    
    # update c
    c = quantum_rk4(c, H_CI, dt * 0.5)
    
    # second half step for P 
    P += 0.5 * dt * P_dot(grad_H_CI, c) + 0.5 * dP_langevin
    
    t += dt
    return t, R, P, c, H_CI, grad_H_CI


def dynamics_one(
    R0: NDArray[np.float64],
    P0: NDArray[np.float64],
    psi0: NDArray[np.complex128],
    hamiltonian: NewnsAndersonHarmonic,
    tf: float,
    dt: float,
    n_quantum_steps: int = 1,
    out_freq: int = 100,
) -> Tuple[NDArray[np.float64]]:
    # copy the hamiltonian object
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
    c = psi0
    
    H, gradH = evaluate_hamiltonian_diabatic(R, P, hami)
    
    # print(f"{P_dot(gradH, psi0)=}")
    
    # loop over the steps
    for istep in range(nsteps):
        if istep % out_freq == 0:
            iout = istep // out_freq
            t_out[iout] = t
            R_out[iout] = R
            P_out[iout] = P
            pop_out[iout, :] = reduce_state_populations(c, hami.no, hami.states)
            KE_out[iout] = compute_KE(P, hami.mass)
            PE_out[iout] = np.dot(c.conj(), np.dot(H, c)).real  
            
        t, R, P, c, H, gradH = verlet_diabatic(t, R, P, c, dt, hami, H, gradH)
        
    return t_out, R_out, P_out, pop_out, KE_out, PE_out
   

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
    
    t, R, P, pop, KE, PE = dynamics_one(R0, P0, psi0, hamiltonian, tf=100000, dt=1, out_freq=100)
    
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
