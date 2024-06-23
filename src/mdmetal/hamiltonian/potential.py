# %%
from numba import njit

@njit
def U0(
    R: float,
    mass: float,
    Omega_B: float,
) -> float:
    return 0.5 * mass * Omega_B**2 * R**2

@njit
def grad_U0(
    R: float,
    mass: float,
    Omega_B: float,
) -> float:
    return mass * Omega_B**2 * R

@njit
def U1(
    R: float,
    g: float,
    mass: float,
    Omega_B: float,
    dG: float,
) -> float:
    return 0.5 * mass * Omega_B**2 * (R - g)**2 + dG

@njit
def grad_U1(
    R: float,
    g: float,
    mass: float,
    Omega_B: float,
) -> float:
    return mass * Omega_B**2 * (R - g)
# %%
