# %%
import numpy as np
from numpy import ndarray as NDArray

# ------ Equations of motion ------
def schrodinger_adiabatic(
    E: NDArray[np.float64],
    v_dot_d: NDArray[np.float64],
    psi: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    return -1.j * E * psi - np.dot(v_dot_d, psi)

def schrodinger_diabatic(
    H: NDArray[np.float64],
    psi: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    return -1.j * np.dot(H, psi)
