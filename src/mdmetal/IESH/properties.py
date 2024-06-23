# %%
import numpy as np
from numpy.typing import NDArray
from numba import njit

from typing import Tuple

@njit
def populations_iesh_landry(
    c: NDArray[np.complex128], 
    U: NDArray[np.float64],
    active_state: NDArray[np.int64],
) -> NDArray[np.float64]:
    no, ne = c.shape
    populations = np.zeros(no, dtype=np.float64)
    rho_i = np.zeros((no, no), dtype=np.complex128)
    for ie in range(ne):
        active_orb_i = active_state[ie]
        rho_i[:] = np.outer(c[:, ie], c[:, ie].conj())
        for io in range(no):
            populations[io] += np.abs(U[io, active_orb_i])**2 
            for jj in range(no):
                for kk in range(jj+1, no):
                    populations[io] += 2.0 * np.real(U[io, jj] * np.conj(U[io, kk]) * rho_i[jj, kk])
    return populations