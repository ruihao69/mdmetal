# %%
import numpy as np
from numpy.typing import NDArray

from enum import Enum
from dataclasses import dataclass, field
import warnings

class Method(Enum):
    # CI hamiltonian (from diabatic orbitals), Ehrenfest dynamics in Diabatic CI basis
    CI_EH_D = "CI-Eh-D" 
    # CI hamiltonian (from diabatic orbitals), Ehrenfest dynamics in Adiabatic CI basis
    CI_EH_A = "CI-Eh-A" 
    # CI hamiltonian (from diabatic orbitals), Tully's surface hopping in Adiabatic CI basis
    CI_FSSH_D = "CI-FSSH-D"
    # CI hamiltonian (from ADIABATIC orbitals), Tully's surface hopping in Adiabatic CI basis
    # particularly, since the CI hamiltonian is from adiabatic orbitals, this mehtod doesn't
    # require diagonalization of the CI hamiltonian
    CI_FSSH_A = "CI-FSSH-A"
    # The good old Independent Electron Surface Hopping (IESH)
    IESH = "IESH"
    
    @property
    def is_CI(self):
        return self in [Method.CI_EH_D, Method.CI_EH_A, Method.CI_FSSH_D, Method.CI_FSSH_A]
    
    @property
    def is_IESH(self):
        return self == Method.IESH  
    
    @property
    def is_diabatic(self):
        return self == Method.CI_EH_D

@dataclass
class InputParameters: 
    ntrajs: int # number of trajectories
    dt: float # time step in au
    tf: float # final time in au, t0 = 0
    flag_boltzmann: bool # flag for boltzmann / wigner  initial conditions
    flag_rev_when_frust: bool
    no: int # number of quantum dofs
    ne: int # number of electrons (occupied orbitals)
    nk: int # number of quadrature sampling points for the electronic bath
    kT: float # thermal energy in au
    W: float # bandwith of the bath in au
    Gamma: float # molecular-bath hybridization Gamma (wide-band limit) in au   
    flag_langevin: bool # flag for langevin dynamics
    init_state:  NDArray[np.float64] 
    mass: float = 2000.0 # mass of the classical dof
    output_dir: str = field(init=False, default=None)   
    method: Method = Method.CI_EH_D
    
    
    @classmethod    
    def from_file(cls, file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        # the input was meant for fortran
        # the format is 
        # number ! comment
        
        # 1st to 4th row are irelevant for this python code
        
        # 5th row is the number of trajectories
        ntrajs = int(lines[4].split()[0])
        
        # skip 6th row (was clasical step)
        # 7th row is dt_quantum
        dt_quantum_in_s = fortran_float_converter(lines[6].split()[0])
        dt_quantum = fs_to_au(dt_quantum_in_s * 1e15)   
        
        tf_in_s = fortran_float_converter(lines[7].split()[0])
        tf = fs_to_au(tf_in_s * 1e15)
        
        # skip the 9-11th row
        # 12th row is the flag for boltzmann / wigner  initial conditions
        flag_boltzmann = bool(int(lines[11].split()[0]))
        
        # 13th row is whether to applied velocity reverse when surface hopping is frustrated
        flag_rev_when_frust = bool(int(lines[12].split()[0]))
        
        # 14-15th are irrelevant
        # 16th row is the number of classical dofs, only support 1 for this python code
        nclassical = int(lines[15].split()[0])
        if nclassical != 1:
            raise ValueError("Only support 1 classical dof")
        
        # 17th row is the number of quantum dofs 
        no = int(lines[16].split()[0])
        
        # 18th row is the number of electronic bath dofs
        nbath = int(lines[17].split()[0])
        
        if no - nbath != 1:
            raise ValueError(f"Only support 1 molecular dof, but got {no} quantum dofs and {nbath} electronic bath dofs")
        
        
        # 19th row is the number of electrons (occupied orbitals)
        ne = int(lines[18].split()[0])
        
        if ne != (nbath // 2) or nbath % 2 != 0:
            warnings.warn(f"You sure you want to have {ne} electrons and {nbath} electronic bath obitals? By default, they shall satisfy nbath = 2 * ne")
            
        nk = nbath // 2
        
        # 20th row is the thermal energy in au
        thermal_energy = fortran_float_converter(lines[19].split()[0])
        
        # 21st row is the bandwith of the bath in au
        W = fortran_float_converter(lines[20].split()[0])
        
        # 22nd row is molecular-bath hybridization Gamma (wide-band limit) in au
        Gamma = fortran_float_converter(lines[21].split()[0])
        
        # 23-24th row are irrelevant to this python code
        # 25th row is the flag for langevin dynamics
        # the frequency of the bath will be 2 * omega_B, where omega_B is the frequency of the harmonic potential
        flag_langevin = bool(lines[24].split()[0])
        
        # 26th row is the seed for random number generator
        # which is irelevant to this python code
        
        # the default initial state is filling the metal orbitals below the fermi level
        # i.e., |011...100...0>, 
        # where the first 0 means the molecular orbital is unoccupied
        # the consecutive 1s mean the metal orbitals are occupied (below the fermi level)
        # the rest 0s mean the metal orbitals are unoccupied (above the fermi level)
        initial_state = np.arange(1, nbath // 2 + 1)
        
        return cls(ntrajs, dt_quantum, tf, flag_boltzmann, flag_rev_when_frust, no, ne, nk, thermal_energy, W, Gamma, flag_langevin, initial_state)
        
def fs_to_au(fs):
    return fs * 41.3413733

def fortran_float_converter(fortran_float: str):
    return float(fortran_float.replace('d', 'e'))
        
def main():
    from tempfile import NamedTemporaryFile
    content = """3                         !! iflow - 1 - serial; 2 - parallel; 3 - averages
8                          !! iproc - max number of jobs running at a time
8                          !! iparallel - number of jobs to parallelize
120                        !! iwait
256                         !! N_traj - number of trajectories
0.25d-15                   !! dtc
0.025d-15                  !! dtq
15.d-12                    !! total_time (*hbar/kT)
0                          !! iwrite: 1 - writes output_cl,output_qm
2                          !! nstep_write - if iwrite=1, write output after every $nstep_write steps
40                         !! nstep_avg - averaging frequency
1                          !! idistribution - 1-classical, 0-Wigner (0 not implemented yet)
1                          !! flag_frust - 0: reverse velocity, 1 - do nothing on frustrated hop. 0 recommended
0                          !! flag_ortho - 1: orthogonalized overlap matrix
100.1d0                    !! energy_cutoff (cm-1) energy conservation threshold at each time-step
1                          !! nclass - number of classical d.o.f.
5                          !! nquant - number of orbitals
4                          !! nmetal - number of metal orbitals
2                          !! n_el - number of electorns
9.5d-4                     !! temperature (au) (here 300K is set)
16.d-3                     !! band_width (au)
1.d-5                      !! gama_coup (au)
1                          !! iforward: 1- left to right, 0 - right to left - not used
0                          !! icollapse: 1 - A-FSSH, 0 - FSSH
1                          !! ifriction: 1 - random forces (Langevin dynamics)
1729 3141                  !! seed for random number
x"""
    file = NamedTemporaryFile()
    file.write(content.encode())
    file.seek(0)
    
    inp = InputParameters.from_file(file.name)
    print(inp)

#%% 
if __name__ == "__main__":
    main()
# %%
