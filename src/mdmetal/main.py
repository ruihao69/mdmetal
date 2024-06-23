# %%
import numpy as np
from joblib import Parallel, delayed

from mdmetal.hamiltonian import NewnsAndersonHarmonic
from mdmetal.initcond import boltzmann_sampling, wigner_sampling, init_amplitudes
from mdmetal.parse_inp import InputParameters, Method
# from dynamics import evaluate_ci_newns_anderson_harmonic
# from dynamics import dynamics_one_traj_adiab, dynamics_one_traj_diab

import os
import argparse
from typing import Tuple

def preprocess() -> InputParameters:
    parser = argparse.ArgumentParser()
    # -i input file
    parser.add_argument("-i", "--input", help="input file")
    # -o output directory
    parser.add_argument("-o", "--output", help="output directory", default="./data")
    # -m method to use, valid options are:
    # CI-Eh-D, full CI hamiltonian, Ehrenfest dynamics, diabatic basis
    # CI-Eh-A, full CI hamiltonian, Ehrenfest dynamics, adiabatic basis
    # CI-SH-A, full CI hamiltonian, Tully's surface hopping, adiabatic basis
    # IESH, one-electron hamiltonian, independent electron surface hopping 
    parser.add_argument("-m", "--method", help="method for MD. Contains: ", default="CI-Eh-D")

    # parse the arguments
    args = parser.parse_args()
    if args.input is None:
        raise ValueError(r"Input file is required. use -h for help.")

    # parse the input file
    inp = InputParameters.from_file(args.input)

    # prepare the output directory if not exist
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    inp.output_dir = args.output
    
    try: 
        method = Method(inp.method)
    except ValueError:
        raise ValueError(f"Invalid method: {inp.method}. Available methods are: {', '.join([m.value for m in Method])}")
    
    inp.method = method
    
    return inp

def initial_conditions(inp: InputParameters) -> Tuple[NewnsAndersonHarmonic, np.ndarray, np.ndarray, np.ndarray]:
    hami = NewnsAndersonHarmonic.initialize(nk=inp.nk, ne=inp.ne, kT=inp.kT, W=inp.W, Gamma=inp.Gamma, mass=inp.mass)

    # classical initial conditions
    if inp.flag_boltzmann:
        R0, P0 = boltzmann_sampling(inp.ntrajs, inp.kT, inp.mass, inp.W)
    else:
        R0, P0 = wigner_sampling(inp.ntrajs, inp.kT, inp.mass, inp.W)

    # quantum initial conditions
    initial_diabatic_states = inp.init_state
    states = hami.states
    nstates = states.shape[0]

    psi0 = np.zeros(nstates, dtype=np.complex128)
    psi0 = init_amplitudes(
        no=inp.no,
        ne=inp.ne,
        diabatic_state=initial_diabatic_states,
        method=inp.method,
        R0=R0,
        hami=hami
    )
    
    return hami, R0, P0, psi0


def main():
    inp = preprocess()
    hami, R0, P0, psi0 = initial_conditions(inp)

    kwargs = {
        "hamiltonian": hami,
        "tf": inp.tf,
        "dt": inp.dt,
    }
    
    # method choices based on the method
    runner = None
    if inp.method == Method.CI_EH_D:
        from mdmetal.CI.ehrenfest_diabatic import dynamics_one as runner
    elif inp.method == Method.CI_EH_A:
        from mdmetal.CI.ehrenfest_adiabatic import dynamics_one as runner
    elif inp.method == Method.CI_SH_A:
        # from mdmetal.CI.surface_hopping_adiabatic import dynamics_one as runner
        raise NotImplementedError("Surface hopping is not implemented yet")
    elif inp.method == Method.IESH:
        from mdmetal.IESH.iesh import dynamics_one as runner
    else:
        raise ValueError(f"method {inp.method} not implemented")
    
    assert runner is not None, f"runner is not defined for method {inp.method}"
        
    
        
        

    # for each initial condition (R0, P0, psi0), run the dynamics
    # parallelize the dynamics for each initial condition
    result = Parallel(n_jobs=-1, verbose=10, return_as='generator')(
        # delayed(dynamics_one_traj_adiab)(R0[itraj], P0[itraj], psi0[itraj], **kwargs) for itraj in range(inp.ntrajs)
        delayed(runner)(R0[itraj], P0[itraj], psi0[itraj], **kwargs) for itraj in range(inp.ntrajs)
    )

    # reduce the results generator
    t_out = R_out = P_out = pop_out = KE_out = PE_out = None
    for item in result:
        if t_out is None:
            t_out, R_out, P_out, pop_out, KE_out, PE_out = item
        else:
            t, R, P, pop, KE, PE = item
            t_out += t
            R_out += R
            P_out += P
            pop_out += pop
            KE_out += KE
            PE_out += PE
    t_out /= inp.ntrajs
    R_out /= inp.ntrajs
    P_out /= inp.ntrajs
    pop_out /= inp.ntrajs
    KE_out /= inp.ntrajs
    PE_out /= inp.ntrajs


    # t_out *= hami.omega_B

    # save the results
    np.savez(os.path.join(inp.output_dir, "output.npz"), t=t_out, R=R_out, P=P_out, pop=pop_out, KE=KE_out, PE=PE_out)


# %%
if __name__ == "__main__":
    main()
