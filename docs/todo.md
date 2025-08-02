## TODOs

+ Observables:
    + handle energy current better, remove nested dicts

+ LatticeDict
  + make it possible to init a LatticeDict from an iterator?
  + LatticeDictKey as class (implement in more places)

+ Other notes:
  + rho_at_minimizer_level(self.hamiltonian.max_l, <- this should be dyn_max_l ??
  + not sure we need `case` in `State`
  + Some strange error with the MPI logger
  + change state.current_time into time? state.current_sum_info to total_information_at_dyn_max_l? state.sum_info to total_information?
  + implement commutator method on operator (?)
  + change state.current_time into time? state.current_sum_info to total_information_at_dyn_max_l? state.sum_info to total_information?
  + do we need t_ind? 
  + use Ray insted of mpi4py? Maybe for some later version it could be useful

+ Hamiltonian
  + parsing the hamiltonian: allow for x11111x to be interpreted as long range
    coupling. Ambiguity for of the Hamiltonian definition for many-bodz terms?
