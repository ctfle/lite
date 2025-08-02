from __future__ import annotations

import logging
from typing import Callable

from local_information.lattice.lattice_dict import LatticeDict
from local_information.mpi.mpi_funcs import get_mpi_variables
from local_information.mpi.distribute import Distributor

logger = logging.getLogger()
COMM, RANK, SIZE, NAME, PARALLEL = get_mpi_variables()


class MultiProcessing:
    """
    Decorator class to execute functions in parallel.

    Note: We scatter parts of the input LatticeDict to each worker process which each executes
    the corresponding computation. The wrapped function HAS to return a Lattice dict which after
    computation is gathered at RANK=0. All other worker processes return None.

    With OpenMPI the operators size is limited even if chunking is applied:
    the maximum operators size \f$L\f$ depends on \f$max_l\f$ as \f$L < 2^{23-max_l}\f$
    """

    def __init__(self, shift: int = 1, method=False):
        # 'shift' discerns whether data is scattered overlapping (shift=1) or non-overlapping (shift=0)
        self.shift = shift
        # determines if the decorated function is bound or not
        self.method = method

    def __call__(self, function: Callable) -> Callable:
        """
        Defines wrapper for calling 'function' in parallel. 'function' must be callable with the
        arguments in the 'wrapper' function and must return a LatticeDict. If parallel computing
        is switched off, passes through the function call.
        """
        if self.method:
            wrapper = self._get_method_wrapper(function)
        else:
            wrapper = self._get_function_wrapper(function)
        return wrapper

    def _get_method_wrapper(self, function: Callable) -> Callable:
        def wrapper(slf, input_dict: LatticeDict, level: int, *args, **kwargs):
            return self._execute(function, input_dict, level, *args, slf=slf, **kwargs)

        return wrapper

    def _get_function_wrapper(self, function: Callable) -> Callable:
        def wrapper(input_dict: LatticeDict, level: int, *args, **kwargs):
            return self._execute(function, input_dict, level, *args, slf=None, **kwargs)

        return wrapper

    def _execute(
        self,
        function: Callable,
        input_dict: LatticeDict,
        level: int,
        *args,
        slf: object | None = None,
        **kwargs,
    ) -> LatticeDict | None:
        if PARALLEL:
            return self._execute_in_parallel(
                function, input_dict, level, *args, slf=slf, **kwargs
            )
        else:
            return self._call_function(
                function, slf, input_dict, level, *args, **kwargs
            )

    def _execute_in_parallel(
        self,
        function: Callable,
        input_dict: LatticeDict,
        level: int,
        *args,
        slf: object | None = None,
        **kwargs,
    ) -> LatticeDict | None:
        level = COMM.bcast(level, root=0)
        distributor = Distributor(input_dict, level)

        # distribute tasks to workers
        lattice_on_worker = distributor.scatter(
            number_of_workers=SIZE, shift=self.shift
        )
        result_on_worker = self._call_function(
            function, slf, lattice_on_worker, level, *args, **kwargs
        )
        # gather processed data at RANK 0. All other worker processes have result = None
        result = distributor.gather(result_on_worker)

        return result

    def _call_function(self, function: Callable, slf: object | None, *args, **kwargs):
        if slf is None:
            return function(*args, **kwargs)
        else:
            return function(slf, *args, **kwargs)
