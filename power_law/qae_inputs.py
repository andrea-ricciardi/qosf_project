#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
from typing import Dict, List, Optional

class QAEInputs:
    
    def __init__(self, n_qubits_algo: int, qubits_per_qpu: List[int],
                 problem: str, param: Dict[str, float] = {},
                 evaluation_schedule: Optional[List[int]] = None) -> None:
        self.algo = QAEAlgoInputs(n_qubits_algo, 
                                  evaluation_schedule=evaluation_schedule)
        self.hardware = HardwareInputs(qubits_per_qpu)
        self.integral = IntegralInputs(problem, param)
    
class QAEAlgoInputs:
    
    def __init__(self, n_qubits: int, shots_list: Optional[List[int]] = None, 
                 evaluation_schedule: Optional[List[int]] = None) -> None:
        """
        Algorithm inputs.

        Parameters
        ----------
        n_qubits : int
            DESCRIPTION.
        shots_list : Optional[List[int]], optional
            If None, 100 shots per evaluation. The default is None.
        evaluation_schedule : Optional[List[int]], optional
            If None, Suzuki method. The default is None.

        """
        self.n_qubits = n_qubits
        self.oracle_size = 2 * n_qubits - 1 # number of qubits needed to query the oracle
        if shots_list is None:
            if evaluation_schedule is None:
                self.shots_list = [100, 100, 100, 100, 100, 100, 100]
            else:
                self.shots_list = [100] * len(evaluation_schedule)
        else:
            self.shots_list = shots_list
        if evaluation_schedule is None:
            self.evaluation_schedule = [0, 1, 2, 4, 8, 16, 32]
        else:
            self.evaluation_schedule = evaluation_schedule
        self.num_paulis = len(self.evaluation_schedule)
        
        assert len(self.shots_list) == len(self.evaluation_schedule) 
        
class HardwareInputs:
    
    def __init__(self, qubits_per_qpu: List[int]) -> None:
        self.qubits_per_qpu = qubits_per_qpu
        
    @property
    def num_qpus(self) -> int:
        return len(self.qubits_per_qpu)
        
class IntegralInputs:
    
    def __init__(self, problem_tag: str, 
                 param: Dict[str, float] = {}) -> None:
        self.problem = problem_tag
        self.param = param
        if problem_tag == 'sine_squared':
            # upper limit of the integral
            if 'upper_limit' not in param:
                self.param['upper_limit'] = math.pi / 5.0
        else:
            raise ValueError("Problem {} not implemented".format(problem_tag))
            
    