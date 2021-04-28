#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
from typing import Dict, List, Optional

class QAEInputs:
    
    def __init__(self, n_qubits_algo: int, qubits_per_qpu: List[int],
                 problem: str, param: Dict[str, float] = {}) -> None:
        self.algo = QAEAlgoInputs(n_qubits_algo)
        self.hardware = HardwareInputs(qubits_per_qpu)
        self.integral = IntegralInputs(problem, param)
    
class QAEAlgoInputs:
    
    def __init__(self, n_qubits: int, shots_list: Optional[List[int]] = None, 
                 number_grover_list: Optional[List[int]] = None) -> None:
        self.n_qubits = n_qubits
        self.oracle_size = 2 * n_qubits - 1 # number of qubits needed to query the oracle
        if shots_list is None:
            self.shots_list = [100, 100, 100, 100, 100, 100, 100]
        else:
            self.shots_list = shots_list
        if number_grover_list is None:
            self.number_grover_list = [0, 1, 2, 4, 8, 16, 32]
        else:
            self.number_grover_list = number_grover_list
        self.num_paulis = len(self.number_grover_list)
        
        assert len(self.shots_list) == len(self.number_grover_list) 
        
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
            
    