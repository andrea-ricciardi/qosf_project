#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import abc
from copy import deepcopy
from qae_inputs import HardwareInputs, QAEAlgoInputs
from typing import Dict, List, Tuple

class Schedule:
    """
    Represents an Ansatz distribution schedule. One can run over the 
    distributed system to obtain an estimate to the desired expectation value.
    
    """
    
    def __init__(self, algo_inputs: QAEAlgoInputs, 
                 hardware_inputs: HardwareInputs, 
                 allow_distributed: bool) -> None:
        """
        Parameters
        ----------
        algo_inputs : QAEAlgoInputs
        hardware_inputs : HardwareInputs
        allow_distributed : bool
            True if distributed computing is allowed, False otherwise.
            Distributed computing requires CCNOT gates for entanglement,
            Qiskit does not currently (easily) support it.

        """
        self.algo_inputs = algo_inputs
        self.hardware_inputs = hardware_inputs
        self.allow_distributed = allow_distributed
        self._schedule: Dict[int, List[Tuple[int, List]]] = {}
        
    @abc.abstractmethod
    def make_schedule(self) -> None:
        """
        Abstract method for making the schedule in self._schedule
        """
        return
    
    @property
    def schedule(self) -> Dict[int, List[Tuple[int, List[int]]]]:
        return self._schedule
    
class GreedySchedule(Schedule):
    """    
    Represents the schedule obtained through a greedy algorithm.
    The QPUs are greedily filled with as many Ansatz states that can possibly
    fit; the remaining needed qubits are split across the QPUs. When the QPUs
    cannot fit any more Ansatze, the execution of those estimations are moved
    to the next round.
    
    """
    
    def make_schedule(self) -> None:
        """
        Prepare the greedy schedule and saves it into self.schedule.

        """
        self.__make_schedule(self.algo_inputs.num_paulis)
        
    def __make_schedule(self, num_paulis: int,
                        round: int = 1):
        """
        Recursively makes the schedule.
        
        """
        if num_paulis == 0 or self.algo_inputs.oracle_size == 0:
            return
        
        qpus = self.hardware_inputs.qubits_per_qpu
        
        modified_qpus = [[bits, i] for i, bits in enumerate(qpus)]
        self._schedule[round] = []
        couldNotFit = 0
        
        for i in range(num_paulis):
            modified_qpus.sort(key=lambda q: q[0], reverse=True)
            if len(modified_qpus) == 0 or self.__does_not_fit(deepcopy(modified_qpus)):
                if round == 1 and i == 0:
                    # The Ansatz does not fit, the problem cannot be solved
                    return
                couldNotFit += 1
                continue
            
            distribution = [0] * self.hardware_inputs.num_qpus
            for j in range(len(modified_qpus)):
                possible_qpus = modified_qpus[:j+1]
                curAllocation = self.__fill_allocation(possible_qpus)
                
                if sum(curAllocation) >= self.algo_inputs.oracle_size:
                    # An allocation is possible
                    remaining_bits = self.algo_inputs.oracle_size
                    for idx, bits_index in enumerate(possible_qpus):
                        t = min(remaining_bits, curAllocation[bits_index[1]])
                        distribution[bits_index[1]] += t
                        remaining_bits -= t
                        possible_qpus[idx][0] -= t
                        if remaining_bits == 0:
                            break
                    break
                
            modified_qpus = [bits_idx for bits_idx in modified_qpus if bits_idx[0] != 0]
            self._schedule[round].append((i, distribution))
        
        self.__make_schedule(couldNotFit, round+1)
    
    def __fill_allocation(self, possible_qpus: List[List[int]]) -> List[int]:
        """
        Allocate qubits

        Parameters
        ----------
        possible_qpus : List of two-elements list [bits, index]

        Returns
        -------
        List[int]
            Allocation of qubits per qpu.

        """
        curAllocation = [0] * self.hardware_inputs.num_qpus
        curAllocation[possible_qpus[0][1]] = possible_qpus[0][0]
        for bits_index in possible_qpus[1:]:
            curAllocation[bits_index[1]] += bits_index[0]
        return curAllocation
    
    def __does_not_fit(self, modified_qpus: List[List[int]]) -> bool:
        """
        Returns True if the Ansatz cannot fit in the distributed QPU
    
        Parameters
        ----------
        modified_qpus : List of two-elements list [bits, index]
            Collection of QPUs in the distributed system, non-increasingly sorted
            by the number of available qubits.
    
        Returns
        -------
        True if it cannot fit, False otherwise.
    
        """
        
        def is_not_distributed_computing(allocation: List[int]) -> bool:
            # [5, 0, 0] is not distributed computing, [2, 2, 1] is
            return sum([1 if x != 0 else 0 for x in allocation]) == 1
        
        if len(modified_qpus) == 0:
            return True
        
        for idx in range(len(modified_qpus)):
            possible_qpus = modified_qpus[:idx+1]
            curAllocation = self.__fill_allocation(possible_qpus)
            if sum(curAllocation) >= self.algo_inputs.oracle_size and\
                (self.allow_distributed or is_not_distributed_computing(curAllocation)):
                return False
        
        return True