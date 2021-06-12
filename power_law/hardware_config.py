#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List

class HardwareConfig:
    
    def __init__(self, qubits_per_qpu: List[int]) -> None:
        self.qubits_per_qpu = qubits_per_qpu
        
    @property
    def num_qpus(self) -> int:
        return len(self.qubits_per_qpu)
    
    def print_configuration(self) -> None:
        print("# Hardware Configuration #")
        print("N QPUs: {}, qubits per qpu: {}".format(
            self.num_qpus, self.qubits_per_qpu
        ))
            