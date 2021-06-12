#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import abc
import numpy as np
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from typing import Dict, Tuple

class UnitaryOperator:
    """
    Represents the A operator.
    
    """
    def __init__(self, n_qubits: int, param: Dict[str, float]) -> None:
        """

        Parameters
        ----------
        n_qubits : int
            Number of qubits.
        param : Dict[str, float]
            Operator's parameters.

        """
        self.n_qubits = n_qubits
        self.param = param
        self._oracle_size = None
    
    def apply_P(self, qc: QuantumCircuit, qx: QuantumRegister) -> None:
        """
        Generating uniform probability distribution.
        The inverse of P is P.
        
        """
        qc.h(qx)
        
    @abc.abstractmethod
    def apply_R(self, qc: QuantumCircuit, qx: QuantumRegister,
                qx_measure: QuantumRegister) -> None:
        """
        Apply R to qc
        
        """
        return
    
    @abc.abstractmethod
    def apply_Rinv(self, qc: QuantumCircuit, qx: QuantumRegister,
                   qx_measure: QuantumRegister) -> None:
        """
        Apply the inverse of R to qc
        
        """
        return
    
    def apply_reflection(self, qc: QuantumCircuit, qx: QuantumRegister,
                         qx_measure: QuantumRegister, 
                         qx_ancilla: QuantumRegister) -> None:
        """
        Apply reflection operator (I - 2|0><0|) to qc
    
        """
        for i in range(self.n_qubits):
            qc.x(qx[i])
        qc.x(qx_measure[0])
        qc.barrier()
        self.apply_multi_control_NOT(qc, qx, qx_measure, qx_ancilla)
        qc.x(qx_measure[0])
        for i in range(self.n_qubits):
            qc.x(qx[i])
            
    def apply_multi_control_NOT(self, qc: QuantumCircuit, qx: QuantumRegister,
                                qx_measure: QuantumRegister,
                                qx_ancilla: QuantumRegister) -> None:
        """
        Apply multi controlled NOT gate to qc
    
        """
        if self.n_qubits == 1:
            qc.cz(qx[0], qx[1], qx_measure[0])
        elif self.n_qubits == 2:
            qc.h(qx_measure[0])
            qc.ccx(qx[0], qx[1], qx_measure[0])
            qc.h(qx_measure[0])
        elif self.n_qubits > 2:
            qc.ccx(qx[0], qx[1], qx_ancilla[0])
            for i in range(self.n_qubits - 3):
                qc.ccx(qx[i + 2], qx_ancilla[i], qx_ancilla[i + 1])
            qc.h(qx_measure[0])
            qc.ccx(qx[self.n_qubits - 1], qx_ancilla[self.n_qubits - 3], qx_measure[0])
            qc.h(qx_measure[0])
            for i in range(self.n_qubits - 3)[::-1]:
                qc.ccx(qx[i + 2], qx_ancilla[i], qx_ancilla[i + 1])
            qc.ccx(qx[0], qx[1], qx_ancilla[0])
            
    def _prepare_circuit(self) -> Tuple[QuantumCircuit, QuantumRegister, 
                                        QuantumRegister, QuantumRegister]:
        qx = QuantumRegister(self.n_qubits)
        qx_measure = QuantumRegister(1)
        cr = ClassicalRegister(1)
        if self.n_qubits > 2:
            qx_ancilla = QuantumRegister(self.n_qubits - 2)
            qc = QuantumCircuit(qx, qx_ancilla, qx_measure, cr)
        else:
            qx_ancilla = 0
            qc = QuantumCircuit(qx, qx_measure, cr)
        return qc, qx, qx_measure, qx_ancilla
        
    def prepare_state(self) -> QuantumCircuit:
        """
        Prepare the quantum state by applying the operators P and R.
        See P and R in Fig (6) in Suzuki for references.

        Returns
        -------
        QuantumCircuit
            Prepared state.

        """
        state, qx, qx_measure, _ = self._prepare_circuit()
        self.apply_P(state, qx)
        self.apply_R(state, qx, qx_measure)
        return state
    
    def make_grover(self) -> QuantumCircuit:
        """
        Make the Q Grover operator.
        See everything after P and R in Fig (6) in Suzuki for references.

        Returns
        -------
        QuantumCircuit
            Grover operator.

        """
        grover, qx, qx_measure, qx_ancilla = self._prepare_circuit()
        grover.z(qx_measure[0])
        self.apply_Rinv(grover, qx, qx_measure)
        grover.barrier()
        self.apply_P(grover, qx)
        self.apply_reflection(grover, qx, qx_measure, qx_ancilla)
        self.apply_P(grover, qx)
        grover.barrier()
        self.apply_R(grover, qx, qx_measure)
        return grover
    
    @property
    def oracle_size(self) -> int:
        return self._oracle_size
    
    @oracle_size.setter
    def oracle_size(self, value: int) -> None:
        if value >= 0:
            self._oracle_size = value
            
    def analytical_result(self) -> float:
        """
        Analytical result of the problem to estimate. This is specific to the
        problem being tackled, therefore this method should be overridden
        by the children with an available analytical result.

        """
        print("Analytical result not available. Return NaN")
        return np.nan
        
    def discretized_result(self) -> float:
        """
        Discretized result of the problem to estimate. This is specific to the
        problem being tackled, therefore this method should be overridden
        by the children with an available analytical result.

        """
        print("Discretized result not available. Return NaN")
        return np.nan
    