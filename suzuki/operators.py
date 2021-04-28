#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import abc
from qiskit import QuantumRegister, QuantumCircuit
from typing import Dict

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
        

class SineSquaredOperator(UnitaryOperator):
    """
    Represents the A operator for the problem of computing the integral
    
            I = (1/upper_limit) * int_0^upper_limit (sin(x))^2 dx
    """
    
    def __init__(self, n_qubits: int, param: Dict[str, float]) -> None:
        assert 'upper_limit' in param
        super().__init__(n_qubits, param)
    
    def apply_R(self, qc: QuantumCircuit, qx: QuantumRegister,
                qx_measure: QuantumRegister) -> None:
        """
        Computing the integral function f()
    
        """
        qc.ry(self.param['upper_limit'] / 2**self.n_qubits * 2 * 0.5, qx_measure)
        for i in range(self.n_qubits):
            qc.cu3(2**i * self.param['upper_limit'] / 2**self.n_qubits * 2, 
                   0, 0, qx[i], qx_measure[0])

    def apply_Rinv(self, qc: QuantumCircuit, qx: QuantumRegister,
                   qx_measure: QuantumRegister) -> None:
        """
        Apply the inverse of R to qc
    
        """
        for i in range(self.n_qubits)[::-1]:
            qc.cu3(-2**i * self.param['upper_limit'] / 2**self.n_qubits * 2, 
                   0, 0, qx[i], qx_measure[0])
        qc.ry(-self.param['upper_limit'] / 2**self.n_qubits * 2 * 0.5, qx_measure)

