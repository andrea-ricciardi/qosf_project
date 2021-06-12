#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys  # TODO ugly
sys.path.append("../../")

import math
from operators import UnitaryOperator
from qiskit import QuantumRegister, QuantumCircuit
from typing import Dict


class SineSquaredOperator(UnitaryOperator):
    """
    Represents the A operator for the problem of computing the integral

            I = (1/upper_limit) * int_0^upper_limit (sin(x))^2 dx
    """

    def __init__(self, n_qubits: int, param: Dict[str, float]) -> None:
        assert 'upper_limit' in param
        super().__init__(n_qubits, param)
        # Oracle size includes the ancillary qubits needed for the Q operator,
        # as well as the measurement qubit
        self.oracle_size = 2 * n_qubits - 1

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

    def analytical_result(self) -> float:
        """
        I = 1/b * (b / 2 - sin(2*b) / 4)

        """
        b_max = self.param['upper_limit']
        return (b_max / 2.0 - math.sin(2*b_max) / 4.0) / b_max

    def discretized_result(self) -> float:
        """
        Returns the integral's discretized approximation (Eq. (23) of Suzuki paper).
        By using more qubits, we can approximate better the integral.

        """
        ndiv = 2 ** self.n_qubits  # number of discretization
        b_max = self.param['upper_limit']
        res = 0.0
        for i in range(ndiv):
            res += math.sin(b_max / ndiv * (i + 0.5))**2
        res = res / ndiv
        return res
