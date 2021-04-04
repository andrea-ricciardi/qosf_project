#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def P(qc, qx, nbit):
    """
    Generating uniform probability distribution.
    The inverse of P is P.

    Parameters
    ----------
    qc : quantum circuit
    qx : quantum register
    nbit : int
        Number of qubits.
        
    """
    qc.h(qx)
    
def R(qc, qx, qx_measure, nbit, b_max):
    """
    Computing the integral function f()

    Parameters
    ----------
    qc : quantum circuit
    qx : quantum register
    qx_measure : quantum register
        Quantum register for measurement.
    nbit : int
        Number of qubits.
    b_max : float
        Upper limit of integral.

    """
    qc.ry(b_max / 2**nbit * 2 * 0.5, qx_measure)
    for i in range(nbit):
        qc.cu3(2**i * b_max / 2**nbit * 2, 0, 0, qx[i], qx_measure[0])
        
def Rinv(qc, qx, qx_measure, nbit, b_max):
    """
    The inverse of R

    Parameters
    ----------
    qc : quantum circuit
    qx : quantum register
    qx_measure : quantum register
        Quantum register for measurement.
    nbit : int
        Number of qubits.
    b_max : float
        Upper limit of integral.

    """
    for i in range(nbit)[::-1]:
        qc.cu3(-2**i * b_max / 2**nbit * 2, 0, 0, qx[i], qx_measure[0])
    qc.ry(-b_max / 2**nbit * 2 * 0.5, qx_measure)

    