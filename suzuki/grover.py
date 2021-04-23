#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import itertools
import multiprocessing as mp
import operators
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit import execute
from qiskit.tools.parallel import parallel_map
import time

def run_single_grover(qc_shots_backend):
    job = execute(qc_shots_backend[0], backend=qc_shots_backend[2], shots=qc_shots_backend[1])
    interval = 0.00001
    time.sleep(interval)
    while job.status().name != 'DONE':
        # Gets stuck here
        time.sleep(interval)
    counts = job.result().get_counts(qc_shots_backend[0]).get("1", 0)
    return counts

def run_grover(
        qc_list, number_grover_list, shots_list, backend, max_qpus, 
        parallel_method, parallel=False, schedule={}
    ):
    """
    Run the quantum circuits returned by create_grover_circuit()

    Parameters
    ----------
    qc_list : list of quantum circuits
    number_grover_list : list of int
        List of number of Grover operators.
    shots_list : list of int
        List of number of shots.
    backend : string
        Name of backends.

    Returns
    -------
    hit_list: list of int
        List of count of observing "1" for qc_list

    """        
    qc_shots_backend_list = list(zip(qc_list, shots_list, [backend for _ in qc_list]))
    if parallel:
        if parallel_method == 'mp':
            print("Parallelization: multiprocessing")
            with mp.Pool(processes=max_qpus) as pool:
                hit_list = pool.map(run_single_grover, qc_shots_backend_list)
        elif parallel_method == 'parallel_map':
            print("Parallelization: parallel_map")
            hit_list = parallel_map(run_single_grover, qc_shots_backend_list, 
                                    num_processes=max_qpus)
        else:
            raise ValueError("Parallel_method {} not recognized".format(parallel_method))
        
    else:
        hit_list = [run_single_grover(qc_shots_backend) for qc_shots_backend in qc_shots_backend_list]
    return hit_list

def create_grover_circuit(number_grover_list, nbit, b_max):
    """
    Generate quantum circuits running Grover operators with number of
    iterations in number_grover_list

    Parameters
    ----------
    number_grover_list : list of int
        List of number of Grover operators.
    nbit : int
        Number of qubits (2**nbit = ndiv is the number of discretization in
                          the Monte Carlo integration).
    b_max : float
        Upper limit of the integral.

    Returns
    -------
    qc_list : list of quantum circuits
        Quantum circuits with Grover operators as in number_grover_list.

    """
    qc_list = []
    for igrover in range(len(number_grover_list)):
        qx = QuantumRegister(nbit)
        qx_measure = QuantumRegister(1)
        cr = ClassicalRegister(1)
        if nbit > 2:
            qx_ancilla = QuantumRegister(nbit - 2)
            qc = QuantumCircuit(qx, qx_ancilla, qx_measure, cr)
        else:
            qx_ancilla = 0
            qc = QuantumCircuit(qx, qx_measure, cr)
        operators.P(qc, qx, nbit)
        operators.R(qc, qx, qx_measure, nbit, b_max)
        for ikAA in range(number_grover_list[igrover]):
            Q_grover(qc, qx, qx_measure, qx_ancilla, nbit, b_max)
        qc.measure(qx_measure[0], cr[0])
        qc_list.append(qc)
    return qc_list

def Q_grover(qc, qx, qx_measure, qx_ancilla, nbit, b_max):
    """
    The Grover operator: R P (I - 2|0><0|) P^+ R^+ U_psi_0

    Parameters
    ----------
    qc : quantum circuit
    qx : quantum register
    qx_measure : quantum register
        Quantum register for measurement.
    qx_ancilla : quantum register
        Temporary quantum register for decomposing multi controlled NOT gate.
    nbit : int
        Number of qubits.
    b_max : float
        Upper limit of integral.

    """
    qc.z(qx_measure[0])
    operators.Rinv(qc, qx, qx_measure, nbit, b_max)
    qc.barrier() # format the circuits visualization
    operators.P(qc, qx, nbit)
    reflect(qc, qx, qx_measure, qx_ancilla, nbit, b_max)
    operators.P(qc, qx, nbit)
    qc.barrier() # format the circuits visualization
    operators.R(qc, qx, qx_measure, nbit, b_max)
            
def reflect(qc, qx, qx_measure, qx_ancilla, nbit, b_max):
    """
    Computing reflection operator (I - 2|0><0|)

    Parameters
    ----------
    qc : quantum circuit
    qx : quantum register
    qx_measure : quantum register
        Quantum register for measurement.
    qx_ancilla : quantum register
        Temporary quantum register for decomposing multi controlled NOT gate.
    nbit : int
        Number of qubits.
    b_max : float
        Upper limit of integral.

    """
    for i in range(nbit):
        qc.x(qx[i])
    qc.x(qx_measure[0])
    qc.barrier()
    multi_control_NOT(qc, qx, qx_measure, qx_ancilla, nbit, b_max)
    qc.x(qx_measure[0])
    for i in range(nbit):
        qc.x(qx[i])
        
def multi_control_NOT(qc, qx, qx_measure, qx_ancilla, nbit, b_max):
    """
    Computing multi controlled NOT gate

    Parameters
    ----------
    qc : quantum circuit
    qx : quantum register
    qx_measure : quantum register
        Quantum register for measurement.
    qx_ancilla : quantum register
        Temporary quantum register for decomposing multi controlled NOT gate.
    nbit : int
        Number of qubits.
    b_max : float
        Upper limit of integral.

    """
    
    if nbit == 1:
        qc.cz(qx[0], qx[1], qx_measure[0])
    elif nbit == 2:
        qc.h(qx_measure[0])
        qc.ccx(qx[0], qx[1], qx_measure[0])
        qc.h(qx_measure[0])
    elif nbit > 2.0:
        qc.ccx(qx[0], qx[1], qx_ancilla[0])
        for i in range(nbit - 3):
            qc.ccx(qx[i + 2], qx_ancilla[i], qx_ancilla[i + 1])
        qc.h(qx_measure[0])
        qc.ccx(qx[nbit - 1], qx_ancilla[nbit - 3], qx_measure[0])
        qc.h(qx_measure[0])
        for i in range(nbit - 3)[::-1]:
            qc.ccx(qx[i + 2], qx_ancilla[i], qx_ancilla[i + 1])
        qc.ccx(qx[0], qx[1], qx_ancilla[0])
