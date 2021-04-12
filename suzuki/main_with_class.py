#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import grover
import math
import operators
from qiskit import Aer
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.algorithms import MaximumLikelihoodAmplitudeEstimation
from qiskit.algorithms.amplitude_estimators import EstimationProblem

def main():
    """
    Run MLAE to compute the integral
        I = (1 / b_max) * /int_0^b_max (sin(x)) ** 2 dx
        
    Use the "suzuki" method (exponential incremental sequence) or the 
    "power_law" method (from "Low depth algorithms for quantum amplitude estimation" paper).
    
    """
    nbit = 3
    b_max = math.pi / 5 # upper limit of integral
    method = 'power_law' # suzuki or power_law

    if method == 'suzuki':
        evaluation_schedule = 7 # M in Suzuki. Q^0, Q^1, ... Q^(M-1)
    elif method == 'power_law':
        # Power law:
        eps_precision = 0.01
        beta = 0.455
        max_k = max(int(1/eps_precision ** (2*beta)), int(math.log(1/eps_precision)))
        evaluation_schedule = [int(k ** ((1-beta)/(2*beta))) for k in range(1, max_k + 1)]
    else:
        raise SystemExit('Method {} not recognized'.format(method))
    
    state_preparation = prepare_state(nbit, b_max)
    grover_op = make_grover_operator(nbit, b_max)
    
    mlae = MaximumLikelihoodAmplitudeEstimation(
        evaluation_schedule=evaluation_schedule,
        quantum_instance=Aer.get_backend('qasm_simulator')
        )
    problem = EstimationProblem(
        state_preparation=state_preparation,
        objective_qubits=[nbit+1],
        grover_operator=grover_op
        )
    
    result = mlae.estimate(problem)
    
    # Returns the result of equation (23) in Suzuki
    approximate_integral(nbit, b_max)
    print("Estimation result: {}".format(result.estimation))

def prepare_state(nbit, b_max):
    # state_preparation: P R in Fig (6) Suzuki
    qx = QuantumRegister(nbit)
    qx_measure = QuantumRegister(1)
    cr = ClassicalRegister(1)
    if nbit > 2:
        qx_ancilla = QuantumRegister(nbit - 2)
        state_preparation = QuantumCircuit(qx, qx_ancilla, qx_measure, cr)
    else:
        qx_ancilla = 0
        state_preparation = QuantumCircuit(qx, qx_measure, cr)
    operators.P(state_preparation, qx, nbit)
    operators.R(state_preparation, qx, qx_measure, nbit, b_max)
    return state_preparation

def make_grover_operator(nbit, b_max):
    # grover_operator: Q operator. Everything after P R in Fig (6) Suzuki
    qx = QuantumRegister(nbit)
    qx_measure = QuantumRegister(1)
    cr = ClassicalRegister(1)
    if nbit > 2:
        qx_ancilla = QuantumRegister(nbit - 2)
        grover_op = QuantumCircuit(qx, qx_ancilla, qx_measure, cr)
    else:
        qx_ancilla = 0
        grover_op = QuantumCircuit(qx, qx_measure, cr)
    grover_op.z(qx_measure[0])
    operators.Rinv(grover_op, qx, qx_measure, nbit, b_max)
    grover_op.barrier() # format the circuits visualization
    operators.P(grover_op, qx, nbit)
    grover.reflect(grover_op, qx, qx_measure, qx_ancilla, nbit, b_max)
    operators.P(grover_op, qx, nbit)
    grover_op.barrier() # format the circuits visualization
    operators.R(grover_op, qx, qx_measure, nbit, b_max)
    return grover_op

def approximate_integral(nbit, b_max):
    """
    Approximate the integral to a discretized result. By using more qubits,
    we can approximate the integral better

    Parameters
    ----------
    nbit : int
        Number of qubits.
    b_max : float
        Upper limit of the integral.

    Returns
    -------
    discretizedResult : float
        Integral approximation.

    """
    analyticResult = (b_max / 2.0 - math.sin(2*b_max) / 4.0) / b_max
    print("Analytical result: {}".format(analyticResult))
    
    ndiv = 2**nbit # number of discretization
    discretizedResult = 0.0
    for i in range(ndiv):
        discretizedResult += math.sin(b_max / ndiv * (i + 0.5))**2
    discretizedResult = discretizedResult / ndiv
    print("Discretized Result: {}".format(discretizedResult))
    return discretizedResult

if __name__ == '__main__':
    main()


