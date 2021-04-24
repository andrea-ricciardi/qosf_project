#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections.abc import Iterable
import grover
import math
import matplotlib.pyplot as plt
import numpy as np
import postprocessing
from qiskit import Aer
import sys
import greedy_ansatz


def main():
    nbit = 3
    num_qpus = 3
    
    qpu_sizes = [10] * num_qpus
    
    ansatz_size = 2 * nbit - 1 # number of qubits needed to query the oracle

    parallel_method = 'schedule' # mp (multiprocessing), parallel_map (qiskit parallel_map) or schedule

    b_max = math.pi / 5 # upper limit of integral
    shots_list = [100, 100, 100, 100, 100, 100, 100]
    number_grover_list = [0, 1, 2, 4, 8, 16, 32]
    
    num_paulis = len(number_grover_list)
    
    if len(shots_list) != len(number_grover_list):
        raise Exception(
            'The length of shots_list should be equal to the length of \
            number_grover_list'
        )
            
    # The sum of the lengths of the lists == num circuits 
    schedule = greedy_ansatz.greedy_distribution(
        qpu_sizes, ansatz_size, num_paulis, allow_distributed=False
    )
            
    # if not isinstance(nbit, Iterable):
    #     nbit = [nbit for _ in range(num_qpus)]
            
    backend = Aer.get_backend('qasm_simulator')
    
    # Creates the circuit in Fig 6 of Suzuki paper
    qc_list = grover.create_grover_circuit(number_grover_list, nbit, b_max)
    
    # Run the circuit and returns list of count of observing "1" for qc_list
    hit_list = grover.run_grover(
        qc_list, number_grover_list, shots_list, backend, num_qpus, 
        parallel_method, parallel=True, schedule=schedule
    )
    
    # Returns a list of len(number_grover_list) with all the values of theta
    thetaCandidate_list = postprocessing.calculate_theta(
        hit_list, number_grover_list, shots_list
        )
    
    # Returns the result of equation (23) in Suzuki
    discretizedResult = approximate_integral(nbit, b_max)
    
    # Plot to find the correlation between the number of oracle calls and the
    # approximation error of theta_a, as well as the lower bound of such error
    # provided by the Cramer-Rao
    
    # list of estimation errors
    error_list = np.abs(np.sin(thetaCandidate_list)**2 - discretizedResult)
    OracleCall_list = [] # list of Cramer-Rao lower bound
    ErrorCramerRao_list = [] # list of Cramer-Rao lower bound
    for i in range(len(number_grover_list)):
        OracleCall_list.append(
            postprocessing.CalcNumberOracleCalls(
                i, shots_list, number_grover_list))
        ErrorCramerRao_list.append(
            postprocessing.CalcErrorCramerRao(
                i, shots_list, discretizedResult, number_grover_list))
        
    p1 = plt.plot(OracleCall_list, error_list, 'o')
    p2 = plt.plot(OracleCall_list, ErrorCramerRao_list)
    plt.xscale('log')
    plt.xlabel("Number of oracle calls")
    plt.yscale('log')
    plt.ylabel("Estimation error")
    plt.legend((p1[0], p2[0]), ("Estimated Value", "Cramer-Rao"))
    plt.show()
    
    # Repeat the above algorithm n_trial = 100 times to estimate the
    # statistical mean of errors.
    # n_trial = 100
    # error_list = np.zeros_like(number_grover_list, dtype=float)
    # qc_list = grover.create_grover_circuit(number_grover_list, nbit, b_max)
    # for i in range(n_trial):
    #     sys.stdout.write("n_trial=(%d/%d)\r" % ((i + 1), n_trial))
    #     sys.stdout.flush()
    #     hit_list = grover.run_grover(
    #         qc_list, number_grover_list, shots_list, backend)
    #     thetaCandidate_list = postprocessing.calculate_theta(
    #         hit_list, number_grover_list, shots_list)
    #     error_list += (np.sin(thetaCandidate_list)**2 - discretizedResult)**2  # list of estimation errors
    
    # error_list = (error_list / (n_trial-1))**(1/2)
    
    # p1 = plt.plot(OracleCall_list, error_list, 'o')
    # p2 = plt.plot(OracleCall_list, ErrorCramerRao_list)
    # plt.xscale('log')
    # plt.xlabel("Number of oracle calls")
    # plt.yscale('log')
    # plt.ylabel("Estimation Error")
    # plt.legend((p1[0], p2[0]), ("Estimated Value", "Cramér-Rao"))
    # plt.show()


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