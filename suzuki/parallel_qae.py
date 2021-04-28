#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from amplitude_amplification_circuit import AmplitudeAmplificationMonteCarloCircuit
import logging
import math
from postprocessing import MLEPostProcessing
from qae_inputs import QAEInputs
from qiskit import Aer
from schedule import GreedySchedule

def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    logging.info("Start program")
    """ Start Inputs """
    n_qubits_algo = 3
    qubits_per_qpu = [10, 10, 10]
    problem = 'sine_squared'
    """ End Inputs """
    inputs = QAEInputs(n_qubits_algo, qubits_per_qpu, problem)
    
    # Schedule
    greedy_schedule = GreedySchedule(inputs.algo, inputs.hardware, False)
    greedy_schedule.make_schedule()
    logging.info("Schedule is {}".format(greedy_schedule.schedule))
    
    # Circuit
    monte_carlo_circuit = AmplitudeAmplificationMonteCarloCircuit(
        inputs.algo, inputs.integral, Aer.get_backend('qasm_simulator')
    )
    monte_carlo_circuit.create_circuit()
    monte_carlo_circuit.run_in_parallel(greedy_schedule)
    hit_list = monte_carlo_circuit.get_counts()
    logging.info("Hit list is {}".format(hit_list))
    
    # Postprocessing
    mle = MLEPostProcessing(inputs.algo)
    mle.calculate_theta_candidates(hit_list)
    discretized_result = approximate_sine_squared_integral(
        n_qubits_algo, inputs.integral.param['upper_limit']
    )
    mle.calculate_errors(discretized_result)
    logging.info("Errors are {}".format(mle.errors))
    mle.plot_errors()
    logging.info("Program terminated")
    
def approximate_sine_squared_integral(n_qubits: int, upper_limit: float) -> float:
    """
    Approximate the integral to a discretized result (equation (23) in the Suzuki paper)

    Parameters
    ----------
    n_qubits : int
        Number of qubits. By using more qubits, we can better approximate the
        integral.
    upper_limit : float
        Upper limit of the integral.

    Returns
    -------
    float
        Discretized result.

    """
    analytic_result = (upper_limit / 2.0 - math.sin(2*upper_limit) / 4.0) / upper_limit
    logging.info("Analytical result: {}".format(analytic_result))
    
    ndiv = 2**n_qubits # number of discretization
    discretized_result = 0.0
    for i in range(ndiv):
        discretized_result += math.sin(upper_limit / ndiv * (i + 0.5))**2
    discretized_result = discretized_result / ndiv
    logging.info("Discretized result: {}".format(discretized_result))
    return discretized_result

if __name__ == '__main__':
    main()
