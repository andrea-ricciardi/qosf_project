#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import math
from operators import UnitaryOperator, SineSquaredOperator
from qae_inputs import QAEInputs, IntegralInputs
from qiskit import Aer
from qiskit.algorithms.amplitude_estimators import EstimationProblem
from parallel_mlae import ParallelMaximumLikelihoodAmplitudeEstimation
from parallel_quantum_instance import ParallelQuantumInstance
from schedule import GreedySchedule
from typing import List
import warnings

warnings.filterwarnings("ignore")

def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    logging.info("Start program")
    
    """ Start Inputs """
    n_qubits_algo = 3
    qubits_per_qpu = [10, 10, 10]
    problem = 'sine_squared' # 'sine_squared'
    evaluation_schedule = power_law_schedule() # power_law_schedule() for power law, None for Suzuki
    """ End Inputs """
    
    inputs = QAEInputs(n_qubits_algo, qubits_per_qpu, problem, 
                       evaluation_schedule=evaluation_schedule)
    # Schedule for distributed computing
    greedy_schedule = GreedySchedule(inputs.algo, inputs.hardware, False)
    greedy_schedule.make_schedule()
    logging.info("Schedule is {}".format(greedy_schedule.schedule))
    
    # SineSquaredOperator
    problem_operator = make_operator(inputs.integral, inputs.algo.n_qubits)
    
    mlae = ParallelMaximumLikelihoodAmplitudeEstimation(
        evaluation_schedule=inputs.algo.evaluation_schedule,
        schedule=greedy_schedule,
        quantum_instance=ParallelQuantumInstance(Aer.get_backend('qasm_simulator'))
        )
    problem = EstimationProblem(
        state_preparation=problem_operator.prepare_state(),
        objective_qubits=[n_qubits_algo + 1],
        grover_operator=problem_operator.make_grover()
        )
    
    result = mlae.estimate(problem)
    # Returns the result of equation (23) in Suzuki
    approximate_integral(n_qubits_algo, inputs.integral.param['upper_limit'])
    print("Estimation result: {}".format(result.estimation))
    
def make_operator(integral_inputs: IntegralInputs, n_qubits: int) -> UnitaryOperator:
    """
    Make problem operator. Only SineSquaredOperator supported so far.

    Parameters
    ----------
    integral_inputs : IntegralInputs
    n_qubits : int

    Raises
    ------
    ValueError
        If integral_inputs.problem is not sine_squared.

    Returns
    -------
    Type[UnitaryOperator]
        Problem operator.

    """
    if integral_inputs.problem == 'sine_squared':
        return SineSquaredOperator(n_qubits, integral_inputs.param)
    else:
        raise ValueError("Operator for problem {} not implemented".format(integral_inputs.problem))
    
def approximate_integral(nbit: int, b_max: float) -> None:
    """
    Approximate the integral to a discretized result. By using more qubits,
    we can approximate the integral better

    Parameters
    ----------
    nbit : int
        Number of qubits.
    b_max : float
        Upper limit of the integral.

    """
    analyticResult = (b_max / 2.0 - math.sin(2*b_max) / 4.0) / b_max
    print("Analytical result: {}".format(analyticResult))
    
    ndiv = 2 ** nbit # number of discretization
    discretizedResult = 0.0
    for i in range(ndiv):
        discretizedResult += math.sin(b_max / ndiv * (i + 0.5))**2
    discretizedResult = discretizedResult / ndiv
    print("Discretized Result: {}".format(discretizedResult))
    
def power_law_schedule(eps_precision: float = 0.01,
                       beta: float = 0.455) -> List[int]:
    """
    Power Law schedule

    Parameters
    ----------
    eps_precision : float, optional
        Epsilon precision. The default is 0.01.
    beta : float, optional
        Beta. The default is 0.455.

    Returns
    -------
    List[int]
        Power Law schedule.

    """
    max_k = max(int(1/eps_precision ** (2*beta)), int(math.log(1/eps_precision)))
    evaluation_schedule = [int(k ** ((1-beta)/(2*beta))) for k in range(1, max_k+1)]
    return evaluation_schedule     
    
if __name__ == '__main__':
    main()