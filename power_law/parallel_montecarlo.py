#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
    """
    Run the MaximumLikelihoodAmplitudeEstimation parallely. Steps:
    
    (1) The user can change the inputs between 'Start Inputs' and 'End Inputs'.
    (2) Make the schedule for distributed computing (resource allocation problem).
    (3) Set up the parallel MLAE algorithm.
    (4) Make the A operator specific to the problem to solve. Currently only the operator
        for the integral of (sin(x))^2 is implemented. In order to tackle new problems,
        the user must derive his/her own operator from operators.UnitaryOperator.
    (5) Set up the estimation problem through the EstimationProblem Qiskit class.
    (6) Run the estimation problem.
    (7) Print the analytical result, the discretized result and the estimation result.
    
    """
    ##############
    # (1) Inputs #
    ##############
    """ Start Inputs """
    n_qubits_algo = 3 # number of qubits needed by the algorithm
    qubits_per_qpu = [10, 10, 10]
    problem = 'sine_squared' # 'sine_squared' for computing the integral of (sin(x))^2
    evaluation_schedule = None # power_law_schedule() for power law, None for Suzuki
    """ End Inputs """
    
    # Make inputs for the program
    inputs = QAEInputs(n_qubits_algo, qubits_per_qpu, problem, 
                       evaluation_schedule=evaluation_schedule)
    
    ##########################################
    # (2) Schedule for distributed computing #
    ##########################################
    # A greedy schedule is a schedule that greedily fills the QPUs with as many qubits
    # that can possibly fit; when the QPUs cannot fit any more qubits, the execution
    # of those estimations are moved to the next round.
    # Other approaches are possible, even though not implemented, such as constraint programming.
    greedy_schedule = GreedySchedule(inputs.algo, inputs.hardware, False)
    greedy_schedule.make_schedule()
    # TODO greedy schedule figure it out
    # greedy_schedule.schedule is a dictionary having:
    # - rounds as keys
    # - list of tuples ()
    
    ##########################################
    # (3) Set up the parallel MLAE algorithm #
    ##########################################
    # PMLAE needs, as well as an evaluation schedule, a parallelization schedule
    # and a ParallelQuantumInstance
    pmlae = ParallelMaximumLikelihoodAmplitudeEstimation(
        evaluation_schedule=inputs.algo.evaluation_schedule,
        parallelization_schedule=greedy_schedule,
        quantum_instance=ParallelQuantumInstance(Aer.get_backend('qasm_simulator'))
        )
    
    ##################
    # (4) Operator A #
    ##################
    # Make the A operator specific to the problem.
    # Currently only SineSquaredOperator is implemented.
    problem_operator = make_operator(inputs.integral, inputs.algo.n_qubits)
    
    #####################################
    # (5) Set up the estimation problem #
    #####################################
    problem = EstimationProblem(
        state_preparation=problem_operator.prepare_state(),
        objective_qubits=[n_qubits_algo + 1], # include the ancilla qubit
        grover_operator=problem_operator.make_grover()
        )
    
    ##################################
    # (6) Run the estimation problem #
    ##################################
    result = pmlae.estimate(problem)
    
    #####################
    # (7) Print results #
    #####################
    # Print the result of equation (23) in Suzuki (discretized result) and the analytical result
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