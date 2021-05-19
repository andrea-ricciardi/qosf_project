#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from qiskit.algorithms.amplitude_estimators import EstimationProblem
from qiskit.algorithms.amplitude_estimators import MaximumLikelihoodAmplitudeEstimation
from qiskit.algorithms.amplitude_estimators import MaximumLikelihoodAmplitudeEstimationResult
from qiskit.algorithms.exceptions import AlgorithmError
from schedule import Schedule
from typing import Callable, List, Optional, Tuple, Union

from parallel_quantum_instance import ParallelQuantumInstance


MINIMIZER = Callable[[Callable[[float], float], List[Tuple[float, float]]], float]

class ParallelMaximumLikelihoodAmplitudeEstimation(MaximumLikelihoodAmplitudeEstimation):
    
    def __init__(self, evaluation_schedule: Union[List[int], int],
                 schedule: Schedule,
                 minimizer: Optional[MINIMIZER] = None,
                 quantum_instance: Optional[ParallelQuantumInstance] = None) -> None:
        super().__init__(evaluation_schedule, minimizer, quantum_instance)
        self._schedule = schedule
        
    @property
    def schedule(self):
        return self._schedule
    
    @schedule.setter
    def schedule(self, schedule: Schedule):
        self._schedule = schedule
        
    def estimate(self, estimation_problem: EstimationProblem
                 ) -> MaximumLikelihoodAmplitudeEstimationResult:
        if estimation_problem.state_preparation is None:
            raise AlgorithmError('Either the state_preparation variable or the a_factory '
                                 '(deprecated) must be set to run the algorithm.')

        result = MaximumLikelihoodAmplitudeEstimationResult()
        result.evaluation_schedule = self._evaluation_schedule
        result.minimizer = self._minimizer
        result.evaluation_schedule = self._evaluation_schedule
        result.post_processing = estimation_problem.post_processing

        if self._quantum_instance.is_statevector:
            # run circuit on statevector simulator
            circuits = self.construct_circuits(estimation_problem, measurement=False)
            ret = self._quantum_instance.execute(circuits, parallel_schedule=self._schedule)

            # get statevectors and construct MLE input
            statevectors = [np.asarray(ret.get_statevector(circuit)) for circuit in circuits]
            result.circuit_results = statevectors

            # to count the number of Q-oracle calls (don't count shots)
            result.shots = 1

        else:
            # run circuit on QASM simulator
            circuits = self.construct_circuits(estimation_problem, measurement=True)
            ret = self._quantum_instance.execute(circuits, parallel_schedule=self._schedule)

            # get counts and construct MLE input
            result.circuit_results = [ret.get_counts(circuit) for circuit in circuits]

            # to count the number of Q-oracle calls
            result.shots = self._quantum_instance._run_config.shots

        # run maximum likelihood estimation
        num_state_qubits = circuits[0].num_qubits - circuits[0].num_ancillas
        theta, good_counts = self.compute_mle(result.circuit_results, estimation_problem,
                                              num_state_qubits, True)

        # store results
        result.theta = theta
        result.good_counts = good_counts
        result.estimation = np.sin(result.theta)**2

        # not sure why pylint complains, this is a callable and the tests pass
        # pylint: disable=not-callable
        result.estimation_processed = result.post_processing(result.estimation)

        result.fisher_information = _compute_fisher_information(result)
        result.num_oracle_queries = result.shots * sum(k for k in result.evaluation_schedule)

        # compute and store confidence interval
        confidence_interval = self.compute_confidence_interval(result, alpha=0.05, kind='fisher')
        result.confidence_interval = confidence_interval
        result.confidence_interval_processed = tuple(estimation_problem.post_processing(value)
                                                     for value in confidence_interval)

        return result
        
def _compute_fisher_information(result: 'MaximumLikelihoodAmplitudeEstimationResult',
                                num_sum_terms: Optional[int] = None,
                                observed: bool = False) -> float:
    """Compute the Fisher information.

    Args:
        result: A maximum likelihood amplitude estimation result.
        num_sum_terms: The number of sum terms to be included in the calculation of the
            Fisher information. By default all values are included.
        observed: If True, compute the observed Fisher information, otherwise the theoretical
            one.

    Returns:
        The computed Fisher information, or np.inf if statevector simulation was used.

    Raises:
        KeyError: Call run() first!
    """
    a = result.estimation

    # Corresponding angle to the value a (only use real part of 'a')
    theta_a = np.arcsin(np.sqrt(np.real(a)))

    # Get the number of hits (shots_k) and one-hits (h_k)
    one_hits = result.good_counts
    all_hits = [result.shots] * len(one_hits)

    # Include all sum terms or just up to a certain term?
    evaluation_schedule = result.evaluation_schedule
    if num_sum_terms is not None:
        evaluation_schedule = evaluation_schedule[:num_sum_terms]
        # not necessary since zip goes as far as shortest list:
        # all_hits = all_hits[:num_sum_terms]
        # one_hits = one_hits[:num_sum_terms]

    # Compute the Fisher information
    if observed:
        # Note, that the observed Fisher information is very unreliable in this algorithm!
        d_loglik = 0
        for shots_k, h_k, m_k in zip(all_hits, one_hits, evaluation_schedule):
            tan = np.tan((2 * m_k + 1) * theta_a)
            d_loglik += (2 * m_k + 1) * (h_k / tan + (shots_k - h_k) * tan)

        d_loglik /= np.sqrt(a * (1 - a))
        fisher_information = d_loglik ** 2 / len(all_hits)

    else:
        fisher_information = sum(shots_k * (2 * m_k + 1)**2
                                 for shots_k, m_k in zip(all_hits, evaluation_schedule))
        fisher_information /= a * (1 - a)

    return fisher_information
        