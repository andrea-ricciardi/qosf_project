#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from qae_inputs import QAEAlgoInputs
from scipy import optimize
from typing import List

class MLEPostProcessing:
    """
    Represents the postprocessing via Maximum Likelihood Estimation.
    
    """
    
    def __init__(self, algo_inputs: QAEAlgoInputs) -> None:
        self.__small = 1.e-15 # small valued parameter to avoid zero division
        self.__confidenceLevel = 5 # confidence level to determine the search range
        self.algo_inputs = algo_inputs
        self.theta_candidates: List[float] = []
        self.errors: List[float] = []
        self.cramer_rao: List[float] = []
        self.n_oracle_calls: List[int] = []
        
    def calculate_theta_candidates(self, hit_list: List[int]) -> None:
        """
        Calculate and return optimal theta values.

        Parameters
        ----------
        hit_list : List[int]
            List of count of observing "1" for each circuit.

        Returns
        -------
        List[float]
            List of optimal thetas.

        """
        rangeMin = 0.0 + self.__small
        rangeMax = 1.0 - self.__small
        for idx_grover in range(len(self.algo_inputs.number_grover_list)):
    
            # alpha: sin^2(theta) = alpha
            # sum(k=0 .. M) (ln (L_k(h_k, theta_a)))
            def loglikelihood(alpha):
                ret = np.zeros_like(alpha)
                theta = np.arcsin(np.sqrt(alpha))
                for n in range(idx_grover + 1):
                    ihit = hit_list[n]
                    # 2m_k + 1 * theta_a
                    arg = (2 * self.algo_inputs.number_grover_list[n] + 1) * theta
                    ret = ret + 2 * ihit * np.log(np.abs(np.sin(arg))) + 2 * (
                        self.algo_inputs.shots_list[n] - ihit) * np.log(np.abs(np.cos(arg)))
                return -ret
    
            searchRange = (rangeMin, rangeMax)
            # searchResult is array([alpha_value])
            searchResult = optimize.brute(loglikelihood, [searchRange])
            alpha_candidate = searchResult[0]
            # Get theta from alpha
            self.theta_candidates.append(np.arcsin(np.sqrt(alpha_candidate)))
            # CramerRao is 1/sqrt(Fisher_information(alpha))
            perror = self.__get_cramer_rao_error(idx_grover, alpha_candidate)
            # Restrict the range to [alpha-confidence*perror, alpha+confidence*perror]
            rangeMax = min(alpha_candidate+self.__confidenceLevel*perror,1.0 - self.__small)
            rangeMin = max(alpha_candidate-self.__confidenceLevel*perror,0.0 + self.__small)
            
    def calculate_errors(self, discretized_result: float) -> None:
        """
        1) Calculate the errors between theta_candidates and discretized result,
           and store them in self.errors
        2) Calculate the cramer rao bounds and store them in self.cramer_rao
        3) Calculate the number of oracle calls and store them in
           self.n_oracle_calls

        Parameters
        ----------
        discretized_result : float

        """
        self.errors = np.abs(np.sin(self.theta_candidates)**2 - discretized_result)
        self.cramer_rao = []
        self.n_oracle_calls = []
        for i in range(len(self.algo_inputs.number_grover_list)):
            self.cramer_rao.append(self.__get_cramer_rao_error(i, discretized_result))
            self.n_oracle_calls.append(self.__get_number_oracle_calls(i))
            
    def plot_errors(self) -> None:
        """
        Plot estimation errors and Cramer-Rao bounds VS number of oracle calls

        """
        p1 = plt.plot(self.n_oracle_calls, self.errors, 'o')
        p2 = plt.plot(self.n_oracle_calls, self.cramer_rao)
        plt.xscale('log')
        plt.xlabel("Number of oracle calls")
        plt.yscale('log')
        plt.ylabel("Estimation error")
        plt.legend((p1[0], p2[0]), ("Estimated Value", "Cramer-Rao"))
        plt.show()
            
    def __get_cramer_rao_error(self, idx_grover: int, alpha: float) -> float:
        """
        Calculate Cramer-Rao lower bound.

        Parameters
        ----------
        idx_grover : int
            Upper limit of the sum in Fisher information.
        alpha : float
            The true parameter value to be estimated.

        Returns
        -------
        float
            Square root of Cramer-Rao lower bound: lower bound on the
            standard deviation of unbiased estimators.

        """
        FisherInfo = 0.0
        for k in range(idx_grover + 1):
            Nk = self.algo_inputs.shots_list[k]
            mk = self.algo_inputs.number_grover_list[k]
            # Equation (12) in Suzuki
            FisherInfo += Nk / (alpha * (1 - alpha)) * (2 * mk + 1)**2
        return np.sqrt(1 / FisherInfo)
    
    def __get_number_oracle_calls(self, idx_grover: int) -> int:
        """
        Calculate the total number of oracle calls.

        Parameters
        ----------
        idx_grover : int
            Upper limit of the sum in Fisher information.

        Returns
        -------
        int
            Total number of oracle calls.

        """
        n_oracle_calls = 0
        for k in range(idx_grover + 1):
            Nk = self.algo_inputs.shots_list[k]
            mk = self.algo_inputs.number_grover_list[k]
            n_oracle_calls += Nk * (2 * mk + 1)
        return n_oracle_calls
            