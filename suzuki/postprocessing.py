#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from scipy import optimize

def calculate_theta(hit_list, number_grover_list, shots_list):
    """
    Calculate optimal theta values

    Parameters
    ----------
    hit_list : list of int
        List of count of observing "1" for qc_list.
    number_grover_list : list of int
        List of number of Grover operators.
    shots_list : list of int
        List of number of shots.

    Returns
    -------
    thetaCandidate_list : list of float
        List of optimal theta.

    """
    small = 1.e-15  # small valued parameter to avoid zero division
    confidenceLevel = 5  # confidence level to determine the search range

    thetaCandidate_list = []
    rangeMin = 0.0 + small
    rangeMax = 1.0 - small
    for igrover in range(len(number_grover_list)):

        # p is the alpha: sin^2(theta) = alpha
        # sum(k=0 .. M) (ln (L_k(h_k, theta_a)))
        def loglikelihood(p):
            ret = np.zeros_like(p)
            theta = np.arcsin(np.sqrt(p))
            for n in range(igrover + 1):
                ihit = hit_list[n]
                # 2m_k + 1 * theta_a
                arg = (2 * number_grover_list[n] + 1) * theta
                ret = ret + 2 * ihit * np.log(np.abs(np.sin(arg))) + 2 * (
                    shots_list[n] - ihit) * np.log(np.abs(np.cos(arg)))
            return -ret

        searchRange = (rangeMin, rangeMax)
        # searchResult is array([alpha_value])
        searchResult = optimize.brute(loglikelihood, [searchRange])
        pCandidate = searchResult[0]
        # Get theta from alpha
        thetaCandidate_list.append(np.arcsin(np.sqrt(pCandidate)))
        # CramerRao is 1/sqrt(Fisher_information(alpha))
        perror = CalcErrorCramerRao(igrover, shots_list, pCandidate, number_grover_list)
        # Restrict the range to [alpha-confidence*perror, alpha+confidence*perror]
        rangeMax = min(pCandidate+confidenceLevel*perror,1.0 - small)
        rangeMin = max(pCandidate-confidenceLevel*perror,0.0 + small)
    return thetaCandidate_list

def CalcErrorCramerRao(M, shots_list, p0, number_grover_list):
    """
    Calculate Cramer-Rao lower bound

    Parameters
    ----------
    M : float
        Upper limit of the sum in Fisher information.
    shots_list : list of int
        List of number of shots.
    p0 : float
        The true parameter value to be estimated. (alpha)
    number_grover_list : list of int
        List of number of Grover operators.

    Returns
    -------
    ErrorCramerRao : float
        Square root of Cramer-Rao lower bound: lower bound on the
        standard deviation of unbiased estimators.

    """
    FisherInfo = 0
    for k in range(M + 1):
        Nk = shots_list[k]
        mk = number_grover_list[k]
        # Equation (12) in Suzuki
        FisherInfo += Nk / (p0 * (1 - p0)) * (2 * mk + 1)**2
    return np.sqrt(1 / FisherInfo)
    
def CalcNumberOracleCalls(M, shots_list, number_grover_list):
    """
    Calculate the total number of oracle calls

    Parameters
    ----------
    M : float
        Upper limit of the sum in Fisher information.
    shots_list : list of int
        List of number of shots.
    number_grover_list : list of int
        List of number of Grover operators.

    Returns
    -------
    Norac : int
        Total number of oracle calls.

    """
    Norac = 0
    for k in range(M + 1):
        Nk = shots_list[k]
        mk = number_grover_list[k]
        Norac += Nk * (2 * mk + 1)
    return Norac
    
    