#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from parallelization import HardwareConfig
from qae import QAEAlgoConfig

import math
from typing import Dict, List, Optional


class QAEMonteCarloConfig:

    def __init__(self, n_qubits_algo: int, qubits_per_qpu: List[int],
                 problem: str, param: Dict[str, float] = {},
                 evaluation_schedule: Optional[List[int]] = None) -> None:
        if n_qubits_algo < 2:
            raise ValueError("At least two qubits are needed for the operator.")
        self.algo = QAEAlgoConfig(n_qubits_algo,
                                  evaluation_schedule=evaluation_schedule)
        self.hardware = HardwareConfig(qubits_per_qpu)
        self.integral = IntegralConfig(problem, param)

    def print_configuration(self) -> None:
        print("### Configuration ###")
        self.algo.print_configuration()
        self.hardware.print_configuration()
        self.integral.print_configuration()
        print("")


class IntegralConfig:

    def __init__(self, problem_tag: str,
                 param: Dict[str, float] = {}) -> None:
        self.problem = problem_tag
        self.param = param
        if problem_tag == 'sine_squared':
            # upper limit of the integral
            if 'upper_limit' not in param:
                self.param['upper_limit'] = math.pi / 5.0
        else:
            raise ValueError("Problem {} not implemented".format(problem_tag))

    def print_configuration(self) -> None:
        print("# Integral Configuration #")
        print("Problem: {}, b_max: {}".format(self.problem, self.param['upper_limit']))
