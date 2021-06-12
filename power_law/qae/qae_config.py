#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import List, Optional
    
class QAEAlgoConfig:
    
    def __init__(self, n_qubits: int, shots_list: Optional[List[int]] = None, 
                 evaluation_schedule: Optional[List[int]] = None) -> None:
        """
        Algorithm configuration.

        Parameters
        ----------
        n_qubits : int
            DESCRIPTION.
        shots_list : Optional[List[int]], optional
            If None, 100 shots per evaluation. The default is None.
        evaluation_schedule : Optional[List[int]], optional
            If None, Suzuki method. The default is None.

        """
        self.n_qubits = n_qubits
        if shots_list is None:
            if evaluation_schedule is None:
                self.shots_list = [100, 100, 100, 100, 100, 100, 100]
            else:
                self.shots_list = [100] * len(evaluation_schedule)
        else:
            self.shots_list = shots_list
        if evaluation_schedule is None:
            self.evaluation_schedule = [0, 1, 2, 4, 8, 16, 32]
        else:
            self.evaluation_schedule = evaluation_schedule
        self.num_likelihoods = len(self.evaluation_schedule)
        
        assert len(self.shots_list) == len(self.evaluation_schedule) 
        
    def print_configuration(self) -> None:
        print("# Algorithm Configuration #")
        print("N qubits: {}, N likelihood functions: {}".format(
            self.n_qubits, self.num_likelihoods)
        )
    