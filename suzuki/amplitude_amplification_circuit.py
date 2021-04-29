#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from operators import SineSquaredOperator
from qae_inputs import IntegralInputs, QAEAlgoInputs
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit import execute
from qiskit.providers import BaseBackend, Backend
from qiskit.tools.parallel import parallel_map
from schedule import GreedySchedule
import time
from typing import List, Tuple, Union

class AmplitudeAmplificationMonteCarloCircuit:
    """
    Quantum circuit of amplitude amplification for the Monte Carlo integration.
    It allows to create the circuit and to run it in parallel.
    
    """
    
    def __init__(self, algo_inputs: QAEAlgoInputs,
                 integral_inputs: IntegralInputs,
                 backend: Union[BaseBackend, Backend]) -> None:
        self.circuits: List[QuantumCircuit] = []
        self.algo_inputs = algo_inputs
        self.__hit_list: List[int] = []
        self.__backend = backend
        if integral_inputs.problem == 'sine_squared':
            self.operator: SineSquaredOperator = SineSquaredOperator(
                algo_inputs.n_qubits, integral_inputs.param
            )
        else:
            raise ValueError("Integral problem {} not recognized".format(
                integral_inputs.problem
            ))
    
    def create_circuit(self) -> None:
        """
        Generate quantum circuits running Grover operators with number of
        iterations in number_grover_list.
        The generated quantum circuits are appended to self.circuits

        """
        for idx, n_grover in enumerate(self.algo_inputs.number_grover_list):
            qx = QuantumRegister(self.algo_inputs.n_qubits)
            qx_measure = QuantumRegister(1)
            cr = ClassicalRegister(1)
            if self.algo_inputs.n_qubits > 2:
                qx_ancilla = QuantumRegister(self.algo_inputs.n_qubits - 2)
                qc = QuantumCircuit(qx, qx_ancilla, qx_measure, cr)
            else:
                qx_ancilla = 0
                qc = QuantumCircuit(qx, qx_measure, cr)
            self.operator.apply_P(qc, qx)
            self.operator.apply_R(qc, qx, qx_measure)
            for _ in range(n_grover):
                self.__apply_grover(qc, qx, qx_measure, qx_ancilla)
            qc.measure(qx_measure[0], cr[0])
            self.circuits.append(qc)
            
    def run_in_parallel(self, parallel_schedule: GreedySchedule) -> None:
        """
        Run in parallel the quantum circuits built by create_circuit.
        The number of parallel runs is determined by parallel_schedule.

        Parameters
        ----------
        parallel_schedule : GreedySchedule
            Distribution schedule.

        """
        self.__hit_list = []
        schedule = parallel_schedule.schedule.copy()
        qc_shots = list(zip(self.circuits, self.algo_inputs.shots_list))
        for round in schedule:
            num_processes = len(schedule[round])
            self.__hit_list = self.__hit_list + parallel_map(
                self.run_single_circuit, qc_shots[:num_processes],
                num_processes=num_processes
            )
            qc_shots = qc_shots[num_processes:]
            
    def get_counts(self) -> List[int]:
        """
        Get the list of counts of observing "1"
        """
        return self.__hit_list
    
    def __apply_grover(self, qc: QuantumCircuit, qx: QuantumRegister,
                       qx_measure: QuantumRegister, 
                       qx_ancilla: QuantumRegister) -> None:
        """
        Apply the Grover operator: R P (I - 2|0><0|) P^+ R^+ U_psi_0 to qc

        """
        qc.z(qx_measure[0])
        self.operator.apply_Rinv(qc, qx, qx_measure)
        qc.barrier()
        self.operator.apply_P(qc, qx)
        self.operator.apply_reflection(qc, qx, qx_measure, qx_ancilla)
        self.operator.apply_P(qc, qx)
        qc.barrier()
        self.operator.apply_R(qc, qx, qx_measure)

    def run_single_circuit(self, 
                           qc_shots: Tuple[QuantumCircuit, int]) -> List[int]:
        """
        Run a single circuit

        Parameters
        ----------
        qc_shots : tuple[QuantumCircuit, int]
            Quantum circuit and number of shots.

        Returns
        -------
        List[int]
            Counts of observing the good state.

        """
        job = execute(qc_shots[0], backend=self.__backend, shots=qc_shots[1])
        interval = 0.00001
        time.sleep(interval)
        while job.status().name != 'DONE':
            time.sleep(interval)
        counts = job.result().get_counts(qc_shots[0]).get("1", 0)
        return counts
