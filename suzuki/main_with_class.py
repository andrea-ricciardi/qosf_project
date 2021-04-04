#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import grover
import math
import operators
from qiskit import Aer
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.aqua.algorithms import MaximumLikelihoodAmplitudeEstimation

nbit = 3
b_max = math.pi / 5 # upper limit of integral
num_circuits = 6


# state_preparation
qx = QuantumRegister(nbit)
qx_measure = QuantumRegister(1)
cr = ClassicalRegister(1)
if nbit > 2:
    qx_ancilla = QuantumRegister(nbit - 2)
    state_preparation = QuantumCircuit(qx, qx_ancilla, qx_measure, cr)
else:
    qx_ancilla = 0
    state_preparation = QuantumCircuit(qx, qx_measure, cr)
operators.P(state_preparation, qx, nbit)
operators.R(state_preparation, qx, qx_measure, nbit, b_max)

# grover_operator
qx = QuantumRegister(nbit)
qx_measure = QuantumRegister(1)
cr = ClassicalRegister(1)
if nbit > 2:
    qx_ancilla = QuantumRegister(nbit - 2)
    grover_op = QuantumCircuit(qx, qx_ancilla, qx_measure, cr)
else:
    qx_ancilla = 0
    grover_op = QuantumCircuit(qx, qx_measure, cr)
grover_op.z(qx_measure[0])
operators.Rinv(grover_op, qx, qx_measure, nbit, b_max)
grover_op.barrier() # format the circuits visualization
operators.P(grover_op, qx, nbit)
grover.reflect(grover_op, qx, qx_measure, qx_ancilla, nbit, b_max)
operators.P(grover_op, qx, nbit)
grover_op.barrier() # format the circuits visualization
operators.R(grover_op, qx, qx_measure, nbit, b_max)

qae = MaximumLikelihoodAmplitudeEstimation(
    num_oracle_circuits=num_circuits + 1,
    state_preparation=state_preparation,
    grover_operator=grover_op
    )

qae.set_backend(Aer.get_backend('qasm_simulator'))
qae.construct_circuits(measurement=True)
qae.run()



