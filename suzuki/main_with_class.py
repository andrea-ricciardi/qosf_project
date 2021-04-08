#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import grover
import math
import operators
from qiskit import Aer
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.algorithms import MaximumLikelihoodAmplitudeEstimation
from qiskit.algorithms.amplitude_estimators import EstimationProblem

nbit = 3
b_max = math.pi / 5 # upper limit of integral
method = 'suzuki' # suzuki or power_law

if method == 'suzuki':
    evaluation_schedule = 7 # M in Suzuki. Q^0, Q^1, ... Q^(M-1)
elif method == 'power_law':
    # Power law:
    eps_precision = 0.01
    beta = 0.455
    max_k = max(int(1/eps_precision ** (2*beta)), int(math.log(1/eps_precision)))
    evaluation_schedule = [int(k ** ((1-beta)/(2*beta))) for k in range(1, max_k + 1)]
else:
    raise SystemExit('')


# state_preparation: P R in Fig (6) Suzuki
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

# grover_operator: Q operator. Everything after P R in Fig (6) Suzuki
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
    evaluation_schedule=evaluation_schedule,
    quantum_instance=Aer.get_backend('qasm_simulator')
    )
problem = EstimationProblem(
    state_preparation=state_preparation,
    objective_qubits=[nbit+1],
    grover_operator=grover_op
    )

result = qae.estimate(problem)

