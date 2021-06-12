#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys  # TODO ugly
sys.path.append("../")

import copy
import logging
from qiskit.exceptions import QiskitError
from qiskit.providers import Backend, BaseBackend, JobStatus
from qiskit.providers.jobstatus import JOB_FINAL_STATES
from qiskit.qobj import QasmQobj
from qiskit.result import Result
from qiskit.tools.parallel import parallel_map
from qiskit.utils import QuantumInstance
from qiskit.utils.backend_utils import (is_aer_qasm, is_simulator_backend,
                                        is_local_backend)
from qiskit.utils import run_circuits
from scheduling.schedule import Schedule
import time
from typing import Callable, Dict, Optional, Union

logger = logging.getLogger(__name__)

MAX_CIRCUITS_PER_JOB = run_circuits.MAX_CIRCUITS_PER_JOB


class ParallelQuantumInstance(QuantumInstance):
    """
    Represents a QuantumInstance that can be executed parallely by providing
    a distributed computing schedule.

    """
    
    def execute(self,
                circuits,
                parallel_schedule: Schedule,
                had_transpiled: bool = False):
        """
        A wrapper to interface with quantum backend.

        Args:
            circuits (Union['QuantumCircuit', List['QuantumCircuit']]):
                        circuits to execute
            parallel_schedule: schedule for distributed computing
            had_transpiled: whether or not circuits had been transpiled

        Returns:
            Result: result object
            
        """

        from qiskit.utils.measurement_error_mitigation import \
            (get_measured_qubits_from_qobj, build_measurement_error_mitigation_qobj)

        # maybe compile
        if not had_transpiled:
            circuits = self.transpile(circuits)

        # assemble
        qobj = self.assemble(circuits)

        if self._meas_error_mitigation_cls is not None:
            qubit_index, qubit_mappings = get_measured_qubits_from_qobj(qobj)
            qubit_index_str = '_'.join([str(x) for x in qubit_index]) + \
                "_{}".format(self._meas_error_mitigation_shots or self._run_config.shots)
            meas_error_mitigation_fitter, timestamp = \
                self._meas_error_mitigation_fitters.get(qubit_index_str, (None, 0.))

            if meas_error_mitigation_fitter is None:
                # check the asked qubit_index are the subset of build matrices
                for key, _ in self._meas_error_mitigation_fitters.items():
                    stored_qubit_index = [int(x) for x in key.split("_")[:-1]]
                    stored_shots = int(key.split("_")[-1])
                    if len(qubit_index) < len(stored_qubit_index):
                        tmp = list(set(qubit_index + stored_qubit_index))
                        if sorted(tmp) == sorted(stored_qubit_index) and \
                                self._run_config.shots == stored_shots:
                            # the qubit used in current job is the subset and shots are the same
                            meas_error_mitigation_fitter, timestamp = \
                                self._meas_error_mitigation_fitters.get(key, (None, 0.))
                            meas_error_mitigation_fitter = \
                                meas_error_mitigation_fitter.subset_fitter(
                                    qubit_sublist=qubit_index)
                            logger.info("The qubits used in the current job is the subset of "
                                        "previous jobs, "
                                        "reusing the calibration matrix if it is not out-of-date.")

            build_cals_matrix = self.maybe_refresh_cals_matrix(timestamp) or \
                meas_error_mitigation_fitter is None

            if build_cals_matrix:
                logger.info("Updating qobj with the circuits for measurement error mitigation.")
                use_different_shots = not (
                    self._meas_error_mitigation_shots is None
                    or self._meas_error_mitigation_shots == self._run_config.shots)
                temp_run_config = copy.deepcopy(self._run_config)
                if use_different_shots:
                    temp_run_config.shots = self._meas_error_mitigation_shots

                cals_qobj, state_labels, circuit_labels = \
                    build_measurement_error_mitigation_qobj(qubit_index,
                                                            self._meas_error_mitigation_cls,
                                                            self._backend,
                                                            self._backend_config,
                                                            self._compile_config,
                                                            temp_run_config)
                if use_different_shots or is_aer_qasm(self._backend):
                    cals_result = run_qobj_parallel(cals_qobj, parallel_schedule,
                                                    self._backend, self._qjob_config,
                                                    self._backend_options,
                                                    self._noise_config,
                                                    self._skip_qobj_validation, self._job_callback)
                    self._time_taken += cals_result.time_taken
                    result = run_qobj_parallel(qobj, parallel_schedule, self._backend,
                                               self._qjob_config,
                                               self._backend_options, self._noise_config,
                                               self._skip_qobj_validation, self._job_callback)
                    self._time_taken += result.time_taken
                else:
                    # insert the calibration circuit into main qobj if the shots are the same
                    qobj.experiments[0:0] = cals_qobj.experiments
                    result = run_qobj_parallel(qobj, parallel_schedule, self._backend,
                                               self._qjob_config,
                                               self._backend_options, self._noise_config,
                                               self._skip_qobj_validation, self._job_callback)
                    self._time_taken += result.time_taken
                    cals_result = result

                logger.info("Building calibration matrix for measurement error mitigation.")
                meas_error_mitigation_fitter = \
                    self._meas_error_mitigation_cls(cals_result,
                                                    state_labels,
                                                    qubit_list=qubit_index,
                                                    circlabel=circuit_labels)
                self._meas_error_mitigation_fitters[qubit_index_str] = \
                    (meas_error_mitigation_fitter, time.time())
            else:
                result = run_qobj_parallel(qobj, parallel_schedule, self._backend,
                                           self._qjob_config,
                                           self._backend_options, self._noise_config,
                                           self._skip_qobj_validation, self._job_callback)
                self._time_taken += result.time_taken

            if meas_error_mitigation_fitter is not None:
                logger.info("Performing measurement error mitigation.")
                skip_num_circuits = len(result.results) - len(circuits)
                #  remove the calibration counts from result object to assure the length of
                #  ExperimentalResult is equal length to input circuits
                result.results = result.results[skip_num_circuits:]
                tmp_result = copy.deepcopy(result)
                for qubit_index_str, c_idx in qubit_mappings.items():
                    curr_qubit_index = [int(x) for x in qubit_index_str.split("_")]
                    tmp_result.results = [result.results[i] for i in c_idx]
                    if curr_qubit_index == qubit_index:
                        tmp_fitter = meas_error_mitigation_fitter
                    else:
                        tmp_fitter = meas_error_mitigation_fitter.subset_fitter(curr_qubit_index)
                    tmp_result = tmp_fitter.filter.apply(
                        tmp_result, self._meas_error_mitigation_method
                    )
                    for i, n in enumerate(c_idx):
                        result.results[n] = tmp_result.results[i]

        else:
            result = run_qobj_parallel(qobj, parallel_schedule, self._backend,
                                       self._qjob_config,
                                       self._backend_options, self._noise_config,
                                       self._skip_qobj_validation, self._job_callback)
            self._time_taken += result.time_taken

        if self._circuit_summary:
            self._circuit_summary = False

        return result


def run_qobj_parallel(qobj: QasmQobj,
                      parallel_schedule: Schedule,
                      backend: Union[Backend, BaseBackend],
                      qjob_config: Optional[Dict] = None,
                      backend_options: Optional[Dict] = None,
                      noise_config: Optional[Dict] = None,
                      skip_qobj_validation: bool = False,
                      job_callback: Optional[Callable] = None) -> Result:
    """
    An execution wrapper with Qiskit-Terra, with job auto recover capability.

    The auto-recovery feature is only applied for non-simulator backend.
    This wrapper will try to get the result no matter how long it takes.

    Args:
        qobj: qobj to execute
        parallel_schedule: schedule for distributed computing
        backend: backend instance
        qjob_config: configuration for quantum job object
        backend_options: configuration for simulator
        noise_config: configuration for noise model
        skip_qobj_validation: Bypass Qobj validation to decrease submission time,
                                               only works for Aer and BasicAer providers
        job_callback: callback used in querying info of the submitted job, and
                                           providing the following arguments:
                                            job_id, job_status, queue_position, job

    Returns:
        Result object

    Raises:
        ValueError: invalid backend
        QiskitError: Any error except for JobError raised by Qiskit Terra
    """
    qjob_config = qjob_config or {}
    backend_options = backend_options or {}
    noise_config = noise_config or {}

    if backend is None or not isinstance(backend, (Backend, BaseBackend)):
        raise ValueError('Backend is missing or not an instance of BaseBackend')

    with_autorecover = not is_simulator_backend(backend)

    if MAX_CIRCUITS_PER_JOB is not None:
        max_circuits_per_job = int(MAX_CIRCUITS_PER_JOB)
    else:
        if is_local_backend(backend):
            max_circuits_per_job = sys.maxsize
        else:
            max_circuits_per_job = backend.configuration().max_experiments

    # split qobj if it exceeds the payload of the backend

    qobjs = run_circuits._split_qobj_to_qobjs(qobj, max_circuits_per_job)
    jobs = []
    job_ids = []
    
    schedule = parallel_schedule.schedule.copy()
    for round in schedule:
        num_processes = len(schedule[round])
        task_args = (backend, backend_options, noise_config, skip_qobj_validation)
        these_jobs_and_job_ids = parallel_map(run_circuits._safe_submit_qobj,
                                              values=qobjs[:num_processes],
                                              task_args=task_args)
        qobjs = qobjs[num_processes:]
        if len(these_jobs_and_job_ids) > 0:
            jobs.append(these_jobs_and_job_ids[0][0])
            job_ids.append(these_jobs_and_job_ids[0][1])

    # for qob in qobjs:
    #     job, job_id = run_circuits._safe_submit_qobj(qob, backend,
    #                                                  backend_options, noise_config,
    #                                                  skip_qobj_validation)
    #     job_ids.append(job_id)
    #     jobs.append(job)

    results = []
    if with_autorecover:
        logger.info("Backend status: %s", backend.status())
        logger.info("There are %s jobs are submitted.", len(jobs))
        logger.info("All job ids:\n%s", job_ids)
        for idx, _ in enumerate(jobs):
            job = jobs[idx]
            job_id = job_ids[idx]
            while True:
                logger.info("Running %s-th qobj, job id: %s", idx, job_id)
                # try to get result if possible
                while True:
                    job_status = run_circuits._safe_get_job_status(job, job_id)
                    queue_position = 0
                    if job_status in JOB_FINAL_STATES:
                        # do callback again after the job is in the final states
                        if job_callback is not None:
                            job_callback(job_id, job_status, queue_position, job)
                        break
                    if job_status == JobStatus.QUEUED:
                        queue_position = job.queue_position()
                        logger.info("Job id: %s is queued at position %s", job_id, queue_position)
                    else:
                        logger.info("Job id: %s, status: %s", job_id, job_status)
                    if job_callback is not None:
                        job_callback(job_id, job_status, queue_position, job)
                    time.sleep(qjob_config['wait'])

                # get result after the status is DONE
                if job_status == JobStatus.DONE:
                    while True:
                        result = job.result(**qjob_config)
                        if result.success:
                            results.append(result)
                            logger.info("COMPLETED the %s-th qobj, job id: %s", idx, job_id)
                            break

                        logger.warning("FAILURE: Job id: %s", job_id)
                        logger.warning("Job (%s) is completed anyway, retrieve result "
                                       "from backend again.", job_id)
                        job = backend.retrieve_job(job_id)
                    break
                # for other cases, resubmit the qobj until the result is available.
                # since if there is no result returned, there is no way algorithm can do any process
                # get back the qobj first to avoid for job is consumed
                qobj = job.qobj()
                if job_status == JobStatus.CANCELLED:
                    logger.warning("FAILURE: Job id: %s is cancelled. Re-submit the Qobj.",
                                   job_id)
                elif job_status == JobStatus.ERROR:
                    logger.warning("FAILURE: Job id: %s encounters the error. "
                                   "Error is : %s. Re-submit the Qobj.",
                                   job_id, job.error_message())
                else:
                    logging.warning("FAILURE: Job id: %s. Unknown status: %s. "
                                    "Re-submit the Qobj.", job_id, job_status)

                job, job_id = run_circuits._safe_submit_qobj(qobj, backend,
                                                             backend_options,
                                                             noise_config, skip_qobj_validation)
                jobs[idx] = job
                job_ids[idx] = job_id
    else:
        results = []
        for job in jobs:
            results.append(job.result(**qjob_config))

    result = run_circuits._combine_result_objects(results) if results else None

    # If result was not successful then raise an exception with either the status msg or
    # extra information if this was an Aer partial result return
    if result is None:
        raise QiskitError('Circuit execution failed')
    if not result.success:
        msg = result.status
        if result.status == 'PARTIAL COMPLETED':
            # Aer can return partial results which Aqua algorithms cannot process and signals
            # using partial completed status where each returned result has a success and status.
            # We use the status from the first result that was not successful
            for res in result.results:
                if not res.success:
                    msg += ', ' + res.status
                    break
        raise QiskitError('Circuit execution failed: {}'.format(msg))

    if not hasattr(result, 'time_taken'):
        setattr(result, 'time_taken', 0.)

    return result
