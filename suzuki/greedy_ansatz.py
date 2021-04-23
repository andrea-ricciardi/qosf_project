#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from copy import deepcopy

def doesNotFit(n, modified_qpus, num_qpus, allow_distributed):
    """
    Returns True if the Ansatz can fit in the distributed QPU Q

    Parameters
    ----------
    n : int
        Size of the Ansatz. It is the number of qubits needed to query the oracle.
    modified_Q : list of two-elements list [bits, index]
        Collection of QPUs in the distributed system, non-increasingly sorted
        by the number of available qubits.

    Returns
    -------
    True if it cannot fit, False otherwise.

    """
    
    def is_not_distributed_computing(allocation):
        return sum([1 if x != 0 else 0 for x in allocation]) == 1
    
    if len(modified_qpus) == 0:
        return True
    
    for idx in range(len(modified_qpus)):
        curAllocation = [0] * num_qpus
        possible_qpus = modified_qpus[:idx+1]
        k = possible_qpus[0][1] # original QPU index
        if idx == 0:
            curAllocation[k] = possible_qpus[0][0]
        else:
            curAllocation[k] = possible_qpus[0][0]
            for bits_index in possible_qpus[1:]:
                # Reserve 2 qubits from the QPUs
                curAllocation[bits_index[1]] += bits_index[0]
            
        if sum(curAllocation) >= n and \
            (allow_distributed or is_not_distributed_computing(curAllocation)):
            return False
        
    return True
    

def greedy_distribution(qpus, ansatz_size, num_paulis, allow_distributed=True, 
                        schedule={}, round=1):
    if num_paulis == 0 or ansatz_size == 0:
        return schedule
    
    modified_qpus = [[bits, i] for i, bits in enumerate(qpus)]
    schedule[round] = []
    couldNotFit = 0
    
    for i in range(num_paulis):
        modified_qpus.sort(key=lambda q: q[0], reverse=True)
        if len(modified_qpus) == 0 or \
            doesNotFit(ansatz_size, deepcopy(modified_qpus), len(qpus), allow_distributed):
            if round == 1 and i == 0:
                # The Ansatz does not fit, the problem cannot be solved
                return schedule
            couldNotFit += 1
            continue
        
        distribution = [0] * len(qpus)
        for j in range(len(modified_qpus)):
            curAllocation = [0] * len(qpus)
            # Restrict to the first j available QPUs
            possible_qpus = modified_qpus[:j+1]
            k = possible_qpus[0][1] # original QPU index
            if j == 0:
                # No split needed
                curAllocation[k] = possible_qpus[0][0]
            else:
                # At least one split needed
                curAllocation[k] = possible_qpus[0][0]
                for bits_index in possible_qpus[1:]:
                    # Reserve 2 qubits from the QPUs
                    curAllocation[bits_index[1]] += bits_index[0]

            if sum(curAllocation) >= ansatz_size:
                # An allocation is possible
                remaining_bits = ansatz_size
                for idx, bits_index in enumerate(possible_qpus):
                    t = min(remaining_bits, curAllocation[bits_index[1]])
                    distribution[bits_index[1]] += t
                    remaining_bits -= t
                    if idx == 0:
                        # Remove the respective quqpu_copybits from the first QPU
                        if len(possible_qpus) == 1:
                        #if j == 0: TODO this is what's in the pseudocode
                            possible_qpus[idx][0] -= t
                        else:
                            possible_qpus[idx][0] -= t
                    else:
                        possible_qpus[idx][0] -= t
                    if remaining_bits == 0:
                        break
                break
                
        modified_qpus = [bits_idx for bits_idx in modified_qpus if bits_idx[0] != 0]
        schedule[round].append((i, distribution))
    return greedy_distribution(
        qpus, ansatz_size, couldNotFit, allow_distributed, schedule, round + 1
    )
