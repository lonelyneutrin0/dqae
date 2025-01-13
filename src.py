import numpy as np 
import qiskit
from qiskit_aer import Aer
from scipy.linalg import expm
import itertools 
from typing import Dict
from dataclasses import dataclass
import qiskit.visualization
import matplotlib.pyplot as plt 
import pprint

@dataclass
class QUBO_solution: 
    prob: Dict 
    eigs: np.array

def q_anneal(Q: np.array)->Dict: 
    """
    Quantum Adiabatic Evolution Algorithm to solve QUBO Problems:  
    """
    # Make sure the matrix is square 
    if(Q.shape[0] != Q.shape[1]): raise ValueError('Improper matrix dimensionality!')

    # Convert QUBO to Ising 
    J = 0.25 * Q 
    h = 0.25 * (np.sum(Q, axis=0) + np.sum(Q, axis=1))
    order = Q.shape[0]

    # Qiskit Circuit Creation 
    qubits = qiskit.QuantumRegister(order)
    circuit = qiskit.QuantumCircuit(qubits)

    # Hamiltonian Formulation 
    pauli_X = np.array([[0, 1], [1, 0]])
    pauli_Z = np.array([[1, 0], [0, -1]])

    # Create the Hamiltonia 
    initial_matrix = np.zeros(shape=(2**order, 2**order))
    final_matrix = np.zeros_like(initial_matrix)
    
    # Create final matrix O(n^2) for now, will implement einsum later 
    for i in range(order): 
        for j in range(order): 
            Z_i = np.kron(np.eye(2**i), np.kron(pauli_Z, np.eye((2**(order-1-i)))))
            Z_j = np.kron(np.eye(2**j), np.kron(pauli_Z, np.eye((2**(order-1-j)))))
            final_matrix += J[i, j] * (Z_i @ Z_j)

    # Add transverse magnetic field terms O(n) for now, will implement einsum later 
    for i in range(order): 
        
        initial_matrix += np.kron(np.eye(2**i), np.kron(pauli_Z, np.eye((2**(order-1-i)))))
        final_matrix  += h[i] * np.kron(np.eye(2**i), np.kron(pauli_X, np.eye((2**(order-1-i)))))

        # Hadamardize the circuit. Initially, we want each state to be equiprobable. 
        circuit.h(i)

    # Adiabatic Setup 
    iterations = 10000
    timesteps = np.linspace(0, iterations, iterations)/iterations
    
    # The timestep 
    delta_t = timesteps[1]-timesteps[0] 

    eigseries = []
    idx = np.argsort(np.linalg.eigvals(initial_matrix))

    for step in timesteps: 
        # Time-evolution operator discretized using the Trotter-Suzuki Approximation
        adiabatic_matrix =  (1-step) * initial_matrix + step * final_matrix
        adiabatic_expm = expm(-1j * delta_t * adiabatic_matrix)
        interpolated_gate = qiskit.circuit.library.UnitaryGate(adiabatic_expm)
        interpolated_gate.name = f'U({step:.3f})'
        circuit.append(interpolated_gate, qubits)

        # Diagnostic Eigenvalue Logging
        tempeig = np.linalg.eigvals(adiabatic_matrix)[idx]
        eigseries.append(tempeig)
    
    #  Setup the Aer Noisy Quantum Simulator 
    sim =  Aer.get_backend('aer_simulator')
    circuit_init = circuit.copy() 
    circuit_init.save_statevector()
    
    # Result processing
    result = sim.run(circuit_init.decompose().decompose(), shots=1024).result() 
    state = np.array(result.get_statevector())
    prob = np.abs(state)**2
    
    #temp 
    qiskit.visualization.plot_histogram(result.get_counts())
    plt.show() 

    result_dict = {list(itertools.product([0, 1], repeat=order))[i] : prob[i] for i in range(prob.size)}
    eigseries = np.array(eigseries).T

    return QUBO_solution(result_dict, eigseries)

