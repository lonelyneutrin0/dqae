#############################################################
#                    Required Libraries                     #
#############################################################
import numpy as np 
import qiskit
import qiskit.quantum_info 
from qiskit_aer import Aer
from scipy.linalg import expm
from typing import Dict
from dataclasses import dataclass

#############################################################
#                      Utility Classes                      #
#############################################################
@dataclass 
class Query: 
    """ 
    Query Class to handle algorithm parameters

    Q: QUBO matrix
    iterations: Number of adiabatic steps. Defaults to 10000
    shots: Number of experiment shots on Aer. Defaults to 1024
    """
    Q: np.array
    iterations:int=10000
    shots:int=1024
@dataclass
class Result: 
    """
    Result Class to handle algorithm output 

    prob: Probability of each bit vector 
    eigs: Instantaneous eigenvalue series throughout evolution 
    counts: Experiment count results
    statevector: Final statevector 
    """
    prob: Dict 
    eigs: np.array
    counts: qiskit.result.Counts
    statevector: qiskit.quantum_info.Statevector

#############################################################
#                       DQAE Algorithm                      #
#############################################################
def q_anneal(Query)->Result: 
    """
    Quantum Adiabatic Evolution Algorithm to solve QUBO Problems:  
    -------------------------------------------------------------
    """
    Q = Query.Q

    # Make sure the matrix is square 
    if(Q.shape[0] != Q.shape[1]): raise ValueError('Improper matrix dimensionality!')

    # Convert QUBO to Ising Model
    J = 0.25 * Q 
    h = 0.25 * (np.sum(Q, axis=0) + np.sum(Q, axis=1))
    order = Q.shape[0]

    # Qiskit Circuit Creation 
    qubits = qiskit.QuantumRegister(order)
    circuit = qiskit.QuantumCircuit(qubits)

    # Hamiltonian Formulation 
    pauli_X = np.array([[0, 1], [1, 0]])
    pauli_Z = np.array([[1, 0], [0, -1]])

    Z = np.zeros((order, 2**order, 2**order))
    X = np.zeros((order, 2**order, 2**order))

    for i in range(order):

        # Multipartite Hilbert Operators
        Z[i] = np.kron(
            np.eye(2**i), 
            np.kron(pauli_Z, np.eye(2**(order - 1 - i)))
        )
        X[i] = np.kron(
            np.eye(2**i), 
            np.kron(pauli_X, np.eye(2**(order - 1 - i)))
        )

        # Hadamardize the circuit to ensure equiprobable initial state
        circuit.h(i)

    # Create the Hamiltonian
    x_term = np.einsum('i,ikl->kl', h, X)
    zz_term = np.einsum('ij,ikl,jkl->kl', J, Z, Z)
    z_term = np.sum(Z, axis=0)
    
    initial_matrix = z_term
    final_matrix = zz_term + x_term

    # Adiabatic Setup 
    iterations = Query.iterations
    timesteps = np.linspace(0, 1, iterations)
    delta_t = timesteps[1]-timesteps[0] # Δt
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
    result = sim.run(circuit_init.decompose().decompose(), shots=Query.shots).result() 
    state = np.array(result.get_statevector())
    prob = np.abs(state)**2
    result_dict = {bin(i)[2:].zfill(order) : prob[i] for i in range(prob.size)}
    eigseries = np.array(eigseries).T

    return Result(result_dict, eigseries, result.get_counts(), result.get_statevector())

