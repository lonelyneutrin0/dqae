import numpy as np 
import qiskit
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram, plot_bloch_multivector
from scipy.linalg import expm

# QUBO Formulation 
order = 3
Q = np.random.uniform(low=-1.0, high=1.0, size=(order, order))
Q *= 1e6
J = 0.25 * Q 
h = 0.25 * (np.sum(Q, axis=0) + np.sum(Q, axis=1))

# Initial Hamiltonian Formulation 
pauli_X = np.array([[0, 1], [1, 0]])
pauli_Z = np.array([[1, 0], [0, -1]])

initial_matrix = np.zeros(shape=(2**order, 2**order))

for i in range(order): 
    for j in range(order): 
        Z_i = np.kron(np.eye(2**i), np.kron(pauli_Z, np.eye((2**(order-1-i)))))
        Z_j = np.kron(np.eye(2**j), np.kron(pauli_Z, np.eye((2**(order-1-j)))))
        initial_matrix += J[i, j] * (Z_i @ Z_j)

# Final Hamiltonian Formulation 

final_matrix = np.copy(initial_matrix)

for i in range(order): 
     final_matrix += h[i] * np.kron(np.eye(2**i), np.kron(pauli_X, np.eye((2**(order-1-i)))))

# Circuit Formulation 
qubits = qiskit.QuantumRegister(order)
circuit = qiskit.QuantumCircuit(qubits)

for i in qubits: 
    circuit.h(i)

# Adiabatic Setup 
iterations = 1000
timesteps = np.linspace(0, iterations, iterations)/iterations
delta_t = timesteps[1]-timesteps[0]

for step in timesteps: 
     adiabatic_matrix = (1-step) * initial_matrix + step * final_matrix
     adiabatic_expm = expm(-1j * delta_t * adiabatic_matrix)
     interpolated_gate = qiskit.circuit.library.UnitaryGate(adiabatic_expm)
     interpolated_gate.name = f'U({step:.3f})'
     circuit.append(interpolated_gate, qubits)

sim =  Aer.get_backend('aer_simulator')
circuit_init = circuit.copy() 
circuit_init.save_statevector()

result = sim.run(circuit_init.decompose().decompose(), shots=1024).result() 
plot_histogram(result.get_counts())

output_array = -np.array([-1, -1, -1])
print(J*4e-6)
print((output_array.T @ J @ output_array + np.dot(h, output_array) + np.sum(J))*1e-6)