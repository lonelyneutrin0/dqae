import numpy as np 

# Fundamentals 

# Standard Computational Basis 
ket_zero = np.array([1, 0], dtype=complex)
ket_one = np.array([0, 1], dtype=complex)

# Hadamard Basis
ket_plus = 1/np.sqrt(2) * np.array([1, 1], dtype=complex)
ket_minus = 1/np.sqrt(2) * np.array([1, -1], dtype=complex)

# Quantum Gates 
pauli_X = np.array([[0, 1], [1, 0]], dtype=complex)
pauli_Z = np.array([[1, 0], [0, -1]], dtype=complex)
hadamard = 1/np.sqrt(2) * np.array([[1,1], [1,-1]], dtype=complex)

class Hamiltonian: 
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.matrix = np.zeros((2**num_qubits,2**num_qubits), dtype=complex) # Implement in subclasses
        self.groundstate = np.zeros(2**num_qubits, dtype=complex) # Implement in subclasses
    
    def operate(): 
        pass
    
    def eigenvalues(self): 
        return np.real(np.round(np.linalg.eigvalsh(self.matrix), 15))
    
    def eigenvectors(self): 
        _, vecs = np.linalg.eigh(self.matrix)
        return np.real(np.round(vecs.T, 15))
    

class H_i(Hamiltonian): 
    def __init__(self, num_qubits):
        super().__init__(num_qubits)
        self.groundstate = ket_plus 

        for i in range(num_qubits): 
            # Construct the controlled gates for each qubit
            C_X = np.kron(np.eye(2**i), np.kron(pauli_X, np.eye((2**(num_qubits-1-i)))))
            self.matrix -= C_X
            if(i != num_qubits-1): 
                self.groundstate = np.kron(self.groundstate, ket_plus)

e = H_i(5)
