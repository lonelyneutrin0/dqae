import numpy as np 
from scipy.linalg import expm

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
    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.matrix = np.zeros((2**num_qubits,2**num_qubits), dtype=complex) # Implement in subclasses
        self.groundstate = np.zeros(2**num_qubits, dtype=complex) # Implement in subclasses
    
    def eigenvalues(self): 
        # Rounding to prevent precision errors
        return np.real(np.round(np.linalg.eigvalsh(self.matrix), 15))
    
    def eigenvectors(self): 
        # Rounding to prevent precision errors
        _, vecs = np.linalg.eigh(self.matrix)
        return np.real(np.round(vecs.T, 15))
    

class H_i(Hamiltonian): 
    def __init__(self, num_qubits: int):
        super().__init__(num_qubits)
        self.groundstate = ket_plus 
        
        for i in range(num_qubits): 
            # Construct the controlled gates for each qubit
            C_X = np.kron(np.eye(2**i), np.kron(pauli_X, np.eye((2**(num_qubits-1-i)))))
            self.matrix -= C_X
            
            # Creating the proposed initial groundstate
            if(i != num_qubits-1): 
                self.groundstate = np.kron(self.groundstate, ket_plus)

class H_f(Hamiltonian): 
    def __init__(self, num_qubits: int, Q: np.array):
        if Q.shape[0] != Q.shape[1] or Q.shape[0] != num_qubits: raise ValueError("The given matrix is not a square matrix!")
        super().__init__(num_qubits)
        
        # Ising Parameters
        self.J = -Q/4
        self.h = 0.25 * np.array([np.sum(Q[i, :]) + np.sum(Q[:, i]) for i in range(num_qubits)])
        self.C = np.sum(Q)/4
        self.groundstate = None # Determined through the computation 

class H_T(Hamiltonian): 
    def __init__(self, num_qubits: int, H_i: Hamiltonian, H_f: Hamiltonian):
        super().__init__(num_qubits)
        self.H_i = H_i 
        self.H_f = H_f 
        self.H_t = H_i 
        self.groundstate = H_i.groundstate
        self.U = None 
    
    def update(self, s: float): 
        self.H_t = (1-s)*H_i + s * H_f
    
    def evolve(self, dT: float, s:float): 
        self.U = expm(-0.5j*s*dT*self.H_f) * expm(-1j * dT * (1-s)*self.H_i) * expm(-0.5j*s*dT*self.H_f)
    
    def fidelity(diag: np.array, evolver: np.array):
        # Should remain close to 1 throughout the evolution
        return np.vdot(diag, evolver)**2  

############################################################
#                   Adiabatic Annealer                     #
############################################################

