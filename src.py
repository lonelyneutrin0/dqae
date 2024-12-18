import numpy as np 
import matplotlib.pyplot as plt 
from scipy.linalg import expm
from dataclasses import dataclass

############################################################
#                      Fundamentals                        #
############################################################ 

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
        self.groundstate = np.zeros(2**num_qubits, dtype=complex) # Implement in subclasses
        self.matrix = np.zeros((2**num_qubits,2**num_qubits), dtype=complex) 
    
    def eigenvalues(self): 
        # Rounding to prevent precision errors
        return np.round(np.linalg.eigvalsh(self.matrix), 15) 
    
    def eigenvectors(self): 
        # Rounding to prevent precision errors
        _, vecs = np.linalg.eigh(self.matrix)
        return np.round(vecs.T, 15)
    

class H_i(Hamiltonian): 
    def __init__(self, num_qubits: int):
        super().__init__(num_qubits)
        self.groundstate = ket_plus
        self.matrix = np.zeros((2**num_qubits,2**num_qubits), dtype=complex) 
        
        for i in range(num_qubits): 
            # Construct the controlled gates for each qubit
            C_X = np.kron(np.eye(2**i), np.kron(pauli_X, np.eye((2**(num_qubits-1-i)))))
            self.matrix += C_X
            
            # Creating the proposed initial groundstate
            if(i != num_qubits-1): 
                self.groundstate = np.kron(self.groundstate, ket_plus)
    
    def operate(self, vec: np.array): 
        return self.matrix @ vec

class H_f(Hamiltonian): 
    def __init__(self, num_qubits: int, Q: np.array):
        if Q.shape[0] != Q.shape[1] or Q.shape[0] != num_qubits: raise ValueError("The given matrix is not a square matrix!")
        super().__init__(num_qubits)
        
        # Ising Parameters
        self.J = Q/4
        self.h = 0.25 * np.array([np.sum(Q[i, :]) + np.sum(Q[:, i]) for i in range(num_qubits)])
        self.C = np.sum(Q)/4
        self.groundstate = None # Determined through the computation 
        
        for i in range(num_qubits): 
            # Construct the controlled gates for each qubit
            C_X = np.kron(np.eye(2**i), np.kron(pauli_X, np.eye((2**(num_qubits-1-i)))))
            C_Z_i = np.kron(np.eye(2**i), np.kron(pauli_Z, np.eye((2**(num_qubits-1-i)))))
            self.matrix -= self.h[i] * C_X
            
            for j in range(num_qubits): 
                C_Z_j = np.kron(np.eye(2**j), np.kron(pauli_Z, np.eye((2**(num_qubits-1-j)))))
                self.matrix +=  self.J[i,j] * C_Z_i @ C_Z_j

class H_T(Hamiltonian): 
    def __init__(self, H_i: Hamiltonian, H_f: Hamiltonian, dT, T):
        
        super().__init__(H_i.num_qubits)
        self.H_i = H_i 
        self.H_f = H_f 
        self.matrix = H_i.matrix 
        self.groundstate = H_i.groundstate 
        self.dT = dT
        self.T = T
    
    def update(self, s: float): 
        self.matrix = (1-s)*self.H_i.matrix + s * self.H_f.matrix   
    
    def evolve(self, s:float): 
        # return expm(-0.5j*s*self.dT*self.H_f.matrix) @ expm(-1j * self.dT * (1-s)*self.H_i.matrix) @ expm(-0.5j*s*self.dT*self.H_f.matrix)
        return expm(-1j*self.matrix*self.dT)
    
    def fidelity(self, eig):
        # Should remain close to 1 throughout the evolution
        return np.abs(np.vdot(eig, self.groundstate))**2  

@dataclass
class quantumOutput: 
    eigs: np.array
    groundstate: np.array
    norms: np.array
    fidelities: np.array

############################################################
#                   Adiabatic Annealer                     #
############################################################ 

def annealer(H_problem: H_f):
    # Create the initial Hamiltonian 
    H_initial = H_i(H_problem.num_qubits)

    print(f'H_i Frob: {np.linalg.norm(H_initial.matrix)}')
    print(f'H_f Frob: {np.linalg.norm(H_problem.matrix)}')

    # Determine the gap size 
    energies = np.sort(H_initial.eigenvalues())
    min_gap = energies[1] - energies[0]
    T = (int) (1/min_gap * 1e5)
    
    # 0, Δt/T, 2Δt/T ..... nΔt/T (1)
    timesteps = np.arange(T)/T
    Δt = timesteps[1]
    fidelities = np.zeros(T)
    norms = np.zeros(T)
    crossings = []

    H_adiabatic = H_T(H_i=H_initial, H_f=H_problem, dT=Δt, T=T)
    
    for timestep in range(timesteps.size): 
        # Update the adiabatic Hamiltonian
        H_adiabatic.update(timesteps[timestep])
        norms[timestep] = np.linalg.norm(H_adiabatic.matrix)
        
        # Evolve the state 
        H_adiabatic.groundstate = H_adiabatic.evolve(timesteps[timestep]) @ H_adiabatic.groundstate 
        
        # Determine the groundstate eigenvector 
        sorted_eigs = H_adiabatic.eigenvectors()[np.argsort(H_adiabatic.eigenvalues())]
        diag_eig = sorted_eigs[0]
        crossings.append(H_adiabatic.eigenvalues())
        # print(np.real(H_adiabatic.groundstate.conj().T @ H_adiabatic.matrix @ H_adiabatic.groundstate) + H_problem.C)
        
        #Evaluate the fidelity 
        fidelities[timestep] = H_adiabatic.fidelity(diag_eig)
    
    output = { 
        'eigs': np.array(crossings),
        'groundstate': H_adiabatic.groundstate,
        'norms': norms, 
        'fidelities': fidelities, 
    }
    n_hadamard = np.copy(hadamard)
    for i in range(H_problem.num_qubits-1): 
        n_hadamard = np.kron(n_hadamard, hadamard)
    
    H_adiabatic.groundstate = n_hadamard @ H_adiabatic.groundstate
    print(np.round(H_adiabatic.groundstate, 2))
    p = np.round(np.abs(H_adiabatic.groundstate)**2, 2)
    print(p)
    # print(f'|{bin(np.argmax(p))[2:].zfill(H_problem.num_qubits)}⟩')
    return quantumOutput(**output)
    


