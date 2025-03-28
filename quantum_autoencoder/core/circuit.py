"""
Quantum Autoencoder Circuit Implementation
"""

from typing import Optional, Tuple, List

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import Statevector

class QuantumAutoencoder:
    """
    Quantum Autoencoder implementation using Qiskit.
    
    This class implements a quantum autoencoder that can compress quantum states
    into a lower-dimensional representation while preserving essential information.
    """
    
    def __init__(
        self,
        n_qubits: int,
        n_latent: int,
        reps: int = 5
    ):
        """
        Initialize the quantum autoencoder.
        
        Args:
            n_qubits: Number of qubits in the input state
            n_latent: Number of qubits in the latent (compressed) space
            reps: Number of repetitions in the parameterized circuit
        """
        self.n_qubits = n_qubits
        self.n_latent = n_latent
        self.n_trash = n_qubits - n_latent
        self.reps = reps
        
        # Create the full circuit
        self._create_circuit()
    
    def _create_circuit(self) -> None:
        """Create the quantum autoencoder circuit."""
        # Initialize registers
        qr = QuantumRegister(self.n_qubits + 2 * self.n_trash + 1, "q")  # +1 for auxiliary
        cr = ClassicalRegister(1, "c")
        self.circuit = QuantumCircuit(qr, cr)
        
        # Add encoder
        encoder = RealAmplitudes(self.n_qubits, reps=self.reps)
        self.circuit.compose(encoder, range(self.n_qubits), inplace=True)
        
        # Add barrier for clarity
        self.circuit.barrier()
        
        # Add SWAP test components
        auxiliary_qubit = self.n_qubits + 2 * self.n_trash
        self.circuit.h(auxiliary_qubit)
        
        for i in range(self.n_trash):
            self.circuit.cswap(
                auxiliary_qubit,
                self.n_latent + i,
                self.n_latent + self.n_trash + i
            )
        
        self.circuit.h(auxiliary_qubit)
        self.circuit.measure(auxiliary_qubit, cr[0])
    
    def encode(self, state: QuantumCircuit) -> QuantumCircuit:
        """
        Encode a quantum state using the trained autoencoder.
        
        Args:
            state: Input quantum state to encode
            
        Returns:
            Encoded quantum circuit
        """
        qc = QuantumCircuit(self.n_qubits)
        qc = qc.compose(state)
        encoder = RealAmplitudes(self.n_qubits, reps=self.reps)
        qc = qc.compose(encoder)
        return qc
    
    def decode(self, encoded_state: QuantumCircuit) -> QuantumCircuit:
        """
        Decode an encoded quantum state.
        
        Args:
            encoded_state: Previously encoded quantum state
            
        Returns:
            Decoded quantum circuit
        """
        qc = QuantumCircuit(self.n_qubits)
        qc = qc.compose(encoded_state)
        
        # Reset trash qubits
        for i in range(self.n_trash):
            qc.reset(self.n_latent + i)
        
        # Apply inverse encoder
        encoder = RealAmplitudes(self.n_qubits, reps=self.reps)
        qc = qc.compose(encoder.inverse())
        return qc
    
    def get_fidelity(self, original_state: QuantumCircuit, 
                     reconstructed_state: QuantumCircuit) -> float:
        """
        Calculate the fidelity between original and reconstructed states.
        
        Args:
            original_state: Original quantum state
            reconstructed_state: Reconstructed quantum state
            
        Returns:
            Fidelity between the states
        """
        original_sv = Statevector(original_state).data
        reconstructed_sv = Statevector(reconstructed_state).data
        
        fidelity = np.sqrt(np.abs(np.dot(original_sv.conj(), reconstructed_sv)) ** 2)
        return float(fidelity.real)
    
    def get_training_circuit(self) -> QuantumCircuit:
        """
        Get the circuit used for training the autoencoder.
        
        Returns:
            Training circuit with SWAP test
        """
        return self.circuit.copy() 