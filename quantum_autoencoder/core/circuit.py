"""
Quantum Autoencoder Circuit Implementation
"""

from typing import Optional, Tuple, List

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import Statevector
from qiskit.primitives import SamplerV2 as Sampler
from qiskit.primitives import EstimatorV2 as Estimator

class QuantumAutoencoder:
    """
    Quantum Autoencoder implementation using Qiskit.
    
    This class implements a quantum autoencoder that can compress quantum states
    into a lower-dimensional representation while preserving essential information.
    Uses V2 primitives for improved hardware compatibility and performance.
    """
    
    def __init__(
        self,
        n_qubits: int,
        n_latent: int,
        reps: int = 5,
        options: Optional[dict] = None
    ):
        """
        Initialize the quantum autoencoder.
        
        Args:
            n_qubits: Number of qubits in the input state
            n_latent: Number of qubits in the latent (compressed) space
            reps: Number of repetitions in the parameterized circuit
            options: Dictionary of options for the primitives
        """
        self.n_qubits = n_qubits
        self.n_latent = n_latent
        self.n_trash = n_qubits - n_latent
        self.reps = reps
        
        # Initialize primitives with options
        self.sampler = Sampler(options=options)
        self.estimator = Estimator(options=options)
        
        # Create the full circuit
        self._create_circuit()
    
    def _create_circuit(self) -> None:
        """Create the quantum autoencoder circuit."""
        # Initialize registers with descriptive names
        qr = QuantumRegister(self.n_qubits + 2 * self.n_trash + 1, "q")
        cr = ClassicalRegister(1, "meas")  # Using 'meas' as per V2 convention
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
    
    def encode(self, state: QuantumCircuit, parameter_values: Optional[np.ndarray] = None) -> QuantumCircuit:
        """
        Encode a quantum state using the trained autoencoder.
        
        Args:
            state: Input quantum state to encode
            parameter_values: Optional parameters for the encoder circuit
            
        Returns:
            Encoded quantum circuit
        """
        qc = QuantumCircuit(self.n_qubits)
        qc = qc.compose(state)
        encoder = RealAmplitudes(self.n_qubits, reps=self.reps)
        
        if parameter_values is not None:
            encoder = encoder.assign_parameters(parameter_values)
            
        qc = qc.compose(encoder)
        return qc
    
    def decode(self, encoded_state: QuantumCircuit, parameter_values: Optional[np.ndarray] = None) -> QuantumCircuit:
        """
        Decode an encoded quantum state.
        
        Args:
            encoded_state: Previously encoded quantum state
            parameter_values: Optional parameters for the decoder circuit
            
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
        if parameter_values is not None:
            encoder = encoder.assign_parameters(parameter_values)
            
        qc = qc.compose(encoder.inverse())
        return qc
    
    def get_fidelity(self, original_state: QuantumCircuit, 
                     reconstructed_state: QuantumCircuit) -> float:
        """
        Calculate the fidelity between original and reconstructed states.
        Uses V2 Estimator for improved accuracy.
        
        Args:
            original_state: Original quantum state
            reconstructed_state: Reconstructed quantum state
            
        Returns:
            Fidelity between the states
        """
        # Create an observable for fidelity measurement
        original_sv = Statevector(original_state)
        observable = original_sv.projector()
        
        # Use Estimator V2 to calculate fidelity
        job = self.estimator.run([(reconstructed_state, observable)])
        result = job.result()
        fidelity = np.sqrt(np.abs(result[0].data.evs[0]))
        
        return float(fidelity.real)
    
    def get_training_circuit(self) -> QuantumCircuit:
        """
        Get the circuit used for training the autoencoder.
        
        Returns:
            Training circuit with SWAP test
        """
        return self.circuit.copy() 