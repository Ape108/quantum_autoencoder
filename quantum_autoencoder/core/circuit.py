"""
Quantum Autoencoder Circuit Implementation
"""

from typing import Optional, Tuple, List

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.primitives import Sampler
from qiskit.primitives import Estimator

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
        n_auxiliary: int = 2,
        reps: int = 5,  # Match notebook's repetitions
        options: Optional[dict] = None
    ):
        """
        Initialize the quantum autoencoder.
        
        Args:
            n_qubits: Number of qubits in the input state
            n_latent: Number of qubits in the latent (compressed) space
            n_auxiliary: Number of auxiliary qubits
            reps: Number of repetitions in the parameterized circuit
            options: Dictionary of options for the primitives
        """
        self.n_qubits = n_qubits
        self.n_latent = n_latent
        self.n_trash = n_qubits - n_latent
        self.n_auxiliary = n_auxiliary
        self.reps = reps
        
        # Initialize primitives with options
        self.sampler = Sampler(options=options)
        self.estimator = Estimator(options=options)
        
        # Create the full circuit
        self._create_circuit()
    
    def _create_circuit(self) -> None:
        """Create the quantum autoencoder circuit with RealAmplitudes ansatz."""
        # Initialize registers with descriptive names
        total_qubits = self.n_qubits + self.n_auxiliary + self.n_trash + 1
        qr = QuantumRegister(total_qubits, "q")
        cr = ClassicalRegister(1, "meas")  # Using 'meas' as per V2 convention
        self.circuit = QuantumCircuit(qr, cr)
        
        # Add encoder with RealAmplitudes ansatz
        encoder = RealAmplitudes(
            self.n_qubits + self.n_auxiliary,
            reps=self.reps,
            insert_barriers=True  # Help with optimization
        )
        
        # Add the encoder
        self.circuit.compose(encoder, range(self.n_qubits + self.n_auxiliary), inplace=True)
        
        # Add barrier for clarity
        self.circuit.barrier()
        
        # Add SWAP test components
        auxiliary_qubit = self.n_qubits + self.n_auxiliary + self.n_trash
        self.circuit.h(auxiliary_qubit)
        
        # SWAP test between trash and reference qubits
        for i in range(self.n_trash):
            self.circuit.cswap(
                auxiliary_qubit,
                self.n_latent + i,
                self.n_qubits + self.n_auxiliary + i
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
        qc = QuantumCircuit(self.n_qubits + self.n_auxiliary)
        qc = qc.compose(state)
            
        # Create encoder with RealAmplitudes
        encoder = RealAmplitudes(
            self.n_qubits + self.n_auxiliary,
            reps=self.reps,
            insert_barriers=True
        )
        
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
        qc = QuantumCircuit(self.n_qubits + self.n_auxiliary)
        qc = qc.compose(encoded_state)
        
        # Reset trash qubits
        for i in range(self.n_trash):
            qc.reset(self.n_latent + i)
            
        # Create encoder with RealAmplitudes
        encoder = RealAmplitudes(
            self.n_qubits + self.n_auxiliary,
            reps=self.reps,
            insert_barriers=True
        )
        
        if parameter_values is not None:
            encoder = encoder.assign_parameters(parameter_values)
            
        qc = qc.compose(encoder.inverse())
        
        # Trace out auxiliary qubits
        for i in range(self.n_qubits, self.n_qubits + self.n_auxiliary):
            qc.reset(i)
            
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
        operator = original_sv.to_operator()
        observable = SparsePauliOp.from_operator(operator)
        
        # Ensure reconstructed state has same number of qubits as original
        if reconstructed_state.num_qubits != original_state.num_qubits:
            raise ValueError(
                f"Number of qubits in reconstructed state ({reconstructed_state.num_qubits}) "
                f"does not match original state ({original_state.num_qubits})"
            )
        
        # Use Estimator to calculate fidelity
        job = self.estimator.run(
            circuits=[reconstructed_state],
            observables=[observable],
            parameter_values=None
        )
        result = job.result()
        fidelity = np.sqrt(np.abs(result.values[0]))
        
        return float(fidelity.real)
    
    def get_training_circuit(self) -> QuantumCircuit:
        """
        Get the circuit used for training the autoencoder.
        
        Returns:
            Training circuit with SWAP test
        """
        return self.circuit.copy() 