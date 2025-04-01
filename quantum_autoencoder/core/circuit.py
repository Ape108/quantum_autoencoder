"""
Quantum Autoencoder Circuit Implementation using U-V encoder architecture
"""

from typing import Optional, Tuple, List

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector
from qiskit.primitives import SamplerV2 as Sampler
from qiskit.primitives import EstimatorV2 as Estimator

class QuantumAutoencoder:
    """
    Quantum Autoencoder implementation using Qiskit.
    
    This class implements a quantum autoencoder that can compress quantum states
    into a lower-dimensional representation while preserving essential information.
    Uses U-V encoder architecture for better compression.
    """
    
    def __init__(
        self,
        n_qubits: int,
        n_latent: int,
        reps: int = 2,
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
        if options is None:
            options = {
                "optimization_level": 3,
                "resilience_level": 1,
                "shots": 1024,  # Reduced shots for faster training
                "dynamical_decoupling": {"enable": True}
            }
        self.sampler = Sampler(options=options)
        self.estimator = Estimator(options=options)
        
        # Create encoders with distinct parameter names
        self.encoder_u = self._create_encoder('u')
        self.encoder_v = self._create_encoder('v')
        
        # Number of parameters for each encoder
        self.n_params_u = len(self.encoder_u.parameters)
        self.n_params_v = len(self.encoder_v.parameters)
        
        # Create the full circuit
        self._create_circuit()
    
    def _create_encoder(self, name: str) -> QuantumCircuit:
        """Create an encoder circuit with unique parameter names."""
        qc = QuantumCircuit(self.n_qubits, name=f'encoder_{name}')
        
        # Calculate number of parameters
        n_layers = self.reps
        n_qubits = self.n_qubits
        
        # Create parameters with unique names
        params = [Parameter(f'{name}_{i}') for i in range(n_layers * n_qubits)]
        param_index = 0
        
        # Build the circuit layer by layer
        for layer in range(n_layers):
            # Rotation layer (only RY gates)
            for qubit in range(n_qubits):
                qc.ry(params[param_index], qubit)
                param_index += 1
            
            # Entanglement layer (linear)
            for i in range(0, n_qubits - 1, 2):
                qc.cx(i, i + 1)
            for i in range(1, n_qubits - 1, 2):
                qc.cx(i, i + 1)
        
        return qc
    
    def _create_circuit(self) -> None:
        """Create the quantum autoencoder circuit."""
        # Initialize registers
        qr = QuantumRegister(self.n_qubits, "q")
        cr = ClassicalRegister(1, "meas")
        self.circuit = QuantumCircuit(qr, cr)
        
        # Add encoder U
        self.circuit.compose(self.encoder_u, range(self.n_qubits), inplace=True)
        
        # Add barrier for clarity
        self.circuit.barrier()
        
        # Add encoder V
        self.circuit.compose(self.encoder_v, range(self.n_qubits), inplace=True)
        
        # Add measurement
        self.circuit.measure(self.n_latent, cr[0])
    
    def encode(self, state: QuantumCircuit, parameter_values: Optional[np.ndarray] = None) -> QuantumCircuit:
        """
        Encode a quantum state using the trained autoencoder.
        
        Args:
            state: Input quantum state to encode
            parameter_values: Optional parameters for the encoder circuit
            
        Returns:
            Encoded quantum circuit
        """
        if parameter_values is not None:
            # Split parameters between U and V encoders
            params_u = parameter_values[:self.n_params_u]
            params_v = parameter_values[self.n_params_u:]
        else:
            params_u = np.zeros(self.n_params_u)
            params_v = np.zeros(self.n_params_v)
        
        # Create encoding circuit
        qc = QuantumCircuit(self.n_qubits)
        qc = qc.compose(state)
        
        # Apply U encoder
        u_encoder = self.encoder_u.assign_parameters(params_u)
        qc = qc.compose(u_encoder)
        
        # Apply V encoder
        v_encoder = self.encoder_v.assign_parameters(params_v)
        qc = qc.compose(v_encoder)
        
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
        if parameter_values is not None:
            # Split parameters between U and V encoders
            params_u = parameter_values[:self.n_params_u]
            params_v = parameter_values[self.n_params_u:]
        else:
            params_u = np.zeros(self.n_params_u)
            params_v = np.zeros(self.n_params_v)
        
        # Create decoding circuit
        qc = QuantumCircuit(self.n_qubits)
        qc = qc.compose(encoded_state)
        
        # Apply inverse V encoder
        v_decoder = self.encoder_v.assign_parameters(params_v).inverse()
        qc = qc.compose(v_decoder)
        
        # Reset trash qubits
        for i in range(self.n_trash):
            qc.reset(self.n_latent + i)
        
        # Apply inverse U encoder
        u_decoder = self.encoder_u.assign_parameters(params_u).inverse()
        qc = qc.compose(u_decoder)
        
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
        # Get statevectors
        sv_original = Statevector(original_state)
        sv_reconstructed = Statevector(reconstructed_state)
        
        # Calculate fidelity directly
        fidelity = np.abs(sv_original.inner(sv_reconstructed)) ** 2
        
        return float(fidelity.real)
    
    def get_training_circuit(self) -> QuantumCircuit:
        """
        Get the circuit used for training the autoencoder.
        
        Returns:
            Training circuit
        """
        return self.circuit.copy() 