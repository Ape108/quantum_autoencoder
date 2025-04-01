"""
Quantum Autoencoder Circuit Implementation using U-V encoder architecture
"""

from typing import Optional, Tuple, List

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector
from qiskit.primitives import Sampler
from qiskit.primitives import Estimator

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
        
        # Total number of parameters
        self.n_parameters = self.n_params_u + self.n_params_v
        
        # Create the full circuit
        self._create_circuit()
    
    def _create_encoder(self, name: str) -> QuantumCircuit:
        """Create an encoder circuit optimized for graph structure."""
        qc = QuantumCircuit(self.n_qubits, name=f'encoder_{name}')
        
        # Calculate number of parameters
        n_layers = self.reps
        n_qubits = self.n_qubits
        
        # Create parameters with unique names
        # Each layer has:
        # - Initial rotation layer: 2 gates per qubit (RY, RZ)
        # - Entanglement layer: 2 rotations per pair of qubits
        # Final layer (outside loop):
        # - 2 gates per qubit (RY, RZ)
        n_params_per_layer = (2 * n_qubits) + (2 * n_qubits * (n_qubits - 1) // 2)
        n_final_params = 2 * n_qubits
        total_params = n_layers * n_params_per_layer + n_final_params
        
        params = [Parameter(f'{name}_{i}') for i in range(total_params)]
        param_index = 0
        
        # Build the circuit layer by layer
        for layer in range(n_layers):
            # Initial rotation layer - encode table properties
            for qubit in range(n_qubits):
                qc.ry(params[param_index], qubit)  # Encode table size
                param_index += 1
                qc.rz(params[param_index], qubit)  # Encode query frequency
                param_index += 1
            
            # Entanglement layer - encode relationships
            for i in range(n_qubits):
                for j in range(i + 1, n_qubits):
                    # Create entanglement to represent relationship
                    qc.cx(i, j)
                    qc.rz(params[param_index], j)  # Encode relationship strength
                    param_index += 1
                    qc.cx(j, i)
                    qc.rz(params[param_index], i)  # Encode relationship type
                    param_index += 1
            
            # Add barrier for better visualization
            qc.barrier()
        
        # Final rotation layer - prepare for measurement
        for qubit in range(n_qubits):
            qc.ry(params[param_index], qubit)
            param_index += 1
            qc.rz(params[param_index], qubit)
            param_index += 1
        
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
        
        # Add measurement on the first trash qubit (qubit after the latent space)
        self.circuit.measure(0, cr[0])  # Measure first qubit as trash
    
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
    
    def compress_entry(self, features: np.ndarray, parameters: np.ndarray = None) -> Statevector:
        """
        Compress a single data entry using the trained autoencoder.
        
        Args:
            features: Input features to compress
            parameters: Optional parameters for the encoder circuit
            
        Returns:
            Compressed quantum state
        """
        # Normalize input features
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
            
        # Create input circuit
        qc = QuantumCircuit(self.n_qubits)
        qc.initialize(features, range(self.n_qubits))
        
        # Apply encoder with parameters if provided
        if parameters is not None:
            params_u = parameters[:len(self.encoder_u.parameters)]
            qc.compose(self.encoder_u.assign_parameters(params_u), inplace=True)
            
        # Get statevector of latent space
        sv = Statevector.from_instruction(qc)
        latent_data = sv.data[:2**self.n_latent]
        
        # Normalize latent state
        latent_norm = np.linalg.norm(latent_data)
        if latent_norm > 0:
            latent_data = latent_data / latent_norm
            
        return Statevector(latent_data)
        
    def decode_features(self, latent_state: Statevector, parameters: np.ndarray = None) -> np.ndarray:
        """
        Decode compressed features back to original space.
        
        Args:
            latent_state: Compressed quantum state
            parameters: Optional parameters for the decoder circuit
            
        Returns:
            Decoded features
        """
        # Normalize latent state
        latent_data = latent_state.data
        norm = np.linalg.norm(latent_data)
        if norm > 0:
            latent_data = latent_data / norm
            
        # Create circuit with latent state
        qc = QuantumCircuit(self.n_qubits)
        qc.initialize(latent_data, range(self.n_latent))
        
        # Reset trash qubits to |0>
        for i in range(self.n_latent, self.n_qubits):
            qc.reset(i)
            
        # Apply decoder with parameters if provided
        if parameters is not None:
            params_v = parameters[len(self.encoder_u.parameters):]
            qc.compose(self.encoder_v.assign_parameters(params_v).inverse(), inplace=True)
            
        # Get final state
        sv = Statevector.from_instruction(qc)
        return sv.data 