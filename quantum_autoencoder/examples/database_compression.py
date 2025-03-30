"""
Database Compression using Quantum Autoencoder

This example demonstrates how to use the quantum autoencoder for compressing
database entries by converting them to quantum states and back.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import EfficientSU2

from quantum_autoencoder.core.circuit import QuantumAutoencoder
from quantum_autoencoder.core.training import train_autoencoder

class DatabaseCompressor:
    """
    A class for compressing database entries using quantum autoencoder.
    """
    
    def __init__(
        self,
        n_features: int,
        n_latent: int,
        feature_encoding: str = "amplitude",
        reps: int = 3
    ):
        """
        Initialize the database compressor.
        
        Args:
            n_features: Number of features in each database entry
            n_latent: Number of qubits in latent space
            feature_encoding: Encoding method ('amplitude', 'basis', or 'angle')
            reps: Number of repetitions in the autoencoder circuit
        """
        self.n_features = n_features
        self.n_latent = n_latent
        self.feature_encoding = feature_encoding
        
        # Calculate required number of qubits
        if feature_encoding == "amplitude":
            self.n_qubits = int(np.ceil(np.log2(n_features)))
        else:
            self.n_qubits = n_features
            
        # Initialize autoencoder
        self.autoencoder = QuantumAutoencoder(
            n_qubits=self.n_qubits,
            n_latent=n_latent,
            reps=reps
        )
        
        # Store optimal parameters after training
        self.optimal_params = None
        
    def encode_features(self, features: np.ndarray) -> QuantumCircuit:
        """
        Convert classical features to quantum state.
        
        Args:
            features: Array of classical features
            
        Returns:
            Quantum circuit representing the features
        """
        qc = QuantumCircuit(self.n_qubits)
        
        if self.feature_encoding == "amplitude":
            # Normalize features
            norm = np.sqrt(np.sum(np.abs(features) ** 2))
            if norm > 1e-10:
                features = features / norm
                
            # Create state preparation circuit
            qc.initialize(features, range(self.n_qubits))
            
        elif self.feature_encoding == "basis":
            # Convert features to binary and apply X gates
            for i, feature in enumerate(features):
                if feature > 0.5:  # Threshold for binary conversion
                    qc.x(i)
                    
        else:  # angle encoding
            # Use angle encoding for continuous features
            for i, feature in enumerate(features):
                qc.ry(feature * np.pi, i)
                
        return qc
    
    def decode_features(self, quantum_state: QuantumCircuit) -> np.ndarray:
        """
        Convert quantum state back to classical features.
        
        Args:
            quantum_state: Quantum circuit representing the state
            
        Returns:
            Array of classical features
        """
        # Get statevector
        sv = Statevector(quantum_state)
        
        if self.feature_encoding == "amplitude":
            # Return amplitudes directly
            return np.array(sv.data)
            
        elif self.feature_encoding == "basis":
            # Measure in computational basis
            counts = sv.probabilities_dict()
            features = np.zeros(self.n_features)
            for bitstring, prob in counts.items():
                idx = int(bitstring, 2)
                if idx < self.n_features:
                    features[idx] = prob
            return features
            
        else:  # angle encoding
            # Extract angles from statevector
            features = np.zeros(self.n_features)
            for i in range(self.n_features):
                # Estimate angle from statevector components
                features[i] = np.angle(sv.data[i]) / np.pi
            return features
    
    def compress_entry(self, entry: np.ndarray) -> Tuple[QuantumCircuit, np.ndarray]:
        """
        Compress a single database entry.
        
        Args:
            entry: Array of features for one database entry
            
        Returns:
            Tuple of (compressed quantum state, reconstruction error)
        """
        # Convert to quantum state
        qc = self.encode_features(entry)
        
        # Compress using autoencoder
        encoded = self.autoencoder.encode(qc, self.optimal_params)
        decoded = self.autoencoder.decode(encoded, self.optimal_params)
        
        # Calculate reconstruction error
        original_sv = Statevector(qc)
        reconstructed_sv = Statevector(decoded)
        error = 1 - np.abs(original_sv.inner(reconstructed_sv)) ** 2
        
        return encoded, error
    
    def train(self, training_data: List[np.ndarray], **kwargs) -> Dict[str, Any]:
        """
        Train the autoencoder on a dataset.
        
        Args:
            training_data: List of feature arrays
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary of training results
        """
        # Convert training data to quantum states
        training_states = [self.encode_features(entry) for entry in training_data]
        
        # Train on first state (can be extended to multiple states)
        optimal_params, final_cost = train_autoencoder(
            self.autoencoder,
            training_states[0],
            **kwargs
        )
        
        self.optimal_params = optimal_params
        
        # Calculate average reconstruction error
        errors = []
        for entry in training_data:
            _, error = self.compress_entry(entry)
            errors.append(error)
            
        return {
            "final_cost": final_cost,
            "fidelity": 1 - final_cost,
            "avg_reconstruction_error": np.mean(errors),
            "max_reconstruction_error": np.max(errors)
        }

def example_usage():
    """Example usage of the DatabaseCompressor."""
    # Generate sample database entries
    n_entries = 100
    n_features = 8
    training_data = [np.random.rand(n_features) for _ in range(n_entries)]
    
    # Initialize compressor
    compressor = DatabaseCompressor(
        n_features=n_features,
        n_latent=4,  # 50% compression
        feature_encoding="amplitude"
    )
    
    # Train the compressor
    results = compressor.train(
        training_data,
        maxiter=500,
        n_trials=5
    )
    
    print("Training Results:")
    print(f"Final fidelity: {results['fidelity']:.4f}")
    print(f"Average reconstruction error: {results['avg_reconstruction_error']:.4f}")
    
    # Test compression on a new entry
    test_entry = np.random.rand(n_features)
    compressed_state, error = compressor.compress_entry(test_entry)
    
    print(f"\nTest Entry Compression:")
    print(f"Reconstruction error: {error:.4f}")
    
    # Reconstruct the entry
    reconstructed = compressor.decode_features(compressed_state)
    print(f"Original vs Reconstructed:")
    print(f"Original: {test_entry}")
    print(f"Reconstructed: {reconstructed}")

if __name__ == "__main__":
    example_usage() 