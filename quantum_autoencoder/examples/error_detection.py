"""
Quantum Error Detection using Quantum Autoencoder

This example demonstrates how to use the quantum autoencoder
for detecting errors in quantum states by comparing fidelities.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import random_statevector, Statevector
from qiskit.quantum_info.operators import Operator
from qiskit.quantum_info import state_fidelity
import matplotlib.pyplot as plt

from quantum_autoencoder.core.circuit import QuantumAutoencoder
from quantum_autoencoder.examples.domain_wall import create_domain_wall_state

def add_noise(state: QuantumCircuit, error_rate: float = 0.1) -> QuantumCircuit:
    """Add random X, Y, Z errors to the quantum state."""
    num_qubits = state.num_qubits
    noisy_state = state.copy()
    
    # Pauli operators
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    paulis = [X, Y, Z]
    
    # Apply random Pauli errors with given probability
    for qubit in range(num_qubits):
        if np.random.random() < error_rate:
            # Choose random Pauli error
            error_type = np.random.choice(['x', 'y', 'z'])
            if error_type == 'x':
                noisy_state.x(qubit)
            elif error_type == 'y':
                noisy_state.y(qubit)
            else:
                noisy_state.z(qubit)
            
    return noisy_state

def test_error_detection(error_rates: list = None) -> None:
    """Test the autoencoder's ability to detect errors."""
    if error_rates is None:
        error_rates = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    
    print("Loading trained autoencoder parameters...")
    best_params = np.load('best_parameters.npy')
    
    # Create quantum autoencoder
    autoencoder = QuantumAutoencoder(5, 3, reps=3)
    
    # Create original state
    original_state = create_domain_wall_state()
    print("\nTesting with domain wall state |00111⟩")
    
    # Test different error rates
    fidelities = []
    for error_rate in error_rates:
        # Create noisy state
        noisy_state = add_noise(original_state, error_rate)
        
        # Encode and decode noisy state
        encoded_state = autoencoder.encode(noisy_state, parameter_values=best_params)
        reconstructed_state = autoencoder.decode(encoded_state, parameter_values=best_params)
        
        # Calculate fidelity
        fidelity = state_fidelity(
            Statevector(original_state),
            Statevector(reconstructed_state)
        )
        fidelities.append(float(fidelity.real))
        
        print(f"\nError rate: {error_rate:.2f}")
        print(f"Reconstruction fidelity: {fidelity:.6f}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(error_rates, fidelities, 'bo-')
    plt.xlabel('Error Rate')
    plt.ylabel('Reconstruction Fidelity')
    plt.title('Quantum Autoencoder Error Detection')
    plt.grid(True)
    plt.savefig('error_detection.png', dpi=300, bbox_inches='tight')
    print("\nError detection plot saved as 'error_detection.png'")

if __name__ == '__main__':
    test_error_detection() 