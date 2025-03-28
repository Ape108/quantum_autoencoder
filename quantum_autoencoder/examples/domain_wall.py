"""
Domain Wall State Compression Example

This example demonstrates compressing a domain wall state |00111⟩ from 5 qubits to 3 qubits.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from quantum_autoencoder.core.circuit import QuantumAutoencoder
from quantum_autoencoder.core.training import train_autoencoder

def create_domain_wall_state(n_qubits: int = 5) -> QuantumCircuit:
    """
    Create a domain wall state |00111⟩.
    
    Args:
        n_qubits: Number of qubits (default: 5)
        
    Returns:
        Quantum circuit implementing the domain wall state
    """
    qc = QuantumCircuit(n_qubits)
    # Apply X gates to the last half of qubits
    for i in range(n_qubits // 2, n_qubits):
        qc.x(i)
    return qc

def run_domain_wall_example():
    """Run the domain wall compression example."""
    # Parameters
    n_qubits = 5
    n_latent = 3
    
    # Create domain wall state
    domain_wall = create_domain_wall_state(n_qubits)
    print("Created domain wall state |00111⟩")
    
    # Create autoencoder
    autoencoder = QuantumAutoencoder(n_qubits, n_latent)
    print(f"Created quantum autoencoder: {n_qubits} qubits → {n_latent} qubits")
    
    # Get training circuit
    training_circuit = autoencoder.get_training_circuit()
    
    # Train the autoencoder
    print("\nTraining autoencoder...")
    optimal_params, final_cost = train_autoencoder(
        training_circuit,
        maxiter=150,
        plot_progress=True
    )
    
    # Encode and decode
    print("\nTesting compression...")
    encoded_state = autoencoder.encode(domain_wall)
    decoded_state = autoencoder.decode(encoded_state)
    
    # Calculate fidelity
    fidelity = autoencoder.get_fidelity(domain_wall, decoded_state)
    print(f"\nFidelity between input and output states: {fidelity:.4f}")

if __name__ == "__main__":
    run_domain_wall_example() 