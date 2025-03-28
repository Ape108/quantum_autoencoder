"""
Domain Wall State Compression Example

This example demonstrates compressing a domain wall state |00111⟩ from 5 qubits to 3 qubits.
Uses Qiskit V2 primitives for improved hardware compatibility.
"""

from typing import Optional, Dict, Any
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

def run_domain_wall_example(options: Optional[Dict[str, Any]] = None):
    """
    Run the domain wall compression example.
    
    Args:
        options: Optional dictionary of options for the V2 primitives
    """
    # Default options for V2 primitives
    if options is None:
        options = {
            "optimization_level": 3,  # Maximum optimization
            "resilience_level": 1,    # Basic error mitigation
            "shots": 1024            # Number of shots per circuit
        }
    
    # Parameters
    n_qubits = 5
    n_latent = 3
    
    # Create domain wall state
    domain_wall = create_domain_wall_state(n_qubits)
    print("Created domain wall state |00111⟩")
    
    # Create autoencoder with V2 primitive options
    autoencoder = QuantumAutoencoder(n_qubits, n_latent, options=options)
    print(f"Created quantum autoencoder: {n_qubits} qubits → {n_latent} qubits")
    
    # Get training circuit
    training_circuit = autoencoder.get_training_circuit()
    
    # Train the autoencoder
    print("\nTraining autoencoder...")
    optimal_params, final_cost = train_autoencoder(
        training_circuit,
        maxiter=150,
        plot_progress=True,
        options=options
    )
    
    # Encode and decode
    print("\nTesting compression...")
    encoded_state = autoencoder.encode(domain_wall, parameter_values=optimal_params)
    decoded_state = autoencoder.decode(encoded_state, parameter_values=optimal_params)
    
    # Calculate fidelity
    fidelity = autoencoder.get_fidelity(domain_wall, decoded_state)
    print(f"\nFidelity between input and output states: {fidelity:.4f}")

if __name__ == "__main__":
    # Example with custom options
    custom_options = {
        "optimization_level": 3,
        "resilience_level": 1,
        "shots": 2048,
        "dynamical_decoupling": {"enable": True}  # Advanced error mitigation
    }
    run_domain_wall_example(options=custom_options) 