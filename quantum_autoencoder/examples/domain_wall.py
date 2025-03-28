"""
Domain Wall State Compression Example

This example demonstrates compressing a domain wall state |00111⟩ from 5 qubits to 3 qubits,
using a 10-qubit circuit for improved fidelity. Uses Qiskit V2 primitives for improved hardware compatibility.
"""

from typing import Optional, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix
from qiskit.visualization import plot_state_city

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

def visualize_state(circuit: QuantumCircuit, title: str, filename: str):
    """
    Visualize a quantum state using city plot with improved readability.
    
    Args:
        circuit: Quantum circuit containing the state
        title: Title for the plot
        filename: Filename to save the plot
    """
    state = Statevector(circuit)
    
    # Create figure with larger size
    plt.figure(figsize=(12, 8))
    
    # Create city plot with customized style
    fig = plot_state_city(state, title=title)
    
    # Save with high DPI for better quality
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def normalize_state(state_vector):
    """Normalize a state vector."""
    norm = np.sqrt(np.sum(np.abs(state_vector) ** 2))
    if norm > 1e-10:  # Avoid division by zero
        return state_vector / norm
    return state_vector

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
            "shots": 8192            # Increased number of shots for better statistics
        }
    
    # Parameters for the domain wall state
    n_qubits = 5      # Size of input state
    n_latent = 3      # Size of compressed state
    n_auxiliary = 2   # Additional auxiliary qubits for improved compression
    n_reps = 12       # Number of repetitions
    
    # Create domain wall state
    domain_wall = create_domain_wall_state(n_qubits)
    print("Created domain wall state |00111⟩")
    
    # Visualize input state
    visualize_state(domain_wall, "Input Domain Wall State |00111⟩", "input_state.png")
    print("Input state visualization saved as 'input_state.png'")
    
    # Create autoencoder with V2 primitive options and auxiliary qubits
    autoencoder = QuantumAutoencoder(
        n_qubits, 
        n_latent,
        n_auxiliary=n_auxiliary,  # Using auxiliary qubits for better compression
        reps=n_reps,
        options=options
    )
    print(f"Created quantum autoencoder: {n_qubits} qubits → {n_latent} qubits")
    print(f"Using {n_reps} repetitions and {n_auxiliary} auxiliary qubits")
    
    # Get training circuit and verify width
    training_circuit = autoencoder.get_training_circuit()
    total_width = training_circuit.num_qubits
    print(f"\nCircuit composition:")
    print(f"- {n_qubits} input qubits")
    print(f"- {n_auxiliary} auxiliary qubits")
    print(f"- {n_qubits - n_latent} trash qubits")
    print(f"- {n_qubits - n_latent} reference qubits")
    print(f"- 1 measurement qubit")
    print(f"Total width: {total_width} qubits")
    
    if total_width > 12:
        raise ValueError(f"Circuit width ({total_width}) exceeds maximum width (12)")
    
    # Train the autoencoder with more iterations
    print("\nTraining autoencoder...")
    optimal_params, final_cost = train_autoencoder(
        training_circuit,
        maxiter=500,  # Increased iterations
        plot_progress=True,
        options=options
    )
    
    # Encode and decode
    print("\nTesting compression...")
    encoded_state = autoencoder.encode(domain_wall, parameter_values=optimal_params)
    decoded_state = autoencoder.decode(encoded_state, parameter_values=optimal_params)
    
    # Get statevectors
    encoded_sv = Statevector(encoded_state)
    decoded_sv = Statevector(decoded_state)
    
    # Create circuits for visualization (without auxiliary qubits)
    encoded_data = normalize_state(encoded_sv.data[:2**n_latent])
    encoded_qc = QuantumCircuit(n_latent)
    encoded_qc.initialize(encoded_data)
    
    decoded_data = normalize_state(decoded_sv.data[:2**n_qubits])
    decoded_qc = QuantumCircuit(n_qubits)
    decoded_qc.initialize(decoded_data)
    
    # Visualize states
    visualize_state(domain_wall, "Input Domain Wall State |00111⟩", "input_state.png")
    print("Input state visualization saved as 'input_state.png'")
    
    visualize_state(encoded_qc, "Encoded State (3 qubits)", "encoded_state.png")
    print("Encoded state visualization saved as 'encoded_state.png'")
    
    visualize_state(decoded_qc, "Reconstructed State (5 qubits)", "decoded_state.png")
    print("Reconstructed state visualization saved as 'decoded_state.png'")
    
    # Calculate fidelity using states without auxiliary qubits
    fidelity = autoencoder.get_fidelity(domain_wall, decoded_qc)
    print(f"\nFidelity between input and output states: {fidelity:.4f}")

if __name__ == "__main__":
    # Example with custom options
    custom_options = {
        "optimization_level": 3,
        "resilience_level": 1,
        "shots": 8192,
        "dynamical_decoupling": {"enable": True}  # Advanced error mitigation
    }
    run_domain_wall_example(options=custom_options) 