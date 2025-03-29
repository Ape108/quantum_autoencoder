"""
Domain Wall State Compression Example

This example demonstrates compressing a domain wall state |00111⟩ from 5 qubits to 3 qubits,
using a U-V encoder architecture for improved fidelity.
"""

from typing import Optional, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix
from qiskit.visualization import plot_state_city
from qiskit_algorithms.optimizers import COBYLA, SPSA

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

def visualize_state(state_vector: np.ndarray, title: str, filename: str):
    """
    Visualize a quantum state using city plot with improved readability.
    
    Args:
        state_vector: State vector to visualize
        title: Title for the plot
        filename: Filename to save the plot
    """
    # Create figure with larger size
    plt.figure(figsize=(12, 8))
    
    # Create city plot with customized style
    fig = plot_state_city(state_vector, title=title)
    
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
        options: Optional dictionary of options for the primitives
    """
    # Default options for primitives
    if options is None:
        options = {
            "optimization_level": 3,    # Maximum optimization
            "resilience_level": 1,      # Basic error mitigation
            "shots": 1024,              # Reduced shots for faster training
            "dynamical_decoupling": {   # Advanced dynamical decoupling
                "enable": True,
                "scheme": "XX"
            },
            "seed_simulator": 42        # For reproducibility
        }
    
    # Parameters for the domain wall state
    n_qubits = 5      # Size of input state
    n_latent = 3      # Size of compressed state
    n_reps = 3        # Number of repetitions (start small, increase if needed)
    
    # Create domain wall state
    domain_wall = create_domain_wall_state(n_qubits)
    print("Created domain wall state |00111⟩")
    
    # Visualize input state
    input_sv = Statevector(domain_wall)
    visualize_state(input_sv, "Input Domain Wall State |00111⟩", "input_state.png")
    print("Input state visualization saved as 'input_state.png'")
    
    # Create autoencoder
    autoencoder = QuantumAutoencoder(
        n_qubits=n_qubits,
        n_latent=n_latent,
        reps=n_reps,
        options=options
    )
    print(f"\nCreated quantum autoencoder: {n_qubits} qubits → {n_latent} qubits")
    print(f"Using {n_reps} repetitions")
    print(f"Total parameters: {autoencoder.n_params_u + autoencoder.n_params_v}")
    
    # Use SPSA with tuned parameters
    optimizer = SPSA(
        maxiter=500,                # Moderate number of iterations
        learning_rate=0.15,         # Slightly higher learning rate
        perturbation=0.1,           # Standard perturbation
        resamplings=1,              # Single resampling for speed
        trust_region=True           # Use trust region for stability
    )
    
    print(f"\nTraining with SPSA...")
    
    # Train the autoencoder
    optimal_params, final_cost = train_autoencoder(
        autoencoder=autoencoder,
        input_state=domain_wall,
        maxiter=500,   # Moderate iterations
        n_trials=5,    # Multiple trials
        plot_progress=True,
        optimizer=optimizer,
        options=options
    )
    
    # Test compression with best parameters
    print("\nTesting compression...")
    encoded_state = autoencoder.encode(domain_wall, parameter_values=optimal_params)
    decoded_state = autoencoder.decode(encoded_state, parameter_values=optimal_params)
    
    # Calculate fidelity
    fidelity = 1.0 - final_cost
    print(f"Final fidelity: {fidelity:.4f}")
    
    # Save best parameters
    np.save("best_parameters.npy", optimal_params)
    print("\nBest parameters saved to 'best_parameters.npy'")
    
    # Get statevectors for visualization
    encoded_sv = Statevector(encoded_state)
    decoded_sv = Statevector(decoded_state)
    
    # Visualize states
    visualize_state(input_sv, "Input Domain Wall State |00111⟩", "input_state.png")
    print("Input state visualization saved as 'input_state.png'")
    
    visualize_state(encoded_sv, "Encoded State (3 qubits)", "encoded_state.png")
    print("Encoded state visualization saved as 'encoded_state.png'")
    
    visualize_state(decoded_sv, "Reconstructed State (5 qubits)", "decoded_state.png")
    print("Reconstructed state visualization saved as 'decoded_state.png'")

if __name__ == "__main__":
    run_domain_wall_example() 