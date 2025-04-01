"""
Domain Wall State Compression Example using IBM Quantum Hardware
"""

from typing import Optional, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix, state_fidelity
from qiskit_algorithms.optimizers import SPSA
from qiskit_ibm_runtime import Options

from quantum_autoencoder.core.circuit import QuantumAutoencoder
from quantum_autoencoder.core.training import train_autoencoder
from quantum_autoencoder.ibm_backend import get_ibm_primitives
from quantum_autoencoder.examples.domain_wall import create_domain_wall_state, visualize_state

def run_ibm_domain_wall_example(backend_name: str = "ibm_brisbane", **user_options):
    """
    Run the domain wall state compression example on IBM Quantum hardware.
    
    Args:
        backend_name: Name of the IBM Quantum backend to use
        **user_options: Additional options for execution
    """
    # Create domain wall state circuit
    qc = QuantumCircuit(5)
    qc.x(2)  # |00100> state
    
    # Get primitives with proper options
    sampler, estimator, session = get_ibm_primitives(
        backend_name=backend_name,
        **user_options
    )
    
    # Run the circuit
    print("\nExecuting domain wall state preparation...")
    job = sampler.run([qc])
    result = job.result()
    
    # Calculate state fidelity
    ideal_state = DensityMatrix.from_label('00100')
    measured_state = result.quasi_dists[0]
    fidelity = state_fidelity(ideal_state, measured_state)
    
    print(f"\nResults:")
    print(f"State preparation fidelity: {fidelity:.4f}")
    print(f"Measurement counts: {measured_state}")

if __name__ == "__main__":
    # Run with default options
    run_ibm_domain_wall_example() 