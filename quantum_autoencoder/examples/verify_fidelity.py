"""
Verification script for quantum autoencoder fidelity.
This script performs multiple checks to validate the high fidelity results:
1. Statevector comparison
2. Measurement statistics comparison
3. Visual probability distribution comparison
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, state_fidelity
from qiskit.primitives import Sampler
from qiskit.visualization import plot_histogram

from quantum_autoencoder.core.circuit import QuantumAutoencoder
from quantum_autoencoder.examples.domain_wall import create_domain_wall_state

def verify_statevector_fidelity(original_state, reconstructed_state):
    """Verify fidelity using statevector simulation."""
    print("\n=== Statevector Verification ===")
    
    # Calculate fidelity using built-in function
    fid = state_fidelity(original_state, reconstructed_state)
    print(f"Built-in fidelity calculation: {fid:.6f}")
    
    # Manual verification of state vectors
    orig_vec = original_state.data
    recon_vec = reconstructed_state.data
    
    print("\nState vector comparison:")
    print("Original state amplitudes:")
    for i, amp in enumerate(orig_vec):
        if abs(amp) > 1e-6:  # Only show non-zero amplitudes
            print(f"|{i:05b}⟩: {amp:.6f}")
            
    print("\nReconstructed state amplitudes:")
    for i, amp in enumerate(recon_vec):
        if abs(amp) > 1e-6:  # Only show non-zero amplitudes
            print(f"|{i:05b}⟩: {amp:.6f}")
    
    return fid

def verify_measurement_statistics(original_state, reconstructed_state, shots=8192):
    """Verify fidelity using measurement statistics."""
    print("\n=== Measurement Statistics Verification ===")
    
    # Add measurements to both circuits
    orig_qc = original_state.copy()
    orig_qc.measure_all()
    
    recon_qc = reconstructed_state.copy()
    recon_qc.measure_all()
    
    # Execute circuits
    sampler = Sampler(options={'shots': shots})
    orig_counts = sampler.run([orig_qc]).result().quasi_dists[0]
    recon_counts = sampler.run([recon_qc]).result().quasi_dists[0]
    
    # Calculate statistical fidelity
    total_variation = 0
    all_states = set(orig_counts.keys()) | set(recon_counts.keys())
    
    for state in all_states:
        p1 = orig_counts.get(state, 0)
        p2 = recon_counts.get(state, 0)
        total_variation += abs(p1 - p2)
    
    statistical_fidelity = 1 - total_variation/2
    print(f"Statistical fidelity from {shots} measurements: {statistical_fidelity:.6f}")
    
    # Plot histograms
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Convert quasi-distributions to regular dictionaries with binary keys
    orig_hist = {format(int(k), '05b'): v for k, v in orig_counts.items()}
    recon_hist = {format(int(k), '05b'): v for k, v in recon_counts.items()}
    
    # Plot histograms
    ax1.bar(orig_hist.keys(), orig_hist.values())
    ax1.set_title('Original State Measurements')
    ax1.tick_params(axis='x', rotation=45)
    
    ax2.bar(recon_hist.keys(), recon_hist.values())
    ax2.set_title('Reconstructed State Measurements')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('measurement_comparison.png', dpi=300, bbox_inches='tight')
    print("Measurement comparison plot saved as 'measurement_comparison.png'")
    
    return statistical_fidelity

def main():
    """Run verification tests."""
    print("Loading best parameters and creating quantum autoencoder...")
    best_params = np.load('best_parameters.npy')
    
    # Create quantum autoencoder
    autoencoder = QuantumAutoencoder(5, 3, reps=3)
    
    # Create domain wall state
    input_state = create_domain_wall_state()
    print("\nInput state created:", input_state)
    
    # Get reconstructed state
    encoded_state = autoencoder.encode(input_state, parameter_values=best_params)
    reconstructed_state = autoencoder.decode(encoded_state, parameter_values=best_params)
    print("Reconstructed state obtained")
    
    # Run verification tests
    sv_fidelity = verify_statevector_fidelity(Statevector(input_state), Statevector(reconstructed_state))
    stat_fidelity = verify_measurement_statistics(input_state, reconstructed_state)
    
    print("\n=== Final Results ===")
    print(f"Statevector fidelity: {sv_fidelity:.6f}")
    print(f"Statistical fidelity: {stat_fidelity:.6f}")

if __name__ == '__main__':
    main() 