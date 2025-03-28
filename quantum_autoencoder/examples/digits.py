"""
Digit Compression Example

This example demonstrates compressing noisy images of digits 0 and 1 using a quantum autoencoder.
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.circuit.library import RawFeatureVector
from qiskit_machine_learning.utils import algorithm_globals

from quantum_autoencoder.core.circuit import QuantumAutoencoder
from quantum_autoencoder.core.training import train_autoencoder

def create_digit_dataset(num_samples: int = 2, draw: bool = True) -> tuple:
    """
    Create a dataset of noisy 0 and 1 digits.
    
    Args:
        num_samples: Number of samples per digit
        draw: Whether to display the generated digits
        
    Returns:
        Tuple of (images, labels)
    """
    def zero_idx(j: int, i: int) -> list:
        """Get pixel indices for zero digit."""
        return [
            [i, j],
            [i - 1, j - 1],
            [i - 1, j + 1],
            [i - 2, j - 1],
            [i - 2, j + 1],
            [i - 3, j - 1],
            [i - 3, j + 1],
            [i - 4, j - 1],
            [i - 4, j + 1],
            [i - 5, j],
        ]

    def one_idx(i: int, j: int) -> list:
        """Get pixel indices for one digit."""
        return [[i, j - 1], [i, j - 2], [i, j - 3], 
                [i, j - 4], [i, j - 5], [i - 1, j - 4], [i, j]]

    train_images = []
    train_labels = []
    
    # Generate ones
    for _ in range(num_samples // 2):
        # Add background noise
        empty = np.array([algorithm_globals.random.uniform(0, 0.1) 
                         for _ in range(32)]).reshape(8, 4)
        
        # Add digit pixels
        for i, j in one_idx(2, 6):
            empty[j][i] = algorithm_globals.random.uniform(0.9, 1)
        
        train_images.append(empty)
        train_labels.append(1)
        
        if draw:
            plt.imshow(empty)
            plt.title("One")
            plt.show()
    
    # Generate zeros
    for _ in range(num_samples // 2):
        empty = np.array([algorithm_globals.random.uniform(0, 0.1) 
                         for _ in range(32)]).reshape(8, 4)
        
        for k, j in zero_idx(2, 6):
            empty[k][j] = algorithm_globals.random.uniform(0.9, 1)
        
        train_images.append(empty)
        train_labels.append(0)
        
        if draw:
            plt.imshow(empty)
            plt.title("Zero")
            plt.show()
    
    # Normalize
    train_images = np.array(train_images)
    train_images = train_images.reshape(len(train_images), 32)
    
    for i in range(len(train_images)):
        sum_sq = np.sum(train_images[i] ** 2)
        train_images[i] = train_images[i] / np.sqrt(sum_sq)
    
    return train_images, np.array(train_labels)

def run_digits_example():
    """Run the digit compression example."""
    # Parameters
    n_qubits = 5  # 32 amplitudes = 2^5
    n_latent = 3
    
    # Create dataset
    print("Generating digit dataset...")
    images, labels = create_digit_dataset(num_samples=4, draw=True)
    
    # Create feature map
    feature_map = RawFeatureVector(2 ** n_qubits)
    
    # Create autoencoder
    autoencoder = QuantumAutoencoder(n_qubits, n_latent)
    print(f"\nCreated quantum autoencoder: {n_qubits} qubits → {n_latent} qubits")
    
    # Get training circuit
    training_circuit = autoencoder.get_training_circuit()
    
    # Train the autoencoder
    print("\nTraining autoencoder...")
    optimal_params, final_cost = train_autoencoder(
        training_circuit,
        input_data=images,
        maxiter=150,
        plot_progress=True
    )
    
    # Test compression on new samples
    print("\nTesting compression on new samples...")
    test_images, test_labels = create_digit_dataset(num_samples=2, draw=False)
    
    for image, label in zip(test_images, test_labels):
        # Create input state
        input_circuit = feature_map.bind_parameters(image)
        
        # Encode and decode
        encoded_state = autoencoder.encode(input_circuit)
        decoded_state = autoencoder.decode(encoded_state)
        
        # Calculate fidelity
        fidelity = autoencoder.get_fidelity(input_circuit, decoded_state)
        print(f"Fidelity for digit {label}: {fidelity:.4f}")
        
        # Visualize
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(image.reshape(8, 4))
        plt.title("Original")
        
        plt.subplot(1, 2, 2)
        reconstructed = Statevector(decoded_state).data
        plt.imshow(np.abs(reconstructed).reshape(8, 4))
        plt.title("Reconstructed")
        plt.show()

if __name__ == "__main__":
    run_digits_example() 