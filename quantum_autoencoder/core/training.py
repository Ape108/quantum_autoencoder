"""
Training functionality for quantum autoencoder.
"""

from typing import List, Optional, Tuple, Callable
import time
import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals

def create_cost_function(qnn: SamplerQNN, input_data: Optional[np.ndarray] = None) -> Callable:
    """
    Create a cost function for training the autoencoder.
    
    Args:
        qnn: Quantum Neural Network with SWAP test circuit
        input_data: Optional input data for batch training
        
    Returns:
        Cost function that calculates fidelity between trash and reference states
    """
    objective_func_vals = []
    
    def cost_function(params_values: np.ndarray) -> float:
        if input_data is not None:
            # Batch training
            probabilities = qnn.forward(input_data, params_values)
            cost = np.sum(probabilities[:, 1]) / input_data.shape[0]
        else:
            # Single state training
            probabilities = qnn.forward([], params_values)
            cost = np.sum(probabilities[:, 1])
            
        # Store for plotting
        objective_func_vals.append(cost)
        
        return cost
        
    return cost_function

def train_autoencoder(
    circuit: QuantumCircuit,
    input_data: Optional[np.ndarray] = None,
    maxiter: int = 150,
    seed: int = 42,
    plot_progress: bool = True
) -> Tuple[np.ndarray, float]:
    """
    Train the quantum autoencoder.
    
    Args:
        circuit: Quantum circuit with SWAP test
        input_data: Optional input data for batch training
        maxiter: Maximum number of iterations
        seed: Random seed
        plot_progress: Whether to plot training progress
        
    Returns:
        Tuple of (optimal parameters, final cost)
    """
    # Set random seed
    algorithm_globals.random_seed = seed
    
    # Create QNN
    sampler = Sampler()
    qnn = SamplerQNN(
        circuit=circuit,
        input_params=[],
        weight_params=circuit.parameters,
        interpret=lambda x: x,
        output_shape=2,
        sampler=sampler,
    )
    
    # Create optimizer
    optimizer = COBYLA(maxiter=maxiter)
    
    # Initialize parameters
    initial_point = algorithm_globals.random.random(circuit.num_parameters)
    
    # Create and get cost function
    cost_func = create_cost_function(qnn, input_data)
    
    # Train
    start = time.time()
    result = optimizer.minimize(cost_func, initial_point)
    elapsed = time.time() - start
    
    print(f"Training completed in {elapsed:.2f} seconds")
    print(f"Final cost: {result.fun:.4f}")
    
    if plot_progress:
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(result.history)), result.history)
        plt.title("Training Progress")
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.show()
    
    return result.x, result.fun 