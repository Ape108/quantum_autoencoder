"""
Training functionality for quantum autoencoder.
"""

from typing import List, Optional, Tuple, Callable, Dict, Any
import time
import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.primitives import SamplerV2 as Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals

def create_cost_function(
    qnn: SamplerQNN, 
    input_data: Optional[np.ndarray] = None,
    options: Optional[Dict[str, Any]] = None
) -> Callable:
    """
    Create a cost function for training the autoencoder using V2 primitives.
    
    Args:
        qnn: Quantum Neural Network with SWAP test circuit
        input_data: Optional input data for batch training
        options: Optional dictionary of options for the sampler
        
    Returns:
        Cost function that calculates fidelity between trash and reference states
    """
    objective_func_vals = []
    sampler = Sampler(options=options)
    
    def cost_function(params_values: np.ndarray) -> float:
        if input_data is not None:
            # Batch training - create PUBs for each input
            pubs = [(qnn.circuit, input_data[i]) for i in range(len(input_data))]
            job = sampler.run(pubs)
            results = job.result()
            probabilities = np.array([res.data.meas.get_counts().get('1', 0) / 
                                    sum(res.data.meas.get_counts().values())
                                    for res in results])
            cost = np.mean(probabilities)
        else:
            # Single state training
            job = sampler.run([(qnn.circuit, [])])
            result = job.result()[0]
            counts = result.data.meas.get_counts()
            cost = counts.get('1', 0) / sum(counts.values())
            
        # Store for plotting
        objective_func_vals.append(cost)
        
        return cost
        
    return cost_function

def train_autoencoder(
    circuit: QuantumCircuit,
    input_data: Optional[np.ndarray] = None,
    maxiter: int = 150,
    seed: int = 42,
    plot_progress: bool = True,
    options: Optional[Dict[str, Any]] = None
) -> Tuple[np.ndarray, float]:
    """
    Train the quantum autoencoder using V2 primitives.
    
    Args:
        circuit: Quantum circuit with SWAP test
        input_data: Optional input data for batch training
        maxiter: Maximum number of iterations
        seed: Random seed
        plot_progress: Whether to plot training progress
        options: Optional dictionary of options for the primitives
        
    Returns:
        Tuple of (optimal parameters, final cost)
    """
    # Set random seed
    algorithm_globals.random_seed = seed
    
    # Create QNN with V2 sampler
    sampler = Sampler(options=options)
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
    cost_func = create_cost_function(qnn, input_data, options)
    
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