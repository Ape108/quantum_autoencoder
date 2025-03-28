"""
Training functionality for quantum autoencoder.
"""

from typing import List, Optional, Tuple, Callable, Dict, Any
import time
import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA, SPSA, ADAM
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
        # Bind parameters to circuit
        bound_circuit = qnn.circuit.assign_parameters(params_values)
        
        if input_data is not None:
            # Batch training - create PUBs for each input
            pubs = [bound_circuit] * len(input_data)
            job = sampler.run(pubs)
            results = job.result()
            probabilities = np.array([dist[1] if 1 in dist else 0.0 
                                    for dist in results.quasi_dists])
            cost = np.mean(probabilities)
        else:
            # Single state training
            job = sampler.run([bound_circuit])
            result = job.result()
            quasi_dist = result.quasi_dists[0]
            cost = quasi_dist[1] if 1 in quasi_dist else 0.0
            
        # Store for plotting
        objective_func_vals.append(cost)
        
        return cost
        
    return cost_function

def train_autoencoder(
    circuit: QuantumCircuit,
    input_data: Optional[np.ndarray] = None,
    maxiter: int = 200,  # Reduced iterations but multiple trials
    n_trials: int = 5,   # Multiple random initializations
    seed: int = 42,
    plot_progress: bool = True,
    options: Optional[Dict[str, Any]] = None
) -> Tuple[np.ndarray, float]:
    """
    Train the quantum autoencoder using V2 primitives.
    
    Args:
        circuit: Quantum circuit with SWAP test
        input_data: Optional input data for batch training
        maxiter: Maximum number of iterations per trial
        n_trials: Number of random initializations to try
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
    
    # Storage for best result
    best_params = None
    best_cost = float('inf')
    all_costs = []
    
    # Try multiple random initializations
    for trial in range(n_trials):
        print(f"\nStarting trial {trial + 1}/{n_trials}")
        
        # Create COBYLA optimizer with settings from notebook
        optimizer = COBYLA(
            maxiter=maxiter,
            tol=1e-6,      # Final accuracy
            disp=False     # Don't print intermediate results
        )
        
        # Initialize parameters in [-π, π]
        num_params = circuit.num_parameters
        initial_point = np.random.uniform(-np.pi, np.pi, size=num_params)
        
        # Create and get cost function for this trial
        cost_func = create_cost_function(qnn, input_data, options)
        
        # Train
        start = time.time()
        result = optimizer.minimize(cost_func, initial_point)
        elapsed = time.time() - start
        
        print(f"Trial {trial + 1} completed in {elapsed:.2f} seconds")
        print(f"Final cost: {result.fun:.4f}")
        
        # Store costs for plotting
        if hasattr(cost_func, '__closure__') and cost_func.__closure__:
            for cell in cost_func.__closure__:
                if isinstance(cell.cell_contents, list):
                    all_costs.extend(cell.cell_contents)
        
        # Update best result
        if result.fun < best_cost:
            best_cost = result.fun
            best_params = result.x
            print("New best result found!")
    
    print(f"\nBest cost achieved: {best_cost:.4f}")
    
    if plot_progress and all_costs:
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(all_costs)), all_costs)
        plt.title("Training Progress Across All Trials")
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.savefig('training_progress.png')
        plt.close()
        print("\nTraining progress plot saved as 'training_progress.png'")
    
    return best_params, best_cost 