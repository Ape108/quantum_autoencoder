"""
Training functionality for quantum autoencoder.
"""

from typing import List, Optional, Tuple, Callable, Dict, Any, Union
import os
import time
import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.primitives import Sampler
from qiskit_algorithms.optimizers import Optimizer, COBYLA, SPSA
from qiskit_machine_learning.utils import algorithm_globals
from qiskit.quantum_info import Statevector

def create_cost_function(
    autoencoder,
    input_state: QuantumCircuit,
    options: Optional[Dict[str, Any]] = None,
    callback: Optional[Callable] = None
) -> Callable:
    """
    Create a cost function for training the autoencoder using statevector fidelity.
    
    Args:
        autoencoder: QuantumAutoencoder instance
        input_state: Input quantum state to compress
        options: Optional dictionary of options for the sampler
        callback: Optional callback function for tracking progress
        
    Returns:
        Cost function that calculates fidelity between input and reconstructed states
    """
    objective_func_vals = []
    
    # Pre-compute input statevector
    sv_input = Statevector(input_state)
    
    def cost_function(params_values: np.ndarray) -> float:
        # Encode and decode the state
        encoded_state = autoencoder.encode(input_state, parameter_values=params_values)
        decoded_state = autoencoder.decode(encoded_state, parameter_values=params_values)
        
        # Calculate fidelity using statevectors
        sv_decoded = Statevector(decoded_state)
        fidelity = np.abs(sv_input.inner(sv_decoded)) ** 2
        
        # Loss is 1 - fidelity
        cost = 1.0 - float(fidelity.real)
        
        # Store for plotting
        objective_func_vals.append(cost)
        
        # Call callback if provided
        if callback is not None:
            callback(cost)
        
        return cost
        
    return cost_function

def train_autoencoder(
    autoencoder,
    input_state: QuantumCircuit,
    maxiter: int = 200,
    n_trials: int = 5,
    seed: int = 42,
    plot_progress: bool = True,
    optimizer: Optional[Union[str, Optimizer]] = None,
    options: Optional[Dict[str, Any]] = None,
    save_dir: str = "outputs/training_plots"
) -> Tuple[np.ndarray, float]:
    """
    Train the quantum autoencoder using statevector fidelity.
    
    Args:
        autoencoder: QuantumAutoencoder instance
        input_state: Input quantum state to compress
        maxiter: Maximum number of iterations per trial
        n_trials: Number of random initializations to try
        seed: Random seed
        plot_progress: Whether to plot training progress
        optimizer: Optimizer to use (string name or instance)
        options: Optional dictionary of options for the primitives
        save_dir: Directory to save training plots
        
    Returns:
        Tuple of (optimal parameters, final cost)
    """
    # Set random seed
    algorithm_globals.random_seed = seed
    
    # Storage for best result
    best_params = None
    best_cost = float('inf')
    all_costs = []
    trial_costs = []  # Store costs for each trial
    
    # Create optimizer
    if optimizer is None or isinstance(optimizer, str):
        if optimizer == "SPSA":
            optimizer = SPSA(
                maxiter=maxiter,
                learning_rate=0.15,
                perturbation=0.1,
                resamplings=1,  # Reduced resampling for speed
                trust_region=True,
                second_order=False  # Disable second-order for speed
            )
        else:
            optimizer = COBYLA(maxiter=maxiter, tol=1e-4)  # Relaxed tolerance
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Total number of parameters
    num_params = autoencoder.n_params_u + autoencoder.n_params_v
    
    # Try multiple random initializations
    for trial in range(n_trials):
        print(f"\nStarting trial {trial + 1}/{n_trials}")
        trial_start = time.time()
        
        # Initialize parameters in [-π/4, π/4] for better convergence
        initial_point = np.random.uniform(-np.pi/4, np.pi/4, size=num_params)
        
        # Create callback for this trial
        trial_costs.clear()
        def callback(cost: float):
            trial_costs.append(cost)
            # Print progress every 50 iterations
            if len(trial_costs) % 50 == 0:
                print(f"Iteration {len(trial_costs)}: cost = {cost:.4f}")
        
        # Create and get cost function for this trial
        cost_func = create_cost_function(autoencoder, input_state, options, callback)
        
        # Train
        result = optimizer.minimize(cost_func, initial_point)
        elapsed = time.time() - trial_start
        
        print(f"Trial {trial + 1} completed in {elapsed:.2f} seconds")
        print(f"Final cost: {result.fun:.4f}")
        
        # Store costs for plotting
        all_costs.extend(trial_costs)
        
        # Update best result
        if result.fun < best_cost:
            best_cost = result.fun
            best_params = result.x
            print("New best result found!")
            print(f"Current fidelity: {1.0 - result.fun:.4f}")
            
            # Plot progress for best trial
            if plot_progress:
                plt.figure(figsize=(12, 6))
                plt.plot(range(len(trial_costs)), trial_costs)
                plt.title(f"Training Progress - Trial {trial + 1} (Best So Far)")
                plt.xlabel("Iteration")
                plt.ylabel("Cost")
                plt.savefig(f"{save_dir}/trial_{trial + 1}_progress.png")
                plt.close()
    
    print(f"\nBest cost achieved: {best_cost:.4f}")
    print(f"Best fidelity: {1.0 - best_cost:.4f}")
    
    if plot_progress and all_costs:
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(all_costs)), all_costs)
        plt.title("Training Progress Across All Trials")
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.savefig(f"{save_dir}/training_progress.png")
        plt.close()
        print("\nTraining progress plot saved as 'training_progress.png'")
    
    return best_params, best_cost 