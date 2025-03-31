"""
Quantum schema optimization.

This module provides functionality for optimizing database schemas using
quantum autoencoders to find improved schema designs.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from qiskit.primitives import Sampler, Estimator
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize

from ..schema.graph import SchemaGraph
from ..schema.metrics import SchemaMetrics
from .state import QuantumStateConverter
from .circuit import QuantumCircuitBuilder

class QuantumSchemaOptimizer:
    """Optimizes database schemas using quantum autoencoders."""
    
    def __init__(
        self,
        n_qubits: int = 4,
        n_latent: int = 2,
        shots: int = 1024,
        tol: float = 1e-4
    ):
        """
        Initialize the schema optimizer.
        
        Args:
            n_qubits: Number of qubits for quantum state representation
            n_latent: Number of latent qubits for compression
            shots: Number of shots for quantum measurements
            tol: Tolerance for optimization convergence
        """
        self.n_qubits = n_qubits
        self.n_latent = n_latent
        self.shots = shots
        self.tol = tol
        
        # Initialize components
        self.state_converter = QuantumStateConverter(n_qubits)
        self.circuit_builder = QuantumCircuitBuilder(n_qubits, n_latent)
        
        # Initialize primitives with options
        self.sampler = Sampler()
        self.estimator = Estimator()
        
        # Initialize optimization state
        self.best_params = None
        self.best_cost = float('inf')
        self.training_history = []
        self.current_iteration = 0
        
    def _cost_function(self, params: np.ndarray, state: Statevector) -> float:
        """
        Calculate the cost function for optimization.
        
        Args:
            params: Circuit parameters
            state: Input quantum state
            
        Returns:
            Cost value
        """
        # Build and assign parameters to circuit
        circuit = self.circuit_builder.build_full_circuit()
        assigned_circuit = self.circuit_builder.assign_parameters(circuit, params)
        
        # Run SWAP test with specified shots
        job = self.sampler.run([assigned_circuit], shots=self.shots)
        result = job.result()
        
        # Get measurement counts from result
        counts = result.quasi_dists[0]
        
        # Calculate fidelity from measurement results
        # Assuming '0' state indicates successful SWAP test
        fidelity = counts.get(0, 0.0)  # Get probability of '0' state, default to 0
        
        # Cost is 1 - fidelity (we want to maximize fidelity)
        cost = 1 - fidelity
        
        # Update training history
        self.training_history.append({
            'iteration': self.current_iteration,
            'cost': cost,
            'params': params.copy()
        })
        self.current_iteration += 1
        
        # Update best cost if needed
        if cost < self.best_cost:
            self.best_cost = cost
            self.best_params = params.copy()
            
        return cost
        
    def _callback(self, xk: np.ndarray) -> bool:
        """
        Callback function for optimization to track progress.
        
        Args:
            xk: Current parameter values
            
        Returns:
            True if optimization should stop, False otherwise
        """
        # Check if we've improved significantly
        if len(self.training_history) > 1:
            last_cost = self.training_history[-1]['cost']
            if abs(last_cost - self.best_cost) < self.tol:
                return True
        return False
        
    def optimize_schema(
        self,
        graph: SchemaGraph,
        max_iterations: int = 100,
        initial_params: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Optimize a database schema.
        
        Args:
            graph: Schema graph to optimize
            max_iterations: Maximum number of optimization iterations
            initial_params: Initial circuit parameters
            
        Returns:
            Tuple of (best parameters, best cost)
        """
        # Reset optimization state
        self.best_params = None
        self.best_cost = float('inf')
        self.training_history = []
        self.current_iteration = 0
        
        # Convert schema to quantum state
        state = self.state_converter.to_quantum_state(graph)
        
        # Initialize parameters if not provided
        if initial_params is None:
            initial_params = np.random.randn(self.circuit_builder.get_parameter_count())
            
        # Define optimization bounds
        bounds = [(-np.pi, np.pi)] * len(initial_params)
        
        # Run optimization with SLSQP
        result = minimize(
            self._cost_function,
            initial_params,
            args=(state,),
            method='SLSQP',
            bounds=bounds,
            callback=self._callback,
            options={
                'maxiter': max_iterations,
                'ftol': self.tol,
                'disp': True
            }
        )
        
        return self.best_params, self.best_cost
        
    def get_optimized_schema(self, graph: SchemaGraph) -> Dict[str, np.ndarray]:
        """
        Get the optimized schema features.
        
        Args:
            graph: Original schema graph
            
        Returns:
            Dictionary of optimized features
        """
        if self.best_params is None:
            raise ValueError("Schema must be optimized first")
            
        # Build and assign parameters to circuit
        circuit = self.circuit_builder.build_full_circuit()
        assigned_circuit = self.circuit_builder.assign_parameters(circuit, self.best_params)
        
        # Convert input state
        state = self.state_converter.to_quantum_state(graph)
        
        # Run circuit with specified shots
        job = self.sampler.run([assigned_circuit], shots=self.shots)
        result = job.result()
        
        # Get measurement outcomes from result
        quasi_dist = result.quasi_dists[0]
        
        # Convert quasi-distribution to statevector amplitudes
        n_states = 2 ** self.n_qubits
        amplitudes = np.zeros(n_states, dtype=complex)
        for state_idx, prob in quasi_dist.items():
            amplitudes[state_idx] = np.sqrt(prob)
            
        # Create statevector from amplitudes
        reconstructed_state = Statevector(amplitudes)
        
        # Convert back to schema features
        return self.state_converter.from_quantum_state(reconstructed_state, graph)
        
    def analyze_optimization(self) -> Dict[str, float]:
        """
        Analyze the optimization results.
        
        Returns:
            Dictionary of analysis metrics
        """
        if not self.training_history:
            raise ValueError("No optimization history available")
            
        # Extract costs
        costs = [entry['cost'] for entry in self.training_history]
        
        # Calculate metrics
        metrics = {
            'initial_cost': costs[0],
            'final_cost': costs[-1],
            'best_cost': min(costs),
            'cost_improvement': (costs[0] - costs[-1]) / costs[0],
            'iterations': len(costs)
        }
        
        return metrics
        
    def get_optimization_history(self) -> List[Dict]:
        """
        Get the optimization history.
        
        Returns:
            List of optimization history entries
        """
        return self.training_history.copy() 