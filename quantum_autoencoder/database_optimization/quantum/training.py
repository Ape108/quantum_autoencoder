"""
Quantum Autoencoder Training for Query Path Optimization.

This module handles the training of the quantum autoencoder to learn
patterns in query execution paths. It specifically focuses on:
1. Preserving quantum correlations between execution steps
2. Learning optimal compression of query patterns
3. Maintaining execution path coherence in latent space
"""

from typing import List, Dict, Optional
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, state_fidelity
import logging
from qiskit.circuit.library import RealAmplitudes
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

class QuantumTrainer:
    """Trains quantum autoencoder circuits."""
    
    def __init__(self, n_qubits: int, n_latent: int):
        """
        Initialize trainer.
        
        Args:
            n_qubits: Number of qubits
            n_latent: Number of latent qubits
        """
        self.n_qubits = n_qubits
        self.n_latent = n_latent
        
        # Initialize encoder/decoder circuits
        self.encoder = RealAmplitudes(n_qubits, reps=2)
        self.decoder = RealAmplitudes(n_qubits, reps=2)
        
        # Store best parameters
        self.best_params = None
        self.best_cost = float('inf')
    
    def _cost_function(self, params: np.ndarray, circuit: QuantumCircuit) -> float:
        """Calculate reconstruction cost."""
        n_params = len(params) // 2
        encoder_params = params[:n_params]
        decoder_params = params[n_params:]
        
        # Get input state
        input_state = Statevector(circuit)
        
        # Encode
        encoded = self.encoder.assign_parameters(encoder_params)
        encoded_state = input_state.evolve(encoded)
        
        # Decode
        decoded = self.decoder.assign_parameters(decoder_params)
        output_state = encoded_state.evolve(decoded)
        
        # Calculate fidelity
        fidelity = abs(input_state.inner(output_state)) ** 2
        return 1.0 - fidelity
    
    def train(
        self,
        circuits: List[QuantumCircuit],
        n_epochs: int = 100,
        batch_size: int = 4
    ) -> List[float]:
        """
        Train autoencoder.
        
        Args:
            circuits: Training circuits
            n_epochs: Number of epochs
            batch_size: Batch size
            
        Returns:
            Training history
        """
        history = []
        
        # Initialize parameters
        n_params = len(self.encoder.parameters) + len(self.decoder.parameters)
        initial_params = np.random.random(n_params)
        
        # Train on first circuit (can be extended to multiple)
        if not circuits:
            raise ValueError("No circuits provided for training")
            
        circuit = circuits[0]
        
        # Optimize
        def objective(params):
            cost = self._cost_function(params, circuit)
            if cost < self.best_cost:
                self.best_cost = cost
                self.best_params = params
            return cost
        
        result = minimize(objective, initial_params, method='COBYLA', options={'maxiter': 100})
        
        # Store best parameters
        self.best_params = result.x
        self.best_cost = result.fun
        
        return [self.best_cost]
    
    def get_latent_representation(self, circuit: QuantumCircuit) -> Statevector:
        """Get latent space representation."""
        if self.best_params is None:
            raise ValueError("Model must be trained first")
            
        n_params = len(self.best_params) // 2
        encoder_params = self.best_params[:n_params]
        
        # Get encoded state
        input_state = Statevector(circuit)
        encoded = self.encoder.assign_parameters(encoder_params)
        encoded_state = input_state.evolve(encoded)
        
        return encoded_state

class QueryPathTrainer:
    """Trains quantum autoencoder on query execution paths."""
    
    def __init__(
        self,
        n_qubits: int,
        n_latent: int,
        options: Optional[Dict] = None
    ):
        """
        Initialize trainer.
        
        Args:
            n_qubits: Number of qubits in input states
            n_latent: Number of latent qubits for compression
            options: Training options
        """
        self.n_qubits = n_qubits
        self.n_latent = n_latent
        
        # Default options focused on preserving quantum properties
        if options is None:
            options = {
                "optimization_level": 3,
                "resilience_level": 1,
                "shots": 1024,
                "dynamical_decoupling": {"enable": True}
            }
        
        # Initialize quantum autoencoder parameters
        self.params = np.random.randn(n_qubits * n_latent * 3)  # 3 parameters per qubit pair
        self.best_params = None
        self.best_fidelity = 0.0
        
        # Training history
        self.history = {
            'fidelity': [],
            'entanglement': [],
            'coherence': []
        }
    
    def _create_encoder_circuit(self, params: np.ndarray) -> QuantumCircuit:
        """Create encoder circuit with given parameters."""
        qc = QuantumCircuit(self.n_qubits)
        param_idx = 0
        
        # Add encoding layers
        for i in range(3):  # 3 layers
            # Single qubit rotations
            for j in range(self.n_qubits):
                qc.ry(params[param_idx], j)
                param_idx += 1
            
            # Entangling layer
            for j in range(self.n_qubits - 1):
                qc.cx(j, j + 1)
            qc.cx(self.n_qubits - 1, 0)  # Close the chain
        
        return qc
    
    def _create_decoder_circuit(self, params: np.ndarray) -> QuantumCircuit:
        """Create decoder circuit with given parameters."""
        qc = QuantumCircuit(self.n_qubits)
        param_idx = len(params) // 2
        
        # Add decoding layers (mirror of encoding)
        for i in range(3):  # 3 layers
            # Entangling layer
            for j in range(self.n_qubits - 1, 0, -1):
                qc.cx(j - 1, j)
            qc.cx(self.n_qubits - 1, 0)
            
            # Single qubit rotations
            for j in range(self.n_qubits):
                qc.ry(params[param_idx], j)
                param_idx += 1
        
        return qc
    
    def encode(self, circuit: QuantumCircuit, params: Optional[np.ndarray] = None) -> QuantumCircuit:
        """Encode quantum state."""
        if params is None:
            params = self.best_params if self.best_params is not None else self.params
        
        encoder = self._create_encoder_circuit(params)
        return encoder.compose(circuit)
    
    def decode(self, circuit: QuantumCircuit, params: Optional[np.ndarray] = None) -> QuantumCircuit:
        """Decode quantum state."""
        if params is None:
            params = self.best_params if self.best_params is not None else self.params
        
        decoder = self._create_decoder_circuit(params)
        return decoder.compose(circuit)
    
    def calculate_fidelity(
        self,
        original: QuantumCircuit,
        encoded: QuantumCircuit,
        decoded: QuantumCircuit
    ) -> float:
        """Calculate fidelity between original and reconstructed states."""
        orig_state = Statevector(original)
        decoded_state = Statevector(decoded)
        return float(state_fidelity(orig_state, decoded_state))
    
    def _calculate_cost(self, circuits: List[QuantumCircuit], params: np.ndarray) -> float:
        """Calculate cost function for given parameters."""
        total_cost = 0.0
        
        for circuit in circuits:
            # Encode and decode
            encoded = self.encode(circuit, params)
            decoded = self.decode(encoded, params)
            
            # Calculate fidelity
            fidelity = self.calculate_fidelity(circuit, encoded, decoded)
            total_cost += (1.0 - fidelity)
        
        return total_cost / len(circuits)
    
    def _optimize_parameters(
        self,
        circuits: List[QuantumCircuit],
        learning_rate: float = 0.01,
        n_steps: int = 100
    ) -> None:
        """Optimize autoencoder parameters."""
        for step in range(n_steps):
            # Calculate gradients numerically
            grads = np.zeros_like(self.params)
            epsilon = 1e-7
            
            for i in range(len(self.params)):
                params_plus = self.params.copy()
                params_plus[i] += epsilon
                params_minus = self.params.copy()
                params_minus[i] -= epsilon
                
                cost_plus = self._calculate_cost(circuits, params_plus)
                cost_minus = self._calculate_cost(circuits, params_minus)
                
                grads[i] = (cost_plus - cost_minus) / (2 * epsilon)
            
            # Update parameters
            self.params -= learning_rate * grads
            
            # Calculate current cost
            current_cost = self._calculate_cost(circuits, self.params)
            
            # Update best parameters if improved
            if current_cost < self.best_fidelity or self.best_params is None:
                self.best_fidelity = current_cost
                self.best_params = self.params.copy()
            
            if step % 10 == 0:
                logger.info(f"Step {step}, Cost: {current_cost:.6f}")
    
    def train(
        self,
        circuits: List[QuantumCircuit],
        n_epochs: int = 100,
        batch_size: int = 4
    ) -> Dict:
        """
        Train the quantum autoencoder.
        
        Args:
            circuits: Training circuits
            n_epochs: Number of training epochs
            batch_size: Batch size
            
        Returns:
            Training history
        """
        if not circuits:
            raise ValueError("No circuits provided for training")
        
        n_batches = len(circuits) // batch_size + (1 if len(circuits) % batch_size else 0)
        
        for epoch in range(n_epochs):
            logger.info(f"Epoch {epoch + 1}/{n_epochs}")
            
            # Shuffle circuits
            np.random.shuffle(circuits)
            
            epoch_fidelity = 0.0
            n_valid = 0
            
            # Train on batches
            for i in range(0, len(circuits), batch_size):
                batch = circuits[i:i + batch_size]
                try:
                    self._optimize_parameters(batch)
                    
                    # Calculate batch fidelity
                    batch_fidelity = 0.0
                    for circuit in batch:
                        try:
                            encoded = self.encode(circuit)
                            decoded = self.decode(encoded)
                            fidelity = self.calculate_fidelity(circuit, encoded, decoded)
                            batch_fidelity += fidelity
                            n_valid += 1
                        except Exception as e:
                            logger.warning(f"Failed to process circuit in batch: {e}")
                    
                    if n_valid > 0:
                        epoch_fidelity += batch_fidelity
                
                except Exception as e:
                    logger.warning(f"Failed to optimize batch: {e}")
            
            # Calculate average fidelity
            if n_valid > 0:
                avg_fidelity = epoch_fidelity / n_valid
                self.history['fidelity'].append(avg_fidelity)
                logger.info(f"Average Fidelity: {avg_fidelity:.6f}")
            else:
                logger.warning("No valid circuits processed in this epoch")
        
        if not self.history['fidelity']:
            raise ValueError("Training failed: no valid fidelity measurements")
        
        return self.history
    
    def get_latent_representation(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Get the latent space representation of a circuit."""
        if self.best_params is None:
            raise ValueError("Model must be trained before getting latent representation")
        return self.encode(circuit, self.best_params)
    
    def calculate_path_metrics(
        self,
        original: QuantumCircuit,
        encoded: QuantumCircuit,
        decoded: QuantumCircuit
    ) -> Dict:
        """
        Calculate metrics that matter for query execution.
        
        Args:
            original: Original query path circuit
            encoded: Encoded (compressed) circuit
            decoded: Decoded (reconstructed) circuit
            
        Returns:
            Dictionary of metrics including:
            - Fidelity: How well the path is preserved
            - Entanglement: Quantum correlations between steps
            - Coherence: Preservation of phase relationships
        """
        metrics = {}
        
        # Calculate fidelity
        orig_state = Statevector(original)
        decoded_state = Statevector(decoded)
        metrics['fidelity'] = state_fidelity(orig_state, decoded_state)
        
        # Calculate entanglement preservation
        orig_entanglement = self._calculate_entanglement(orig_state)
        decoded_entanglement = self._calculate_entanglement(decoded_state)
        metrics['entanglement_preservation'] = decoded_entanglement / orig_entanglement
        
        # Calculate coherence preservation
        orig_coherence = self._calculate_coherence(orig_state)
        decoded_coherence = self._calculate_coherence(decoded_state)
        metrics['coherence_preservation'] = decoded_coherence / orig_coherence
        
        return metrics
    
    def _calculate_entanglement(self, state: Statevector) -> float:
        """
        Calculate entanglement metric for a state.
        Uses partial trace and von Neumann entropy.
        """
        # Convert to density matrix
        rho = state.to_operator().data
        
        # Calculate reduced density matrix for first qubit
        reduced_rho = np.zeros((2, 2), dtype=complex)
        for i in range(2):
            for j in range(2):
                # Partial trace over other qubits
                trace = 0
                for k in range(2**(self.n_qubits-1)):
                    idx1 = i * 2**(self.n_qubits-1) + k
                    idx2 = j * 2**(self.n_qubits-1) + k
                    trace += rho[idx1][idx2]
                reduced_rho[i][j] = trace
        
        # Calculate von Neumann entropy
        eigenvals = np.linalg.eigvalsh(reduced_rho)
        entropy = -np.sum(eigenvals * np.log2(eigenvals + 1e-10))
        
        return entropy
    
    def _calculate_coherence(self, state: Statevector) -> float:
        """
        Calculate quantum coherence metric.
        Uses l1-norm of coherence.
        """
        # Get state vector
        psi = state.data
        
        # Calculate l1-norm of coherence
        rho = np.outer(psi, psi.conj())
        coherence = np.sum(np.abs(rho)) - np.sum(np.abs(np.diag(rho)))
        
        return coherence
    
    def analyze_compression(self, state: QuantumCircuit) -> Dict:
        """
        Analyze how the query path is compressed.
        
        Returns detailed metrics about the compression quality
        and quantum properties preserved.
        """
        encoded = self.encode(state)
        decoded = self.decode(encoded)
        
        metrics = self.calculate_path_metrics(state, encoded, decoded)
        
        # Add compression ratio
        metrics['compression_ratio'] = self.n_qubits / self.n_latent
        
        return metrics 