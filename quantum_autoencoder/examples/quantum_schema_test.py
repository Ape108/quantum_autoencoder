"""
Quantum Autoencoder Database Schema Compression Test

This script demonstrates and measures the quantum autoencoder's ability to:
1. Compress database schema features into a minimal quantum representation
2. Reconstruct the schema accurately from the compressed state
3. Measure the quantum advantage quantitatively
"""

import numpy as np
from pathlib import Path
import sqlite3
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from qiskit.quantum_info import state_fidelity
from quantum_autoencoder.core.circuit import QuantumAutoencoder
from qiskit.quantum_info import Statevector
from qiskit.circuit import QuantumCircuit

def extract_schema_features(db_path: str) -> np.ndarray:
    """Extract normalized features from database schema."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    features = []
    
    # Get tables
    cursor.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name NOT LIKE 'sqlite_%'
    """)
    tables = cursor.fetchall()
    
    for table_name in tables:
        # Get table size
        cursor.execute(f"SELECT COUNT(*) FROM {table_name[0]}")
        size = cursor.fetchone()[0]
        
        # Get column count
        cursor.execute(f"PRAGMA table_info({table_name[0]})")
        columns = cursor.fetchall()
        column_count = len(columns)
        
        # Get foreign key count
        cursor.execute(f"PRAGMA foreign_key_list({table_name[0]})")
        fk_count = len(cursor.fetchall())
        
        # Get index count
        cursor.execute(f"PRAGMA index_list({table_name[0]})")
        index_count = len(cursor.fetchall())
        
        # Normalize features
        features.extend([
            size / 1000.0,  # Normalize size
            column_count / 20.0,  # Normalize column count
            fk_count / 5.0,  # Normalize FK count
            index_count / 5.0  # Normalize index count
        ])
    
    conn.close()
    return np.array(features)

def cost_function(params: np.ndarray, autoencoder: QuantumAutoencoder, features: np.ndarray) -> float:
    """Calculate cost (negative fidelity) for optimization."""
    # Split parameters for U and V encoders
    params_u = params[:autoencoder.n_params_u]
    params_v = params[autoencoder.n_params_u:]
    
    # Normalize features
    norm_features = features / np.linalg.norm(features)
    
    # Create input state circuit
    input_circuit = QuantumCircuit(autoencoder.n_qubits)
    for i, amplitude in enumerate(norm_features):
        if i < autoencoder.n_qubits:
            input_circuit.ry(2 * np.arccos(amplitude), i)
    
    # Encode and decode
    encoded_state = autoencoder.encode(input_circuit, params)
    reconstructed_state = autoencoder.decode(encoded_state, params)
    
    # Calculate fidelity
    fidelity = autoencoder.get_fidelity(input_circuit, reconstructed_state)
    
    # Return negative fidelity for minimization
    return -fidelity

def test_quantum_compression(features: np.ndarray, n_latent: int) -> Dict:
    """Test quantum autoencoder compression of schema features."""
    # Calculate required qubits
    n_features = len(features)
    n_qubits = int(np.ceil(np.log2(n_features)))
    
    print(f"\nCompression Configuration:")
    print(f"- Original features: {n_features}")
    print(f"- Input qubits: {n_qubits}")
    print(f"- Latent qubits: {n_latent}")
    print(f"- Compression ratio: {2**n_qubits}/{2**n_latent} = {2**(n_qubits-n_latent)}:1")
    
    # Initialize quantum autoencoder with proper options
    autoencoder = QuantumAutoencoder(
        n_qubits=n_qubits,
        n_latent=n_latent,
        reps=3,  # Increased for better compression
        options={
            "optimization_level": 3,
            "resilience_level": 1,
            "shots": 1024,
            "dynamical_decoupling": {"enable": True}
        }
    )
    
    # Total number of parameters
    n_params_total = autoencoder.n_params_u + autoencoder.n_params_v
    print(f"- Total parameters: {n_params_total}")
    
    # Initialize parameters
    initial_params = np.random.randn(n_params_total) * 0.1
    
    # Training history
    training_history = []
    
    # Callback to track progress
    def callback(xk):
        fidelity = -cost_function(xk, autoencoder, features)
        training_history.append(fidelity)
        if len(training_history) % 10 == 0:
            print(f"Iteration {len(training_history)}: Fidelity = {fidelity:.4f}")
    
    # Optimize parameters using COBYLA
    result = minimize(
        cost_function,
        initial_params,
        args=(autoencoder, features),
        method='COBYLA',
        callback=callback,
        options={'maxiter': 200}
    )
    
    # Get final results with optimized parameters
    best_params = result.x
    
    # Create input state circuit
    norm_features = features / np.linalg.norm(features)
    input_circuit = QuantumCircuit(autoencoder.n_qubits)
    for i, amplitude in enumerate(norm_features):
        if i < autoencoder.n_qubits:
            input_circuit.ry(2 * np.arccos(amplitude), i)
    
    # Final compression and reconstruction
    final_encoded = autoencoder.encode(input_circuit, best_params)
    final_reconstructed = autoencoder.decode(final_encoded, best_params)
    final_fidelity = autoencoder.get_fidelity(input_circuit, final_reconstructed)
    
    # Get state vectors for visualization
    encoded_state = Statevector.from_instruction(final_encoded)
    reconstructed_state = Statevector.from_instruction(final_reconstructed)
    
    print(f"\nOptimization completed:")
    print(f"- Final fidelity: {final_fidelity:.4f}")
    print(f"- Optimization success: {result.success}")
    print(f"- Number of iterations: {len(training_history)}")
    
    return {
        "compression_ratio": 2**(n_qubits-n_latent),
        "final_fidelity": final_fidelity,
        "training_history": training_history,
        "compressed_state": encoded_state.data,
        "reconstructed_state": reconstructed_state.data,
        "best_parameters": best_params,
        "optimization_result": result
    }

def plot_results(results: Dict, save_path: str = "quantum_compression_results.png"):
    """Plot training progress and compression results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training history
    ax1.plot(results["training_history"])
    ax1.set_title("Training Progress")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("State Fidelity")
    ax1.grid(True)
    
    # Plot original vs reconstructed features
    x = np.arange(len(results["reconstructed_state"]))
    ax2.plot(x, np.abs(results["reconstructed_state"]), 'b-', label='Reconstructed', alpha=0.7)
    ax2.plot(x[:len(results["compressed_state"])], np.abs(results["compressed_state"]), 'r--', label='Compressed', alpha=0.7)
    ax2.set_title(f"Feature Comparison (Compression {results['compression_ratio']}:1)")
    ax2.set_xlabel("Feature Index")
    ax2.set_ylabel("Magnitude")
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    """Run quantum compression test on Northwind database."""
    # Load database schema features
    db_path = "tests/database_optimization/ExampleDB/northwind.db"
    features = extract_schema_features(db_path)
    
    print("\nDatabase Schema Features:")
    print(f"- Number of features: {len(features)}")
    print(f"- Feature vector: {features}")
    
    # Test different compression levels
    latent_qubits = [2, 3, 4]  # Test different compression ratios
    compression_results = {}
    
    for n_latent in latent_qubits:
        print(f"\nTesting compression with {n_latent} latent qubits...")
        results = test_quantum_compression(features, n_latent)
        compression_results[n_latent] = results
        
        print(f"\nResults for {n_latent} latent qubits:")
        print(f"- Compression ratio: {results['compression_ratio']}:1")
        print(f"- Final fidelity: {results['final_fidelity']:.4f}")
        
        # Plot results for this compression level
        plot_results(results, f"quantum_compression_{n_latent}qubits.png")
    
    # Print summary
    print("\nCompression Summary:")
    print("===================")
    for n_latent, results in compression_results.items():
        print(f"\n{n_latent} latent qubits:")
        print(f"- Compression ratio: {results['compression_ratio']}:1")
        print(f"- Final fidelity: {results['final_fidelity']:.4f}")
        print(f"- Training iterations: {len(results['training_history'])}")

if __name__ == "__main__":
    main() 