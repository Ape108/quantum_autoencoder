"""
Code Compression using Quantum Autoencoder

This example demonstrates how to use the quantum autoencoder for compressing
source code by converting it to binary representation and then to quantum states.
"""

import os
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from sklearn.model_selection import train_test_split
from datetime import datetime

from quantum_autoencoder.core.circuit import QuantumAutoencoder
from quantum_autoencoder.core.training import train_autoencoder
from quantum_autoencoder.examples.code_analysis import CodeCompressionAnalyzer, CompressionMetrics

class CodeCompressor:
    def __init__(
        self,
        n_latent: int = 8,
        feature_encoding: str = "amplitude",
        reps: int = 5
    ):
        """
        Initialize the code compressor.
        
        Args:
            n_latent: Number of latent qubits
            feature_encoding: Encoding method for quantum states
            reps: Number of circuit repetitions
        """
        self.n_latent = n_latent
        self.feature_encoding = feature_encoding
        self.reps = reps
        self.compressor = None
        
    def _file_to_binary(self, file_path: str) -> np.ndarray:
        """
        Convert a file to binary representation.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Binary array representation of the file
        """
        with open(file_path, 'rb') as f:
            binary_data = np.frombuffer(f.read(), dtype=np.uint8)
        return binary_data
    
    def _binary_to_file(self, binary_data: np.ndarray, file_path: str) -> None:
        """
        Convert binary data back to a file.
        
        Args:
            binary_data: Binary array representation
            file_path: Path to save the file
        """
        with open(file_path, 'wb') as f:
            f.write(binary_data.astype(np.uint8).tobytes())
    
    def _prepare_features(self, binary_data: np.ndarray, n_qubits: int) -> np.ndarray:
        """
        Prepare binary data for quantum state encoding.
        
        Args:
            binary_data: Binary array representation
            n_qubits: Number of qubits to use
            
        Returns:
            Normalized features for quantum state preparation
        """
        # Pad or truncate to match the number of qubits
        target_size = 2**n_qubits
        if len(binary_data) < target_size:
            padded = np.zeros(target_size, dtype=np.float64)
            padded[:len(binary_data)] = binary_data
        else:
            padded = binary_data[:target_size]
        
        # Normalize
        norm = np.linalg.norm(padded)
        if norm > 0:
            padded = padded / norm
            
        return padded
    
    def train(self, code_files: List[str], **kwargs) -> Dict:
        """
        Train the quantum autoencoder on code files.
        
        Args:
            code_files: List of paths to code files
            **kwargs: Additional training parameters
            
        Returns:
            Training results
        """
        # Convert all files to binary and prepare features
        features = []
        for file_path in code_files:
            binary_data = self._file_to_binary(file_path)
            features.append(self._prepare_features(binary_data, self.n_latent))
        
        features = np.array(features)
        
        # Initialize compressor if not already done
        if self.compressor is None:
            self.compressor = QuantumAutoencoder(
                n_features=features.shape[1],
                n_latent=self.n_latent,
                feature_encoding=self.feature_encoding,
                reps=self.reps
            )
        
        # Train the autoencoder
        results = train_autoencoder(
            self.compressor,
            features,
            **kwargs
        )
        
        return results
    
    def compress_file(self, file_path: str) -> Tuple[np.ndarray, float]:
        """
        Compress a single code file.
        
        Args:
            file_path: Path to the file to compress
            
        Returns:
            Tuple of (compressed state, reconstruction error)
        """
        if self.compressor is None:
            raise ValueError("Compressor not trained. Call train() first.")
            
        binary_data = self._file_to_binary(file_path)
        features = self._prepare_features(binary_data, self.n_latent)
        
        compressed_state, error = self.compressor.compress_entry(features)
        return compressed_state, error
    
    def decompress_file(self, compressed_state: np.ndarray, output_path: str) -> None:
        """
        Decompress a quantum state back to a file.
        
        Args:
            compressed_state: Compressed quantum state
            output_path: Path to save the decompressed file
        """
        if self.compressor is None:
            raise ValueError("Compressor not trained. Call train() first.")
            
        reconstructed = self.compressor.decode_features(compressed_state)
        self._binary_to_file(reconstructed, output_path)

def example_usage():
    """Example usage of the CodeCompressor."""
    # Create results directory
    results_dir = Path(__file__).parent / "compression_results"
    results_dir.mkdir(exist_ok=True)
    
    # Initialize analyzer
    analyzer = CodeCompressionAnalyzer(str(results_dir))
    
    # Get all code sample files
    code_dir = Path(__file__).parent / "code_samples"
    code_files = [str(f) for f in code_dir.glob("example.*")]
    
    print(f"Found {len(code_files)} code samples:")
    for file in code_files:
        print(f"- {os.path.basename(file)}")
    
    # Initialize compressor
    compressor = CodeCompressor(
        n_latent=8,  # 256 possible states
        feature_encoding="amplitude",
        reps=5
    )
    
    # Train the compressor
    print("\nTraining quantum autoencoder...")
    results = compressor.train(
        code_files,
        maxiter=1000,
        n_trials=5,
        optimizer="COBYLA",
        options={"shots": 1024}
    )
    
    print("\nTraining results:")
    print(f"Final cost: {results['final_cost']:.4f}")
    print(f"Fidelity: {results['fidelity']:.4f}")
    print(f"Average reconstruction error: {results['avg_reconstruction_error']:.4f}")
    
    # Test compression and decompression
    print("\nTesting compression and decompression...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for file_path in code_files:
        print(f"\nProcessing {os.path.basename(file_path)}:")
        
        # Compress
        compressed_state, error = compressor.compress_file(file_path)
        print(f"Compression error: {error:.4f}")
        
        # Decompress
        output_path = str(results_dir / f"reconstructed_{os.path.basename(file_path)}")
        compressor.decompress_file(compressed_state, output_path)
        
        # Compare file sizes
        original_size = os.path.getsize(file_path)
        reconstructed_size = os.path.getsize(output_path)
        compression_ratio = (1 - reconstructed_size / original_size) * 100
        
        print(f"Original size: {original_size} bytes")
        print(f"Reconstructed size: {reconstructed_size} bytes")
        print(f"Compression ratio: {compression_ratio:.2f}%")
        
        # Add metrics to analyzer
        metrics = CompressionMetrics(
            file_name=os.path.basename(file_path),
            original_size=original_size,
            compressed_size=reconstructed_size,
            compression_ratio=compression_ratio,
            reconstruction_error=error,
            language=analyzer._get_language_from_extension(file_path),
            timestamp=timestamp
        )
        analyzer.add_metrics(metrics)
    
    # Generate analysis report
    print("\nGenerating analysis report...")
    analyzer.generate_report(str(results_dir))
    print(f"Analysis report and plots saved in: {results_dir}")

if __name__ == "__main__":
    example_usage() 