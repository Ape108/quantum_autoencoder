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
import chardet

from quantum_autoencoder.core.circuit import QuantumAutoencoder
from quantum_autoencoder.core.training import train_autoencoder
from quantum_autoencoder.examples.code_analysis import CodeCompressionAnalyzer, CompressionMetrics

class CodeCompressor:
    def __init__(
        self,
        n_latent: int = 5,  # Balanced between 4 and 6 qubits
        chunk_size: int = 7,  # Balanced between 6 and 8 qubits
        feature_encoding: str = "amplitude",
        reps: int = 6  # Balanced between 5 and 8
    ):
        """
        Initialize the code compressor.
        
        Args:
            n_latent: Number of latent qubits
            chunk_size: Size of data chunks in qubits
            feature_encoding: Encoding method for quantum states
            reps: Number of circuit repetitions
        """
        self.n_latent = n_latent
        self.chunk_size = chunk_size
        self.feature_encoding = feature_encoding
        self.reps = reps
        self.compressor = None
        self.trained_params = None
        self.original_norm = None
        self.original_size = None
        self.is_text_file = None
        self.file_encoding = None
        
    def _detect_encoding(self, file_path: str) -> str:
        """
        Detect the encoding of a text file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Detected encoding
        """
        # Read raw bytes
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            
        # Detect encoding
        result = chardet.detect(raw_data)
        return result['encoding'] or 'utf-8'
    
    def _file_to_binary(self, file_path: str) -> np.ndarray:
        """
        Convert a file to binary representation.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Binary array representation of the file
        """
        # For text files, read as text first
        text_extensions = {'.py', '.js', '.java', '.cpp', '.h', '.c', '.hpp', '.txt', '.json', '.xml', '.html', '.css'}
        is_text = any(file_path.endswith(ext) for ext in text_extensions)
        
        try:
            if is_text:
                # Detect encoding
                encoding = self._detect_encoding(file_path)
                with open(file_path, 'r', encoding=encoding) as f:
                    # Normalize line endings and encode with detected encoding
                    text = f.read().replace('\r\n', '\n')
                    data = text.encode(encoding)
                    self.file_encoding = encoding
            else:
                with open(file_path, 'rb') as f:
                    data = f.read()
                self.file_encoding = None
                
            # Store original size and text flag
            self.original_size = len(data)
            self.is_text_file = is_text
            
            return np.frombuffer(data, dtype=np.uint8)
        except (UnicodeDecodeError, LookupError):
            # Fallback to binary if encoding detection/decode fails
            with open(file_path, 'rb') as f:
                data = f.read()
            self.original_size = len(data)
            self.is_text_file = False
            self.file_encoding = None
            return np.frombuffer(data, dtype=np.uint8)
    
    def _binary_to_file(self, binary_data: np.ndarray, output_path: str) -> None:
        """
        Convert binary data back to a file.
        
        Args:
            binary_data: Binary array representation
            file_path: Path to save the file
        """
        # Convert to bytes
        byte_data = binary_data.tobytes()
        
        # For text files, decode using original encoding
        if hasattr(self, 'is_text_file') and self.is_text_file:
            try:
                encoding = getattr(self, 'file_encoding', 'utf-8')
                text_data = byte_data.decode(encoding)
                
                # Ensure proper line endings for the platform
                if os.name == 'nt':  # Windows
                    text_data = text_data.replace('\n', '\r\n')
                    
                with open(output_path, 'w', encoding=encoding, newline='') as f:
                    f.write(text_data)
            except UnicodeDecodeError:
                # Fallback to binary if decode fails
                with open(output_path, 'wb') as f:
                    f.write(byte_data)
        else:
            with open(output_path, 'wb') as f:
                f.write(byte_data)
    
    def _prepare_chunk(self, binary_data: np.ndarray, start_idx: int) -> np.ndarray:
        """
        Prepare a chunk of binary data for quantum state encoding.
        
        Args:
            binary_data: Binary array representation
            start_idx: Starting index of the chunk
            
        Returns:
            Normalized features for quantum state encoding
        """
        # Get chunk of data
        chunk = binary_data[start_idx:start_idx + 2**self.chunk_size]
        
        # Pad with zeros if necessary
        if len(chunk) < 2**self.chunk_size:
            chunk = np.pad(chunk, (0, 2**self.chunk_size - len(chunk)))
            
        # Normalize the chunk
        norm = np.linalg.norm(chunk)
        if norm > 0:
            chunk = chunk / norm
            
        return chunk
    
    def _features_to_circuit(self, features: np.ndarray) -> QuantumCircuit:
        """
        Convert feature vector to quantum circuit.
        
        Args:
            features: Normalized feature vector
            
        Returns:
            Quantum circuit initialized with the features
        """
        # Create statevector from features
        sv = Statevector(features)
        
        # Create quantum circuit
        n_qubits = int(np.log2(len(features)))
        qc = QuantumCircuit(n_qubits)
        
        # Initialize circuit with statevector
        qc.initialize(sv, range(n_qubits))
        
        return qc
    
    def train(self, code_files: List[str], **kwargs) -> Dict:
        """
        Train the quantum autoencoder on code files.
        
        Args:
            code_files: List of paths to code files
            **kwargs: Additional training parameters
            
        Returns:
            Training results
        """
        # Initialize compressor if not already done
        if self.compressor is None:
            self.compressor = QuantumAutoencoder(
                n_qubits=self.chunk_size,
                n_latent=self.n_latent,
                reps=self.reps
            )
        
        # Train on first chunk of first file
        binary_data = self._file_to_binary(code_files[0])
        chunk = self._prepare_chunk(binary_data, 0)
        input_circuit = self._features_to_circuit(chunk)
        
        # Train the autoencoder
        self.trained_params, final_cost = train_autoencoder(
            self.compressor,
            input_circuit,
            **kwargs
        )
        
        return {
            'final_cost': final_cost,
            'fidelity': 1.0 - final_cost,
            'avg_reconstruction_error': final_cost
        }
    
    def compress_file(self, file_path: str) -> Tuple[np.ndarray, float]:
        """
        Compress a single code file using quantum feature extraction.
        
        Args:
            file_path: Path to the file to compress
            
        Returns:
            Tuple of (compressed data, average reconstruction error)
        """
        if self.trained_params is None:
            raise ValueError("Autoencoder must be trained before compression")
            
        # For text files, handle text content separately
        text_extensions = {'.py', '.js', '.java', '.cpp', '.h', '.c', '.hpp', '.txt', '.json', '.xml', '.html', '.css'}
        is_text = any(file_path.endswith(ext) for ext in text_extensions)
        
        if is_text:
            try:
                # Read text content
                with open(file_path, 'r', encoding='utf-8') as f:
                    text_content = f.read()
                
                # Store text content as UTF-8 bytes
                text_bytes = text_content.encode('utf-8')
                data_array = np.frombuffer(text_bytes, dtype=np.uint8)
                
                # Store metadata
                self.is_text_file = True
                self.file_encoding = 'utf-8'
                self.original_size = len(data_array)
                self.original_norm = np.linalg.norm(data_array)
                
            except UnicodeDecodeError:
                # Fallback to binary if UTF-8 decode fails
                with open(file_path, 'rb') as f:
                    data_array = np.frombuffer(f.read(), dtype=np.uint8)
                self.is_text_file = False
                self.file_encoding = None
                self.original_size = len(data_array)
                self.original_norm = np.linalg.norm(data_array)
        else:
            # Binary file
            with open(file_path, 'rb') as f:
                data_array = np.frombuffer(f.read(), dtype=np.uint8)
            self.is_text_file = False
            self.file_encoding = None
            self.original_size = len(data_array)
            self.original_norm = np.linalg.norm(data_array)
        
        # Process in chunks
        compressed_chunks = []
        total_error = 0
        n_chunks = 0
        
        # Track repeated chunks for run-length encoding
        last_chunk = None
        repeat_count = 0
        
        for i in range(0, len(data_array), 2**self.chunk_size):
            # Prepare chunk
            features = self._prepare_chunk(data_array, i)
            
            # Skip chunks that are all zeros
            if np.all(features == 0):
                if last_chunk is None or not np.all(last_chunk == 0):
                    compressed_chunks.append(np.zeros(2**self.n_latent, dtype=np.uint8))
                    last_chunk = np.zeros(2**self.n_latent, dtype=np.uint8)
                    repeat_count = 1
                else:
                    repeat_count += 1
                continue
                
            # Compress chunk using quantum circuit
            compressed_state = self.compressor.compress_entry(
                features,
                self.trained_params
            )
            
            # Extract binary features from latent space
            latent_data = np.abs(compressed_state.data[:2**self.n_latent])
            binary_features = (latent_data > 0.5).astype(np.uint8)
            
            # Run-length encoding
            if last_chunk is not None and np.array_equal(binary_features, last_chunk):
                repeat_count += 1
            else:
                if repeat_count > 1:
                    # Store repeat count
                    compressed_chunks.append(np.array([repeat_count], dtype=np.uint8))
                compressed_chunks.append(binary_features)
                last_chunk = binary_features.copy()
                repeat_count = 1
            
            # Calculate reconstruction error
            reconstructed = self.compressor.decode_features(
                compressed_state,
                self.trained_params
            )
            error = np.linalg.norm(features - reconstructed)
            total_error += error
            n_chunks += 1
            
        # Handle final repeat count
        if repeat_count > 1:
            compressed_chunks.append(np.array([repeat_count], dtype=np.uint8))
            
        # Pack bits to bytes for better compression
        packed_chunks = []
        for chunk in compressed_chunks:
            if len(chunk) == 1:  # Repeat count
                packed_chunks.append(chunk.astype(np.uint8))
            else:
                # Pack 8 bits into each byte
                packed = np.packbits(chunk).astype(np.uint8)
                packed_chunks.append(packed)
        
        # Add metadata
        metadata = np.array([
            len(data_array),  # Original size
            float(self.original_norm),  # Original norm
            len(compressed_chunks),  # Number of chunks
            1 if self.is_text_file else 0,  # Is text file
        ], dtype=np.float64)
        
        # Ensure all arrays are 1D
        compressed_data = np.concatenate([
            metadata,
            np.concatenate(packed_chunks).astype(np.uint8)  # Packed compressed chunks
        ])
        
        return compressed_data, total_error / max(n_chunks, 1)
    
    def decompress_file(self, compressed_data: np.ndarray, output_path: str) -> None:
        """
        Decompress data back to a file.
        
        Args:
            compressed_data: Compressed data array
            output_path: Path to save the file
        """
        if self.trained_params is None:
            raise ValueError("Autoencoder must be trained before decompression")
            
        # Extract metadata
        original_size = int(compressed_data[0])
        self.original_norm = float(compressed_data[1])
        n_chunks = int(compressed_data[2])
        self.is_text_file = bool(compressed_data[3])
        
        # Process compressed chunks
        packed_data = compressed_data[4:].astype(np.uint8)
        
        # Process chunks
        reconstructed_chunks = []
        bytes_per_chunk = (2**self.n_latent + 7) // 8  # Ceiling division
        i = 0
        
        while len(reconstructed_chunks) < n_chunks:
            # Check if this is a repeat count
            if i + 1 <= len(packed_data):
                repeat_count = int(packed_data[i])
                if repeat_count > 0 and repeat_count < 2**self.n_latent:
                    # This is a repeat count
                    if reconstructed_chunks:  # Only try to repeat if we have chunks
                        last_chunk = reconstructed_chunks[-1]
                        reconstructed_chunks.extend([last_chunk.copy() for _ in range(repeat_count - 1)])
                    i += 1
                    continue
            
            # Regular chunk
            if i + bytes_per_chunk <= len(packed_data):
                packed_chunk = packed_data[i:i + bytes_per_chunk]
                # Unpack bytes to bits
                binary_features = np.unpackbits(packed_chunk)[:2**self.n_latent]
                
                if np.all(binary_features == 0):
                    reconstructed_chunks.append(np.zeros(2**self.chunk_size, dtype=np.uint8))
                else:
                    # Create quantum state from binary features
                    latent_state = Statevector(binary_features.astype(np.float32))
                    
                    # Decode chunk
                    reconstructed = self.compressor.decode_features(
                        latent_state,
                        self.trained_params
                    )
                    
                    # Take real part and scale back
                    chunk = np.real(reconstructed)
                    chunk = (chunk * self.original_norm).astype(np.uint8)
                    reconstructed_chunks.append(chunk)
                
                i += bytes_per_chunk
            else:
                break
        
        if not reconstructed_chunks:
            raise ValueError("No chunks were successfully reconstructed")
            
        # Combine chunks and truncate to original size
        reconstructed_data = np.concatenate(reconstructed_chunks)[:original_size]
        
        # Convert back to bytes
        byte_data = reconstructed_data.tobytes()
        
        # Handle text files
        if self.is_text_file:
            try:
                # Decode bytes as UTF-8 text
                text = byte_data.decode('utf-8')
                
                # Ensure proper line endings for the platform
                if os.name == 'nt':  # Windows
                    text = text.replace('\n', '\r\n')
                    
                # Write text with UTF-8 encoding
                with open(output_path, 'w', encoding='utf-8', newline='') as f:
                    f.write(text)
                return
            except UnicodeDecodeError:
                # Fallback to binary if decode fails
                pass
        
        # Write binary data (fallback)
        with open(output_path, 'wb') as f:
            f.write(byte_data)

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
    
    # Initialize compressor with optimized parameters
    compressor = CodeCompressor(
        n_latent=4,  # Reduced latent space for better compression
        chunk_size=7,  # Process 128 bytes at a time
        feature_encoding="amplitude",
        reps=6
    )
    
    # Train the compressor
    print("\nTraining quantum autoencoder...")
    results = compressor.train(
        code_files,
        maxiter=800,
        n_trials=4,
        optimizer="COBYLA",
        options={"shots": 1536}
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
        compressed_data, error = compressor.compress_file(file_path)
        print(f"Average compression error: {error:.4f}")
        
        # Decompress
        output_path = str(results_dir / f"reconstructed_{os.path.basename(file_path)}")
        compressor.decompress_file(compressed_data, output_path)
        
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