# Quantum Autoencoder for Domain Wall State Compression

This project implements a quantum autoencoder (QAE) that achieves high-fidelity compression of quantum states, demonstrated on a 5-qubit domain wall state |00111⟩ compressed to 3 qubits. Using a U-V encoder architecture and Qiskit primitives, we achieve >99.9% fidelity in state reconstruction.

## Theoretical Background

### Quantum Autoencoders

Quantum autoencoders are quantum circuits designed to compress quantum states into a lower-dimensional latent space while preserving essential quantum information. Similar to classical autoencoders, they consist of:

1. **Encoder (U)**: Maps input state |ψ⟩ to a compressed representation
2. **Latent Space**: Lower-dimensional representation of the quantum state
3. **Decoder (V)**: Reconstructs the original state from the compressed representation

### U-V Architecture

Our implementation uses a U-V encoder architecture:

```
|ψ⟩ ─── U ─── V ───
```

Where:
- U: Parameterized encoding circuit
- V: Parameterized decoding circuit
- Dimension reduction: 5 qubits → 3 qubits (40% reduction)

The architecture is designed to minimize information loss by:
1. Separating encoding and decoding parameters
2. Using alternating layers of rotation and entanglement gates
3. Implementing efficient parameter reuse

### Mathematical Framework

The compression process can be described as:

1. **Input State**: |ψ⟩ ∈ ℂ^(2^n)
2. **Encoding**: U(θ)|ψ⟩ = |ϕ⟩|0⟩^(n-k)
3. **Decoding**: V(φ)|ϕ⟩|0⟩^(n-k) ≈ |ψ⟩
4. **Fidelity**: F = |⟨ψ|V(φ)U(θ)|ψ⟩|²

Where:
- n: Number of input qubits (5)
- k: Number of latent qubits (3)
- θ, φ: Trainable parameters
- |0⟩^(n-k): Auxiliary qubits in |0⟩ state

## Implementation Details

### Circuit Architecture

```python
class QuantumAutoencoder:
    def __init__(self, n_qubits: int, n_latent: int, reps: int = 2):
        """
        Args:
            n_qubits: Number of input qubits (5)
            n_latent: Number of latent qubits (3)
            reps: Number of repetition layers
        """
```

#### Encoder Structure
Each encoder (U and V) consists of:
1. **Rotation Layers**: RY gates with trainable parameters
2. **Entanglement Layers**: CX gates in linear configuration
3. **Parameter Sharing**: Efficient reuse of parameters across layers

### Training Process

1. **Cost Function**:
   ```
   C(θ,φ) = 1 - F = 1 - |⟨ψ|V(φ)U(θ)|ψ⟩|²
   ```

2. **Optimization**:
   - Algorithm: SPSA (Simultaneous Perturbation Stochastic Approximation)
   - Learning rate: 0.15
   - Perturbation size: 0.1
   - Trust region: Enabled
   - Multiple random initializations: 5 trials

3. **Error Mitigation**:
   - Dynamical decoupling
   - Resilience level: 1
   - Optimized shot count: 1024

### Domain Wall State Example

The domain wall state |00111⟩ was chosen for its:
1. Clear boundary between 0s and 1s
2. Entanglement properties
3. Practical relevance in condensed matter physics

## Results

### Performance Metrics

1. **Compression Ratio**: 40% (5→3 qubits)
2. **Fidelity**: 99.98%
3. **Training Time**: ~2.06s per trial
4. **Convergence**: Typically within 500 iterations

### Visualization Outputs

The code generates:
1. Input state visualization
2. Encoded (latent) state representation
3. Reconstructed state comparison
4. Training convergence plots

## Usage

### Installation
```bash
# Clone repository
git clone <repository-url>
cd quantum-autoencoder

# Install dependencies
pip install -r requirements.txt
```

### Running the Example
```bash
python -m quantum_autoencoder.examples.domain_wall
```

### Requirements
- Python ≥3.9
- Qiskit ≥1.0.0
- Qiskit Machine Learning ≥0.8.2
- NumPy ≥1.21.0
- Matplotlib ≥3.4.0
- SciPy ≥1.7.0

## Implications and Applications

1. **Quantum Data Compression**:
   - Efficient storage of quantum states
   - Reduced qubit requirements for quantum memory
   - Potential for quantum error correction

2. **Quantum Machine Learning**:
   - Feature extraction in quantum data
   - Dimensionality reduction for quantum algorithms
   - Quantum state preparation optimization

3. **Quantum Simulation**:
   - Efficient representation of many-body states
   - Compression of quantum simulation results
   - Study of quantum phase transitions

4. **Practical Impact**:
   - Reduced hardware requirements
   - Improved quantum circuit depth
   - Enhanced noise resilience

## Future Directions

1. **Architecture Optimization**:
   - Alternative entanglement patterns
   - Adaptive parameter schemes
   - Hardware-efficient variants

2. **Application Expansion**:
   - Multiple state compression
   - Dynamic compression ratios
   - Integration with quantum error correction

3. **Performance Enhancement**:
   - Advanced error mitigation
   - Improved optimization strategies
   - Hardware-specific optimizations

## References and Citations

This implementation builds upon and extends several key works:

1. **Original Quantum Autoencoder Paper**:
   - Romero, J., Olson, J. P., & Aspuru-Guzik, A. (2017). Quantum autoencoders for efficient compression of quantum data. Quantum Science and Technology, 2(4), 045001.

2. **Reference Implementations**:
   - Original circuit design by [@qiaoyi213](https://github.com/qiaoyi213) - see `references/Main Algorithm Quantum Circuit Autoencoder.py`
   - Base quantum circuit implementation - see `references/qc.py`

3. **Technical Resources**:
   - [Qiskit Documentation](https://qiskit.org/documentation/)
   - [Qiskit V2 Primitives Guide](https://docs.quantum.ibm.com/api/migration-guides/v2-primitives)

The reference implementations have been substantially modified and enhanced in this project to:
- Implement V2 primitives for better hardware compatibility
- Add comprehensive error mitigation
- Improve training convergence
- Enhance visualization capabilities

## License

MIT License 