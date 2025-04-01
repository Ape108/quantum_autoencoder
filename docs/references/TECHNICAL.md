# Technical Documentation

This document provides detailed technical information about the quantum autoencoder implementation.

## Circuit Architecture

### U-V Encoder Design
The quantum autoencoder uses a U-V encoder architecture, which consists of:

1. **Encoder U**
   - Acts on input qubits
   - Parameterized rotation layers (RY gates)
   - Linear entanglement pattern
   - Number of parameters: n_qubits * reps

2. **Encoder V**
   - Acts on latent and trash qubits
   - Matches U's parameter structure
   - Optimized for state compression
   - Number of parameters: n_qubits * reps

### Circuit Parameters
```python
n_qubits = 5        # Total number of qubits
n_latent = 3        # Number of latent qubits
n_trash = 2         # Number of trash qubits
reps = 3            # Number of repetitions
total_params = 30   # Total trainable parameters
```

### Gate Sequence
1. Input State Preparation
2. Barrier
3. Encoder U Operations:
   - RY rotations on all qubits
   - CX gates for entanglement
4. Barrier
5. Encoder V Operations:
   - RY rotations on latent/trash
   - CX gates for entanglement
6. Measurement on trash qubits

## Training Process

### SPSA Optimizer Configuration
```python
optimizer_config = {
    "maxiter": 1000,
    "blocking": True,
    "allowed_increase": 0.1,
    "trust_region": True,
    "learning_rate": 0.1,
    "perturbation": 0.1
}
```

### Cost Function
The cost function is based on the SWAP test between:
1. Trash qubits and reference state
2. Uses V2 primitives for efficient computation
3. Incorporates error mitigation

```python
def cost_function(parameters, circuit, estimator):
    """
    Args:
        parameters: Circuit parameters
        circuit: Quantum circuit
        estimator: V2 Estimator primitive
    
    Returns:
        float: Cost value (lower is better)
    """
```

### Training Loop
1. Initialize random parameters
2. For each trial:
   - Run 1000 iterations
   - Update parameters using SPSA
   - Calculate cost and fidelity
   - Save best parameters

### Error Mitigation
1. **Dynamical Decoupling**
   ```python
   options = {
       "dynamical_decoupling": {
           "enable": True,
           "scheme": "XY4"
       }
   }
   ```

2. **Resilience Levels**
   - Level 1: Basic error mitigation
   - Level 2: Enhanced error mitigation
   - Level 3: Maximum error mitigation

## Performance Analysis

### Fidelity Metrics
1. **Statevector Fidelity**
   - Direct state comparison
   - No measurement noise
   - Theoretical maximum

2. **Statistical Fidelity**
   - Based on measurements
   - Includes noise effects
   - Practical performance

### Resource Requirements
1. **Circuit Depth**
   - Linear in number of repetitions
   - Constant overhead for SWAP test

2. **Gate Counts**
   - RY gates: 2 * n_qubits * reps
   - CX gates: (n_qubits - 1) * reps
   - Measurement gates: n_trash

3. **Classical Resources**
   - Training time: ~1.7s per trial
   - Memory: O(2^n_qubits)

## Error Detection Capabilities

### Error Models Tested
1. Pauli Errors (X, Y, Z)
2. Amplitude Damping
3. Phase Damping
4. Depolarizing Channel

### Detection Thresholds
| Error Type | Detection Threshold |
|------------|-------------------|
| Pauli X    | 20%              |
| Pauli Y    | 15%              |
| Pauli Z    | 25%              |
| Amplitude  | 10%              |

## V2 Primitive Integration

### Sampler Usage
```python
from qiskit.primitives import Sampler

sampler = Sampler(options={
    'shots': 1024,
    'optimization_level': 3
})
```

### Estimator Usage
```python
from qiskit.primitives import Estimator

estimator = Estimator(options={
    'resilience_level': 1,
    'optimization_level': 3
})
```

## Future Improvements

1. **Circuit Optimization**
   - Reduce two-qubit gate count
   - Optimize parameter initialization
   - Implement hardware-specific optimizations

2. **Training Enhancements**
   - Parallel trial execution
   - Adaptive learning rates
   - Early stopping criteria

3. **Error Mitigation**
   - Zero-noise extrapolation
   - Measurement error mitigation
   - Readout error correction

## Hardware Considerations

### IBM Quantum Systems
1. **Required Capabilities**
   - All-to-all connectivity preferred
   - Minimum coherence times
   - Two-qubit gate fidelity requirements

2. **Layout Optimization**
   - Qubit mapping strategy
   - Gate cancellation opportunities
   - Measurement assignment

### Resource Estimation
| Resource           | Requirement    |
|-------------------|----------------|
| Qubits            | n_qubits + 2   |
| Circuit Depth     | O(reps)        |
| Coherence Time    | >100μs         |
| Memory            | 2GB            |
| Training Time     | ~10s           | 