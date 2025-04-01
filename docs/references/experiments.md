# Quantum Autoencoder Experiments and Results

## 1. Domain Wall State Compression
### Configuration
- Input qubits: 5
- Latent qubits: 3
- Compression ratio: 40%
- Circuit repetitions: 3
- Training shots: 1024
- Optimizer: SPSA

### Training Results
- Best cost achieved: 0.0003
- Training time per trial: ~1.7 seconds
- Number of trials: 5
- Best trial fidelity: 0.9997 (99.97%)

### Verification Results
#### Statevector Fidelity Test
- Fidelity: 0.999667 (99.97%)
- Original state: |11100⟩ (equivalent to |00111⟩ when read right-to-left)
- Reconstructed state dominant amplitude: 0.999834 for |11100⟩
- Other amplitudes: All < 1%

#### Statistical Measurement Test
- Fidelity: 0.999878 (99.99%)
- Number of shots: 8192
- Visualization: measurement_comparison.png

## 2. Error Detection Application
### Configuration
- Same autoencoder architecture as above
- Error types: Random X, Y, Z Pauli errors
- Error rates tested: [0%, 5%, 10%, 20%, 30%, 40%, 50%]

### Results
| Error Rate | Reconstruction Fidelity |
|------------|------------------------|
| 0%         | 0.999667 (99.97%)     |
| 5%         | 0.997378 (99.74%)     |
| 10%        | 0.999667 (99.97%)     |
| 20%        | 0.000439 (0.04%)      |
| 30%        | 0.000511 (0.05%)      |
| 40%        | 0.002482 (0.25%)      |
| 50%        | 0.000000 (0.00%)      |

### Key Findings
1. Perfect reconstruction (99.97% fidelity) for error-free states
2. Robust performance up to 10% error rate
3. Sharp transition to low fidelity above 20% error rate
4. Clear error detection capability
5. Visualization: error_detection.png

## Visualizations
All experiment visualizations are stored in the root directory:
1. training_progress.png - Shows cost convergence during training
2. measurement_comparison.png - Compares original and reconstructed state measurements
3. error_detection.png - Shows fidelity vs error rate relationship

## Implementation Details
- Using Qiskit V2 primitives
- U-V encoder architecture
- Dynamical decoupling enabled
- Resilience level: 1
- Optimization level: 3

*Note: This document will be updated as we conduct more experiments and applications.* 