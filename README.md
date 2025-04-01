# Domain Wall State Compression using Quantum Autoencoder

This project implements a quantum autoencoder to compress a 5-qubit domain wall state (|00111⟩) into 3 qubits with high fidelity using Qiskit V2 primitives.

## Features

- High-fidelity quantum state compression (>99.9%)
- U-V encoder architecture
- Error mitigation using V2 primitives
- Visualization tools for quantum states

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd quantum-autoencoder

# Install dependencies
pip install -r requirements.txt
```

## Usage

Run the domain wall example:

```bash
python -m quantum_autoencoder.examples.domain_wall
```

This will:
1. Create a 5-qubit domain wall state (|00111⟩)
2. Train the autoencoder to compress it to 3 qubits
3. Generate visualizations of the input, encoded, and reconstructed states
4. Save the trained model parameters

## Requirements

- Python ≥3.9
- Qiskit ≥1.0.0
- Qiskit Machine Learning ≥0.8.2
- NumPy ≥1.21.0
- Matplotlib ≥3.4.0
- SciPy ≥1.7.0

## License

MIT License 