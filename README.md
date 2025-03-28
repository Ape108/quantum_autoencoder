# Quantum Autoencoder

A Python package implementing quantum autoencoders using Qiskit. This package provides tools for compressing quantum states into lower-dimensional representations while preserving essential quantum information.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/quantum_autoencoder.git
cd quantum_autoencoder
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Example: Domain Wall State

```python
from quantum_autoencoder.examples.domain_wall import run_domain_wall_example

# Run the domain wall compression example
run_domain_wall_example()
```

This will:
1. Create a domain wall state |00111⟩
2. Train a quantum autoencoder to compress it from 5 to 3 qubits
3. Show the training progress
4. Calculate the fidelity between input and reconstructed states

### Digit Compression Example

```python
from quantum_autoencoder.examples.digits import run_digits_example

# Run the digit compression example
run_digits_example()
```

This will:
1. Generate noisy images of digits 0 and 1
2. Train an autoencoder to compress them
3. Show the original and reconstructed images
4. Calculate the fidelity for each test sample

## Package Structure

```
quantum_autoencoder/
├── core/
│   ├── circuit.py     # Main autoencoder implementation
│   ├── ansatz.py      # Parameterized circuit definitions
│   └── training.py    # Training functionality
├── examples/
│   ├── domain_wall.py # Domain wall example
│   └── digits.py      # Digit compression example
└── utils/
    ├── visualization.py
    └── data.py
```

## Features

- Flexible quantum autoencoder implementation
- Customizable ansatz for encoder/decoder
- SWAP test-based training
- Support for both quantum and classical data
- Built-in visualization tools
- Example implementations

## Requirements

- Python 3.7+
- Qiskit 0.43.0+
- Qiskit Machine Learning 0.6.0+
- NumPy 1.21.0+
- Matplotlib 3.4.0+
- SciPy 1.7.0+

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

1. Romero, J., Olson, J. P., & Aspuru-Guzik, A. (2017). Quantum autoencoders for efficient compression of quantum data. Quantum Science and Technology, 2(4), 045001.
2. [Qiskit Documentation](https://qiskit.org/documentation/)
3. [Quantum Autoencoder Tutorial](https://qiskit.org/documentation/tutorials/) 