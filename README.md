# Quantum Autoencoder

A Python package implementing quantum autoencoders using Qiskit V2 primitives. This package provides tools for compressing quantum states into lower-dimensional representations while preserving essential quantum information.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Ape108/quantum_autoencoder.git
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

# Run the domain wall compression example with custom options
custom_options = {
    "optimization_level": 3,
    "resilience_level": 1,
    "shots": 2048,
    "dynamical_decoupling": {"enable": True}  # Advanced error mitigation
}
run_domain_wall_example(options=custom_options)
```

This will:
1. Create a domain wall state |00111⟩
2. Train a quantum autoencoder to compress it from 5 to 3 qubits using V2 primitives
3. Show the training progress with error mitigation
4. Calculate the fidelity between input and reconstructed states

### Digit Compression Example

```python
from quantum_autoencoder.examples.digits import run_digits_example

# Run the digit compression example with optimized settings
options = {
    "optimization_level": 3,
    "resilience_level": 2,  # Enhanced error mitigation for complex states
    "shots": 4096,
    "dynamical_decoupling": {
        "enable": True,
        "scheme": "XY4"
    }
}
run_digits_example(options=options)
```

This will:
1. Generate noisy images of digits 0 and 1
2. Train an autoencoder to compress them using V2 primitives
3. Show the original and reconstructed images with error mitigation
4. Calculate the fidelity for each test sample

## Package Structure

```
quantum_autoencoder/
├── core/
│   ├── circuit.py     # Main autoencoder implementation using V2 primitives
│   ├── ansatz.py      # Parameterized circuit definitions
│   └── training.py    # Training functionality with error mitigation
├── examples/
│   ├── domain_wall.py # Domain wall example
│   └── digits.py      # Digit compression example
└── utils/
    ├── visualization.py
    └── data.py
```

## Features

- Flexible quantum autoencoder implementation using Qiskit V2 primitives
- Customizable ansatz for encoder/decoder
- SWAP test-based training with error mitigation
- Support for both quantum and classical data
- Built-in visualization tools
- Example implementations with error resilience
- Hardware-optimized execution using PUBs (Programs Under Batch)
- Advanced error mitigation options

## Requirements

- Python ≥ 3.9
- Qiskit ≥ 1.0.0
- Qiskit Machine Learning ≥ 0.8.2
- NumPy ≥ 1.21.0
- Matplotlib ≥ 3.4.0
- SciPy ≥ 1.7.0

## Error Mitigation Options

The package supports various error mitigation strategies through V2 primitives:

1. **Basic Error Mitigation** (resilience_level=1):
   - Measurement error mitigation
   - Zero noise extrapolation
   - Dynamical decoupling

2. **Advanced Error Mitigation** (resilience_level=2):
   - All basic strategies
   - Enhanced readout error mitigation
   - Advanced dynamical decoupling schemes

Example configuration:
```python
options = {
    "resilience_level": 2,
    "dynamical_decoupling": {
        "enable": True,
        "scheme": "XY4"
    }
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

1. Romero, J., Olson, J. P., & Aspuru-Guzik, A. (2017). Quantum autoencoders for efficient compression of quantum data. Quantum Science and Technology, 2(4), 045001.
2. [Qiskit Documentation](https://qiskit.org/documentation/)
3. [Qiskit V2 Primitives Guide](https://docs.quantum.ibm.com/api/migration-guides/v2-primitives)
4. [Quantum Autoencoder Tutorial](https://qiskit.org/documentation/tutorials/) 