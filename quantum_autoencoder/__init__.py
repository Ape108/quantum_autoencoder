"""
Quantum Autoencoder Package

A package for implementing and training quantum autoencoders using Qiskit.
"""

from quantum_autoencoder.core.circuit import QuantumAutoencoder
from quantum_autoencoder.core.ansatz import create_encoder_ansatz
from quantum_autoencoder.core.training import train_autoencoder

__version__ = "0.1.0"
__author__ = "Your Name"

__all__ = ["QuantumAutoencoder", "create_encoder_ansatz", "train_autoencoder"] 