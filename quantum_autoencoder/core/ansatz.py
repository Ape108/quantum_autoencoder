"""
Ansatz definitions for quantum autoencoder.
"""

from typing import Optional
from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes

def create_encoder_ansatz(
    n_qubits: int,
    reps: int = 5,
    name: str = "encoder"
) -> QuantumCircuit:
    """
    Create a parameterized ansatz for the encoder circuit.
    
    Args:
        n_qubits: Number of qubits in the circuit
        reps: Number of repetitions of the ansatz
        name: Name of the circuit
        
    Returns:
        Parameterized quantum circuit for encoding
    """
    return RealAmplitudes(n_qubits, reps=reps, name=name)

def create_decoder_ansatz(encoder_ansatz: QuantumCircuit) -> QuantumCircuit:
    """
    Create the decoder ansatz from an encoder ansatz.
    The decoder is the inverse of the encoder.
    
    Args:
        encoder_ansatz: The encoder circuit
        
    Returns:
        Decoder circuit (inverse of encoder)
    """
    return encoder_ansatz.inverse() 