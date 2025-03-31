"""
Quantum circuit construction for schema optimization.

This module provides functionality for building quantum circuits used in
the quantum autoencoder for database schema optimization.
"""

from typing import List, Tuple, Optional
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import TwoLocal
from qiskit.quantum_info import Statevector

class QuantumCircuitBuilder:
    """Builder for quantum circuits used in schema optimization."""
    
    def __init__(self, n_qubits: int = 4, n_latent: int = 2):
        """
        Initialize the circuit builder.
        
        Args:
            n_qubits: Total number of qubits
            n_latent: Number of latent qubits for compression
        """
        self.n_qubits = n_qubits
        self.n_latent = n_latent
        self.n_trash = n_qubits - n_latent
        
        # Create quantum registers
        self.qr_input = QuantumRegister(n_qubits, 'input')
        self.qr_latent = QuantumRegister(n_latent, 'latent')
        self.qr_trash = QuantumRegister(self.n_trash, 'trash')
        self.qr_aux = QuantumRegister(1, 'aux')
        
        # Create classical register for measurements
        self.cr = ClassicalRegister(n_qubits, 'meas')
        
        # Initialize parameter vectors
        self.encoder_params = ParameterVector('θ', 2 * n_qubits)
        self.decoder_params = ParameterVector('φ', 2 * n_qubits)
        
    def build_encoder(self) -> QuantumCircuit:
        """
        Build the encoder circuit.
        
        Returns:
            Quantum circuit implementing the encoder
        """
        # Create circuit
        circuit = QuantumCircuit(
            self.qr_input,
            self.qr_latent,
            self.qr_trash,
            self.qr_aux,
            self.cr
        )
        
        # Add two-local ansatz for encoding
        encoder = TwoLocal(
            self.n_qubits,
            'ry',
            'cz',
            self.encoder_params,
            reps=2
        )
        
        # Add encoder to circuit
        circuit.compose(encoder, inplace=True)
        
        return circuit
        
    def build_decoder(self) -> QuantumCircuit:
        """
        Build the decoder circuit.
        
        Returns:
            Quantum circuit implementing the decoder
        """
        # Create circuit
        circuit = QuantumCircuit(
            self.qr_latent,
            self.qr_trash,
            self.qr_aux,
            self.cr
        )
        
        # Add two-local ansatz for decoding
        decoder = TwoLocal(
            self.n_qubits,
            'ry',
            'cz',
            self.decoder_params,
            reps=2
        )
        
        # Add decoder to circuit
        circuit.compose(decoder, inplace=True)
        
        return circuit
        
    def build_swap_test(self) -> QuantumCircuit:
        """
        Build the SWAP test circuit for fidelity measurement.
        
        Returns:
            Quantum circuit implementing the SWAP test
        """
        # Create circuit
        circuit = QuantumCircuit(
            self.qr_input,
            self.qr_latent,
            self.qr_trash,
            self.qr_aux,
            self.cr
        )
        
        # Prepare auxiliary qubit
        circuit.h(self.qr_aux[0])
        
        # Perform controlled SWAP
        for i in range(self.n_qubits):
            circuit.cswap(
                self.qr_aux[0],
                self.qr_input[i],
                self.qr_trash[i]
            )
            
        # Final Hadamard
        circuit.h(self.qr_aux[0])
        
        # Measure auxiliary qubit
        circuit.measure(self.qr_aux[0], self.cr[0])
        
        return circuit
        
    def build_full_circuit(self) -> QuantumCircuit:
        """
        Build the complete circuit combining encoder, decoder, and SWAP test.
        
        Returns:
            Complete quantum circuit
        """
        # Create circuit
        circuit = QuantumCircuit(
            self.qr_input,
            self.qr_latent,
            self.qr_trash,
            self.qr_aux,
            self.cr
        )
        
        # Add encoder
        encoder = self.build_encoder()
        circuit.compose(encoder, inplace=True)
        
        # Add decoder
        decoder = self.build_decoder()
        circuit.compose(decoder, inplace=True)
        
        # Add SWAP test
        swap_test = self.build_swap_test()
        circuit.compose(swap_test, inplace=True)
        
        return circuit
        
    def bind_parameters(self, circuit: QuantumCircuit, params: np.ndarray) -> QuantumCircuit:
        """
        Bind parameters to a circuit.
        
        Args:
            circuit: Circuit to bind parameters to
            params: Parameter values to bind
            
        Returns:
            Circuit with bound parameters
        """
        # Split parameters into encoder and decoder
        n_params = len(self.encoder_params)
        encoder_params = params[:n_params]
        decoder_params = params[n_params:]
        
        # Create parameter binding dictionary
        param_dict = {}
        param_dict.update(zip(self.encoder_params, encoder_params))
        param_dict.update(zip(self.decoder_params, decoder_params))
        
        # Bind parameters
        return circuit.bind_parameters(param_dict)
        
    def get_parameter_count(self) -> int:
        """
        Get the total number of parameters in the circuit.
        
        Returns:
            Number of parameters
        """
        return len(self.encoder_params) + len(self.decoder_params) 