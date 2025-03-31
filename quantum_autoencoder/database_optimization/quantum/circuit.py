"""
Quantum circuit construction for schema optimization.

This module provides functionality for building quantum circuits used in
the quantum autoencoder for database schema optimization.
"""

from typing import List, Optional
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
from qiskit.circuit.library import TwoLocal

class QuantumCircuitBuilder:
    """Builds quantum circuits for schema optimization."""
    
    def __init__(self, n_qubits: int = 4, n_latent: int = 2):
        """
        Initialize the circuit builder.
        
        Args:
            n_qubits: Number of qubits in the circuit
            n_latent: Number of latent qubits for compression
        """
        self.n_qubits = n_qubits
        self.n_latent = n_latent
        self.n_trash = n_qubits - n_latent
        
        # Create quantum registers
        self.qr_input = QuantumRegister(n_qubits, 'input')
        self.qr_latent = QuantumRegister(n_latent, 'latent')
        self.qr_trash = QuantumRegister(self.n_trash, 'trash')
        self.qr_ref = QuantumRegister(self.n_trash, 'ref')
        self.qr_aux = QuantumRegister(1, 'aux')
        
        # Create classical register for measurement
        self.cr = ClassicalRegister(1, 'meas')
        
    def build_encoder(self) -> QuantumCircuit:
        """
        Build the encoder circuit.
        
        Returns:
            Encoder circuit
        """
        # Create circuit
        circuit = QuantumCircuit(self.qr_input)
        
        # Create variational form with unique parameter prefix
        var_form = TwoLocal(
            self.n_qubits,
            ['ry', 'rz'],
            'cz',
            reps=2,
            entanglement='linear',
            parameter_prefix='enc'
        )
        
        # Add variational form to circuit
        circuit.compose(var_form, inplace=True)
        
        return circuit
        
    def build_decoder(self) -> QuantumCircuit:
        """
        Build the decoder circuit.
        
        Returns:
            Decoder circuit
        """
        # Create circuit
        circuit = QuantumCircuit(self.qr_latent, self.qr_trash)
        
        # Create variational form with unique parameter prefix
        var_form = TwoLocal(
            self.n_qubits,
            ['ry', 'rz'],
            'cz',
            reps=2,
            entanglement='linear',
            parameter_prefix='dec'
        )
        
        # Add variational form to circuit
        circuit.compose(var_form, inplace=True)
        
        return circuit
        
    def build_swap_test(self) -> QuantumCircuit:
        """
        Build the SWAP test circuit.
        
        Returns:
            SWAP test circuit
        """
        # Create circuit
        circuit = QuantumCircuit(
            self.qr_trash,
            self.qr_ref,
            self.qr_aux,
            self.cr
        )
        
        # Apply Hadamard to auxiliary qubit
        circuit.h(self.qr_aux)
        
        # Apply controlled-SWAP operations
        for i in range(self.n_trash):
            circuit.cswap(self.qr_aux[0], self.qr_trash[i], self.qr_ref[i])
            
        # Apply Hadamard to auxiliary qubit
        circuit.h(self.qr_aux)
        
        # Measure auxiliary qubit
        circuit.measure(self.qr_aux, self.cr)
        
        return circuit
        
    def build_full_circuit(self) -> QuantumCircuit:
        """
        Build the full quantum autoencoder circuit.
        
        Returns:
            Complete quantum circuit
        """
        # Create circuit with all registers
        circuit = QuantumCircuit(
            self.qr_input,
            self.qr_latent,
            self.qr_trash,
            self.qr_ref,
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
        
    def assign_parameters(self, circuit: QuantumCircuit, params: np.ndarray) -> QuantumCircuit:
        """
        Assign parameters to the circuit.
        
        Args:
            circuit: Circuit to assign parameters to
            params: Parameter values to assign
            
        Returns:
            Circuit with assigned parameters
        """
        # Get circuit parameters
        parameters = circuit.parameters
        
        # Create parameter dictionary
        param_dict = dict(zip(parameters, params))
        
        # Assign parameters
        assigned_circuit = circuit.assign_parameters(param_dict)
        
        return assigned_circuit
        
    def get_parameter_count(self) -> int:
        """
        Get the number of parameters in the circuit.
        
        Returns:
            Number of parameters
        """
        # Create full circuit
        circuit = self.build_full_circuit()
        
        # Return number of parameters
        return len(circuit.parameters) 