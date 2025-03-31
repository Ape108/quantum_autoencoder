"""Tests for the QuantumCircuitBuilder class."""

import pytest
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from quantum_autoencoder.database_optimization.schema.graph import (
    SchemaGraph,
    NodeProperties,
    EdgeProperties
)
from quantum_autoencoder.database_optimization.quantum.circuit import QuantumCircuitBuilder

def create_test_schema():
    """Create a test schema for quantum circuit building."""
    graph = SchemaGraph()
    
    # Add tables
    graph.add_table('users', NodeProperties(
        size=1000,
        column_count=5,
        query_frequency=0.8,
        update_frequency=0.2,
        primary_key='id',
        indexes=['email']
    ))
    
    graph.add_table('orders', NodeProperties(
        size=5000,
        column_count=8,
        query_frequency=0.7,
        update_frequency=0.3,
        primary_key='id',
        indexes=['user_id', 'status']
    ))
    
    # Add relationship
    graph.add_relationship('users', 'orders', EdgeProperties(
        cardinality='1:N',
        query_frequency=0.9,
        selectivity=0.2,
        foreign_key='user_id'
    ))
    
    return graph

@pytest.mark.quantum
def test_build_encoder():
    """Test building the encoder circuit."""
    graph = create_test_schema()
    builder = QuantumCircuitBuilder(graph)
    
    # Build encoder circuit
    encoder = builder.build_encoder()
    
    # Check circuit properties
    assert isinstance(encoder, QuantumCircuit)
    assert encoder.num_qubits == 2  # 2 qubits for 2 tables
    assert encoder.num_clbits == 2  # 2 classical bits for measurement
    
    # Check parameter names
    param_names = [p.name for p in encoder.parameters]
    assert all(p.startswith('enc') for p in param_names)

@pytest.mark.quantum
def test_build_decoder():
    """Test building the decoder circuit."""
    graph = create_test_schema()
    builder = QuantumCircuitBuilder(graph)
    
    # Build decoder circuit
    decoder = builder.build_decoder()
    
    # Check circuit properties
    assert isinstance(decoder, QuantumCircuit)
    assert decoder.num_qubits == 2  # 2 qubits for 2 tables
    assert decoder.num_clbits == 2  # 2 classical bits for measurement
    
    # Check parameter names
    param_names = [p.name for p in decoder.parameters]
    assert all(p.startswith('dec') for p in param_names)

@pytest.mark.quantum
def test_build_swap_test():
    """Test building the SWAP test circuit."""
    graph = create_test_schema()
    builder = QuantumCircuitBuilder(graph)
    
    # Build SWAP test circuit
    swap_test = builder.build_swap_test()
    
    # Check circuit properties
    assert isinstance(swap_test, QuantumCircuit)
    assert swap_test.num_qubits == 3  # 2 qubits for states + 1 ancilla
    assert swap_test.num_clbits == 1  # 1 classical bit for measurement

@pytest.mark.quantum
def test_build_autoencoder():
    """Test building the complete autoencoder circuit."""
    graph = create_test_schema()
    builder = QuantumCircuitBuilder(graph)
    
    # Build complete circuit
    circuit = builder.build_autoencoder()
    
    # Check circuit properties
    assert isinstance(circuit, QuantumCircuit)
    assert circuit.num_qubits == 3  # 2 qubits for states + 1 ancilla
    assert circuit.num_clbits == 3  # 2 for measurement + 1 for SWAP test
    
    # Check parameter names
    param_names = [p.name for p in circuit.parameters]
    assert all(p.startswith(('enc', 'dec')) for p in param_names)

@pytest.mark.quantum
def test_build_circuit_with_empty_graph():
    """Test building circuits with empty schema."""
    graph = SchemaGraph()
    builder = QuantumCircuitBuilder(graph)
    
    # Build circuits
    encoder = builder.build_encoder()
    decoder = builder.build_decoder()
    swap_test = builder.build_swap_test()
    circuit = builder.build_autoencoder()
    
    # Check circuit properties
    assert encoder.num_qubits == 1  # Minimum 1 qubit
    assert decoder.num_qubits == 1
    assert swap_test.num_qubits == 2  # 1 qubit + 1 ancilla
    assert circuit.num_qubits == 2

@pytest.mark.quantum
def test_build_circuit_with_single_table():
    """Test building circuits with single table."""
    graph = SchemaGraph()
    
    # Add single table
    graph.add_table('users', NodeProperties(
        size=1000,
        column_count=5,
        query_frequency=0.8,
        update_frequency=0.2,
        primary_key='id',
        indexes=['email']
    ))
    
    builder = QuantumCircuitBuilder(graph)
    
    # Build circuits
    encoder = builder.build_encoder()
    decoder = builder.build_decoder()
    swap_test = builder.build_swap_test()
    circuit = builder.build_autoencoder()
    
    # Check circuit properties
    assert encoder.num_qubits == 1  # 1 qubit for 1 table
    assert decoder.num_qubits == 1
    assert swap_test.num_qubits == 2  # 1 qubit + 1 ancilla
    assert circuit.num_qubits == 2

@pytest.mark.quantum
def test_build_circuit_with_large_schema():
    """Test building circuits with large schema."""
    graph = SchemaGraph()
    
    # Add multiple tables
    for i in range(4):
        graph.add_table(f'table_{i}', NodeProperties(
            size=1000,
            column_count=5,
            query_frequency=0.8,
            update_frequency=0.2,
            primary_key='id',
            indexes=['email']
        ))
    
    # Add relationships
    for i in range(3):
        graph.add_relationship(f'table_{i}', f'table_{i+1}', EdgeProperties(
            cardinality='1:N',
            query_frequency=0.9,
            selectivity=0.2,
            foreign_key='id'
        ))
    
    builder = QuantumCircuitBuilder(graph)
    
    # Build circuits
    encoder = builder.build_encoder()
    decoder = builder.build_decoder()
    swap_test = builder.build_swap_test()
    circuit = builder.build_autoencoder()
    
    # Check circuit properties
    assert encoder.num_qubits == 2  # 2 qubits for 4 tables (log2(4))
    assert decoder.num_qubits == 2
    assert swap_test.num_qubits == 3  # 2 qubits + 1 ancilla
    assert circuit.num_qubits == 3

@pytest.mark.quantum
def test_build_circuit_with_complex_relationships():
    """Test building circuits with complex relationships."""
    graph = SchemaGraph()
    
    # Add tables
    graph.add_table('A', NodeProperties(
        size=1000,
        column_count=5,
        query_frequency=0.8,
        update_frequency=0.2,
        primary_key='id',
        indexes=['email']
    ))
    
    graph.add_table('B', NodeProperties(
        size=1000,
        column_count=5,
        query_frequency=0.8,
        update_frequency=0.2,
        primary_key='id',
        indexes=['email']
    ))
    
    graph.add_table('C', NodeProperties(
        size=1000,
        column_count=5,
        query_frequency=0.8,
        update_frequency=0.2,
        primary_key='id',
        indexes=['email']
    ))
    
    # Add complex relationships (cycle)
    graph.add_relationship('A', 'B', EdgeProperties(
        cardinality='1:N',
        query_frequency=0.9,
        selectivity=0.2,
        foreign_key='b_id'
    ))
    
    graph.add_relationship('B', 'C', EdgeProperties(
        cardinality='1:N',
        query_frequency=0.9,
        selectivity=0.2,
        foreign_key='c_id'
    ))
    
    graph.add_relationship('C', 'A', EdgeProperties(
        cardinality='1:N',
        query_frequency=0.9,
        selectivity=0.2,
        foreign_key='a_id'
    ))
    
    builder = QuantumCircuitBuilder(graph)
    
    # Build circuits
    encoder = builder.build_encoder()
    decoder = builder.build_decoder()
    swap_test = builder.build_swap_test()
    circuit = builder.build_autoencoder()
    
    # Check circuit properties
    assert encoder.num_qubits == 2  # 2 qubits for 3 tables (log2(3) rounded up)
    assert decoder.num_qubits == 2
    assert swap_test.num_qubits == 3  # 2 qubits + 1 ancilla
    assert circuit.num_qubits == 3 