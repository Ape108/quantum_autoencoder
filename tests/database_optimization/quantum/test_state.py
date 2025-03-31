"""Tests for the QuantumStateConverter class."""

import pytest
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from quantum_autoencoder.database_optimization.schema.graph import (
    SchemaGraph,
    NodeProperties,
    EdgeProperties
)
from quantum_autoencoder.database_optimization.quantum.state import QuantumStateConverter

def create_test_schema():
    """Create a test schema for quantum state conversion."""
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
def test_convert_to_quantum_state():
    """Test converting schema to quantum state."""
    graph = create_test_schema()
    converter = QuantumStateConverter(n_qubits=2)
    
    # Convert to quantum state
    state = converter.convert_to_quantum_state(graph)
    
    assert isinstance(state, np.ndarray)
    assert len(state) == 2**2  # 2^n_qubits
    assert np.abs(np.sum(np.abs(state)**2) - 1.0) < 1e-10  # Check normalization

def test_convert_to_quantum_state_with_empty_graph():
    """Test converting empty schema to quantum state."""
    graph = SchemaGraph()
    converter = QuantumStateConverter(n_qubits=1)
    
    # Convert to quantum state
    state = converter.convert_to_quantum_state(graph)
    
    assert isinstance(state, np.ndarray)
    assert len(state) == 2  # 2^1
    assert np.abs(np.sum(np.abs(state)**2) - 1.0) < 1e-10

def test_convert_to_quantum_state_with_single_table():
    """Test converting schema with single table to quantum state."""
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
    
    converter = QuantumStateConverter(n_qubits=1)
    
    # Convert to quantum state
    state = converter.convert_to_quantum_state(graph)
    
    assert isinstance(state, np.ndarray)
    assert len(state) == 2  # 2^1
    assert np.abs(np.sum(np.abs(state)**2) - 1.0) < 1e-10

def test_convert_to_quantum_state_with_large_schema():
    """Test converting large schema to quantum state."""
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
    
    converter = QuantumStateConverter(n_qubits=2)
    
    # Convert to quantum state
    state = converter.convert_to_quantum_state(graph)
    
    assert isinstance(state, np.ndarray)
    assert len(state) == 4  # 2^2
    assert np.abs(np.sum(np.abs(state)**2) - 1.0) < 1e-10

def test_convert_to_quantum_state_with_complex_relationships():
    """Test converting schema with complex relationships to quantum state."""
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
    
    converter = QuantumStateConverter(n_qubits=2)
    
    # Convert to quantum state
    state = converter.convert_to_quantum_state(graph)
    
    assert isinstance(state, np.ndarray)
    assert len(state) == 4  # 2^2
    assert np.abs(np.sum(np.abs(state)**2) - 1.0) < 1e-10

def test_convert_to_quantum_state_with_query_frequencies():
    """Test converting schema with varying query frequencies to quantum state."""
    graph = SchemaGraph()
    
    # Add tables with different query frequencies
    graph.add_table('high_freq', NodeProperties(
        size=1000,
        column_count=5,
        query_frequency=0.9,
        update_frequency=0.1,
        primary_key='id',
        indexes=['email']
    ))
    
    graph.add_table('low_freq', NodeProperties(
        size=1000,
        column_count=5,
        query_frequency=0.1,
        update_frequency=0.9,
        primary_key='id',
        indexes=['email']
    ))
    
    # Add relationship
    graph.add_relationship('high_freq', 'low_freq', EdgeProperties(
        cardinality='1:N',
        query_frequency=0.5,
        selectivity=0.2,
        foreign_key='high_freq_id'
    ))
    
    converter = QuantumStateConverter(n_qubits=2)
    
    # Convert to quantum state
    state = converter.convert_to_quantum_state(graph)
    
    assert isinstance(state, np.ndarray)
    assert len(state) == 4  # 2^2
    assert np.abs(np.sum(np.abs(state)**2) - 1.0) < 1e-10 