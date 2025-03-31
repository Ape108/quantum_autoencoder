"""Test configuration and fixtures."""

import pytest
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from quantum_autoencoder.database_optimization.schema.graph import (
    SchemaGraph,
    NodeProperties,
    EdgeProperties
)

@pytest.fixture
def sample_schema():
    """Create a sample schema for testing."""
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

@pytest.fixture
def empty_schema():
    """Create an empty schema for testing."""
    return SchemaGraph()

@pytest.fixture
def single_table_schema():
    """Create a schema with a single table for testing."""
    graph = SchemaGraph()
    graph.add_table('users', NodeProperties(
        size=1000,
        column_count=5,
        query_frequency=0.8,
        update_frequency=0.2,
        primary_key='id',
        indexes=['email']
    ))
    return graph

@pytest.fixture
def large_schema():
    """Create a large schema for testing."""
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
    
    return graph

@pytest.fixture
def complex_schema():
    """Create a schema with complex relationships for testing."""
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
    
    return graph

@pytest.fixture
def quantum_circuit():
    """Create a sample quantum circuit for testing."""
    circuit = QuantumCircuit(2, 1)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure(0, 0)
    return circuit

@pytest.fixture
def quantum_state():
    """Create a sample quantum state for testing."""
    return np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])

@pytest.fixture
def optimizer_params():
    """Create optimizer parameters for testing."""
    return {
        'n_qubits': 2,
        'n_latent': 1,
        'shots': 1024,
        'optimizer': 'COBYLA'
    } 