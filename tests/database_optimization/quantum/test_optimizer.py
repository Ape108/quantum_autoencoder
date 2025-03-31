"""Tests for the quantum schema optimizer."""

import pytest
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from quantum_autoencoder.database_optimization.schema.graph import (
    SchemaGraph,
    NodeProperties,
    EdgeProperties
)
from quantum_autoencoder.database_optimization.quantum.optimizer import QuantumSchemaOptimizer

def create_test_schema():
    """Create a test schema for quantum optimization."""
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
def test_initialize_optimizer():
    """Test initializing the quantum schema optimizer."""
    graph = create_test_schema()
    optimizer = QuantumSchemaOptimizer(
        n_qubits=2,
        n_latent=1,
        shots=1024,
        optimizer='COBYLA'
    )
    
    # Check initialization
    assert optimizer.n_qubits == 2
    assert optimizer.n_latent == 1
    assert optimizer.shots == 1024
    assert optimizer.optimizer == 'COBYLA'

@pytest.mark.quantum
def test_optimize_schema():
    """Test optimizing a schema."""
    graph = create_test_schema()
    optimizer = QuantumSchemaOptimizer(
        n_qubits=2,
        n_latent=1,
        shots=1024,
        optimizer='COBYLA'
    )
    
    # Optimize schema
    best_params, best_cost = optimizer.optimize_schema(graph, max_iterations=10)
    
    # Check optimization results
    assert isinstance(best_params, np.ndarray)
    assert isinstance(best_cost, float)
    assert best_cost >= 0  # Cost should be non-negative

@pytest.mark.quantum
def test_get_optimized_schema():
    """Test getting optimized schema features."""
    graph = create_test_schema()
    optimizer = QuantumSchemaOptimizer(
        n_qubits=2,
        n_latent=1,
        shots=1024,
        optimizer='COBYLA'
    )
    
    # Optimize schema
    optimizer.optimize_schema(graph, max_iterations=10)
    
    # Get optimized features
    features = optimizer.get_optimized_schema(graph)
    
    # Check features
    assert isinstance(features, np.ndarray)
    assert len(features) == 2  # Number of qubits

@pytest.mark.quantum
def test_analyze_optimization():
    """Test analyzing optimization results."""
    graph = create_test_schema()
    optimizer = QuantumSchemaOptimizer(
        n_qubits=2,
        n_latent=1,
        shots=1024,
        optimizer='COBYLA'
    )
    
    # Optimize schema
    optimizer.optimize_schema(graph, max_iterations=10)
    
    # Analyze results
    analysis = optimizer.analyze_optimization()
    
    # Check analysis results
    assert isinstance(analysis, dict)
    assert 'initial_cost' in analysis
    assert 'final_cost' in analysis
    assert 'best_cost' in analysis
    assert 'cost_improvement' in analysis
    assert 'iterations' in analysis

def test_get_optimization_history():
    """Test getting optimization history."""
    graph = create_test_schema()
    optimizer = QuantumSchemaOptimizer(
        n_qubits=2,
        n_latent=1,
        shots=1024,
        optimizer='COBYLA'
    )
    
    # Optimize schema
    optimizer.optimize_schema(graph, max_iterations=10)
    
    # Get history
    history = optimizer.get_optimization_history()
    
    # Check history
    assert isinstance(history, list)
    assert len(history) > 0
    assert all(isinstance(entry, dict) for entry in history)
    assert all('iteration' in entry for entry in history)
    assert all('cost' in entry for entry in history)

def test_optimize_empty_schema():
    """Test optimizing an empty schema."""
    graph = SchemaGraph()
    optimizer = QuantumSchemaOptimizer(
        n_qubits=1,  # Minimum 1 qubit
        n_latent=1,
        shots=1024,
        optimizer='COBYLA'
    )
    
    # Optimize schema
    best_params, best_cost = optimizer.optimize_schema(graph, max_iterations=10)
    
    # Check optimization results
    assert isinstance(best_params, np.ndarray)
    assert isinstance(best_cost, float)
    assert best_cost >= 0

def test_optimize_single_table_schema():
    """Test optimizing a schema with a single table."""
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
    
    optimizer = QuantumSchemaOptimizer(
        n_qubits=1,  # 1 qubit for 1 table
        n_latent=1,
        shots=1024,
        optimizer='COBYLA'
    )
    
    # Optimize schema
    best_params, best_cost = optimizer.optimize_schema(graph, max_iterations=10)
    
    # Check optimization results
    assert isinstance(best_params, np.ndarray)
    assert isinstance(best_cost, float)
    assert best_cost >= 0

def test_optimize_large_schema():
    """Test optimizing a large schema."""
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
    
    optimizer = QuantumSchemaOptimizer(
        n_qubits=2,  # 2 qubits for 4 tables (log2(4))
        n_latent=1,
        shots=1024,
        optimizer='COBYLA'
    )
    
    # Optimize schema
    best_params, best_cost = optimizer.optimize_schema(graph, max_iterations=10)
    
    # Check optimization results
    assert isinstance(best_params, np.ndarray)
    assert isinstance(best_cost, float)
    assert best_cost >= 0

def test_optimize_complex_schema():
    """Test optimizing a schema with complex relationships."""
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
    
    optimizer = QuantumSchemaOptimizer(
        n_qubits=2,  # 2 qubits for 3 tables (log2(3) rounded up)
        n_latent=1,
        shots=1024,
        optimizer='COBYLA'
    )
    
    # Optimize schema
    best_params, best_cost = optimizer.optimize_schema(graph, max_iterations=10)
    
    # Check optimization results
    assert isinstance(best_params, np.ndarray)
    assert isinstance(best_cost, float)
    assert best_cost >= 0 