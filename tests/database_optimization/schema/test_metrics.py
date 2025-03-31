"""Tests for the SchemaMetrics class."""

import pytest
from quantum_autoencoder.database_optimization.schema.graph import (
    SchemaGraph,
    NodeProperties,
    EdgeProperties
)
from quantum_autoencoder.database_optimization.schema.metrics import SchemaMetrics

def create_test_schema():
    """Create a test schema for metrics calculation."""
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
    
    graph.add_table('products', NodeProperties(
        size=100,
        column_count=6,
        query_frequency=0.6,
        update_frequency=0.1,
        primary_key='id',
        indexes=['category']
    ))
    
    # Add relationships
    graph.add_relationship('users', 'orders', EdgeProperties(
        cardinality='1:N',
        query_frequency=0.9,
        selectivity=0.2,
        foreign_key='user_id'
    ))
    
    graph.add_relationship('orders', 'products', EdgeProperties(
        cardinality='N:M',
        query_frequency=0.8,
        selectivity=0.3,
        foreign_key='product_id'
    ))
    
    return graph

def test_calculate_query_metrics():
    """Test calculating query-related metrics."""
    graph = create_test_schema()
    metrics = SchemaMetrics(graph)
    
    # Calculate query metrics
    avg_freq = metrics.calculate_query_metrics()['avg_query_frequency']
    freq_var = metrics.calculate_query_metrics()['query_frequency_variance']
    
    # Check average query frequency
    expected_avg = (0.8 + 0.7 + 0.6) / 3
    assert abs(avg_freq - expected_avg) < 1e-6
    
    # Check query frequency variance
    expected_var = sum((f - expected_avg) ** 2 for f in [0.8, 0.7, 0.6]) / 3
    assert abs(freq_var - expected_var) < 1e-6

def test_calculate_storage_metrics():
    """Test calculating storage-related metrics."""
    graph = create_test_schema()
    metrics = SchemaMetrics(graph)
    
    # Calculate storage metrics
    total_size = metrics.calculate_storage_metrics()['total_size']
    avg_size = metrics.calculate_storage_metrics()['avg_table_size']
    size_var = metrics.calculate_storage_metrics()['size_variance']
    
    # Check total size
    assert total_size == 6100
    
    # Check average table size
    assert abs(avg_size - 6100/3) < 1e-6
    
    # Check size variance
    expected_var = sum((s - avg_size) ** 2 for s in [1000, 5000, 100]) / 3
    assert abs(size_var - expected_var) < 1e-6

def test_calculate_relationship_metrics():
    """Test calculating relationship-related metrics."""
    graph = create_test_schema()
    metrics = SchemaMetrics(graph)
    
    # Calculate relationship metrics
    rel_metrics = metrics.calculate_relationship_metrics()
    
    # Check average selectivity
    expected_avg = (0.2 + 0.3) / 2
    assert abs(rel_metrics['avg_selectivity'] - expected_avg) < 1e-6
    
    # Check selectivity variance
    expected_var = sum((s - expected_avg) ** 2 for s in [0.2, 0.3]) / 2
    assert abs(rel_metrics['selectivity_variance'] - expected_var) < 1e-6

def test_calculate_graph_metrics():
    """Test calculating graph-related metrics."""
    graph = create_test_schema()
    metrics = SchemaMetrics(graph)
    
    # Calculate graph metrics
    graph_metrics = metrics.calculate_graph_metrics()
    
    # Check relationship density
    # For 3 nodes, max edges = 6 (directed graph)
    # We have 2 edges
    expected_density = 2 / 6
    assert abs(graph_metrics['relationship_density'] - expected_density) < 1e-6
    
    # Check average degree
    # Total edges = 2, nodes = 3
    expected_degree = 2 / 3
    assert abs(graph_metrics['avg_degree'] - expected_degree) < 1e-6

def test_get_all_metrics():
    """Test getting all metrics together."""
    graph = create_test_schema()
    metrics = SchemaMetrics(graph)
    
    # Get all metrics
    all_metrics = metrics.get_all_metrics()
    
    # Check that all expected metrics are present
    expected_metrics = {
        'avg_query_frequency',
        'query_frequency_variance',
        'avg_relationship_frequency',
        'total_size',
        'avg_table_size',
        'size_variance',
        'avg_column_count',
        'avg_selectivity',
        'selectivity_variance',
        'relationship_density',
        'avg_degree',
        'clustering_coefficient',
        'avg_path_length'
    }
    
    assert all(metric in all_metrics for metric in expected_metrics)

def test_metrics_with_empty_graph():
    """Test metrics calculation with an empty graph."""
    graph = SchemaGraph()
    metrics = SchemaMetrics(graph)
    
    # Get all metrics
    all_metrics = metrics.get_all_metrics()
    
    # Check that metrics are zero or appropriate default values
    assert all_metrics['avg_query_frequency'] == 0
    assert all_metrics['query_frequency_variance'] == 0
    assert all_metrics['avg_relationship_frequency'] == 0
    assert all_metrics['total_size'] == 0
    assert all_metrics['avg_table_size'] == 0
    assert all_metrics['size_variance'] == 0
    assert all_metrics['avg_column_count'] == 0
    assert all_metrics['avg_selectivity'] == 0
    assert all_metrics['selectivity_variance'] == 0
    assert all_metrics['relationship_density'] == 0
    assert all_metrics['avg_degree'] == 0
    assert all_metrics['clustering_coefficient'] == 0
    assert all_metrics['avg_path_length'] == float('inf')

def test_metrics_with_single_table():
    """Test metrics calculation with a single table."""
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
    
    metrics = SchemaMetrics(graph)
    all_metrics = metrics.get_all_metrics()
    
    # Check metrics for single table
    assert all_metrics['avg_query_frequency'] == 0.8
    assert all_metrics['query_frequency_variance'] == 0
    assert all_metrics['avg_relationship_frequency'] == 0
    assert all_metrics['total_size'] == 1000
    assert all_metrics['avg_table_size'] == 1000
    assert all_metrics['size_variance'] == 0
    assert all_metrics['avg_column_count'] == 5
    assert all_metrics['avg_selectivity'] == 0
    assert all_metrics['selectivity_variance'] == 0
    assert all_metrics['relationship_density'] == 0
    assert all_metrics['avg_degree'] == 0
    assert all_metrics['clustering_coefficient'] == 0
    assert all_metrics['avg_path_length'] == float('inf') 