"""Tests for the SchemaAnalyzer class."""

import pytest
from quantum_autoencoder.database_optimization.schema.graph import (
    SchemaGraph,
    NodeProperties,
    EdgeProperties
)
from quantum_autoencoder.database_optimization.schema.analyzer import SchemaAnalyzer

def create_test_schema():
    """Create a test schema for analysis."""
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

def test_analyze_query_patterns():
    """Test analyzing query patterns in the schema."""
    graph = create_test_schema()
    analyzer = SchemaAnalyzer(graph)
    
    patterns = analyzer.analyze_query_patterns()
    
    # Check high frequency tables
    assert 'users' in patterns['high_frequency_tables']
    assert len(patterns['high_frequency_tables']) == 1
    
    # Check low frequency tables
    assert 'products' in patterns['low_frequency_tables']
    assert len(patterns['low_frequency_tables']) == 1
    
    # Check hot paths
    assert ('users', 'orders') in patterns['hot_paths']
    assert len(patterns['hot_paths']) == 1
    
    # Check cold paths
    assert ('orders', 'products') in patterns['cold_paths']
    assert len(patterns['cold_paths']) == 1

def test_analyze_storage_patterns():
    """Test analyzing storage patterns in the schema."""
    graph = create_test_schema()
    analyzer = SchemaAnalyzer(graph)
    
    patterns = analyzer.analyze_storage_patterns()
    
    # Check large tables
    assert 'orders' in patterns['large_tables']
    assert len(patterns['large_tables']) == 1
    
    # Check small tables
    assert 'products' in patterns['small_tables']
    assert len(patterns['small_tables']) == 1
    
    # Check wide tables
    assert 'orders' in patterns['wide_tables']
    assert len(patterns['wide_tables']) == 1
    
    # Check narrow tables
    assert 'users' in patterns['narrow_tables']
    assert len(patterns['narrow_tables']) == 1

def test_analyze_relationship_patterns():
    """Test analyzing relationship patterns in the schema."""
    graph = create_test_schema()
    analyzer = SchemaAnalyzer(graph)
    
    patterns = analyzer.analyze_relationship_patterns()
    
    # Check high selectivity relationships
    assert ('orders', 'products') in patterns['high_selectivity']
    assert len(patterns['high_selectivity']) == 1
    
    # Check low selectivity relationships
    assert ('users', 'orders') in patterns['low_selectivity']
    assert len(patterns['low_selectivity']) == 1
    
    # Check complex relationships
    assert ('orders', 'products') in patterns['complex_relationships']
    assert len(patterns['complex_relationships']) == 1
    
    # Check simple relationships
    assert ('users', 'orders') in patterns['simple_relationships']
    assert len(patterns['simple_relationships']) == 1

def test_validate_schema():
    """Test schema validation."""
    graph = create_test_schema()
    analyzer = SchemaAnalyzer(graph)
    
    validation = analyzer.validate_schema()
    
    # Check validation results
    assert not validation['has_cycles']
    assert not validation['has_isolated_tables']
    assert not validation['has_redundant_indexes']
    assert not validation['has_missing_foreign_keys']

def test_validate_schema_with_cycle():
    """Test schema validation with a cycle."""
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
    
    # Create a cycle
    graph.add_relationship('A', 'B', EdgeProperties(
        cardinality='1:N',
        query_frequency=0.9,
        selectivity=0.2,
        foreign_key='b_id'
    ))
    
    graph.add_relationship('B', 'A', EdgeProperties(
        cardinality='1:N',
        query_frequency=0.9,
        selectivity=0.2,
        foreign_key='a_id'
    ))
    
    analyzer = SchemaAnalyzer(graph)
    validation = analyzer.validate_schema()
    
    assert validation['has_cycles']

def test_validate_schema_with_isolated_table():
    """Test schema validation with an isolated table."""
    graph = create_test_schema()
    
    # Add isolated table
    graph.add_table('isolated', NodeProperties(
        size=1000,
        column_count=5,
        query_frequency=0.8,
        update_frequency=0.2,
        primary_key='id',
        indexes=['email']
    ))
    
    analyzer = SchemaAnalyzer(graph)
    validation = analyzer.validate_schema()
    
    assert validation['has_isolated_tables']

def test_validate_schema_with_redundant_index():
    """Test schema validation with redundant indexes."""
    graph = SchemaGraph()
    
    # Add table with redundant indexes
    graph.add_table('users', NodeProperties(
        size=1000,
        column_count=5,
        query_frequency=0.8,
        update_frequency=0.2,
        primary_key='id',
        indexes=['email', 'email']  # Redundant index
    ))
    
    analyzer = SchemaAnalyzer(graph)
    validation = analyzer.validate_schema()
    
    assert validation['has_redundant_indexes']

def test_validate_schema_with_missing_foreign_key():
    """Test schema validation with missing foreign key."""
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
        indexes=['user_id']
    ))
    
    # Add relationship without foreign key
    graph.add_relationship('users', 'orders', EdgeProperties(
        cardinality='1:N',
        query_frequency=0.9,
        selectivity=0.2,
        foreign_key=None  # Missing foreign key
    ))
    
    analyzer = SchemaAnalyzer(graph)
    validation = analyzer.validate_schema()
    
    assert validation['has_missing_foreign_keys']

def test_suggest_optimizations():
    """Test generating optimization suggestions."""
    graph = create_test_schema()
    analyzer = SchemaAnalyzer(graph)
    
    suggestions = analyzer.suggest_optimizations()
    
    # Check for expected suggestions
    assert any('users' in s for s in suggestions)  # High frequency table
    assert any('orders' in s for s in suggestions)  # Large table
    assert any('users -> orders' in s for s in suggestions)  # Low selectivity relationship
    assert any('orders -> products' in s for s in suggestions)  # Complex relationship 