"""Tests for the SchemaGraph class."""

import pytest
import networkx as nx
from quantum_autoencoder.database_optimization.schema.graph import (
    SchemaGraph,
    NodeProperties,
    EdgeProperties
)

def test_create_empty_graph():
    """Test creating an empty schema graph."""
    graph = SchemaGraph()
    assert isinstance(graph.graph, nx.DiGraph)
    assert len(graph.graph) == 0

def test_add_table():
    """Test adding a table to the schema graph."""
    graph = SchemaGraph()
    
    # Add a table with properties
    properties = NodeProperties(
        size=1000,
        column_count=5,
        query_frequency=0.8,
        update_frequency=0.2,
        primary_key='id',
        indexes=['email']
    )
    
    graph.add_table('users', properties)
    
    # Verify table was added correctly
    assert 'users' in graph.graph
    assert graph.graph.nodes['users'] == properties.__dict__

def test_add_relationship():
    """Test adding a relationship between tables."""
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
    
    # Add relationship
    properties = EdgeProperties(
        cardinality='1:N',
        query_frequency=0.9,
        selectivity=0.2,
        foreign_key='user_id'
    )
    
    graph.add_relationship('users', 'orders', properties)
    
    # Verify relationship was added correctly
    assert graph.graph.has_edge('users', 'orders')
    assert graph.graph.edges['users', 'orders'] == properties.__dict__

def test_get_table_properties():
    """Test retrieving table properties."""
    graph = SchemaGraph()
    
    # Add a table
    properties = NodeProperties(
        size=1000,
        column_count=5,
        query_frequency=0.8,
        update_frequency=0.2,
        primary_key='id',
        indexes=['email']
    )
    
    graph.add_table('users', properties)
    
    # Get properties
    retrieved = graph.get_table_properties('users')
    assert retrieved == properties.__dict__

def test_get_relationship_properties():
    """Test retrieving relationship properties."""
    graph = SchemaGraph()
    
    # Add tables and relationship
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
    
    properties = EdgeProperties(
        cardinality='1:N',
        query_frequency=0.9,
        selectivity=0.2,
        foreign_key='user_id'
    )
    
    graph.add_relationship('users', 'orders', properties)
    
    # Get properties
    retrieved = graph.get_relationship_properties('users', 'orders')
    assert retrieved == properties.__dict__

def test_remove_table():
    """Test removing a table from the schema graph."""
    graph = SchemaGraph()
    
    # Add a table
    graph.add_table('users', NodeProperties(
        size=1000,
        column_count=5,
        query_frequency=0.8,
        update_frequency=0.2,
        primary_key='id',
        indexes=['email']
    ))
    
    # Remove table
    graph.remove_table('users')
    
    # Verify table was removed
    assert 'users' not in graph.graph

def test_remove_relationship():
    """Test removing a relationship from the schema graph."""
    graph = SchemaGraph()
    
    # Add tables and relationship
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
    
    graph.add_relationship('users', 'orders', EdgeProperties(
        cardinality='1:N',
        query_frequency=0.9,
        selectivity=0.2,
        foreign_key='user_id'
    ))
    
    # Remove relationship
    graph.remove_relationship('users', 'orders')
    
    # Verify relationship was removed
    assert not graph.graph.has_edge('users', 'orders')

def test_get_tables():
    """Test getting all tables in the schema graph."""
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
    
    # Get tables
    tables = graph.get_tables()
    assert set(tables) == {'users', 'orders'}

def test_get_relationships():
    """Test getting all relationships in the schema graph."""
    graph = SchemaGraph()
    
    # Add tables and relationships
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
    
    graph.add_table('products', NodeProperties(
        size=100,
        column_count=6,
        query_frequency=0.6,
        update_frequency=0.1,
        primary_key='id',
        indexes=['category']
    ))
    
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
    
    # Get relationships
    relationships = graph.get_relationships()
    assert set(relationships) == {('users', 'orders'), ('orders', 'products')}

def test_has_table():
    """Test checking if a table exists in the schema graph."""
    graph = SchemaGraph()
    
    # Add a table
    graph.add_table('users', NodeProperties(
        size=1000,
        column_count=5,
        query_frequency=0.8,
        update_frequency=0.2,
        primary_key='id',
        indexes=['email']
    ))
    
    # Check table existence
    assert graph.has_table('users')
    assert not graph.has_table('nonexistent')

def test_has_relationship():
    """Test checking if a relationship exists in the schema graph."""
    graph = SchemaGraph()
    
    # Add tables and relationship
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
    
    graph.add_relationship('users', 'orders', EdgeProperties(
        cardinality='1:N',
        query_frequency=0.9,
        selectivity=0.2,
        foreign_key='user_id'
    ))
    
    # Check relationship existence
    assert graph.has_relationship('users', 'orders')
    assert not graph.has_relationship('orders', 'users')
    assert not graph.has_relationship('users', 'nonexistent') 