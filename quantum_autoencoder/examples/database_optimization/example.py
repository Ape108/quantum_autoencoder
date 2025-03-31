"""
Example of using quantum autoencoder for database schema optimization.

This script demonstrates how to:
1. Create a database schema graph
2. Analyze the schema for optimization opportunities
3. Convert it to a quantum state
4. Optimize the schema using quantum autoencoder
5. Analyze the results
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

from quantum_autoencoder.database_optimization.schema.graph import SchemaGraph, NodeProperties, EdgeProperties
from quantum_autoencoder.database_optimization.schema.analyzer import SchemaAnalyzer
from quantum_autoencoder.database_optimization.schema.metrics import SchemaMetrics
from quantum_autoencoder.database_optimization.quantum.optimizer import QuantumSchemaOptimizer

def create_example_schema() -> SchemaGraph:
    """
    Create an example database schema for optimization.
    
    Returns:
        SchemaGraph: Example schema graph
    """
    # Create schema graph
    graph = SchemaGraph()
    
    # Add tables (nodes)
    graph.add_table(
        'users',
        NodeProperties(
            size=1000,
            column_count=5,
            query_frequency=0.8,
            update_frequency=0.2,
            primary_key='id',
            indexes=['email']
        )
    )
    
    graph.add_table(
        'orders',
        NodeProperties(
            size=5000,
            column_count=8,
            query_frequency=0.7,
            update_frequency=0.3,
            primary_key='id',
            indexes=['user_id', 'status']
        )
    )
    
    graph.add_table(
        'products',
        NodeProperties(
            size=100,
            column_count=6,
            query_frequency=0.6,
            update_frequency=0.1,
            primary_key='id',
            indexes=['category']
        )
    )
    
    # Add relationships (edges)
    graph.add_relationship(
        'users',
        'orders',
        EdgeProperties(
            cardinality='1:N',
            query_frequency=0.9,
            selectivity=0.2,
            foreign_key='user_id'
        )
    )
    
    graph.add_relationship(
        'orders',
        'products',
        EdgeProperties(
            cardinality='N:M',
            query_frequency=0.8,
            selectivity=0.3,
            foreign_key='product_id'
        )
    )
    
    return graph

def analyze_schema(graph: SchemaGraph) -> None:
    """
    Analyze the database schema.
    
    Args:
        graph: Schema graph to analyze
    """
    # Create analyzer
    analyzer = SchemaAnalyzer(graph)
    
    # Analyze patterns
    query_patterns = analyzer.analyze_query_patterns()
    storage_patterns = analyzer.analyze_storage_patterns()
    relationship_patterns = analyzer.analyze_relationship_patterns()
    
    # Print analysis results
    print("\nAnalyzing schema...\n")
    
    print("Query patterns:")
    print(f"  high_frequency_tables: {query_patterns['high_frequency_tables']}")
    print(f"  low_frequency_tables: {query_patterns['low_frequency_tables']}")
    print(f"  hot_paths: {query_patterns['hot_paths']}")
    print(f"  cold_paths: {query_patterns['cold_paths']}\n")
    
    print("Storage patterns:")
    print(f"  large_tables: {storage_patterns['large_tables']}")
    print(f"  small_tables: {storage_patterns['small_tables']}")
    print(f"  wide_tables: {storage_patterns['wide_tables']}")
    print(f"  narrow_tables: {storage_patterns['narrow_tables']}\n")
    
    print("Relationship patterns:")
    print(f"  high_selectivity: {relationship_patterns['high_selectivity']}")
    print(f"  low_selectivity: {relationship_patterns['low_selectivity']}")
    print(f"  complex_relationships: {relationship_patterns['complex_relationships']}")
    print(f"  simple_relationships: {relationship_patterns['simple_relationships']}\n")
    
    # Validate schema
    validation = analyzer.validate_schema()
    print("Schema validation:")
    print(f"  has_cycles: {'✓' if validation['has_cycles'] else '✗'}")
    print(f"  has_isolated_tables: {'✓' if validation['has_isolated_tables'] else '✗'}")
    print(f"  has_redundant_indexes: {'✓' if validation['has_redundant_indexes'] else '✗'}")
    print(f"  has_missing_foreign_keys: {'✓' if validation['has_missing_foreign_keys'] else '✗'}\n")
    
    # Get optimization suggestions
    suggestions = analyzer.suggest_optimizations()
    print("Optimization suggestions:")
    for suggestion in suggestions:
        print(f"  • {suggestion}")
    print()

def plot_optimization_progress(history: List[Dict]) -> None:
    """
    Plot the optimization progress.
    
    Args:
        history: List of optimization history entries
    """
    iterations = [entry['iteration'] for entry in history]
    costs = [entry['cost'] for entry in history]
    
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, costs, 'b-', label='Cost')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Optimization Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig('optimization_progress.png')
    plt.close()

def main() -> None:
    """Run the example."""
    print("Creating example schema...")
    graph = create_example_schema()
    
    # Analyze schema
    analyze_schema(graph)
    
    # Calculate initial metrics
    print("Calculating initial metrics...")
    metrics = SchemaMetrics(graph)
    initial_metrics = metrics.get_all_metrics()
    print("Initial metrics:")
    for name, value in initial_metrics.items():
        print(f"  {name}: {value:.4f}")
    print()
    
    # Initialize quantum optimizer
    print("Initializing quantum optimizer...")
    optimizer = QuantumSchemaOptimizer(
        n_qubits=4,
        n_latent=2,
        shots=1024,
        tol=1e-4
    )
    
    # Optimize schema
    print("Optimizing schema...")
    best_params, best_cost = optimizer.optimize_schema(
        graph,
        max_iterations=100
    )
    print(f"Best cost achieved: {best_cost:.4f}\n")
    
    # Get optimized schema
    print("Getting optimized schema...")
    optimized_features = optimizer.get_optimized_schema(graph)
    
    # Analyze optimization results
    print("\nAnalyzing optimization results...")
    analysis = optimizer.analyze_optimization()
    print("Optimization analysis:")
    for name, value in analysis.items():
        print(f"  {name}: {value:.4f}")
    print()
    
    # Print optimization summary
    print(f"Optimization completed in {len(optimizer.get_optimization_history())} iterations\n")
    
    # Plot optimization progress
    plot_optimization_progress(optimizer.get_optimization_history())
    print("Optimization progress plot saved as 'optimization_progress.png'\n")

if __name__ == '__main__':
    main() 