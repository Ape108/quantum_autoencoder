"""
Example of using quantum autoencoder for database schema optimization.

This script demonstrates how to:
1. Create a database schema graph
2. Convert it to a quantum state
3. Optimize the schema using quantum autoencoder
4. Analyze the results
"""

import numpy as np
from quantum_autoencoder.database_optimization import (
    SchemaGraph,
    NodeProperties,
    EdgeProperties,
    SchemaMetrics,
    QuantumSchemaOptimizer
)

def create_example_schema() -> SchemaGraph:
    """Create an example database schema graph."""
    # Create graph
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
        query_frequency=0.6,
        update_frequency=0.4,
        primary_key='id',
        indexes=['user_id', 'status']
    ))
    
    graph.add_table('products', NodeProperties(
        size=100,
        column_count=6,
        query_frequency=0.7,
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
    
    graph.add_relationship('products', 'orders', EdgeProperties(
        cardinality='1:N',
        query_frequency=0.8,
        selectivity=0.3,
        foreign_key='product_id'
    ))
    
    return graph

def main():
    """Run the example."""
    # Create example schema
    print("Creating example schema...")
    graph = create_example_schema()
    
    # Calculate initial metrics
    print("\nCalculating initial metrics...")
    metrics = SchemaMetrics(graph)
    initial_metrics = metrics.get_all_metrics()
    print("Initial metrics:")
    for key, value in initial_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Create and run optimizer
    print("\nInitializing quantum optimizer...")
    optimizer = QuantumSchemaOptimizer(
        n_qubits=4,
        n_latent=2,
        shots=1024,
        optimizer='COBYLA'
    )
    
    print("Optimizing schema...")
    best_params, best_cost = optimizer.optimize_schema(
        graph,
        max_iterations=100
    )
    print(f"Best cost achieved: {best_cost:.4f}")
    
    # Get optimized schema
    print("\nGetting optimized schema...")
    optimized_features = optimizer.get_optimized_schema(graph)
    
    # Analyze optimization
    print("\nAnalyzing optimization results...")
    analysis = optimizer.analyze_optimization()
    print("Optimization analysis:")
    for key, value in analysis.items():
        print(f"  {key}: {value:.4f}")
    
    # Get optimization history
    history = optimizer.get_optimization_history()
    print(f"\nOptimization completed in {len(history)} iterations")
    
    # Plot optimization progress
    try:
        import matplotlib.pyplot as plt
        
        costs = [entry['cost'] for entry in history]
        plt.figure(figsize=(10, 6))
        plt.plot(costs)
        plt.title('Optimization Progress')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.grid(True)
        plt.savefig('optimization_progress.png')
        print("\nOptimization progress plot saved as 'optimization_progress.png'")
    except ImportError:
        print("\nMatplotlib not available, skipping plot generation")

if __name__ == '__main__':
    main() 