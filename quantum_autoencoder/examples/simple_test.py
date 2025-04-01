"""
Simple test script for quantum autoencoder database optimization.
"""

import json
import numpy as np
from pathlib import Path
from quantum_autoencoder.core.circuit import QuantumAutoencoder
from quantum_autoencoder.database_optimization.schema.graph import SchemaGraph, NodeProperties, EdgeProperties
from quantum_autoencoder.database_optimization.schema.validation import SchemaValidator

def load_schema_from_json(json_path: str) -> SchemaGraph:
    """Load schema from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    schema = SchemaGraph()
    
    # Add tables
    for table_name, props in data['tables'].items():
        schema.add_table(table_name, NodeProperties(
            size=props['size'],
            column_count=props['column_count'],
            query_frequency=props['query_frequency'],
            update_frequency=props['update_frequency'],
            primary_key=props['primary_key'],
            indexes=props['indexes']
        ))
    
    # Add relationships
    for rel in data['relationships']:
        schema.add_relationship(rel['source'], rel['target'], EdgeProperties(
            cardinality=rel['cardinality'],
            query_frequency=rel['query_frequency'],
            selectivity=rel['selectivity'],
            foreign_key=rel['foreign_key']
        ))
    
    return schema

def main():
    """Run simple schema optimization test."""
    # Load schema
    json_path = Path(__file__).parent / 'sample_schema.json'
    schema = load_schema_from_json(str(json_path))
    
    # Create validator
    validator = SchemaValidator(schema)
    
    # Print original schema
    print("\nOriginal Schema:")
    print("===============")
    print(validator.generate_human_readable_schema())
    
    # Extract features
    features = []
    for _, props in schema.graph.nodes(data=True):
        # Normalize features
        size_norm = min(props['size'] / 200000, 1.0)
        col_norm = min(props['column_count'] / 20, 1.0)
        features.extend([
            size_norm,
            col_norm,
            props['query_frequency'],
            props['update_frequency']
        ])
    
    for _, _, props in schema.graph.edges(data=True):
        features.extend([
            props['query_frequency'],
            props['selectivity'],
            1.0 if props['cardinality'] == 'N:M' else 0.5
        ])
    
    features = np.array(features)
    
    # Initialize quantum autoencoder
    n_qubits = int(np.ceil(np.log2(len(features))))
    n_latent = n_qubits - 1
    
    print(f"\nCompressing {len(features)} features using {n_qubits} qubits -> {n_latent} qubits")
    
    autoencoder = QuantumAutoencoder(
        n_qubits=n_qubits,
        n_latent=n_latent,
        reps=2,
        options={
            "shots": 1024,
            "optimization_level": 3,
            "resilience_level": 1
        }
    )
    
    # Test compression
    print("\nTesting compression...")
    compressed = autoencoder.compress_entry(features, np.random.randn(autoencoder.n_parameters) * 0.1)
    reconstructed = autoencoder.decode_features(compressed, np.random.randn(autoencoder.n_parameters) * 0.1)
    
    # Generate visualization
    print("\nGenerating visualization...")
    validator.visualize_schema(features, reconstructed)
    print("Visualization saved to visualizations/schema_comparison.png")
    
    # Print comparison
    print("\nOptimization Results:")
    print("===================")
    print(validator.generate_optimization_comparison(features, reconstructed))

if __name__ == "__main__":
    main() 