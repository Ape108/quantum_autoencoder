"""
Test quantum autoencoder with Northwind database.
"""

import numpy as np
from pathlib import Path
import sqlite3
from typing import Dict, List, Tuple

from quantum_autoencoder.core.circuit import QuantumAutoencoder
from quantum_autoencoder.database_optimization.schema.graph import SchemaGraph, NodeProperties, EdgeProperties
from quantum_autoencoder.database_optimization.schema.analyzer import SchemaAnalyzer
from quantum_autoencoder.database_optimization.schema.metrics import OptimizationMetrics
from quantum_autoencoder.database_optimization.schema.validation import SchemaValidator

def pad_to_power_of_two(features: np.ndarray) -> np.ndarray:
    """Pad feature vector to the next power of 2."""
    n_features = len(features)
    next_power_of_two = 2 ** int(np.ceil(np.log2(n_features)))
    padded_features = np.zeros(next_power_of_two)
    padded_features[:n_features] = features
    return padded_features

def create_northwind_schema(db_path: str) -> SchemaGraph:
    """Create schema graph from Northwind database."""
    schema = SchemaGraph()
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get table information
    cursor.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name NOT LIKE 'sqlite_%'
    """)
    tables = cursor.fetchall()
    
    # Add tables to schema
    for (table_name,) in tables:
        # Get row count
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        size = cursor.fetchone()[0]
        
        # Get column count
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        column_count = len(columns)
        
        # Get indexes
        cursor.execute(f"PRAGMA index_list({table_name})")
        indexes = [idx[1] for idx in cursor.fetchall()]
        
        # Determine primary key
        primary_key = next((col[1] for col in columns if col[5] == 1), None)
        
        # Estimate query/update frequencies based on table type
        query_freq = 0.8 if table_name in ['Products', 'Orders', 'Customers'] else 0.5
        update_freq = 0.5 if table_name in ['Orders', 'OrderDetails'] else 0.2
        
        schema.add_table(table_name, NodeProperties(
            size=size,
            column_count=column_count,
            query_frequency=query_freq,
            update_frequency=update_freq,
            primary_key=primary_key,
            indexes=indexes
        ))
    
    # Add relationships based on foreign keys
    for (table_name,) in tables:
        cursor.execute(f"PRAGMA foreign_key_list({table_name})")
        foreign_keys = cursor.fetchall()
        
        for fk in foreign_keys:
            target_table = fk[2]  # Referenced table
            from_col = fk[3]      # Column in current table
            to_col = fk[4]        # Column in referenced table
            
            # Calculate selectivity
            cursor.execute(f"""
                SELECT COUNT(DISTINCT {from_col}) * 1.0 / COUNT({from_col})
                FROM {table_name}
            """)
            selectivity = cursor.fetchone()[0] or 0.5
            
            schema.add_relationship(table_name, target_table, EdgeProperties(
                cardinality="N:1",  # Most common in Northwind
                query_frequency=0.8 if table_name in ['Orders', 'OrderDetails'] else 0.6,
                selectivity=selectivity,
                foreign_key=from_col
            ))
    
    conn.close()
    return schema

def main():
    """Run the test."""
    # Load schema from database
    db_path = Path(__file__).parent.parent.parent / "tests/database_optimization/ExampleDB/northwind.db"
    schema = create_northwind_schema(str(db_path))
    
    # Create validator and analyzer
    validator = SchemaValidator(schema)
    analyzer = SchemaAnalyzer(schema)
    metrics = OptimizationMetrics(schema)
    
    # Print original schema description
    print("\nOriginal Schema Description:")
    print("============================")
    print(validator.generate_human_readable_schema())
    
    # Extract features for quantum encoding
    features = []
    
    # Table features
    for _, props in schema.graph.nodes(data=True):
        # Normalize features
        size_norm = props['size'] / max(1, max(n['size'] for _, n in schema.graph.nodes(data=True)))
        col_norm = props['column_count'] / max(1, max(n['column_count'] for _, n in schema.graph.nodes(data=True)))
        features.extend([
            size_norm,
            col_norm,
            props['query_frequency'],
            props['update_frequency']
        ])
    
    # Relationship features
    for _, _, props in schema.graph.edges(data=True):
        features.extend([
            props['query_frequency'],
            props['selectivity'],
            1.0 if props['cardinality'] == 'N:M' else 0.5
        ])
    
    # Convert to numpy array and pad to power of 2
    features = np.array(features)
    padded_features = pad_to_power_of_two(features)
    
    # Calculate required qubits based on padded features
    n_qubits = int(np.log2(len(padded_features)))
    n_latent = n_qubits - 1
    
    print(f"\nCompressing {len(features)} features (padded to {len(padded_features)}) using {n_qubits} qubits -> {n_latent} qubits")
    
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
    compressed = autoencoder.compress_entry(padded_features, np.random.randn(autoencoder.n_parameters) * 0.1)
    reconstructed = autoencoder.decode_features(compressed, np.random.randn(autoencoder.n_parameters) * 0.1)
    
    # Calculate reconstruction error
    error = np.mean((padded_features - np.abs(reconstructed[:len(padded_features)])) ** 2)
    print(f"Reconstruction error: {error:.3f}")
    
    # Generate visualization
    print("\nGenerating visualization...")
    validator.visualize_schema(padded_features, reconstructed[:len(padded_features)])
    
    # Print optimization suggestions
    print("\nOptimization Suggestions:")
    print("========================")
    for suggestion in validator.suggest_optimizations():
        print(f"- {suggestion}")

if __name__ == "__main__":
    main() 