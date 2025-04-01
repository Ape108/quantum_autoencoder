"""
Test quantum autoencoder for database schema optimization.
"""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from quantum_autoencoder.core.circuit import QuantumAutoencoder
from quantum_autoencoder.database_optimization.schema.graph import SchemaGraph, NodeProperties, EdgeProperties
from quantum_autoencoder.database_optimization.schema.analyzer import SchemaAnalyzer
from quantum_autoencoder.database_optimization.schema.metrics import OptimizationMetrics
from quantum_autoencoder.database_optimization.schema.validation import SchemaValidator

def create_ecommerce_schema() -> SchemaGraph:
    """Create a realistic e-commerce database schema."""
    schema = SchemaGraph()
    
    # Add tables with realistic properties
    schema.add_table("users", NodeProperties(
        size=10000,  # 10K users
        column_count=8,  # id, name, email, password, created_at, updated_at, last_login, status
        query_frequency=0.9,  # High query frequency
        update_frequency=0.3,  # Moderate updates
        primary_key="id",
        indexes=["email", "created_at"]
    ))
    
    schema.add_table("products", NodeProperties(
        size=5000,  # 5K products
        column_count=12,  # id, name, description, price, stock, category_id, etc.
        query_frequency=0.8,  # High query frequency
        update_frequency=0.4,  # Moderate updates
        primary_key="id",
        indexes=["category_id", "price"]
    ))
    
    schema.add_table("orders", NodeProperties(
        size=50000,  # 50K orders
        column_count=10,  # id, user_id, total, status, created_at, etc.
        query_frequency=0.7,  # High query frequency
        update_frequency=0.5,  # High updates
        primary_key="id",
        indexes=["user_id", "created_at"]
    ))
    
    schema.add_table("order_items", NodeProperties(
        size=150000,  # 150K order items
        column_count=6,  # id, order_id, product_id, quantity, price, etc.
        query_frequency=0.6,  # Moderate query frequency
        update_frequency=0.2,  # Low updates
        primary_key="id",
        indexes=["order_id", "product_id"]
    ))
    
    schema.add_table("categories", NodeProperties(
        size=100,  # 100 categories
        column_count=4,  # id, name, parent_id, description
        query_frequency=0.5,  # Moderate query frequency
        update_frequency=0.1,  # Low updates
        primary_key="id",
        indexes=["parent_id"]
    ))
    
    # Add relationships with realistic properties
    schema.add_relationship("orders", "users", EdgeProperties(
        cardinality="N:1",  # Many orders per user
        query_frequency=0.9,  # High query frequency
        selectivity=0.8,  # High selectivity
        foreign_key="user_id"
    ))
    
    schema.add_relationship("order_items", "orders", EdgeProperties(
        cardinality="N:1",  # Many items per order
        query_frequency=0.8,  # High query frequency
        selectivity=0.7,  # High selectivity
        foreign_key="order_id"
    ))
    
    schema.add_relationship("order_items", "products", EdgeProperties(
        cardinality="N:1",  # Many items per product
        query_frequency=0.7,  # High query frequency
        selectivity=0.6,  # Moderate selectivity
        foreign_key="product_id"
    ))
    
    schema.add_relationship("products", "categories", EdgeProperties(
        cardinality="N:1",  # Many products per category
        query_frequency=0.6,  # Moderate query frequency
        selectivity=0.5,  # Moderate selectivity
        foreign_key="category_id"
    ))
    
    schema.add_relationship("categories", "categories", EdgeProperties(
        cardinality="N:1",  # Hierarchical categories
        query_frequency=0.4,  # Low query frequency
        selectivity=0.3,  # Low selectivity
        foreign_key="parent_id"
    ))
    
    return schema

def extract_graph_features(schema: SchemaGraph) -> np.ndarray:
    """Extract features from the schema graph for quantum encoding."""
    n_tables = len(schema.graph.nodes)
    features = []
    
    # Extract node features
    for table, props in schema.graph.nodes(data=True):
        # Normalize features to [0,1] range
        size_norm = min(props['size'] / 200000, 1.0)  # Cap at 200K
        col_norm = min(props['column_count'] / 20, 1.0)  # Cap at 20 columns
        query_freq = props['query_frequency']
        update_freq = props['update_frequency']
        
        features.extend([size_norm, col_norm, query_freq, update_freq])
    
    # Extract edge features
    for edge in schema.graph.edges(data=True):
        source, target, props = edge
        # Normalize features
        query_freq = props['query_frequency']
        selectivity = props['selectivity']
        
        # Encode cardinality
        if props['cardinality'] == '1:1':
            cardinality = 0.0
        elif props['cardinality'] == '1:N':
            cardinality = 0.5
        else:  # N:M
            cardinality = 1.0
            
        features.extend([query_freq, selectivity, cardinality])
    
    return np.array(features)

def test_database_compression():
    """Test quantum autoencoder for database schema optimization."""
    # Create test schema
    schema = create_ecommerce_schema()
    
    # Initialize validators and analyzers
    analyzer = SchemaAnalyzer(schema)
    metrics = OptimizationMetrics(schema)
    validator = SchemaValidator(schema)
    
    # Generate human-readable schema description
    print("\nOriginal Schema Description:")
    print("============================")
    print(validator.generate_human_readable_schema())
    
    # Validate original schema
    print("\nOriginal Schema Validation:")
    print("==========================")
    validation_results = validator.validate_schema()
    for category, messages in validation_results.items():
        if messages:
            print(f"\n{category.title()}:")
            for msg in messages:
                print(f"- {msg}")
    
    # Analyze schema
    query_patterns = analyzer.analyze_query_patterns()
    storage_patterns = analyzer.analyze_storage_patterns()
    relationship_patterns = analyzer.analyze_relationship_patterns()
    
    print("\nSchema Analysis Results:")
    print("=======================")
    print("Query Patterns:", query_patterns)
    print("Storage Patterns:", storage_patterns)
    print("Relationship Patterns:", relationship_patterns)
    
    # Extract features for quantum encoding
    features = extract_graph_features(schema)
    print(f"\nExtracted {len(features)} features from schema")
    
    # Calculate required qubits based on feature dimension
    n_qubits = int(np.ceil(np.log2(len(features))))
    n_latent = n_qubits - 1
    
    print(f"\nUsing {n_qubits} qubits for {len(features)} features")
    print(f"Compressing to {n_latent} qubits")
    
    # Initialize quantum autoencoder
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
    
    # Train the autoencoder
    print("\nTraining quantum autoencoder...")
    print(f"Input qubits: {n_qubits}")
    print(f"Latent qubits: {n_latent}")
    
    # Initialize parameters randomly
    n_params = len(autoencoder.encoder_u.parameters) + len(autoencoder.encoder_v.parameters)
    parameters = np.random.randn(n_params) * 0.1
    
    # Pad features to match qubit count
    padded_features = np.zeros(2**n_qubits, dtype=np.complex128)
    padded_features[:len(features)] = features.astype(np.complex128)
    padded_features = padded_features / np.linalg.norm(padded_features)
    
    # Compress and reconstruct
    print("\nTesting compression and reconstruction...")
    print(f"Original dimension: {len(features)}")
    print(f"Compressed dimension: {2**n_latent}")
    
    # Compress features
    compressed_state = autoencoder.compress_entry(padded_features, parameters)
    
    # Reconstruct features
    reconstructed_features = autoencoder.decode_features(compressed_state, parameters)
    reconstructed_features = reconstructed_features[:len(features)]
    
    # Calculate reconstruction error
    error = np.linalg.norm(features - reconstructed_features)
    print(f"\nReconstruction error: {error}")
    
    # Calculate optimization fidelity
    original_circuit = QuantumCircuit(n_qubits)
    original_circuit.initialize(padded_features, range(n_qubits))
    
    # Prepare reconstructed features for quantum circuit
    padded_reconstructed = np.zeros(2**n_qubits, dtype=np.complex128)
    padded_reconstructed[:len(features)] = reconstructed_features
    padded_reconstructed = padded_reconstructed / np.linalg.norm(padded_reconstructed)
    
    reconstructed_circuit = QuantumCircuit(n_qubits)
    reconstructed_circuit.initialize(padded_reconstructed, range(n_qubits))
    
    optimization_scores = metrics.get_optimization_fidelity(original_circuit, reconstructed_circuit)
    
    print("\nOptimization Scores:")
    print("===================")
    print(f"Query Optimization:      {optimization_scores['query']:.4f}")
    print(f"Storage Optimization:    {optimization_scores['storage']:.4f}")
    print(f"Relationship Optimization: {optimization_scores['relationship']:.4f}")
    print(f"Complexity Reduction:    {optimization_scores['complexity']:.4f}")
    print(f"Total Optimization Score: {optimization_scores['total']:.4f}")
    
    # Generate optimization suggestions
    print("\nOptimization Suggestions:")
    print("========================")
    suggestions = validator.suggest_optimizations()
    for suggestion in suggestions:
        print(f"- {suggestion}")
    
    # Generate before/after comparison
    print("\nBefore/After Comparison:")
    print("=======================")
    print(validator.generate_optimization_comparison(features, reconstructed_features))
    
    # Generate schema visualization
    print("\nGenerating schema visualization...")
    validator.visualize_schema(features, reconstructed_features)
    print("Visualization saved to visualizations/schema_comparison.png")
    
    # Validate optimized schema
    print("\nOptimized Schema Validation:")
    print("===========================")
    validation_results = validator.validate_schema()
    for category, messages in validation_results.items():
        if messages:
            print(f"\n{category.title()}:")
            for msg in messages:
                print(f"- {msg}")

if __name__ == "__main__":
    test_database_compression() 