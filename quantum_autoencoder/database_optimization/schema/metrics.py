"""
Optimization metrics for database schema analysis.

This module provides metrics to evaluate schema optimization quality,
focusing on query efficiency, storage optimization, and relationship structure.
"""

import numpy as np
from typing import Dict, List, Tuple
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from .graph import SchemaGraph

class OptimizationMetrics:
    """Calculate optimization metrics for database schemas."""
    
    def __init__(self, schema: SchemaGraph):
        """
        Initialize optimization metrics calculator.
        
        Args:
            schema: Schema graph to analyze
        """
        self.schema = schema
        self.n_tables = len(schema.graph.nodes)
        self.n_relationships = len(schema.graph.edges)
        
        # Weights for different optimization aspects
        self.weights = {
            'query': 0.4,      # Prioritize query performance
            'storage': 0.2,    # Storage efficiency
            'relationship': 0.3,  # Relationship optimization
            'complexity': 0.1   # Schema simplification
        }
    
    def reshape_features(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reshape feature vector into table and relationship features.
        
        Args:
            features: Feature vector from quantum state
            
        Returns:
            Tuple of (table_features, relationship_features)
        """
        # Table features: [size, columns, query_freq, update_freq]
        table_features = features[:self.n_tables * 4].reshape(-1, 4)
        
        # Relationship features: [query_freq, selectivity, cardinality]
        relationship_features = features[self.n_tables * 4:].reshape(-1, 3)
        
        return table_features, relationship_features
    
    def calculate_query_score(self, 
                            opt_table_features: np.ndarray,
                            orig_table_features: np.ndarray,
                            opt_rel_features: np.ndarray,
                            orig_rel_features: np.ndarray) -> float:
        """
        Calculate query optimization score.
        Higher score means frequently accessed data is more efficiently structured.
        """
        # Table query optimization
        table_score = np.mean(
            opt_table_features[:, 2] * orig_table_features[:, 2]  # Query frequency alignment
        )
        
        # Relationship query optimization
        rel_score = np.mean(
            opt_rel_features[:, 0] * orig_rel_features[:, 0]  # Query frequency alignment
        )
        
        return 0.6 * table_score + 0.4 * rel_score
    
    def calculate_storage_score(self,
                              opt_table_features: np.ndarray,
                              orig_table_features: np.ndarray) -> float:
        """
        Calculate storage optimization score.
        Higher score means better storage efficiency for access patterns.
        """
        # Reward size reduction for infrequently queried tables
        infreq_reduction = np.mean(
            (1 - opt_table_features[:, 0]) * (1 - orig_table_features[:, 2])
        )
        
        # Reward size preservation for frequently queried tables
        freq_preservation = np.mean(
            opt_table_features[:, 0] * orig_table_features[:, 2]
        )
        
        return 0.5 * infreq_reduction + 0.5 * freq_preservation
    
    def calculate_relationship_score(self,
                                  opt_rel_features: np.ndarray,
                                  orig_rel_features: np.ndarray) -> float:
        """
        Calculate relationship optimization score.
        Higher score means better relationship structure for query patterns.
        """
        # Reward high selectivity for frequent queries
        selectivity_score = np.mean(
            opt_rel_features[:, 1] * orig_rel_features[:, 0]
        )
        
        # Reward simpler cardinality for frequent queries
        cardinality_score = np.mean(
            (1 - opt_rel_features[:, 2]) * orig_rel_features[:, 0]
        )
        
        return 0.7 * selectivity_score + 0.3 * cardinality_score
    
    def calculate_complexity_score(self,
                                 opt_table_features: np.ndarray,
                                 orig_table_features: np.ndarray) -> float:
        """
        Calculate complexity reduction score.
        Higher score means simpler schema without losing functionality.
        """
        # Reward column count reduction while preserving query frequency
        return 1 - np.mean(
            (opt_table_features[:, 1] / orig_table_features[:, 1]) *
            (1 - orig_table_features[:, 2])  # Weight by inverse query frequency
        )
    
    def get_optimization_fidelity(self,
                                original_circuit: QuantumCircuit,
                                optimized_circuit: QuantumCircuit) -> Dict[str, float]:
        """
        Calculate optimization fidelity metrics.
        
        Args:
            original_circuit: Original schema quantum circuit
            optimized_circuit: Optimized schema quantum circuit
            
        Returns:
            Dictionary of optimization metrics
        """
        # Get feature vectors
        orig_features = np.abs(Statevector(original_circuit).data)[:self.n_tables * 4 + self.n_relationships * 3]
        opt_features = np.abs(Statevector(optimized_circuit).data)[:self.n_tables * 4 + self.n_relationships * 3]
        
        # Reshape features
        orig_table_features, orig_rel_features = self.reshape_features(orig_features)
        opt_table_features, opt_rel_features = self.reshape_features(opt_features)
        
        # Calculate component scores
        scores = {
            'query': self.calculate_query_score(
                opt_table_features, orig_table_features,
                opt_rel_features, orig_rel_features
            ),
            'storage': self.calculate_storage_score(
                opt_table_features, orig_table_features
            ),
            'relationship': self.calculate_relationship_score(
                opt_rel_features, orig_rel_features
            ),
            'complexity': self.calculate_complexity_score(
                opt_table_features, orig_table_features
            )
        }
        
        # Calculate weighted total
        scores['total'] = sum(
            self.weights[key] * score
            for key, score in scores.items()
            if key != 'total'
        )
        
        return scores 