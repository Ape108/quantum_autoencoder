"""
Quantum state conversion for database schemas.

This module provides functionality for converting database schema graphs into
quantum states suitable for quantum autoencoder optimization.
"""

import numpy as np
from typing import Dict, List, Tuple
from qiskit.quantum_info import Statevector
from ..schema.graph import SchemaGraph, NodeProperties, EdgeProperties

class QuantumStateConverter:
    """Converts database schema graphs to quantum states."""
    
    def __init__(self, n_qubits: int = 4):
        """
        Initialize the state converter.
        
        Args:
            n_qubits: Number of qubits for quantum state representation
        """
        self.n_qubits = n_qubits
        self.state_size = 2**n_qubits
        
    def _extract_node_features(self, graph: SchemaGraph) -> np.ndarray:
        """
        Extract features from graph nodes.
        
        Args:
            graph: Schema graph to analyze
            
        Returns:
            Array of node features
        """
        features = []
        for props in graph.graph.nodes.values():
            node_features = [
                props.size,
                props.column_count,
                props.query_frequency,
                props.update_frequency
            ]
            features.extend(node_features)
        return np.array(features)
        
    def _extract_edge_features(self, graph: SchemaGraph) -> np.ndarray:
        """
        Extract features from graph edges.
        
        Args:
            graph: Schema graph to analyze
            
        Returns:
            Array of edge features
        """
        features = []
        for props in graph.graph.edges.values():
            # Convert cardinality to numeric value
            card_map = {'1:1': 1, '1:N': 2, 'N:M': 3}
            card_value = card_map.get(props['cardinality'], 0)
            
            edge_features = [
                card_value,
                props['query_frequency'],
                props['selectivity']
            ]
            features.extend(edge_features)
        return np.array(features)
        
    def _extract_graph_features(self, graph: SchemaGraph) -> np.ndarray:
        """
        Extract features from the entire graph structure.
        
        Args:
            graph: Schema graph to analyze
            
        Returns:
            Array of graph features
        """
        # Get node and edge features
        node_features = self._extract_node_features(graph)
        edge_features = self._extract_edge_features(graph)
        
        # Combine features
        graph_features = np.concatenate([node_features, edge_features])
        
        # Normalize features
        max_val = np.max(np.abs(graph_features))
        if max_val > 0:
            graph_features = graph_features / max_val
            
        return graph_features
        
    def _pad_features(self, features: np.ndarray) -> np.ndarray:
        """
        Pad features to match quantum state size.
        
        Args:
            features: Feature array to pad
            
        Returns:
            Padded feature array
        """
        if len(features) < self.state_size:
            return np.pad(features, (0, self.state_size - len(features)))
        elif len(features) > self.state_size:
            return features[:self.state_size]
        return features
        
    def to_quantum_state(self, graph: SchemaGraph) -> Statevector:
        """
        Convert schema graph to quantum state.
        
        Args:
            graph: Schema graph to convert
            
        Returns:
            Quantum state vector
        """
        # Extract and normalize features
        features = self._extract_graph_features(graph)
        
        # Pad features to match quantum state size
        padded_features = self._pad_features(features)
        
        # Create quantum state
        return Statevector(padded_features)
        
    def from_quantum_state(self, state: Statevector, graph: SchemaGraph) -> Dict[str, np.ndarray]:
        """
        Convert quantum state back to schema features.
        
        Args:
            state: Quantum state to convert
            graph: Original schema graph for reference
            
        Returns:
            Dictionary of reconstructed features
        """
        # Get original feature sizes
        node_features = self._extract_node_features(graph)
        edge_features = self._extract_edge_features(graph)
        
        # Extract features from quantum state
        state_data = np.abs(state.data)
        
        # Split into node and edge features
        node_size = len(node_features)
        edge_size = len(edge_features)
        
        reconstructed_node = state_data[:node_size]
        reconstructed_edge = state_data[node_size:node_size + edge_size]
        
        return {
            'node_features': reconstructed_node,
            'edge_features': reconstructed_edge
        }
        
    def calculate_fidelity(self, original: Statevector, reconstructed: Statevector) -> float:
        """
        Calculate fidelity between original and reconstructed states.
        
        Args:
            original: Original quantum state
            reconstructed: Reconstructed quantum state
            
        Returns:
            Fidelity value between 0 and 1
        """
        return np.abs(np.vdot(original.data, reconstructed.data))**2 