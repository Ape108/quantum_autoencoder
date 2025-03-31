"""
Quantum state conversion for database schemas.

This module provides functionality for converting database schemas
to quantum states and back.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from qiskit.quantum_info import Statevector
from ..schema.graph import SchemaGraph

class QuantumStateConverter:
    """Converts database schemas to quantum states and back."""
    
    def __init__(self, n_qubits: int = 4):
        """
        Initialize the state converter.
        
        Args:
            n_qubits: Number of qubits for state representation
        """
        self.n_qubits = n_qubits
        
    def _extract_node_features(self, graph: SchemaGraph) -> np.ndarray:
        """
        Extract features from graph nodes.
        
        Args:
            graph: Schema graph
            
        Returns:
            Array of node features
        """
        features = []
        
        for _, props in graph.graph.nodes(data=True):
            node_features = [
                props['size'],
                props['column_count'],
                props['query_frequency'],
                props['update_frequency']
            ]
            features.extend(node_features)
            
        return np.array(features, dtype=np.float64)
        
    def _extract_edge_features(self, graph: SchemaGraph) -> np.ndarray:
        """
        Extract features from graph edges.
        
        Args:
            graph: Schema graph
            
        Returns:
            Array of edge features
        """
        features = []
        
        for edge in graph.graph.edges(data=True):
            _, _, props = edge
            edge_features = [
                1.0 if props['cardinality'] == '1:1' else 0.0,
                1.0 if props['cardinality'] == '1:N' else 0.0,
                1.0 if props['cardinality'] == 'N:M' else 0.0,
                props['query_frequency'],
                props['selectivity']
            ]
            features.extend(edge_features)
            
        return np.array(features, dtype=np.float64)
        
    def _extract_graph_features(self, graph: SchemaGraph) -> np.ndarray:
        """
        Extract all features from the graph.
        
        Args:
            graph: Schema graph
            
        Returns:
            Array of all features
        """
        # Extract features
        node_features = self._extract_node_features(graph)
        edge_features = self._extract_edge_features(graph)
        
        # Combine features
        features = np.concatenate([node_features, edge_features])
        
        # Normalize features
        if len(features) > 0:
            features = (features - np.mean(features)) / (np.std(features) + 1e-8)
        
        return features
        
    def _features_to_amplitudes(self, features: np.ndarray) -> np.ndarray:
        """
        Convert features to quantum state amplitudes.
        
        Args:
            features: Feature array
            
        Returns:
            Array of amplitudes
        """
        # Pad or truncate features to match number of amplitudes
        n_amplitudes = 2 ** self.n_qubits
        if len(features) < n_amplitudes:
            features = np.pad(features, (0, n_amplitudes - len(features)))
        else:
            features = features[:n_amplitudes]
            
        # Normalize to create valid quantum state
        norm = np.sqrt(np.sum(features ** 2) + 1e-8)
        amplitudes = features / norm
        
        return amplitudes
        
    def _amplitudes_to_features(self, amplitudes: np.ndarray, n_features: int) -> np.ndarray:
        """
        Convert quantum state amplitudes back to features.
        
        Args:
            amplitudes: Array of amplitudes
            n_features: Number of features to extract
            
        Returns:
            Array of features
        """
        # Extract and denormalize features
        features = amplitudes[:n_features].real
        if len(features) > 0:
            features = features * np.std(features) + np.mean(features)
        
        return features
        
    def to_quantum_state(self, graph: SchemaGraph) -> Statevector:
        """
        Convert schema graph to quantum state.
        
        Args:
            graph: Schema graph to convert
            
        Returns:
            Quantum state representation
        """
        # Extract and convert features
        features = self._extract_graph_features(graph)
        amplitudes = self._features_to_amplitudes(features)
        
        # Create quantum state
        state = Statevector(amplitudes)
        
        return state
        
    def from_quantum_state(self, state: Statevector, graph: SchemaGraph) -> Dict[str, np.ndarray]:
        """
        Convert quantum state back to schema features.
        
        Args:
            state: Quantum state to convert
            graph: Original schema graph for reference
            
        Returns:
            Dictionary of extracted features
        """
        # Get amplitudes
        amplitudes = state.data
        
        # Calculate number of features
        n_nodes = len(graph.graph.nodes)
        n_edges = len(graph.graph.edges)
        n_node_features = n_nodes * 4  # size, columns, query_freq, update_freq
        n_edge_features = n_edges * 5  # cardinality (3), query_freq, selectivity
        n_total_features = n_node_features + n_edge_features
        
        # Convert amplitudes to features
        all_features = self._amplitudes_to_features(amplitudes, n_total_features)
        
        # Pad features if necessary
        if len(all_features) < n_total_features:
            all_features = np.pad(all_features, (0, n_total_features - len(all_features)))
        
        # Split and reshape features
        if n_node_features > 0:
            node_features = all_features[:n_node_features].reshape(n_nodes, 4)
        else:
            node_features = np.array([])
            
        if n_edge_features > 0:
            edge_features = all_features[n_node_features:n_node_features + n_edge_features].reshape(n_edges, 5)
        else:
            edge_features = np.array([])
        
        return {
            'node_features': node_features,
            'edge_features': edge_features
        } 