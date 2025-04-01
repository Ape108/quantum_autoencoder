"""
Enhanced Quantum-Database Feature Mapping.

This module provides sophisticated bidirectional mapping between database features
and quantum states, ensuring we fully utilize the quantum properties of the autoencoder.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import sqlite3
import logging

logger = logging.getLogger(__name__)

class QueryFeatureMapper:
    """Maps between database features and quantum states."""
    
    def __init__(self, n_qubits: int):
        """
        Initialize mapper.
        
        Args:
            n_qubits: Number of qubits for state representation
        """
        self.n_qubits = n_qubits
    
    def encode_query_features(
        self,
        query: str,
        cursor: sqlite3.Cursor
    ) -> Tuple[QuantumCircuit, Dict]:
        """
        Encode query features into quantum state.
        
        Args:
            query: SQL query to encode
            cursor: Database cursor for analysis
            
        Returns:
            Tuple of (quantum circuit, feature mapping)
        """
        # Analyze query execution plan
        cursor.execute(f"EXPLAIN QUERY PLAN {query}")
        plan = cursor.fetchall()
        
        # Extract key features
        features = self._extract_query_features(query, plan)
        
        # Create quantum circuit
        qc = QuantumCircuit(self.n_qubits)
        
        # Initialize to zero state (all qubits start in |0⟩ by default)
        
        # Encode features
        self._encode_features(qc, features)
        
        return qc, features
    
    def _extract_query_features(
        self,
        query: str,
        plan: List
    ) -> Dict:
        """Extract quantum-relevant features from query."""
        features = {
            'tables': set(),
            'joins': [],
            'conditions': [],
            'costs': []
        }
        
        # Extract features from plan
        for step in plan:
            detail = step[3]  # step[3] contains operation details
            cost = float(step[2]) if step[2] is not None else 0.0  # step[2] is estimated rows
            
            # Extract tables
            words = detail.split()
            for word in words:
                if word not in ['SCAN', 'SEARCH', 'INDEX', 'USING', 'JOIN']:
                    features['tables'].add(word)
            
            # Extract joins
            if 'JOIN' in detail:
                features['joins'].append({
                    'type': 'LOOP' if 'LOOP' in detail else 'HASH',
                    'tables': [t for t in words if t not in ['JOIN', 'LOOP', 'USING']],
                    'cost': cost
                })
            
            # Extract conditions
            if 'WHERE' in detail:
                features['conditions'].append({
                    'column': detail.split('WHERE')[1].split()[0],
                    'cost': cost
                })
            
            features['costs'].append(cost)
        
        return features
    
    def _encode_features(self, qc: QuantumCircuit, features: Dict):
        """Encode features into quantum state."""
        # Normalize costs (avoid division by zero)
        total_cost = sum(features['costs']) or 1.0
        normalized_costs = [c / total_cost for c in features['costs']]
        
        # Encode table access patterns (qubits 0-1)
        n_tables = len(features['tables'])
        if n_tables > 0:
            angle = np.pi * (n_tables / 4)  # Scale based on number of tables
            qc.ry(angle, 0)
            qc.ry(angle, 1)
        
        # Encode join patterns (qubits 2-3)
        if features['joins']:
            # Use different angles for different join types
            for join in features['joins']:
                angle = np.pi/3 if join['type'] == 'LOOP' else 2*np.pi/3
                qc.ry(angle, 2)
                
                # Entangle join qubits
                qc.cx(2, 3)
        
        # Encode conditions (qubits 4-5)
        if features['conditions']:
            for condition in features['conditions']:
                # Create superposition for conditions
                qc.h(4)
                
                # Rotate based on condition cost
                cost = condition['cost'] / total_cost
                angle = np.pi * cost
                qc.ry(angle, 5)
        
        # Add final entangling layer
        for i in range(self.n_qubits - 1):
            qc.cx(i, i + 1)
        qc.cx(self.n_qubits - 1, 0)  # Close the chain

class QueryOptimizationMapper:
    """Maps between query optimization problems and quantum states."""
    
    def __init__(
        self,
        n_qubits: int,
        conn: sqlite3.Connection
    ):
        """
        Initialize mapper.
        
        Args:
            n_qubits: Number of qubits
            conn: Database connection
        """
        self.n_qubits = n_qubits
        self.conn = conn
        self.cursor = conn.cursor()
        self.feature_mapper = QueryFeatureMapper(n_qubits)
    
    def prepare_optimization_batch(
        self,
        queries: List[str]
    ) -> Tuple[List[QuantumCircuit], List[Dict]]:
        """
        Prepare batch of queries for optimization.
        
        Args:
            queries: List of SQL queries
            
        Returns:
            Tuple of (quantum circuits, feature mappings)
        """
        circuits = []
        features = []
        
        for query in queries:
            try:
                qc, feat = self.feature_mapper.encode_query_features(query, self.cursor)
                circuits.append(qc)
                features.append(feat)
            except Exception as e:
                logger.warning(f"Failed to encode query '{query}': {e}")
        
        if not circuits:
            raise ValueError("No queries were successfully encoded into quantum circuits")
        
        return circuits, features
    
    def interpret_results(
        self,
        states: List[Statevector],
        original_features: List[Dict]
    ) -> List[Dict[str, List[Dict]]]:
        """
        Interpret optimization results.
        
        Args:
            states: List of quantum states from autoencoder
            original_features: Original feature mappings
            
        Returns:
            List of optimization strategies for each query
        """
        all_strategies = []
        
        for state, features in zip(states, original_features):
            strategies = self.feature_mapper.decode_quantum_state(
                state,
                features
            )
            all_strategies.append(strategies)
        
        return all_strategies 