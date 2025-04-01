import logging
import sqlite3
from typing import List

import numpy as np
from qiskit import QuantumCircuit

logger = logging.getLogger(__name__)

class QueryFeatureMapper:
    """Maps database features to quantum states."""
    
    def __init__(self, n_qubits: int):
        """
        Initialize mapper.
        
        Args:
            n_qubits: Number of qubits
        """
        self.n_qubits = n_qubits
        
    def extract_features(self, db_path: str) -> List[QuantumCircuit]:
        """
        Extract features from database.
        
        Args:
            db_path: Path to database
            
        Returns:
            List of quantum circuits
        """
        logger.info(f"Extracting features from {db_path}")
        
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Get tables
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
            """)
            tables = cursor.fetchall()
            
            # Extract features for each table
            circuits = []
            for table_name in tables:
                table = table_name[0]
                
                # Get table size
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                size = cursor.fetchone()[0]
                
                # Get column info
                cursor.execute(f"PRAGMA table_info({table})")
                columns = cursor.fetchall()
                
                # Get foreign keys
                cursor.execute(f"PRAGMA foreign_key_list({table})")
                foreign_keys = cursor.fetchall()
                
                # Get indexes
                cursor.execute(f"PRAGMA index_list({table})")
                indexes = cursor.fetchall()
                
                # Extract features
                features = [
                    size / 1000.0,  # Normalize size
                    len(columns) / 20.0,  # Normalize column count
                    len(foreign_keys) / 5.0,  # Normalize FK count
                    len(indexes) / 5.0,  # Normalize index count
                ]
                
                # Create quantum circuit
                circuit = self._features_to_circuit(features)
                circuits.append(circuit)
                
        return circuits
        
    def _features_to_circuit(self, features: List[float]) -> QuantumCircuit:
        """Convert features to quantum circuit."""
        # Pad features to match number of qubits
        n_amplitudes = 2 ** self.n_qubits
        if len(features) < n_amplitudes:
            features = np.pad(features, (0, n_amplitudes - len(features)))
        else:
            features = features[:n_amplitudes]
            
        # Normalize
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
            
        # Create circuit
        qc = QuantumCircuit(self.n_qubits)
        qc.initialize(features, range(self.n_qubits))
        
        return qc 