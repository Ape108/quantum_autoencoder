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
import sqlite3

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

class SchemaMetrics:
    """Analyzes and provides metrics for database schema."""
    
    def __init__(self, conn: sqlite3.Connection):
        """
        Initialize metrics analyzer.
        
        Args:
            conn: Database connection
        """
        self.conn = conn
        self.cursor = conn.cursor()
        
    def get_table_metrics(self, table_name: str) -> Dict:
        """Get metrics for a specific table."""
        metrics = {
            'row_count': 0,
            'column_count': 0,
            'index_count': 0,
            'foreign_keys': [],
            'data_distribution': {}
        }
        
        # Get row count
        self.cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        metrics['row_count'] = self.cursor.fetchone()[0]
        
        # Get column info
        self.cursor.execute(f"PRAGMA table_info({table_name})")
        columns = self.cursor.fetchall()
        metrics['column_count'] = len(columns)
        
        # Get index info
        self.cursor.execute(f"PRAGMA index_list({table_name})")
        indexes = self.cursor.fetchall()
        metrics['index_count'] = len(indexes)
        
        # Get foreign keys
        self.cursor.execute(f"PRAGMA foreign_key_list({table_name})")
        foreign_keys = self.cursor.fetchall()
        metrics['foreign_keys'] = [
            {
                'from': fk[3],
                'to_table': fk[2],
                'to_column': fk[4]
            }
            for fk in foreign_keys
        ]
        
        # Analyze data distribution for each column
        for col in columns:
            col_name = col[1]
            self.cursor.execute(f"""
                SELECT {col_name}, COUNT(*) as freq 
                FROM {table_name} 
                GROUP BY {col_name}
            """)
            distribution = self.cursor.fetchall()
            metrics['data_distribution'][col_name] = {
                'unique_values': len(distribution),
                'max_frequency': max(freq[1] for freq in distribution) if distribution else 0,
                'null_count': sum(1 for freq in distribution if freq[0] is None)
            }
        
        return metrics
    
    def get_join_metrics(self, table1: str, table2: str) -> Dict:
        """Analyze join characteristics between two tables."""
        metrics = {
            'join_size': 0,
            'cardinality_ratio': 0.0,
            'foreign_key_coverage': 0.0
        }
        
        # Find foreign key relationship
        self.cursor.execute(f"PRAGMA foreign_key_list({table1})")
        foreign_keys = [
            fk for fk in self.cursor.fetchall()
            if fk[2] == table2  # fk[2] is referenced table
        ]
        
        if foreign_keys:
            fk = foreign_keys[0]
            from_col = fk[3]  # fk[3] is from column
            to_col = fk[4]    # fk[4] is to column
            
            # Get join size
            self.cursor.execute(f"""
                SELECT COUNT(*)
                FROM {table1} t1
                JOIN {table2} t2
                ON t1.{from_col} = t2.{to_col}
            """)
            metrics['join_size'] = self.cursor.fetchone()[0]
            
            # Get table sizes
            self.cursor.execute(f"SELECT COUNT(*) FROM {table1}")
            size1 = self.cursor.fetchone()[0]
            self.cursor.execute(f"SELECT COUNT(*) FROM {table2}")
            size2 = self.cursor.fetchone()[0]
            
            # Calculate cardinality ratio
            if size2 > 0:
                metrics['cardinality_ratio'] = size1 / size2
            
            # Calculate foreign key coverage
            self.cursor.execute(f"""
                SELECT COUNT(DISTINCT {from_col})
                FROM {table1}
            """)
            distinct_fk = self.cursor.fetchone()[0]
            
            self.cursor.execute(f"""
                SELECT COUNT(DISTINCT {to_col})
                FROM {table2}
            """)
            distinct_pk = self.cursor.fetchone()[0]
            
            if distinct_pk > 0:
                metrics['foreign_key_coverage'] = distinct_fk / distinct_pk
        
        return metrics
    
    def get_query_metrics(self, query: str) -> Dict:
        """Analyze query execution characteristics."""
        metrics = {
            'execution_plan': [],
            'estimated_cost': 0,
            'tables_accessed': [],
            'indexes_used': []
        }
        
        # Get query plan
        self.cursor.execute(f"EXPLAIN QUERY PLAN {query}")
        plan = self.cursor.fetchall()
        
        # Parse plan
        for step in plan:
            detail = step[3]  # step[3] contains the operation details
            metrics['execution_plan'].append({
                'id': step[0],
                'parent': step[1],
                'operation': detail
            })
            
            # Extract cost estimate
            if 'SCAN' in detail:
                metrics['estimated_cost'] += step[2]  # step[2] is estimated rows
            
            # Extract accessed tables
            for word in detail.split():
                if word not in ['SCAN', 'SEARCH', 'JOIN', 'USING', 'INDEX']:
                    metrics['tables_accessed'].append(word)
            
            # Extract used indexes
            if 'INDEX' in detail:
                idx = detail.split('INDEX')[-1].strip()
                metrics['indexes_used'].append(idx)
        
        return metrics
    
    def get_overall_metrics(self) -> Dict:
        """Get overall database metrics."""
        metrics = {
            'tables': {},
            'relationships': {},
            'complexity': {
                'total_tables': 0,
                'total_columns': 0,
                'total_indexes': 0,
                'total_foreign_keys': 0,
                'max_join_depth': 0
            }
        }
        
        # Get all tables
        self.cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
        """)
        tables = [row[0] for row in self.cursor.fetchall()]
        
        # Analyze each table
        for table in tables:
            metrics['tables'][table] = self.get_table_metrics(table)
            
            # Update complexity metrics
            metrics['complexity']['total_tables'] += 1
            metrics['complexity']['total_columns'] += metrics['tables'][table]['column_count']
            metrics['complexity']['total_indexes'] += metrics['tables'][table]['index_count']
            metrics['complexity']['total_foreign_keys'] += len(metrics['tables'][table]['foreign_keys'])
        
        # Analyze relationships between tables
        for table1 in tables:
            for table2 in tables:
                if table1 != table2:
                    join_metrics = self.get_join_metrics(table1, table2)
                    if join_metrics['join_size'] > 0:
                        metrics['relationships'][f"{table1}__{table2}"] = join_metrics
        
        # Calculate max join depth
        def get_join_depth(table: str, visited: set) -> int:
            if table in visited:
                return 0
            visited.add(table)
            max_depth = 0
            for rel in metrics['relationships']:
                if table in rel:
                    other = rel.replace(table, '').replace('__', '')
                    depth = get_join_depth(other, visited.copy())
                    max_depth = max(max_depth, depth + 1)
            return max_depth
        
        for table in tables:
            depth = get_join_depth(table, set())
            metrics['complexity']['max_join_depth'] = max(
                metrics['complexity']['max_join_depth'],
                depth
            )
        
        return metrics 