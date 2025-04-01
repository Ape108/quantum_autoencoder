"""Database schema analysis tools."""

import logging
import sqlite3
from typing import Dict, List, Set, Tuple

import networkx as nx

logger = logging.getLogger(__name__)

class SchemaAnalyzer:
    """Analyzes database schema structure."""
    
    def __init__(self, db_path: str):
        """
        Initialize analyzer.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self.graph = nx.DiGraph()
        
    def analyze(self) -> Dict[str, any]:
        """
        Analyze database schema.
        
        Returns:
            Analysis results
        """
        self._build_schema_graph()
        
        return {
            'tables': self._analyze_tables(),
            'relationships': self._analyze_relationships(),
            'complexity': self._analyze_complexity()
        }
        
    def _build_schema_graph(self):
        """Build graph representation of schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get tables
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
            """)
            tables = cursor.fetchall()
            
            # Add nodes
            for table in tables:
                table_name = table[0]
                self.graph.add_node(
                    table_name,
                    type='table'
                )
                
                # Get columns
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()
                
                for col in columns:
                    col_name = f"{table_name}.{col[1]}"
                    self.graph.add_node(
                        col_name,
                        type='column',
                        data_type=col[2],
                        is_pk=bool(col[5])
                    )
                    self.graph.add_edge(table_name, col_name)
                    
                # Get foreign keys
                cursor.execute(f"PRAGMA foreign_key_list({table_name})")
                foreign_keys = cursor.fetchall()
                
                for fk in foreign_keys:
                    from_col = f"{table_name}.{fk[3]}"
                    to_col = f"{fk[2]}.{fk[4]}"
                    self.graph.add_edge(
                        from_col,
                        to_col,
                        type='foreign_key'
                    )
                    
    def _analyze_tables(self) -> Dict[str, Dict]:
        """Analyze table structure."""
        tables = {}
        
        for node in self.graph.nodes():
            if self.graph.nodes[node]['type'] == 'table':
                # Get columns
                columns = []
                for _, col, data in self.graph.edges(node, data=True):
                    if 'type' not in data:  # Only direct column connections
                        col_name = col.split('.')[1]
                        col_data = self.graph.nodes[col]
                        columns.append({
                            'name': col_name,
                            'type': col_data['data_type'],
                            'is_pk': col_data['is_pk']
                        })
                        
                # Get relationships
                relationships = []
                for col in columns:
                    col_node = f"{node}.{col['name']}"
                    for _, target, data in self.graph.edges(col_node, data=True):
                        if data.get('type') == 'foreign_key':
                            target_table = target.split('.')[0]
                            target_col = target.split('.')[1]
                            relationships.append({
                                'from_column': col['name'],
                                'to_table': target_table,
                                'to_column': target_col
                            })
                            
                tables[node] = {
                    'columns': columns,
                    'relationships': relationships
                }
                
        return tables
        
    def _analyze_relationships(self) -> List[Dict]:
        """Analyze table relationships."""
        relationships = []
        
        for source, target, data in self.graph.edges(data=True):
            if data.get('type') == 'foreign_key':
                source_parts = source.split('.')
                target_parts = target.split('.')
                
                relationships.append({
                    'from_table': source_parts[0],
                    'from_column': source_parts[1],
                    'to_table': target_parts[0],
                    'to_column': target_parts[1]
                })
                
        return relationships
        
    def _analyze_complexity(self) -> Dict[str, float]:
        """Calculate schema complexity metrics."""
        # Get tables and columns
        tables = [
            node for node, data in self.graph.nodes(data=True)
            if data['type'] == 'table'
        ]
        columns = [
            node for node, data in self.graph.nodes(data=True)
            if data['type'] == 'column'
        ]
        
        # Get relationships
        relationships = [
            (source, target) for source, target, data 
            in self.graph.edges(data=True)
            if data.get('type') == 'foreign_key'
        ]
        
        # Calculate metrics
        avg_columns_per_table = len(columns) / len(tables)
        relationship_complexity = len(relationships) / len(tables)
        
        # Calculate connectivity
        table_connectivity = nx.density(
            self.graph.subgraph([
                node for node, data in self.graph.nodes(data=True)
                if data['type'] == 'table'
            ])
        )
        
        return {
            'n_tables': len(tables),
            'n_columns': len(columns),
            'n_relationships': len(relationships),
            'avg_columns_per_table': avg_columns_per_table,
            'relationship_complexity': relationship_complexity,
            'table_connectivity': table_connectivity
        } 