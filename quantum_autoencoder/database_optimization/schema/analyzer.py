"""
Schema analysis module.

This module provides functionality for analyzing database schemas,
including pattern detection, optimization suggestions, and schema validation.
"""

from typing import Dict, List, Optional
import networkx as nx
from .graph import SchemaGraph, NodeProperties, EdgeProperties

class SchemaAnalyzer:
    """Analyzes database schemas for optimization opportunities."""
    
    def __init__(self, graph: SchemaGraph):
        """
        Initialize the schema analyzer.
        
        Args:
            graph: Schema graph to analyze
        """
        self.graph = graph
        
    def analyze_query_patterns(self) -> Dict[str, List[str]]:
        """
        Analyze query patterns in the schema.
        
        Returns:
            Dictionary containing analysis results
        """
        patterns = {
            'high_frequency_tables': [],
            'low_frequency_tables': [],
            'hot_paths': [],
            'cold_paths': []
        }
        
        # Analyze table frequencies
        for table, props in self.graph.graph.nodes(data=True):
            if props['query_frequency'] > 0.7:
                patterns['high_frequency_tables'].append(table)
            elif props['query_frequency'] < 0.3:
                patterns['low_frequency_tables'].append(table)
                
        # Analyze relationship frequencies
        for edge in self.graph.graph.edges(data=True):
            source, target, props = edge
            if props['query_frequency'] > 0.8:
                patterns['hot_paths'].append(f"{source} -> {target}")
            elif props['query_frequency'] < 0.2:
                patterns['cold_paths'].append(f"{source} -> {target}")
                
        return patterns
        
    def analyze_storage_patterns(self) -> Dict[str, List[str]]:
        """
        Analyze storage patterns in the schema.
        
        Returns:
            Dictionary containing analysis results
        """
        patterns = {
            'large_tables': [],
            'small_tables': [],
            'wide_tables': [],
            'narrow_tables': []
        }
        
        # Analyze table sizes and column counts
        for table, props in self.graph.graph.nodes(data=True):
            if props['size'] > 1000:
                patterns['large_tables'].append(table)
            elif props['size'] < 100:
                patterns['small_tables'].append(table)
                
            if props['column_count'] > 10:
                patterns['wide_tables'].append(table)
            elif props['column_count'] < 5:
                patterns['narrow_tables'].append(table)
                
        return patterns
        
    def analyze_relationship_patterns(self) -> Dict[str, List[str]]:
        """
        Analyze relationship patterns in the schema.
        
        Returns:
            Dictionary containing analysis results
        """
        patterns = {
            'high_selectivity': [],
            'low_selectivity': [],
            'complex_relationships': [],
            'simple_relationships': []
        }
        
        # Analyze relationship properties
        for edge in self.graph.graph.edges(data=True):
            source, target, props = edge
            if props['selectivity'] > 0.7:
                patterns['high_selectivity'].append(f"{source} -> {target}")
            elif props['selectivity'] < 0.3:
                patterns['low_selectivity'].append(f"{source} -> {target}")
                
            # Check relationship complexity
            if props['cardinality'] == 'N:M':
                patterns['complex_relationships'].append(f"{source} -> {target}")
            elif props['cardinality'] == '1:1':
                patterns['simple_relationships'].append(f"{source} -> {target}")
                
        return patterns
        
    def suggest_optimizations(self) -> List[str]:
        """
        Generate optimization suggestions for the schema.
        
        Returns:
            List of optimization suggestions
        """
        suggestions = []
        
        # Analyze patterns
        query_patterns = self.analyze_query_patterns()
        storage_patterns = self.analyze_storage_patterns()
        relationship_patterns = self.analyze_relationship_patterns()
        
        # Generate suggestions based on patterns
        for table in query_patterns['high_frequency_tables']:
            suggestions.append(f"Consider adding indexes to frequently queried table: {table}")
            
        for table in storage_patterns['large_tables']:
            suggestions.append(f"Consider partitioning large table: {table}")
            
        for path in relationship_patterns['low_selectivity']:
            suggestions.append(f"Consider adding index for low-selectivity relationship: {path}")
            
        for path in relationship_patterns['complex_relationships']:
            suggestions.append(f"Consider denormalizing complex relationship: {path}")
            
        return suggestions
        
    def validate_schema(self) -> Dict[str, bool]:
        """
        Validate the schema for common issues.
        
        Returns:
            Dictionary of validation results
        """
        validation = {
            'has_cycles': False,
            'has_isolated_tables': False,
            'has_redundant_indexes': False,
            'has_missing_foreign_keys': False
        }
        
        # Check for cycles
        validation['has_cycles'] = self.graph.is_cyclic()
        
        # Check for isolated tables
        validation['has_isolated_tables'] = not nx.is_strongly_connected(self.graph.graph)
        
        # Check for redundant indexes
        for table, props in self.graph.graph.nodes(data=True):
            if 'indexes' in props and len(props['indexes']) > 3:
                validation['has_redundant_indexes'] = True
                break
                
        # Check for missing foreign keys
        for edge in self.graph.graph.edges(data=True):
            source, target, props = edge
            if 'foreign_key' not in props or not props['foreign_key']:
                validation['has_missing_foreign_keys'] = True
                break
                
        return validation 