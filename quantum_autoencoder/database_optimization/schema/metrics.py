"""
Schema metrics calculation.

This module provides functionality for calculating various metrics
about database schemas to evaluate their quality and performance.
"""

from typing import Dict
import networkx as nx
from .graph import SchemaGraph

class SchemaMetrics:
    """Calculates metrics for database schemas."""
    
    def __init__(self, graph: SchemaGraph):
        """
        Initialize the metrics calculator.
        
        Args:
            graph: Schema graph to analyze
        """
        self.graph = graph
        
    def calculate_query_metrics(self) -> Dict[str, float]:
        """
        Calculate query-related metrics.
        
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Calculate average query frequency
        total_freq = 0
        for _, props in self.graph.graph.nodes(data=True):
            total_freq += props['query_frequency']
        metrics['avg_query_frequency'] = total_freq / len(self.graph.graph.nodes)
        
        # Calculate query frequency variance
        var_freq = 0
        for _, props in self.graph.graph.nodes(data=True):
            var_freq += (props['query_frequency'] - metrics['avg_query_frequency']) ** 2
        metrics['query_frequency_variance'] = var_freq / len(self.graph.graph.nodes)
        
        # Calculate relationship query metrics
        total_rel_freq = 0
        for edge in self.graph.graph.edges(data=True):
            _, _, props = edge
            total_rel_freq += props['query_frequency']
        metrics['avg_relationship_frequency'] = total_rel_freq / len(self.graph.graph.edges)
        
        return metrics
        
    def calculate_storage_metrics(self) -> Dict[str, float]:
        """
        Calculate storage-related metrics.
        
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Calculate total size
        total_size = 0
        for _, props in self.graph.graph.nodes(data=True):
            total_size += props['size']
        metrics['total_size'] = total_size
        
        # Calculate average table size
        metrics['avg_table_size'] = total_size / len(self.graph.graph.nodes)
        
        # Calculate size variance
        var_size = 0
        for _, props in self.graph.graph.nodes(data=True):
            var_size += (props['size'] - metrics['avg_table_size']) ** 2
        metrics['size_variance'] = var_size / len(self.graph.graph.nodes)
        
        # Calculate average column count
        total_cols = 0
        for _, props in self.graph.graph.nodes(data=True):
            total_cols += props['column_count']
        metrics['avg_column_count'] = total_cols / len(self.graph.graph.nodes)
        
        return metrics
        
    def calculate_relationship_metrics(self) -> Dict[str, float]:
        """
        Calculate relationship-related metrics.
        
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Calculate average selectivity
        total_sel = 0
        for edge in self.graph.graph.edges(data=True):
            _, _, props = edge
            total_sel += props['selectivity']
        metrics['avg_selectivity'] = total_sel / len(self.graph.graph.edges)
        
        # Calculate selectivity variance
        var_sel = 0
        for edge in self.graph.graph.edges(data=True):
            _, _, props = edge
            var_sel += (props['selectivity'] - metrics['avg_selectivity']) ** 2
        metrics['selectivity_variance'] = var_sel / len(self.graph.graph.edges)
        
        # Calculate relationship density
        n = len(self.graph.graph.nodes)
        m = len(self.graph.graph.edges)
        metrics['relationship_density'] = m / (n * (n - 1)) if n > 1 else 0
        
        return metrics
        
    def calculate_structural_metrics(self) -> Dict[str, float]:
        """
        Calculate structural metrics.
        
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Calculate average degree
        total_degree = 0
        for node in self.graph.graph.nodes:
            total_degree += self.graph.graph.degree(node)
        metrics['avg_degree'] = total_degree / len(self.graph.graph.nodes)
        
        # Calculate clustering coefficient
        metrics['clustering_coefficient'] = nx.average_clustering(self.graph.graph)
        
        # Calculate average path length
        try:
            metrics['avg_path_length'] = nx.average_shortest_path_length(self.graph.graph)
        except nx.NetworkXError:
            metrics['avg_path_length'] = float('inf')
            
        return metrics
        
    def get_all_metrics(self) -> Dict[str, float]:
        """
        Get all available metrics.
        
        Returns:
            Dictionary of all metrics
        """
        metrics = {}
        
        # Calculate all metric categories
        metrics.update(self.calculate_query_metrics())
        metrics.update(self.calculate_storage_metrics())
        metrics.update(self.calculate_relationship_metrics())
        metrics.update(self.calculate_structural_metrics())
        
        return metrics 