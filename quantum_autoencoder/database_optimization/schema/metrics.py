"""
Schema performance metrics calculation.

This module provides functionality for calculating various performance metrics
for database schemas, including query patterns, join complexity, and storage efficiency.
"""

import numpy as np
from typing import Dict, List, Optional
from .graph import SchemaGraph, NodeProperties, EdgeProperties

class SchemaMetrics:
    """Calculator for database schema performance metrics."""
    
    def __init__(self, graph: SchemaGraph):
        """
        Initialize the metrics calculator.
        
        Args:
            graph: Schema graph to analyze
        """
        self.graph = graph
        
    def calculate_query_metrics(self) -> Dict[str, float]:
        """
        Calculate metrics related to query patterns.
        
        Returns:
            Dictionary of query-related metrics
        """
        metrics = {}
        
        # Average query frequency across tables
        query_freqs = [
            props.query_frequency
            for props in self.graph.graph.nodes.values()
        ]
        metrics['avg_query_frequency'] = np.mean(query_freqs)
        
        # Average update frequency
        update_freqs = [
            props.update_frequency
            for props in self.graph.graph.nodes.values()
        ]
        metrics['avg_update_frequency'] = np.mean(update_freqs)
        
        # Average join frequency
        join_freqs = [
            props['query_frequency']
            for props in self.graph.graph.edges.values()
        ]
        metrics['avg_join_frequency'] = np.mean(join_freqs)
        
        # Average join selectivity
        selectivities = [
            props['selectivity']
            for props in self.graph.graph.edges.values()
        ]
        metrics['avg_join_selectivity'] = np.mean(selectivities)
        
        return metrics
        
    def calculate_storage_metrics(self) -> Dict[str, float]:
        """
        Calculate metrics related to storage efficiency.
        
        Returns:
            Dictionary of storage-related metrics
        """
        metrics = {}
        
        # Total size across all tables
        total_size = sum(
            props.size
            for props in self.graph.graph.nodes.values()
        )
        metrics['total_size'] = total_size
        
        # Average table size
        table_sizes = [
            props.size
            for props in self.graph.graph.nodes.values()
        ]
        metrics['avg_table_size'] = np.mean(table_sizes)
        
        # Average columns per table
        column_counts = [
            props.column_count
            for props in self.graph.graph.nodes.values()
        ]
        metrics['avg_columns'] = np.mean(column_counts)
        
        return metrics
        
    def calculate_complexity_metrics(self) -> Dict[str, float]:
        """
        Calculate metrics related to schema complexity.
        
        Returns:
            Dictionary of complexity-related metrics
        """
        metrics = {}
        
        # Graph density
        metrics['density'] = nx.density(self.graph.graph)
        
        # Average node degree
        degrees = [d for n, d in self.graph.graph.degree()]
        metrics['avg_degree'] = np.mean(degrees)
        
        # Average path length (if graph is connected)
        if nx.is_strongly_connected(self.graph.graph):
            metrics['avg_path_length'] = nx.average_shortest_path_length(self.graph.graph)
        else:
            metrics['avg_path_length'] = float('inf')
            
        # Clustering coefficient
        metrics['clustering_coefficient'] = nx.average_clustering(self.graph.graph)
        
        return metrics
        
    def calculate_join_complexity(self) -> Dict[str, float]:
        """
        Calculate metrics related to join complexity.
        
        Returns:
            Dictionary of join complexity metrics
        """
        metrics = {}
        
        # Average number of joins per query path
        paths = []
        for source in self.graph.graph.nodes():
            for target in self.graph.graph.nodes():
                if source != target:
                    try:
                        path = nx.shortest_path(self.graph.graph, source, target)
                        paths.append(len(path) - 1)  # Number of joins is path length - 1
                    except nx.NetworkXNoPath:
                        continue
                        
        if paths:
            metrics['avg_joins_per_path'] = np.mean(paths)
        else:
            metrics['avg_joins_per_path'] = 0
            
        # Maximum number of joins in any path
        if paths:
            metrics['max_joins_in_path'] = max(paths)
        else:
            metrics['max_joins_in_path'] = 0
            
        return metrics
        
    def get_all_metrics(self) -> Dict[str, float]:
        """
        Calculate all available metrics.
        
        Returns:
            Dictionary containing all metrics
        """
        metrics = {}
        metrics.update(self.calculate_query_metrics())
        metrics.update(self.calculate_storage_metrics())
        metrics.update(self.calculate_complexity_metrics())
        metrics.update(self.calculate_join_complexity())
        return metrics 