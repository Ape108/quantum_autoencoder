"""
Schema validation and human-readable output.

This module provides functionality to validate database schemas
and generate human-readable descriptions of the optimized structures.
"""

from typing import Dict, List, Tuple, Optional, Callable, Any
from .graph import SchemaGraph, NodeProperties, EdgeProperties
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os

class SchemaValidator:
    """Validates and generates human-readable database schemas."""
    
    def __init__(self, schema: SchemaGraph):
        """
        Initialize schema validator.
        
        Args:
            schema: Schema graph to validate
        """
        self.schema = schema
        self.table_validation_rules = {
            'table_size': self._validate_table_size,
            'column_count': self._validate_column_count,
            'index_coverage': self._validate_index_coverage,
            'query_patterns': self._validate_query_patterns
        }
        self.relationship_validation_rules = {
            'cardinality': self._validate_relationship_cardinality
        }
    
    def _validate_table_size(self, table: str, props: Dict) -> Tuple[bool, str]:
        """Validate table size is reasonable."""
        if props['size'] < 10:
            return False, f"Table {table} is too small (size: {props['size']})"
        if props['size'] > 1_000_000:
            return False, f"Table {table} is too large (size: {props['size']})"
        return True, f"Table {table} size ({props['size']}) is reasonable"
    
    def _validate_column_count(self, table: str, props: Dict) -> Tuple[bool, str]:
        """Validate column count is reasonable."""
        if props['column_count'] < 2:
            return False, f"Table {table} has too few columns ({props['column_count']})"
        if props['column_count'] > 50:
            return False, f"Table {table} has too many columns ({props['column_count']})"
        return True, f"Table {table} column count ({props['column_count']}) is reasonable"
    
    def _validate_relationship_cardinality(self, source: str, target: str, props: Dict) -> Tuple[bool, str]:
        """Validate relationship cardinality makes sense."""
        if props['cardinality'] not in ['1:1', '1:N', 'N:1', 'N:M']:
            return False, f"Invalid cardinality {props['cardinality']} for {source}->{target}"
        return True, f"Relationship {source}->{target} cardinality ({props['cardinality']}) is valid"
    
    def _validate_index_coverage(self, table: str, props: Dict) -> Tuple[bool, str]:
        """Validate index coverage is appropriate."""
        if not props['indexes']:
            return False, f"Table {table} has no indexes"
        if len(props['indexes']) > 10:
            return False, f"Table {table} has too many indexes ({len(props['indexes'])})"
        return True, f"Table {table} has appropriate index coverage ({len(props['indexes'])} indexes)"
    
    def _validate_query_patterns(self, table: str, props: Dict) -> Tuple[bool, str]:
        """Validate query patterns are reasonable."""
        if props['query_frequency'] < 0 or props['query_frequency'] > 1:
            return False, f"Invalid query frequency {props['query_frequency']} for {table}"
        if props['update_frequency'] < 0 or props['update_frequency'] > 1:
            return False, f"Invalid update frequency {props['update_frequency']} for {table}"
        return True, f"Table {table} has valid query patterns (query: {props['query_frequency']:.2f}, update: {props['update_frequency']:.2f})"
    
    def validate_schema(self) -> Dict[str, List[str]]:
        """
        Validate entire schema.
        
        Returns:
            Dictionary of validation results by category
        """
        results = {
            'warnings': [],
            'errors': [],
            'suggestions': []
        }
        
        # Validate tables
        for table, props in self.schema.graph.nodes(data=True):
            for rule_name, rule_func in self.table_validation_rules.items():
                is_valid, message = rule_func(table, props)
                if not is_valid:
                    results['errors'].append(message)
                elif 'warning' in message.lower():
                    results['warnings'].append(message)
        
        # Validate relationships
        for source, target, props in self.schema.graph.edges(data=True):
            for rule_name, rule_func in self.relationship_validation_rules.items():
                is_valid, message = rule_func(source, target, props)
                if not is_valid:
                    results['errors'].append(message)
                elif 'warning' in message.lower():
                    results['warnings'].append(message)
        
        return results
    
    def generate_human_readable_schema(self) -> str:
        """
        Generate human-readable schema description.
        
        Returns:
            String containing human-readable schema description
        """
        output = ["Database Schema Description", "========================"]
        
        # Tables section
        output.append("\nTables:")
        output.append("------")
        for table, props in self.schema.graph.nodes(data=True):
            output.append(f"\n{table}:")
            output.append(f"  Size: {props['size']:,} rows")
            output.append(f"  Columns: {props['column_count']}")
            output.append(f"  Primary Key: {props['primary_key']}")
            output.append(f"  Indexes: {', '.join(props['indexes'])}")
            output.append(f"  Query Frequency: {props['query_frequency']:.2f}")
            output.append(f"  Update Frequency: {props['update_frequency']:.2f}")
        
        # Relationships section
        output.append("\nRelationships:")
        output.append("-------------")
        for source, target, props in self.schema.graph.edges(data=True):
            output.append(f"\n{source} -> {target}:")
            output.append(f"  Type: {props['cardinality']}")
            output.append(f"  Foreign Key: {props['foreign_key']}")
            output.append(f"  Query Frequency: {props['query_frequency']:.2f}")
            output.append(f"  Selectivity: {props['selectivity']:.2f}")
        
        # Query Patterns section
        output.append("\nQuery Patterns:")
        output.append("--------------")
        high_freq_tables = [t for t, p in self.schema.graph.nodes(data=True) 
                          if p['query_frequency'] > 0.7]
        if high_freq_tables:
            output.append(f"High-frequency tables: {', '.join(high_freq_tables)}")
        
        # Storage Patterns section
        output.append("\nStorage Patterns:")
        output.append("----------------")
        large_tables = [t for t, p in self.schema.graph.nodes(data=True) 
                       if p['size'] > 10000]
        if large_tables:
            output.append(f"Large tables: {', '.join(large_tables)}")
        
        return "\n".join(output)
    
    def suggest_optimizations(self) -> List[str]:
        """
        Generate optimization suggestions.
        
        Returns:
            List of optimization suggestions
        """
        suggestions = []
        
        # Analyze table sizes
        for table, props in self.schema.graph.nodes(data=True):
            if props['size'] > 100000:
                suggestions.append(f"Consider partitioning table {table} (size: {props['size']:,})")
            if props['column_count'] > 20:
                suggestions.append(f"Consider normalizing table {table} (columns: {props['column_count']})")
        
        # Analyze query patterns
        for table, props in self.schema.graph.nodes(data=True):
            if props['query_frequency'] > 0.8 and len(props['indexes']) < 3:
                suggestions.append(f"Add more indexes to frequently queried table {table}")
        
        # Analyze relationships
        for source, target, props in self.schema.graph.edges(data=True):
            if props['query_frequency'] > 0.7 and props['selectivity'] < 0.5:
                suggestions.append(f"Consider optimizing relationship {source}->{target} (low selectivity)")
        
        return suggestions
    
    def generate_optimization_comparison(self, original_features: np.ndarray, optimized_features: np.ndarray) -> str:
        """
        Generate a before/after comparison of schema optimization.
        
        Args:
            original_features: Original feature vector
            optimized_features: Optimized feature vector
            
        Returns:
            String containing before/after comparison
        """
        output = ["Schema Optimization Comparison", "================================"]
        
        # Tables section
        output.append("\nTables:")
        output.append("------")
        n_tables = len(self.schema.graph.nodes)
        for i in range(n_tables):
            table = list(self.schema.graph.nodes)[i]
            base_idx = i * 4
            
            # Size comparison
            orig_size = original_features[base_idx]
            opt_size = optimized_features[base_idx]
            size_change = opt_size - orig_size
            size_impact = "↑" if size_change > 0 else "↓" if size_change < 0 else "="
            
            # Column count comparison
            orig_cols = original_features[base_idx + 1]
            opt_cols = optimized_features[base_idx + 1]
            col_change = opt_cols - orig_cols
            col_impact = "↑" if col_change > 0 else "↓" if col_change < 0 else "="
            
            # Query frequency comparison
            orig_query = original_features[base_idx + 2]
            opt_query = optimized_features[base_idx + 2]
            query_change = opt_query - orig_query
            query_impact = "↑" if query_change > 0 else "↓" if query_change < 0 else "="
            
            # Update frequency comparison
            orig_update = original_features[base_idx + 3]
            opt_update = optimized_features[base_idx + 3]
            update_change = opt_update - orig_update
            update_impact = "↑" if update_change > 0 else "↓" if update_change < 0 else "="
            
            output.append(f"\n{table}:")
            output.append(f"  Size: {orig_size:.4f} -> {opt_size:.4f} ({size_impact})")
            output.append(f"  Columns: {orig_cols:.4f} -> {opt_cols:.4f} ({col_impact})")
            output.append(f"  Query Freq: {orig_query:.4f} -> {opt_query:.4f} ({query_impact})")
            output.append(f"  Update Freq: {orig_update:.4f} -> {opt_update:.4f} ({update_impact})")
            
            # Add interpretation
            if size_change < 0:
                output.append("  → Table size optimized")
            if col_change < 0:
                output.append("  → Schema simplified")
            if query_change > 0:
                output.append("  → Query access improved")
        
        # Relationships section
        output.append("\nRelationships:")
        output.append("-------------")
        base_idx = n_tables * 4
        for i, (source, target) in enumerate(self.schema.graph.edges):
            edge_idx = base_idx + i * 3
            
            # Query frequency comparison
            orig_query = original_features[edge_idx]
            opt_query = optimized_features[edge_idx]
            query_change = opt_query - orig_query
            query_impact = "↑" if query_change > 0 else "↓" if query_change < 0 else "="
            
            # Selectivity comparison
            orig_sel = original_features[edge_idx + 1]
            opt_sel = optimized_features[edge_idx + 1]
            sel_change = opt_sel - orig_sel
            sel_impact = "↑" if sel_change > 0 else "↓" if sel_change < 0 else "="
            
            # Cardinality comparison
            orig_card = original_features[edge_idx + 2]
            opt_card = optimized_features[edge_idx + 2]
            card_change = opt_card - orig_card
            card_impact = "↑" if card_change > 0 else "↓" if card_change < 0 else "="
            
            output.append(f"\n{source} -> {target}:")
            output.append(f"  Query Freq: {orig_query:.4f} -> {opt_query:.4f} ({query_impact})")
            output.append(f"  Selectivity: {orig_sel:.4f} -> {opt_sel:.4f} ({sel_impact})")
            output.append(f"  Cardinality: {orig_card:.4f} -> {opt_card:.4f} ({card_impact})")
            
            # Add interpretation
            if sel_change > 0:
                output.append("  → Relationship efficiency improved")
            if card_change < 0:
                output.append("  → Relationship simplified")
        
        # Summary section
        output.append("\nOptimization Summary:")
        output.append("-------------------")
        
        # Calculate overall changes
        size_reduction = np.mean([opt - orig for orig, opt in zip(original_features[:n_tables*4:4], optimized_features[:n_tables*4:4])])
        col_reduction = np.mean([opt - orig for orig, opt in zip(original_features[1:n_tables*4:4], optimized_features[1:n_tables*4:4])])
        query_improvement = np.mean([opt - orig for orig, opt in zip(original_features[2:n_tables*4:4], optimized_features[2:n_tables*4:4])])
        
        output.append(f"Average Size Reduction: {size_reduction:.4f}")
        output.append(f"Average Column Reduction: {col_reduction:.4f}")
        output.append(f"Average Query Improvement: {query_improvement:.4f}")
        
        return "\n".join(output)
    
    def visualize_schema(self, original_features: Optional[np.ndarray] = None, 
                        optimized_features: Optional[np.ndarray] = None,
                        output_dir: str = "visualizations") -> None:
        """
        Generate visualizations of the database schema.
        
        Args:
            original_features: Original feature vector (for before visualization)
            optimized_features: Optimized feature vector (for after visualization)
            output_dir: Directory to save visualizations
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Before optimization
        self._draw_schema(ax1, original_features, "Original Schema")
        
        # After optimization
        self._draw_schema(ax2, optimized_features, "Optimized Schema")
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "schema_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _draw_schema(self, ax: plt.Axes, features: Optional[np.ndarray], title: str) -> None:
        """
        Draw a single schema visualization.
        
        Args:
            ax: Matplotlib axes to draw on
            features: Feature vector (if None, use original schema)
            title: Title for the visualization
        """
        # Create a new graph for visualization
        G = nx.DiGraph()
        
        # Add nodes
        n_tables = len(self.schema.graph.nodes)
        for i, (table, props) in enumerate(self.schema.graph.nodes(data=True)):
            # Calculate node size based on table size
            if features is not None:
                size = abs(features[i * 4]) * 3000  # Increased scale for better visibility
            else:
                # Use original schema properties
                size = min(props['size'] / 200000, 1.0) * 3000  # Same scaling as features
            
            # Calculate color based on query frequency
            if features is not None:
                query_freq = abs(features[i * 4 + 2])
            else:
                query_freq = props['query_frequency']  # Use original query frequency
            
            # Add node with properties
            G.add_node(table, 
                      size=size,
                      color=query_freq,
                      column_count=props['column_count'])
        
        # Add edges
        for i, (source, target, props) in enumerate(self.schema.graph.edges(data=True)):
            # Calculate edge weight based on query frequency
            if features is not None:
                edge_idx = n_tables * 4 + i * 3
                weight = abs(features[edge_idx]) if edge_idx < len(features) else props['query_frequency']
            else:
                weight = props['query_frequency']
            
            # Add edge with properties
            G.add_edge(source, target, weight=weight)
        
        # Set up the layout with fixed random seed for consistency
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        # Draw nodes with a colormap that shows query frequency
        nodes = nx.draw_networkx_nodes(G, pos, 
                                     node_size=[G.nodes[node]['size'] for node in G.nodes()],
                                     node_color=[G.nodes[node]['color'] for node in G.nodes()],
                                     cmap=plt.cm.YlOrRd,  # Changed to YlOrRd for better visibility
                                     alpha=0.7,
                                     ax=ax)
        
        # Draw edges with varying widths
        nx.draw_networkx_edges(G, pos,
                             width=[G.edges[edge]['weight'] * 3 for edge in G.edges()],  # Increased width
                             edge_color='gray',
                             alpha=0.5,
                             ax=ax)
        
        # Add labels with column counts
        labels = {node: f"{node}\n({G.nodes[node]['column_count']} cols)" 
                 for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
        
        # Add title
        ax.set_title(title, pad=20)
        
        # Add legend
        legend_elements = [
            Rectangle((0, 0), 1, 1, facecolor='white', edgecolor='black', 
                     label='Node size: Table size'),
            Rectangle((0, 0), 1, 1, facecolor='#ffeda0', edgecolor='black',
                     label='Node color: Low query frequency'),
            Rectangle((0, 0), 1, 1, facecolor='#f03b20', edgecolor='black',
                     label='Node color: High query frequency'),
            Rectangle((0, 0), 1, 1, facecolor='white', edgecolor='black',
                     label='Edge width: Relationship strength')
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1))
        
        # Set axis limits and remove axes
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.axis('off') 