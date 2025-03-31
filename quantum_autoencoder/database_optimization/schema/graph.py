"""
Graph-based representation of database schemas.

This module provides functionality for representing database schemas as directed graphs,
where tables are nodes and relationships are edges with properties.
"""

import networkx as nx
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class NodeProperties:
    """Properties of a table node in the schema graph."""
    size: int
    column_count: int
    query_frequency: float
    update_frequency: float
    primary_key: Optional[str] = None
    indexes: List[str] = None

@dataclass
class EdgeProperties:
    """Properties of a relationship edge in the schema graph."""
    cardinality: str  # '1:1', '1:N', 'N:M'
    query_frequency: float
    selectivity: float
    foreign_key: Optional[str] = None

class SchemaGraph:
    """Directed graph representation of a database schema."""
    
    def __init__(self):
        """Initialize an empty schema graph."""
        self.graph = nx.DiGraph()
        
    def add_table(self, name: str, properties: NodeProperties) -> None:
        """
        Add a table node to the graph.
        
        Args:
            name: Name of the table
            properties: Properties of the table
        """
        self.graph.add_node(name, **properties.__dict__)
        
    def add_relationship(self, source: str, target: str, properties: EdgeProperties) -> None:
        """
        Add a relationship edge to the graph.
        
        Args:
            source: Source table name
            target: Target table name
            properties: Properties of the relationship
        """
        self.graph.add_edge(source, target, **properties.__dict__)
        
    def get_table_properties(self, name: str) -> NodeProperties:
        """
        Get properties of a table node.
        
        Args:
            name: Name of the table
            
        Returns:
            Properties of the table
        """
        props = self.graph.nodes[name]
        return NodeProperties(**props)
        
    def get_relationship_properties(self, source: str, target: str) -> EdgeProperties:
        """
        Get properties of a relationship edge.
        
        Args:
            source: Source table name
            target: Target table name
            
        Returns:
            Properties of the relationship
        """
        props = self.graph.edges[source, target]
        return EdgeProperties(**props)
        
    def get_connected_tables(self, table: str) -> List[str]:
        """
        Get all tables connected to a given table.
        
        Args:
            table: Name of the table
            
        Returns:
            List of connected table names
        """
        return list(self.graph.neighbors(table))
        
    def get_incoming_tables(self, table: str) -> List[str]:
        """
        Get all tables with relationships pointing to the given table.
        
        Args:
            table: Name of the table
            
        Returns:
            List of incoming table names
        """
        return list(self.graph.predecessors(table))
        
    def get_outgoing_tables(self, table: str) -> List[str]:
        """
        Get all tables that the given table has relationships to.
        
        Args:
            table: Name of the table
            
        Returns:
            List of outgoing table names
        """
        return list(self.graph.successors(table))
        
    def get_shortest_path(self, source: str, target: str) -> List[str]:
        """
        Get the shortest path between two tables.
        
        Args:
            source: Source table name
            target: Target table name
            
        Returns:
            List of table names in the path
        """
        return nx.shortest_path(self.graph, source, target)
        
    def get_all_paths(self, source: str, target: str) -> List[List[str]]:
        """
        Get all possible paths between two tables.
        
        Args:
            source: Source table name
            target: Target table name
            
        Returns:
            List of paths, where each path is a list of table names
        """
        return list(nx.all_simple_paths(self.graph, source, target))
        
    def is_cyclic(self) -> bool:
        """
        Check if the schema graph contains cycles.
        
        Returns:
            True if the graph is cyclic, False otherwise
        """
        return not nx.is_directed_acyclic_graph(self.graph)
        
    def get_cycles(self) -> List[List[str]]:
        """
        Get all cycles in the schema graph.
        
        Returns:
            List of cycles, where each cycle is a list of table names
        """
        return list(nx.simple_cycles(self.graph))
        
    def get_strongly_connected_components(self) -> List[List[str]]:
        """
        Get strongly connected components in the schema graph.
        
        Returns:
            List of components, where each component is a list of table names
        """
        return list(nx.strongly_connected_components(self.graph)) 