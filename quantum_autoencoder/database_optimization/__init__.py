"""
Database schema optimization using quantum autoencoders.

This package provides functionality for optimizing database schemas using
quantum computing techniques, specifically quantum autoencoders.
"""

from .schema.graph import SchemaGraph, NodeProperties, EdgeProperties
from .schema.metrics import SchemaMetrics
from .schema.analyzer import SchemaAnalyzer
from .quantum.optimizer import QuantumSchemaOptimizer
from .quantum.state import QuantumStateConverter
from .quantum.circuit import QuantumCircuitBuilder

__all__ = [
    'SchemaGraph',
    'NodeProperties',
    'EdgeProperties',
    'SchemaMetrics',
    'SchemaAnalyzer',
    'QuantumSchemaOptimizer',
    'QuantumStateConverter',
    'QuantumCircuitBuilder'
] 